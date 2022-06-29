import os
import inferenceConfig3
from mrcnn import visualize
from mrcnn import model3
import numpy as np
from numba import jit, cuda
import math
import tensorflow as tf
import time



@cuda.jit
def ZeropaddingKernel(input_image, output_image, padding):
  c, r, channel = cuda.grid(3)

  if channel < output_image.shape[3] and r < input_image.shape[1] and c < input_image.shape[2]:
    output_image[0, r+padding, c+padding, channel] += input_image[0,r,c,channel]


def Zeropadding(input_image, padding):
  outputShape = (input_image.shape[0],input_image.shape[1]+2*padding,input_image.shape[2]+2*padding,input_image.shape[3])
  outImage = np.zeros(outputShape)

  block_size = (32,32)
  grid_size = (math.ceil(input_image.shape[2]/block_size[0]),
                math.ceil(input_image.shape[1]/block_size[1]),
                input_image.shape[3])
  ZeropaddingKernel[grid_size, block_size](input_image, outImage, padding)
  return outImage

def readKernelFiltersAndBias():
  import h5py
  with h5py.File('mask_rcnn_coco.h5', 'r') as f:
    base_items = list(f.items())
    conv1 = f.get('conv1')

    conv1Child= conv1.get('conv1')

    kernel = np.array(conv1Child.get('kernel:0'))
    bias = np.array(conv1Child.get('bias:0'))
    #7 7 3 64 -> 64 7 7 3
    kernel = np.transpose(kernel, [3,0,1,2])
    return kernel, bias

def readBatchNorm_C1():
  import h5py
  with h5py.File('mask_rcnn_coco.h5', 'r') as f:
    base_items = list(f.items())
    bn_conv1 = f.get('bn_conv1')

    bn_conv1Child= bn_conv1.get('bn_conv1')

    beta = np.array(bn_conv1Child.get('beta:0'))
    gamma = np.array(bn_conv1Child.get('gamma:0'))
    moving_mean = np.array(bn_conv1Child.get('moving_mean:0'))
    moving_variance = np.array(bn_conv1Child.get('moving_variance:0'))
    return beta, gamma, moving_mean, moving_variance
  

@cuda.jit
def ConvolutionKernel(input_image, output_image, filters, bias, stride):
  c, r, n_f = cuda.grid(3)

  filter_size = filters.shape[1]
  (_, img_h, img_w, n_c) = input_image.shape
  (_, out_h, out_w, _) = output_image.shape

  if n_f < filters.shape[0] and r < out_h and c < out_w:
    row = stride*r
    column = stride*c 
    if row + filter_size <= img_h and column + filter_size <= img_w:
      # output_image[0, r, c, n_f] = np.sum(filters[n_f] * input_image[0, row:row + filters.shape[1], column:column + filters.shape[2],:]) + bias[n_f]
      for channel in range(0, n_c):
        for filterR, inR in zip(range(0, filter_size), range(row,row+filter_size)):
          for filterC,inC in zip(range(0, filter_size), range(column,column+filter_size)):
            output_image[0, r, c, n_f] += filters[n_f,filterR,filterC,channel] * input_image[0, inR, inC, channel]
      output_image[0, r, c, n_f] += bias[n_f]


def convolution(image, Filter, bias, stride=1):
    # convolution of input image with a filter of dimensions(n_f,n_c,f,f)
    # n_f is number filters
    # n_fc is number channels
    # f,f are height & width

    # image dimensions(n_c, image_h, image_w)
    # n_c is number channels in image
    # img_h is height of image
    # img_w is width of image

    (_, img_h, img_w, n_c) = image.shape
    (n_f, f, f, n_fc) = Filter.shape

    # output dimensions after convolution
    out_h = int((img_h - f) / stride) + 1  # height of output matrix
    out_w = int((img_h - f) / stride) + 1  # width of output matrix
    # n_f will be the depth of the matrix

    out = np.zeros((1, out_h, out_w, n_f))

    block_size = (32,32)
    grid_size = (math.ceil(out.shape[2]/block_size[0]),
                math.ceil(out.shape[1]/block_size[1]),
                out.shape[3])
    
    Filter = np.ascontiguousarray(Filter, dtype=np.float32)
    ConvolutionKernel[grid_size, block_size](image, out, Filter, bias, stride)

    return out

@cuda.jit
def batchNorm_kernel(img_conv1, beta, gamma, moving_mean, moving_variance):
  c, r, f = cuda.grid(3)

  (_, img_h, img_w, n_f) = img_conv1.shape
  if f < n_f and r < img_h and c < img_w:
    X_hat = (img_conv1[0, r, c, f] - moving_mean[f]) / math.sqrt(moving_variance[f])
    img_conv1[0, r, c, f] = gamma[f] * X_hat + beta[f]


def batchNorm(img_conv1, beta, gamma, moving_mean, moving_variance):
  (_, img_h, img_w, n_f) = img_conv1.shape

  block_size = (32,32)
  grid_size = ((math.ceil(img_conv1.shape[2]/block_size[0]),
                math.ceil(img_conv1.shape[1]/block_size[1]),
                n_f))

  #out = np.zeros((1, img_h, img_w, n_f))
  batchNorm_kernel[grid_size, block_size](img_conv1, beta, gamma, moving_mean, moving_variance)
  return img_conv1

@cuda.jit
def Activation_Relu_kernel(input, input_h, input_w, n_f):
  c, r, f = cuda.grid(3)

  if f < n_f and r < input_h and c < input_w:
    if input[0, r, c, f] < 0:
      input[0, r, c, f] = 0

def Activation_Relu(input):
  (n, input_h, input_w, n_f) = input.shape
  block_size = (32,32)
  grid_size = ((math.ceil(input.shape[2]/block_size[0]),
                math.ceil(input.shape[1]/block_size[1]),
                n_f))
  Activation_Relu_kernel[grid_size, block_size](input, input_h, input_w, n_f)
  return input

@cuda.jit
def PaddingBeforeMaxPoolKernel(input, output):
  c, r, channel = cuda.grid(3)

  if channel < output.shape[3] and r < input.shape[1] and c < input.shape[2]:
    output[0,r,c,channel] = input[0,r,c,channel]

def PaddingBeforeMaxPool(input):
  outputShape = (input.shape[0],input.shape[1]+1,input.shape[2]+1,input.shape[3])
  outImage = np.zeros(outputShape)
  block_size = (32,32)
  grid_size = (math.ceil(input.shape[2]/block_size[0]),
                math.ceil(input.shape[1]/block_size[1]),
                input.shape[3])
  
  PaddingBeforeMaxPoolKernel[grid_size,block_size](input, outImage)
  return outImage

@cuda.jit
def MaxPool2DKernel(input, output, pool_size, stride):
  c,r,channel = cuda.grid(3)

  (_, img_h, img_w, n_c) = input.shape
  (_, out_h, out_w, _) = output.shape

  if r < out_h and c < out_w and channel < n_c:
    row = stride*r
    column = stride*c 
    if row + pool_size <= img_h and column + pool_size <= img_w:
      max = 0
      for inR in range(row,row+pool_size):
        for inC in range(column,column+pool_size):
          if input[0,inR,inC,channel] > max: max = input[0,inR,inC,channel]
      output[0,r,c,channel] = max      


def MaxPool2D(input, pool_size, stride):
  (_, img_h, img_w, n_c) = input.shape

  input = PaddingBeforeMaxPool(input)

  # output dimensions after maxpool
  out_h = int((img_h - pool_size + 1) / stride) + 1  # height of output matrix
  out_w = int((img_h - pool_size + 1) / stride) + 1  # width of output matrix

  out = np.zeros((1, out_h, out_w, n_c))

  block_size = (32,32)
  grid_size = (math.ceil(out.shape[2]/block_size[0]),
                math.ceil(out.shape[1]/block_size[1]),
                out.shape[3])
  MaxPool2DKernel[grid_size,block_size](input, out, pool_size, stride)

  return out


tic = time.perf_counter()
inferenceConfig3.processed_input_image = Zeropadding(inferenceConfig3.processed_input_image, 3)
toc = time.perf_counter()
zeroPaddingTime = toc - tic
print("Zeropadding timer:", {zeroPaddingTime}, "seconds")

kernels, bias = readKernelFiltersAndBias()
tic = time.perf_counter()
inferenceConfig3.processed_input_image=convolution(inferenceConfig3.processed_input_image, kernels, bias, 2)
toc = time.perf_counter()
convTime = toc - tic
print("Convolution timer:", {convTime}, "seconds")

beta, gamma, moving_mean, moving_variance = readBatchNorm_C1()
tic = time.perf_counter()
inferenceConfig3.processed_input_image=batchNorm(inferenceConfig3.processed_input_image, beta, gamma, moving_mean, moving_variance)
toc = time.perf_counter()
batchNormTime = toc - tic
print("BatchNorm timer:", {batchNormTime}, "seconds")

tic = time.perf_counter()
inferenceConfig3.processed_input_image=Activation_Relu(inferenceConfig3.processed_input_image)
toc = time.perf_counter()
reluTime = toc - tic
print("Relu timer:", {reluTime}, "seconds")

tic = time.perf_counter()
inferenceConfig3.processed_input_image=MaxPool2D(inferenceConfig3.processed_input_image, 3 ,2)
toc = time.perf_counter()
maxpoolTime = toc - tic
print("Maxpooling timer (with padding):", {maxpoolTime}, "seconds")
print ("C1 timer: ", {zeroPaddingTime+convTime+batchNormTime+reluTime+maxpoolTime}, "seconds")

# Run detection
tic = time.perf_counter()
results = model3.detect([inferenceConfig3.image], inferenceConfig3.processed_input_image, verbose=0)
toc = time.perf_counter()
print("Detect time:", {toc - tic}, "seconds")

r = results[0]

# Visualize results
visualize.display_instances(inferenceConfig3.image, 
                    r['rois'], 
                    r['masks'], 
                    r['class_ids'], 
                    inferenceConfig3.CLASS_NAMES, 
                    scores=r['scores'],
                    save_fig_path='output_parallel_1.png')