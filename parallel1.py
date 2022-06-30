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
def Add_kernel(input_1, input_2):
  c, r, f = cuda.grid(3)

  (a_1, b_1, c_1, d_1) = input_1.shape
  (a_2, b_2, c_2, d_2) = input_2.shape

  if c < b_1 and r < c_1 and f < d_1:
    input_1[0, r, c, f] = input_1[0, r, c, f] + input_2[0, r, c, f]


def Add(input_1, input_2):
  (a_1, b_1, c_1, d_1) = input_1.shape
  (a_2, b_2, c_2, d_2) = input_2.shape

  block_size = (32,32)
  grid_size = ((math.ceil(input_1.shape[2]/block_size[0]),
                math.ceil(input_1.shape[1]/block_size[1]),
                d_1))

  Add_kernel[grid_size, block_size](input_1, input_2)
  return input_1


@cuda.jit
def Activation_Relu_kernel(input, input_h, input_w, n_f):
  c, r, f = cuda.grid(3)

  if f < n_f and r < input_h and r < input_w:
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

################################################################################
#C1

kernels, bias = readKernelFiltersAndBias()
beta, gamma, moving_mean, moving_variance = readBatchNorm_C1()
def C1_resnet(input_image):
  input_image = Zeropadding(input_image, 3)
  input_image = convolution(input_image, kernels, bias, 2)
  input_image = batchNorm(input_image, beta, gamma, moving_mean, moving_variance)
  input_image = Activation_Relu(input_image)
  input_image = MaxPool2D(input_image, 3, 2)
  return input_image

#################################################################################
#conv_block

def readKernelFiltersAndBias_C2():
  import h5py
  with h5py.File('mask_rcnn_coco.h5', 'r') as f:
    base_items = list(f.items())

    res2a_branch2a = f.get('res2a_branch2a')
    res2a_branch2a_Child = res2a_branch2a.get('res2a_branch2a')
    kernel_2a = np.array(res2a_branch2a_Child.get('kernel:0'))
    bias_2a = np.array(res2a_branch2a_Child.get('bias:0'))
    kernel_2a = np.transpose(kernel_2a, [3,0,1,2])

    res2a_branch2b = f.get('res2a_branch2b')
    res2a_branch2b_Child = res2a_branch2b.get('res2a_branch2b')
    kernel_2b = np.array(res2a_branch2b_Child.get('kernel:0'))
    bias_2b = np.array(res2a_branch2b_Child.get('bias:0'))
    kernel_2b = np.transpose(kernel_2b, [3,0,1,2])
    
    res2a_branch2c = f.get('res2a_branch2c')
    res2a_branch2c_Child = res2a_branch2c.get('res2a_branch2c')
    kernel_2c = np.array(res2a_branch2c_Child.get('kernel:0'))
    bias_2c = np.array(res2a_branch2c_Child.get('bias:0'))
    kernel_2c = np.transpose(kernel_2c, [3,0,1,2])

    res2a_branch1 = f.get('res2a_branch1')
    res2a_branch1_Child = res2a_branch1.get('res2a_branch1')
    kernel_2_1 = np.array(res2a_branch1_Child.get('kernel:0'))
    bias_2_1 = np.array(res2a_branch1_Child.get('bias:0'))
    kernel_2_1 = np.transpose(kernel_2_1, [3,0,1,2])
    return kernel_2a, bias_2a, kernel_2b, bias_2b, kernel_2c, bias_2c, kernel_2_1, bias_2_1


def readBatchNorm_C2():
  import h5py
  with h5py.File('mask_rcnn_coco.h5', 'r') as f:
    base_items = list(f.items())

    beta_C2 = []
    gamma_C2 = []
    moving_mean_C2 = []
    moving_variance_C2 = []

    bn2a_branch2a = f.get('bn2a_branch2a')
    bn2a_branch2a_Child = bn2a_branch2a.get('bn2a_branch2a')
    beta = np.array(bn2a_branch2a_Child.get('beta:0'))
    gamma = np.array(bn2a_branch2a_Child.get('gamma:0'))
    moving_mean = np.array(bn2a_branch2a_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2a_branch2a_Child.get('moving_variance:0'))

    beta_C2.append(beta)
    gamma_C2.append(gamma)
    moving_mean_C2.append(moving_mean)
    moving_variance_C2.append(moving_variance)

    #####################################

    bn2a_branch2b = f.get('bn2a_branch2b')
    bn2a_branch2b_Child = bn2a_branch2b.get('bn2a_branch2b')
    beta = np.array(bn2a_branch2b_Child.get('beta:0'))
    gamma = np.array(bn2a_branch2b_Child.get('gamma:0'))
    moving_mean = np.array(bn2a_branch2b_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2a_branch2b_Child.get('moving_variance:0'))

    beta_C2.append(beta)
    gamma_C2.append(gamma)
    moving_mean_C2.append(moving_mean)
    moving_variance_C2.append(moving_variance)

    #############################################

    bn2a_branch2c = f.get('bn2a_branch2c')
    bn2a_branch2c_Child = bn2a_branch2c.get('bn2a_branch2c')
    beta = np.array(bn2a_branch2c_Child.get('beta:0'))
    gamma = np.array(bn2a_branch2c_Child.get('gamma:0'))
    moving_mean = np.array(bn2a_branch2c_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2a_branch2c_Child.get('moving_variance:0'))

    beta_C2.append(beta)
    gamma_C2.append(gamma)
    moving_mean_C2.append(moving_mean)
    moving_variance_C2.append(moving_variance)

    ##############################################

    bn2a_branch1 = f.get('bn2a_branch1')
    bn2a_branch1_Child = bn2a_branch1.get('bn2a_branch1')
    beta = np.array(bn2a_branch1_Child.get('beta:0'))
    gamma = np.array(bn2a_branch1_Child.get('gamma:0'))
    moving_mean = np.array(bn2a_branch1_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2a_branch1_Child.get('moving_variance:0'))

    beta_C2.append(beta)
    gamma_C2.append(gamma)
    moving_mean_C2.append(moving_mean)
    moving_variance_C2.append(moving_variance)

    return beta_C2, gamma_C2, moving_mean_C2, moving_variance_C2

def conv_block_C2(C1_res, kernel_2a, bias_2a, kernel_2b, bias_2b, kernel_2c, bias_2c, kernel_2_1, bias_2_1, beta_C2, gamma_C2, moving_mean_C2, moving_variance_C2, strides = 1):

  x = convolution(C1_res, kernel_2a, bias_2a, strides)
  x = batchNorm(x, beta_C2[0], gamma_C2[0], moving_mean_C2[0], moving_variance_C2[0])
  x = Activation_Relu(x)

  x = Zeropadding(x, 1)
  x = convolution(x, kernel_2b, bias_2b, strides)
  x = batchNorm(x, beta_C2[1], gamma_C2[1], moving_mean_C2[1], moving_variance_C2[1])
  x = Activation_Relu(x)

  x = convolution(x, kernel_2c, bias_2c, strides)
  x = batchNorm(x, beta_C2[2], gamma_C2[2], moving_mean_C2[2], moving_variance_C2[2])

  shortcut = convolution(C1_res, kernel_2_1, bias_2_1, strides)
  shortcut = batchNorm(shortcut, beta_C2[3], gamma_C2[3], moving_mean_C2[3], moving_variance_C2[3])

  x = Add(x, shortcut)
  x = Activation_Relu(x)
  return x

########################################################################################
#identity_block C2

def readKernelFiltersAndBias_C2_identity():
  import h5py
  with h5py.File('mask_rcnn_coco.h5', 'r') as f:
    base_items = list(f.items())

    #identity_C2_1
    res2b_branch2a = f.get('res2b_branch2a')
    res2b_branch2a_Child = res2b_branch2a.get('res2b_branch2a')
    kernel_2a_1 = np.array(res2b_branch2a_Child.get('kernel:0'))
    bias_2a_1 = np.array(res2b_branch2a_Child.get('bias:0'))
    kernel_2a_1 = np.transpose(kernel_2a_1, [3,0,1,2])

    res2b_branch2b = f.get('res2b_branch2b')
    res2b_branch2b_Child = res2b_branch2b.get('res2b_branch2b')
    kernel_2b_1 = np.array(res2b_branch2b_Child.get('kernel:0'))
    bias_2b_1 = np.array(res2b_branch2b_Child.get('bias:0'))
    kernel_2b_1 = np.transpose(kernel_2b_1, [3,0,1,2])
    
    res2b_branch2c = f.get('res2b_branch2c')
    res2b_branch2c_Child = res2b_branch2c.get('res2b_branch2c')
    kernel_2c_1 = np.array(res2b_branch2c_Child.get('kernel:0'))
    bias_2c_1 = np.array(res2b_branch2c_Child.get('bias:0'))
    kernel_2c_1 = np.transpose(kernel_2c_1, [3,0,1,2])

    #identity_C2_2
    res2c_branch2a = f.get('res2c_branch2a')
    res2c_branch2a_Child = res2c_branch2a.get('res2c_branch2a')
    kernel_2a_2 = np.array(res2c_branch2a_Child.get('kernel:0'))
    bias_2a_2 = np.array(res2c_branch2a_Child.get('bias:0'))
    kernel_2a_2 = np.transpose(kernel_2a_2, [3,0,1,2])

    res2c_branch2b = f.get('res2c_branch2b')
    res2c_branch2b_Child = res2c_branch2b.get('res2c_branch2b')
    kernel_2b_2 = np.array(res2c_branch2b_Child.get('kernel:0'))
    bias_2b_2 = np.array(res2b_branch2b_Child.get('bias:0'))
    kernel_2b_2 = np.transpose(kernel_2b_2, [3,0,1,2])
    
    res2c_branch2c = f.get('res2c_branch2c')
    res2c_branch2c_Child = res2c_branch2c.get('res2c_branch2c')
    kernel_2c_2 = np.array(res2c_branch2c_Child.get('kernel:0'))
    bias_2c_2 = np.array(res2c_branch2c_Child.get('bias:0'))
    kernel_2c_2 = np.transpose(kernel_2c_2, [3,0,1,2])

    return kernel_2a_1, bias_2a_1, kernel_2b_1, bias_2b_1, kernel_2c_1, bias_2c_1, kernel_2a_2, bias_2a_2, kernel_2b_2, bias_2b_2, kernel_2c_2, bias_2c_2


def readBatchNorm_C2_identity():
  import h5py
  with h5py.File('mask_rcnn_coco.h5', 'r') as f:
    base_items = list(f.items())

    beta_C2_identity = []
    gamma_C2_identity = []
    moving_mean_C2_identity = []
    moving_variance_C2_identity = []

    bn2b_branch2a = f.get('bn2b_branch2a')
    bn2b_branch2a_Child = bn2b_branch2a.get('bn2b_branch2a')
    beta = np.array(bn2b_branch2a_Child.get('beta:0'))
    gamma = np.array(bn2b_branch2a_Child.get('gamma:0'))
    moving_mean = np.array(bn2b_branch2a_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2b_branch2a_Child.get('moving_variance:0'))
    beta_C2_identity.append(beta)
    gamma_C2_identity.append(gamma)
    moving_mean_C2_identity.append(moving_mean)
    moving_variance_C2_identity.append(moving_variance)

    bn2b_branch2b = f.get('bn2b_branch2b')
    bn2b_branch2b_Child = bn2b_branch2b.get('bn2b_branch2b')
    beta = np.array(bn2b_branch2b_Child.get('beta:0'))
    gamma = np.array(bn2b_branch2b_Child.get('gamma:0'))
    moving_mean = np.array(bn2b_branch2b_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2b_branch2b_Child.get('moving_variance:0'))
    beta_C2_identity.append(beta)
    gamma_C2_identity.append(gamma)
    moving_mean_C2_identity.append(moving_mean)
    moving_variance_C2_identity.append(moving_variance)

    bn2b_branch2c = f.get('bn2b_branch2c')
    bn2b_branch2c_Child = bn2b_branch2c.get('bn2b_branch2c')
    beta = np.array(bn2b_branch2c_Child.get('beta:0'))
    gamma = np.array(bn2b_branch2c_Child.get('gamma:0'))
    moving_mean = np.array(bn2b_branch2c_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2b_branch2c_Child.get('moving_variance:0'))
    beta_C2_identity.append(beta)
    gamma_C2_identity.append(gamma)
    moving_mean_C2_identity.append(moving_mean)
    moving_variance_C2_identity.append(moving_variance)

    bn2c_branch2a = f.get('bn2c_branch2a')
    bn2c_branch2a_Child = bn2c_branch2a.get('bn2c_branch2a')
    beta = np.array(bn2c_branch2a_Child.get('beta:0'))
    gamma = np.array(bn2c_branch2a_Child.get('gamma:0'))
    moving_mean = np.array(bn2c_branch2a_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2c_branch2a_Child.get('moving_variance:0'))
    beta_C2_identity.append(beta)
    gamma_C2_identity.append(gamma)
    moving_mean_C2_identity.append(moving_mean)
    moving_variance_C2_identity.append(moving_variance)

    bn2c_branch2b = f.get('bn2c_branch2b')
    bn2c_branch2b_Child = bn2c_branch2b.get('bn2c_branch2b')
    beta = np.array(bn2c_branch2b_Child.get('beta:0'))
    gamma = np.array(bn2c_branch2b_Child.get('gamma:0'))
    moving_mean = np.array(bn2c_branch2b_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2c_branch2b_Child.get('moving_variance:0'))
    beta_C2_identity.append(beta)
    gamma_C2_identity.append(gamma)
    moving_mean_C2_identity.append(moving_mean)
    moving_variance_C2_identity.append(moving_variance)

    bn2c_branch2c = f.get('bn2c_branch2c')
    bn2c_branch2c_Child = bn2c_branch2c.get('bn2c_branch2c')
    beta = np.array(bn2c_branch2c_Child.get('beta:0'))
    gamma = np.array(bn2c_branch2c_Child.get('gamma:0'))
    moving_mean = np.array(bn2c_branch2c_Child.get('moving_mean:0'))
    moving_variance = np.array(bn2c_branch2c_Child.get('moving_variance:0'))
    beta_C2_identity.append(beta)
    gamma_C2_identity.append(gamma)
    moving_mean_C2_identity.append(moving_mean)
    moving_variance_C2_identity.append(moving_variance)

    return beta_C2_identity, gamma_C2_identity, moving_mean_C2_identity, moving_variance_C2_identity


def identity_block_C2(input, kernel_2a, bias_2a, kernel_2b, bias_2b, kernel_2c, bias_2c, beta_C2_identity, gamma_C2_identity, moving_mean_C2_identity, moving_variance_C2_identity, n_identity, strides = 1):
  
  x = convolution(input, kernel_2a, bias_2a, strides)
  x = batchNorm(x, beta_C2_identity[n_identity], gamma_C2_identity[n_identity], moving_mean_C2_identity[n_identity], moving_variance_C2_identity[n_identity])
  x = Activation_Relu(x)

  x = Zeropadding(x, 1)
  x = convolution(x, kernel_2b, bias_2b, strides)
  x = batchNorm(x, beta_C2_identity[n_identity + 1], gamma_C2_identity[n_identity + 1], moving_mean_C2_identity[n_identity + 1], moving_variance_C2_identity[n_identity + 1])
  x = Activation_Relu(x)

  x = convolution(x, kernel_2c, bias_2c, strides)
  x = batchNorm(x, beta_C2_identity[n_identity + 2], gamma_C2_identity[n_identity + 2], moving_mean_C2_identity[n_identity + 2], moving_variance_C2_identity[n_identity + 2])

  x = Add(x, input)
  x = Activation_Relu(x)
  return x

######################################################################################################

tic = time.perf_counter()
inferenceConfig3.processed_input_image = C1_resnet(inferenceConfig3.processed_input_image)
toc = time.perf_counter()
print("C1 timer:", {toc - tic}, "seconds")


kernel_2a, bias_2a, kernel_2b, bias_2b, kernel_2c, bias_2c, kernel_2_1, bias_2_1 = readKernelFiltersAndBias_C2()
beta_C2, gamma_C2, moving_mean_C2, moving_variance_C2 = readBatchNorm_C2()
kernel_2a_1, bias_2a_1, kernel_2b_1, bias_2b_1, kernel_2c_1, bias_2c_1, kernel_2a_2, bias_2a_2, kernel_2b_2, bias_2b_2, kernel_2c_2, bias_2c_2 = readKernelFiltersAndBias_C2_identity()
beta_C2_identity, gamma_C2_identity, moving_mean_C2_identity, moving_variance_C2_identity = readBatchNorm_C2_identity()

tic = time.perf_counter()
inferenceConfig3.processed_input_image = conv_block_C2(inferenceConfig3.processed_input_image, kernel_2a, bias_2a, kernel_2b, bias_2b, kernel_2c, bias_2c, kernel_2_1, bias_2_1, beta_C2, gamma_C2, moving_mean_C2, moving_variance_C2, strides = 1)
inferenceConfig3.processed_input_image = identity_block_C2(inferenceConfig3.processed_input_image, kernel_2a_1, bias_2a_1, kernel_2b_1, bias_2b_1, kernel_2c_1, bias_2c_1, beta_C2_identity, gamma_C2_identity, moving_mean_C2_identity, moving_variance_C2_identity, n_identity = 0 * 3, strides = 1)
inferenceConfig3.processed_input_image = identity_block_C2(inferenceConfig3.processed_input_image, kernel_2a_2, bias_2a_2, kernel_2b_2, bias_2b_2, kernel_2c_2, bias_2c_2, beta_C2_identity, gamma_C2_identity, moving_mean_C2_identity, moving_variance_C2_identity, n_identity = 1 * 3, strides = 1)
toc = time.perf_counter()
print("C2 timer:", {toc - tic}, "seconds")

# Run detection
results = model3.detect([inferenceConfig3.image], inferenceConfig3.processed_input_image, verbose=0)

r = results[0]

# Visualize results
visualize.display_instances(inferenceConfig3.image, 
                    r['rois'], 
                    r['masks'], 
                    r['class_ids'], 
                    inferenceConfig3.CLASS_NAMES, 
                    scores=r['scores'],
                    save_fig_path='output_parallel_1.png')