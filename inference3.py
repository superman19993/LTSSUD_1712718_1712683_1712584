import os
import inferenceConfig3
from mrcnn import visualize
from mrcnn import model3
import numpy as np
from numba import jit
import math
import tensorflow as tf
import time

@jit
def Zeropadding(input_image, padding):
  outputShape = (input_image.shape[0],input_image.shape[1]+2*padding,input_image.shape[2]+2*padding,input_image.shape[3])
  outImage = np.zeros(outputShape)

  for channel in range(0, outputShape[3]):
    for inputCol in range(0, input_image.shape[2]):
      for inputRow in range(0, input_image.shape[1]):
        outImage[0, inputRow+padding, inputCol+padding, channel]+=input_image[0, inputRow, inputCol, channel]

  return outImage


# def generateConvFilter(filter_num, filter_size):
#   # limit = np.sqrt(1 / float(64))
#   # return np.random.normal(0.0, limit, size=(64, 7, 7, 1))
#   F_in = 49*3
#   F_out = 49 * 64
#   limit = np.sqrt(6 / float(F_in + F_out))
#   W = np.random.uniform(low=-limit, high=limit, size=(64,7,7,1))
#   return W

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
 
@jit
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

    # convolution of image_array with filter yeilds out_array
    # for i in range of no.of filters
    # define a row , out_y variabless to hover along rows of image, out_matrix respectively
    # define a column , out_x variables to hover along columns of image, out_matrix respectively
    # convolution is done in the ranges of image_height to image_width
    for i in range(n_f):
        row = out_row = 0

        while row + f <= img_h:

            column = out_column = 0

            while column + f <= img_w:
                out[0, out_row, out_column, i] = np.sum(Filter[i] * image[0, row: row + f, column: column + f,:]) + bias[i]
                column += stride
                out_column += 1

            row += stride
            out_row += 1

    print(out.shape)

    return out

@jit
def convolution2(image, Filter, bias, stride=1):
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

    # convolution of image_array with filter yeilds out_array
    # for i in range of no.of filters
    # define a row , out_y variabless to hover along rows of image, out_matrix respectively
    # define a column , out_x variables to hover along columns of image, out_matrix respectively
    # convolution is done in the ranges of image_height to image_width
    for i in range(n_f):
        row = out_row = 0
        while row + f <= img_h:
            column = out_column = 0

            while column + f <= img_w:
                for channel in range(0, n_c):
                  for filterR, inR in zip(range(0, f), range(row,row+f)):
                    for filterC, inC in zip(range(0, f), range(column,column+f)):
                      out[0, out_row, out_column, i] += Filter[i,filterR,filterC,channel] * image[0, inR, inC, channel]
                out[0, out_row, out_column, i] += bias[i]
                column += stride
                out_column += 1

            row += stride
            out_row += 1

    print(out.shape)

    return out

@jit
def batchNorm(img_conv1, beta, gamma, moving_mean, moving_variance):
  # n = 1, input_h = 512, input_w = 512, n_f = 64
  (n, input_h, input_w, n_f) = img_conv1.shape
  #out = np.zeros((n, input_h, input_w, n_f))

  for f in range(n_f):
    for row in range(input_h):
      for column in range(input_w):
        X_hat = (img_conv1[0, row, column, f] - moving_mean[f]) / np.sqrt(moving_variance[f])
        img_conv1[0, row, column, f] = gamma[f] * X_hat + beta[f]
  return img_conv1

@jit
def Relu(input):
  # n = 1, input_h = 512, input_w = 512, n_f = 64
  (n, input_h, input_w, n_f) = input.shape
  #out = np.zeros((n, input_h, input_w, n_f))
  for f in range(n_f):
    for row in range(input_h):
      for column in range(input_w):
        if input[0, row, column, f] < 0:
          input[0, row, column, f] = 0

  return input

@jit
def PaddingBeforeMaxPool(input):
  outputShape = (input.shape[0],input.shape[1]+1,input.shape[2]+1,input.shape[3])
  outImage = np.zeros(outputShape)
  for channel in range(0, outputShape[3]):
    for inputCol in range(0, input.shape[2]):
      for inputRow in range(0, input.shape[1]):
        outImage[0, inputRow, inputCol, channel]+=input[0, inputRow, inputCol, channel]

  return outImage

@jit
def MaxPool2D(input, pool_size, stride, padding = 'valid'):
  (_, img_h, img_w, n_c) = input.shape

  if padding =='same' and (img_h-pool_size)%2 != 0: 
    input = PaddingBeforeMaxPool(input)
    out_h = int((img_h - pool_size + 1) / stride) + 1  # height of output matrix
    out_w = int((img_h - pool_size + 1) / stride) + 1  # width of output matrix
  else:
  # output dimensions after convolution
    out_h = int((img_h - pool_size) / stride) + 1  # height of output matrix
    out_w = int((img_h - pool_size) / stride) + 1  # width of output matrix

  out = np.zeros((1, out_h, out_w, n_c))

  for i in range(n_c):
      row = out_row = 0
      while row + pool_size <= img_h:
          column = out_column = 0
          while column + pool_size <= img_w:
              max = 0
              for inR in range(row,row+pool_size):
                for inC in range(column,column+pool_size):
                  if input[0,inR,inC,i] > max: max = input[0,inR,inC,i]
              out[0, out_row, out_column, i] = max
              column += stride
              out_column += 1

          row += stride
          out_row += 1

  print(out.shape)

  return out


tic = time.perf_counter()
inferenceConfig3.processed_input_image = Zeropadding(inferenceConfig3.processed_input_image, 3)
toc = time.perf_counter()
print("Zeropadding timer:", {toc - tic}, "seconds")

kernels, bias = readKernelFiltersAndBias()
tic = time.perf_counter()
inferenceConfig3.processed_input_image=convolution2(inferenceConfig3.processed_input_image, kernels, bias, 2)
toc = time.perf_counter()
print("Convolution timer:", {toc - tic}, "seconds")

beta, gamma, moving_mean, moving_variance = readBatchNorm_C1()
tic = time.perf_counter()
inferenceConfig3.processed_input_image=batchNorm(inferenceConfig3.processed_input_image, beta, gamma, moving_mean, moving_variance)
toc = time.perf_counter()
print("BatchNorm timer:", {toc - tic}, "seconds")

tic = time.perf_counter()
inferenceConfig3.processed_input_image=Relu(inferenceConfig3.processed_input_image)
toc = time.perf_counter()
print("Relu timer:", {toc - tic}, "seconds")

tic = time.perf_counter()
inferenceConfig3.processed_input_image=MaxPool2D(inferenceConfig3.processed_input_image, 3 ,2, 'same')
toc = time.perf_counter()
print("Maxpooling timer (with padding):", {toc - tic}, "seconds")
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
                    save_fig_path='output_host_jit.png')

