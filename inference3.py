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



'''
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

    #print(out.shape)

    return out
'''

@jit
def PaddingBeforeMaxPool(input):
  outputShape = (input.shape[0],input.shape[1]+1,input.shape[2]+1,input.shape[3])
  outImage = np.zeros(outputShape)
  for channel in range(0, outputShape[3]):
    for inputCol in range(0, input.shape[2]):
      for inputRow in range(0, input.shape[1]):
        outImage[0, inputRow, inputCol, channel] += input[0, inputRow, inputCol, channel]

  return outImage

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
              for channel in range(0, n_c):
                for filterR, inR in zip(range(0, f), range(row,row+f)):
                  for filterC, inC in zip(range(0, f), range(column,column+f)):
                    out[0, out_row, out_column, i] += Filter[i,filterR,filterC,channel] * image[0, inR, inC, channel]
              out[0, out_row, out_column, i] += bias[i]
              column += stride
              out_column += 1

          row += stride
          out_row += 1
  #print(out.shape)

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
def Activation_Relu(input):
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
def MaxPool2D(input, pool_size, stride, padding = 'valid'):
  (_, img_h, img_w, n_c) = input.shape

  if padding =='same' and (img_h - pool_size)%2 != 0: 
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

  #print(out.shape)

  return out

@jit
def Add(input_1, input_2):
  (a_1, b_1, c_1, d_1) = input_1.shape
  (a_2, b_2, c_2, d_2) = input_2.shape

  #if a_1 == a_2 and b_1 == b_2 and c_1 == c_2 and d_1 == d_2:
  out = np.zeros((a_1, b_1, c_1, d_1)) 

  for i in range(a_1):
    for j in range(b_1):
      for k in range(c_1):
        for l in range(d_1):
          out[i, j, k, l] = input_1[i, j, k, l] + input_2[i, j, k, l]           
  #print(out.shape)
  return out

kernels, bias = readKernelFiltersAndBias()
beta, gamma, moving_mean, moving_variance = readBatchNorm_C1()
def C1_resnet(input_image):
  input_image = Zeropadding(input_image, 3)
  input_image = convolution(input_image, kernels, bias, 2)
  input_image = batchNorm(input_image, beta, gamma, moving_mean, moving_variance)
  input_image = Activation_Relu(input_image)
  input_image = MaxPool2D(input_image, 3, 2, 'same')
  return input_image


#####################################################################
#conv_block C2

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


'''
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
inferenceConfig3.processed_input_image=Activation_Relu(inferenceConfig3.processed_input_image)
toc = time.perf_counter()
print("Relu timer:", {toc - tic}, "seconds")

tic = time.perf_counter()
inferenceConfig3.processed_input_image=MaxPool2D(inferenceConfig3.processed_input_image, 3 ,2, 'same')
toc = time.perf_counter()
print("Maxpooling timer (with padding):", {toc - tic}, "seconds")
'''
###################################################################################

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
                    save_fig_path='output_host_jit.png')

