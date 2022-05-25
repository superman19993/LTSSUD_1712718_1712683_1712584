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
  

@cuda.jit
def ConvolutionKernel(input_image, input_image_h, input_image_w, channel, i, output_image, filters, bias, stride):
  c, r = cuda.grid(2)

  filter_size = filters.shape[1]
  (_, out_h, out_w, _) = output_image.shape

  if r < out_h and c < out_w:
    row = stride*r
    column = stride*c 
    if row + filter_size <= input_image_h and column + filter_size <= input_image_w:
      # output_image[0, r, c, n_f] = np.sum(filters[n_f] * input_image[0, row:row + filters.shape[1], column:column + filters.shape[2],:]) + bias[n_f]
      for filterR, inR in zip(range(0, filter_size), range(row,row+filter_size)):
        for filterC,inC in zip(range(0, filter_size), range(column,column+filter_size)):
          output_image[0, r, c, i] += filters[i,filterR,filterC,channel] * input_image[0, inR, inC, channel]
      output_image[0, r, c, i] += bias[i]


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
                math.ceil(out.shape[1]/block_size[1]))
    
    Filter = np.ascontiguousarray(Filter, dtype=np.float32)
    for i in range(n_f):
      for channel in range(0, n_c):
        ConvolutionKernel[grid_size, block_size](image, img_h, img_w, channel, i, out, Filter, bias, stride)

    print(out.shape)

    return out


tic = time.perf_counter()
inferenceConfig3.processed_input_image = Zeropadding(inferenceConfig3.processed_input_image, 3)
toc = time.perf_counter()
print("Zeropadding timer:", {toc - tic}, "seconds")

kernels, bias = readKernelFiltersAndBias()
tic = time.perf_counter()
inferenceConfig3.processed_input_image=convolution(inferenceConfig3.processed_input_image, kernels, bias, 2)
toc = time.perf_counter()
print("Convolution timer:", {toc - tic}, "seconds")

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
                    save_fig_path='output_parallel_0.png')
