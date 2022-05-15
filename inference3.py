import os
import inferenceConfig3
from mrcnn import visualize
from mrcnn import model3
import numpy as np
from numba import jit

# print(type(inferenceConfig3.processed_input_image))
# print(inferenceConfig3.processed_input_image)
# print(inferenceConfig3.processed_input_image.shape)

@jit
def Zeropadding(input_image, padding):
  outputShape = (input_image.shape[0],input_image.shape[1]+2*padding,input_image.shape[2]+2*padding,input_image.shape[3])
  outImage = np.zeros(outputShape)

  for channel in range(0, outputShape[3]):
    for inputCol in range(0, input_image.shape[2]):
      for inputRow in range(0, input_image.shape[1]):
        outImage[0, inputRow+padding, inputCol+padding, channel]+=input_image[0, inputRow, inputCol, channel]

  return outImage


# Zeropadding(inferenceConfig3.processed_input_image, 3)
inferenceConfig3.processed_input_image = Zeropadding(inferenceConfig3.processed_input_image, 3)
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
                    save_fig_path='output.png')
