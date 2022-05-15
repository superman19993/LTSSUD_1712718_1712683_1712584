import os
import inferenceConfig3
from mrcnn import visualize
from mrcnn import model3

print(type(inferenceConfig3.processed_input_image))
# Run detection
results = model3.detect([inferenceConfig3.image], inferenceConfig3.processed_input_image, verbose=0)
# Visualize results
r = results[0]

visualize.display_instances(inferenceConfig3.image, 
                    r['rois'], 
                    r['masks'], 
                    r['class_ids'], 
                    inferenceConfig3.CLASS_NAMES, 
                    scores=r['scores'],
                    save_fig_path='output.png')
