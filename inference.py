import os
import random
import skimage.io
import inferenceConfig
from mrcnn import visualize
from mrcnn import model2

# Load a random image from the images folder
file_names = next(os.walk(inferenceConfig.IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(inferenceConfig.IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model2.detect([image], verbose=0)

# Visualize results
r = results[0]

visualize.display_instances(image, 
                    r['rois'], 
                    r['masks'], 
                    r['class_ids'], 
                    inferenceConfig.CLASS_NAMES, 
                    scores=r['scores'],
                    save_fig_path='output.png')




                    