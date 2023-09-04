from augment import AugmentedSUASDataset
from visualize import display_boxes 
import cv2 as cv
import numpy as np

from model import SUASYOLO

model = SUASYOLO(num_classes = 14)
dataset = AugmentedSUASDataset("data_v2/images/train", "data_v2/labels/train", 14, model.num_cells)

img, label = dataset[4]
# display the image as a color histogram

boxes, objectness, shape_color, letter_color, shape_class, letter_class = model.process_predictions(label.unsqueeze(0))
boxes = boxes[objectness > 0]
shape_class = shape_class[objectness > 0]
letter_class = letter_class[objectness > 0]
objectness = objectness[objectness > 0]

img = img.permute(1, 2, 0).numpy().astype(np.uint8)
display_boxes(boxes, shape_class, objectness, (255,0,0), 1, img, centers_only = True)
# display the (3,640,640) pytorch tensor
cv.imshow("img", img)
cv.waitKey(0)