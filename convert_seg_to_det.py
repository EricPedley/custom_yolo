# for each text file in the data/labels folder, convert the segmentation labels to detection labels
# and save them in the det_data/labels fodler
# the detection labels are in the format:
# <object-class> <x_center> <y_center> <width> <height>
# where the coordinates are relative to the image size

# the segmentation labels are in the format:
# <object-class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
# where the coordinates are relative to the image size
import os

INPUT_FOLDER = "multilabel_color_data"
OUTPUT_FOLDER = "data_v2"

os.makedirs(f"{OUTPUT_FOLDER}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/labels/validation", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/labels/test", exist_ok=True)

for split in ["train", "validation", "test"]:
    for file_name in os.listdir(f"{INPUT_FOLDER}/labels/{split}"):
        file_contents = open(f"{INPUT_FOLDER}/labels/{split}/{file_name}").read().split("\n")
        file_contents = [line for line in file_contents if line != ""]
        file_contents = [line.split(" ") for line in file_contents]
        with open(f"{OUTPUT_FOLDER}/labels/{split}/{file_name}", "w") as f:
            for shape_class, alphanumeric, shape_color, letter_color, *polygon_coords in file_contents:
                polygon_coords = [float(coord) for coord in polygon_coords]
                x_coords = polygon_coords[::2]
                y_coords = polygon_coords[1::2]
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                f.write(f"{shape_class} {alphanumeric} {shape_color} {letter_color} {x_center} {y_center} {width} {height}\n")