from dataset import SUASDataset
import torch
import os
import cv2 as cv
import torchvision.transforms.functional as TF

alphanumerics = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def box_horizontal_reflection(box_dims: "tuple[float]"):
    x, y, w, h = box_dims
    return 1 - x, y, w, h

def box_vertical_reflection(box_dims: "tuple[float]"):
    x, y, w, h = box_dims
    return x, 1 - y, w, h

def img_horizontal_reflection(img: torch.Tensor):
    return torch.flip(img, (2,))

def img_vertical_reflection(img: torch.Tensor):
    return torch.flip(img, (1,))


class AugmentedSUASDataset(SUASDataset):
    def __init__(
        self, img_dir, label_dir, n_class, n_cells 
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.n_class = n_class
        self.n_cells = n_cells
        # self.imgs = list(sorted(os.listdir(img_dir)))
        self.labels = list(sorted(os.listdir(label_dir)))
        self.imgs = [label.replace(".txt", ".png") for label in self.labels]

    def __getitem__(self, idx) -> "tuple(torch.Tensor, torch.Tensor)":
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        do_reflect_x = torch.rand(1) > 0.5
        do_reflect_y = torch.rand(1) > 0.5
        brightness_modification_amount  = torch.clamp(torch.normal(1,0.5), 0.5 ,1.5)
        contrast_factor  = torch.clamp(torch.normal(1,0.5), 0.5 ,1.5)
        
        boxes = []
        with open(label_path) as f:
            for line in f:
                shape_class, letter_class, shape_color, letter_color, *box_dims_str = line.split(" ")
                shape_class = int(shape_class)
                letter_class = alphanumerics.index(letter_class) if letter_class != "N/A" else -1
                shape_color = list(map(int, shape_color.split(":"))) if shape_color != "N/A" else [-1, -1, -1]
                letter_color = list(map(int, letter_color.split(":"))) if letter_color != "N/A" else [-1, -1, -1]
                box_dims = [float(dim) for dim in box_dims_str]
                if do_reflect_x:
                    box_dims = box_horizontal_reflection(box_dims)
                if do_reflect_y:
                    box_dims = box_vertical_reflection(box_dims)
                boxes.append([shape_class, letter_class, *shape_color, *letter_color, *box_dims])
        boxes = torch.tensor(boxes)
        img = torch.tensor(cv.imread(img_path)).type(torch.FloatTensor).permute(2, 0, 1)
        img = TF.adjust_brightness(img, brightness_modification_amount)
        img = TF.adjust_contrast(img, contrast_factor)
        if do_reflect_x:
            img = img_horizontal_reflection(img)
        if do_reflect_y:
            img = img_vertical_reflection(img)
        #TODO apply transformations here
        num_cells = self.n_cells

        label_matrix = torch.zeros((num_cells, num_cells, 3 + 6 + self.n_class + len(alphanumerics)), dtype=torch.float32)
        for box in boxes:
            class_no, letter_no, shape_r, shape_g, shape_b, letter_r, letter_g, letter_b, x, y, w, h = box
            x_cell, y_cell = int(x * num_cells), int(y * num_cells)
            # convert x,y,w,h to be relative to cell
            x, y = (x * num_cells) - x_cell, (y * num_cells) - y_cell
            # w, h = w * num_cells, h * num_cells
            if label_matrix[y_cell, x_cell, 4] == 1: # if cell already has an object
                # raise NotImplementedError("Need to support more than one object per cell") 
                continue
            label_matrix[y_cell, x_cell, :2] = torch.tensor([x, y])
            label_matrix[y_cell, x_cell, 2] = 1
            label_matrix[y_cell, x_cell, 3:6] = torch.tensor([shape_r, shape_g, shape_b])/255.0
            label_matrix[y_cell, x_cell, 6:9] = torch.tensor([letter_r, letter_g, letter_b])/255.0
            label_matrix[y_cell, x_cell, int(class_no) + 3 + 6] = 1
            label_matrix[y_cell, x_cell, int(letter_no) + 3 + 6 + self.n_class] = 1
        target = label_matrix.permute(2,1,0)

        return img, target
        


    def __len__(self):
        return len(self.imgs)