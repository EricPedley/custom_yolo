import torch
import os
import cv2 as cv

class SUASDataset(torch.utils.data.Dataset):
    def __init__(
        self, img_dir, label_dir, n_class, n_cells=7
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
        boxes = []
        with open(label_path) as f:
            for line in f:
                class_no, *box_dims = line.split(" ")
                class_no = int(class_no)
                boxes.append([class_no, *[float(dim) for dim in box_dims]])
        boxes = torch.tensor(boxes)
        img = torch.tensor(cv.imread(img_path)).type(torch.FloatTensor).permute(2, 0, 1)
        #TODO apply transformations here

        num_cells = self.n_cells

        label_matrix = torch.zeros((num_cells, num_cells, self.n_class + 5))
        for box in boxes:
            class_no, x, y, w, h = box
            x_cell, y_cell = int(x * num_cells), int(y * num_cells)
            # convert x,y,w,h to be relative to cell
            x, y = (x * num_cells) - x_cell, (y * num_cells) - y_cell
            w, h = w * num_cells, h * num_cells
            if label_matrix[y_cell, x_cell, 4] == 1: # if cell already has an object
                # raise NotImplementedError("Need to support more than one object per cell") 
                continue
            label_matrix[y_cell, x_cell, :4] = torch.tensor([x, y, w, h])
            label_matrix[y_cell, x_cell, 4] = 1
            label_matrix[y_cell, x_cell, int(class_no) + 5] = 1

        return img, label_matrix.permute(2, 1, 0)
        


    def __len__(self):
        return len(self.imgs)