import cv2 as cv

from model import SUASYOLO
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


img_file = "data/images/tiny_train/image0.png"
label_file = "data/labels/tiny_train/image0.txt"
img = cv.imread(img_file)

with open(label_file, "r") as f:
    for line in f.readlines():
        line = line.strip().split()
        x, y, w, h = map(float, line[1:])
        x1 = int((x - w / 2) * img.shape[1])
        y1 = int((y - h / 2) * img.shape[0])
        x2 = int((x + w / 2) * img.shape[1])
        y2 = int((y + h / 2) * img.shape[0])
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # draw label
        cv.putText(img, line[0], (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # predict and draw
        model = SUASYOLO(num_classes = 17, cell_resolution=10).to(DEVICE)
        model.load_state_dict(torch.load("tiny_train.pt"))
        model.eval()
        boxes, classes = model.predict(torch.tensor(cv.imread(img_file)).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE))
        for box, classes in zip(boxes, classes):
            x, y, w, h = box 
            x1 = int((x - w / 2) * img.shape[1])
            y1 = int((y - h / 2) * img.shape[0])
            x2 = int((x + w / 2) * img.shape[1])
            y2 = int((y + h / 2) * img.shape[0])
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # draw label
            class_pred = classes.argmax()
            cv.putText(img, str(class_pred), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


cv.imshow("ground truth", img)
cv.waitKey(0)
