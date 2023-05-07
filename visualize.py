import cv2 as cv

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

cv.imshow("img", img)
cv.waitKey(0)
