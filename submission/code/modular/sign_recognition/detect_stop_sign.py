import torch.nn as nn
import torch
import numpy as np
import cv2 as cv
import imutils
import os, sys

# class LogisticRegression(torch.nn.Module):
#     def __init__(self):
#         super(LogisticRegression, self).__init__()
#         self.linear = nn.Linear(3, 1)
#     def forward(self, x):
#         y_pred = torch.sigmoid(self.linear(x))
#         return y_pred

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear_0 = nn.Linear(3, 32)
        self.linear_1 = nn.Linear(32, 64)
        self.linear_2 = nn.Linear(64, 16)
        self.linear_3 = nn.Linear(16, 4)
        self.linear_4 = nn.Linear(4, 1)
    def forward(self, x):
        y = self.linear_0(x)
        y = nn.ReLU()(self.linear_1(y))
        y = nn.ReLU()(self.linear_2(y))
        y = nn.ReLU()(self.linear_3(y))
        y_pred = torch.sigmoid(self.linear_4(y))
        return y_pred

model = LogisticRegression()
# model.load_state_dict(torch.load('RedModel.pt', map_location=torch.device('cpu')))
fpath = os.path.join(sys.path[0], "gym-duckietown/modular/sign_recognition/my_model.pt")
model.load_state_dict(torch.load(fpath, map_location=torch.device('cpu')))

def convert_image(img):
    image = torch.from_numpy(img).float()
    image = torch.reshape(image, (-1, 3))
    prediction = model(image)
    predicted = torch.ge(prediction, 0.5).to(torch.float32)
    predicted = torch.reshape(predicted, (img.shape[0], img.shape[1]))
    return predicted

def sliding_window(img, step_size, window_size):
    for y in range(0, int(img.shape[0]) - window_size, step_size):
        for x in range(0, int(img.shape[1]) - window_size, step_size):
            yield img[y:y+window_size, x:x+window_size]

def create_contours(img):
    thresh = cv.threshold(img, 60, 255, cv.THRESH_BINARY)[1]
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def select_stop_contour(img_height, bound_rect, min_area=500, heigh_crop_ratio=0.5, max_heigh_width_ratio=1.8, min_heigh_width_ratio=0.8):
    ret_contours = []
    y_limit = int(img_height * heigh_crop_ratio)
    for x, y, width, height in bound_rect:
        if min_heigh_width_ratio <= height / width <= max_heigh_width_ratio and height * width >= 500 and \
            y + height <= y_limit:
            ret_contours.append((x, y, width, height))
    return ret_contours

def stop_detect(img, min_area=500, heigh_crop_ratio=0.5, max_heigh_width_ratio=1.8, min_heigh_width_ratio=0.8):
    image = convert_image(img).numpy().astype(np.uint8) * 255
    from PIL import Image

    resized = imutils.resize(image, width=320)
    im = imutils.resize(resized, width=image.shape[1])

    im = cv.GaussianBlur(im, (7,7), 0)
    im = cv.Canny(im, 0, 200)
    contours = create_contours(im)
    bound_rect = [None] * len(contours)
    for i, c in enumerate(contours):
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.01 * peri, True)
        bound_rect[i] = cv.boundingRect(approx)
        cv.fillPoly(im, pts=[c], color=(255,255,255))

    drawing = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    
    bound_rect = select_stop_contour(im.shape[0], bound_rect, min_area, heigh_crop_ratio, max_heigh_width_ratio, min_heigh_width_ratio)
    return len(bound_rect) > 0
    # print(bound_rect)
    # for i in range(len(bound_rect)):
    #     cv.rectangle(drawing, (int(bound_rect[i][0]), int(bound_rect[i][1])), \
    #       (int(bound_rect[i][0]+bound_rect[i][2]), int(bound_rect[i][1]+bound_rect[i][3])), (255, 255, 255), 2)
    # cv.imwrite('image.png', drawing)

    # img = Image.fromarray(im)
    # img.save('screen.png')