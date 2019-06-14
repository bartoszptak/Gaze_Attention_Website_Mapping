import cv2
import glob
import numpy as np
import os

directory_name = "eye_array2"

imgs = glob.glob(os.path.join('..', 'data', 'eyes', directory_name, '*.png'))

array = []

i = 0


def transform(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONUP:
        Xs.append(x)
        Xs.append(y)


cv2.namedWindow('image')
cv2.setMouseCallback('image', transform)
scale = 2

while i < len(imgs):
    Xs = []
    im = cv2.imread(imgs[i])

    imz = cv2.resize(im, None, None, scale, scale)
    cv2.imshow('image', imz)
    cv2.waitKey(0)

    if len(Xs) != 6:
        continue
    array.append([im, int(Xs[0] / scale), int(Xs[1] / scale), int(Xs[2] / scale), int(Xs[3] / scale), int(Xs[4] / scale), int(Xs[5] / scale)])
    print(imgs.__len__() - i)
    i += 1

np.save(os.path.join('..', 'data', 'eyes', directory_name), array)