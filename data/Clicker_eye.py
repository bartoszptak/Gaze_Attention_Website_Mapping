import cv2
import glob
import numpy as np
import pandas as pd
import os

option = 'train' # train, test, valid

files = glob.glob(os.path.join(option+'_imgs', '*.png'))


if os.path.isfile(option+'_landmarks.csv'):
    df = pd.read_csv(option+'_landmarks.csv')
else:
    df = pd.DataFrame(columns=[
        'file', 
        'L_x', 'L_y',
        'R_x', 'R_y',
        'CC_x', 'CC_y',
        'CL_x', 'CL_y',
        'CR_x', 'CR_y',
        'CU_x', 'CU_y',
        'CD_x', 'CD_y'
        ])

imgs = []

for path in files:
    if not (path==df.file).any():
        imgs.append(path)

print(len(imgs))

def transform(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONUP:
        Xs.append(x)
        Xs.append(y)

cv2.namedWindow('image')
cv2.setMouseCallback('image', transform)

scale = 2
i = 0

while i < len(imgs):
    Xs = []
    im = cv2.imread(imgs[i])

    imz = cv2.resize(im, None, fx=scale, fy=scale)
    cv2.imshow('image', imz)
    key = cv2.waitKey(0)

    if key == ord('r'):
        print('Again')
        continue
    elif key == 27:
        print('Save')
        break

    if len(Xs) != 14:
        print('Again')
        continue

    d = {
        'file': imgs[i], 
        'L_x': int(Xs[0] / scale), 
        'L_y': int(Xs[1] / scale),
        'R_x': int(Xs[2] / scale), 
        'R_y': int(Xs[3] / scale),
        'CC_x': int(Xs[4] / scale), 
        'CC_y': int(Xs[5] / scale),
        'CL_x': int(Xs[6] / scale), 
        'CL_y': int(Xs[7] / scale),
        'CR_x': int(Xs[8] / scale), 
        'CR_y': int(Xs[9] / scale),
        'CU_x': int(Xs[10] / scale), 
        'CU_y': int(Xs[11] / scale),
        'CD_x': int(Xs[12] / scale), 
        'CD_y': int(Xs[13] / scale)
    }

    df = df.append(d, ignore_index=True)
    
    print('Ok ({})'.format(len(imgs) - i))
    i += 1

df.to_csv(option+'_landmarks.csv', index=False)
