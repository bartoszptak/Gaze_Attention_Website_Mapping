from imutils import face_utils
import dlib
import cv2
import os
import glob


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join('..', 'shape_predictor_68_face_landmarks.dat'))

cap = cv2.VideoCapture(0)

files = glob.glob(os.path.join('dataset/'+'train_imgs', '*.png'))
name = int(files[-1].split('.')[-2].split('_')[-1])+1

while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, +1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    leftXs = []
    leftYs = []
    rightXs = []
    rightYs = []

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for x, y in shape[36:42]:
            leftXs.append(x)
            leftYs.append(y)
        for x, y in shape[42:48]:
            rightXs.append(x)
            rightYs.append(y)

    key = cv2.waitKey(1)
    if key == 27:
        break

    if leftXs.__len__() > 0:
        left = image[min(leftYs) - 5:max(leftYs) + 5, min(leftXs) - 5:max(leftXs) + 5]
        right = image[min(rightYs) - 5:max(rightYs) + 5, min(rightXs) - 5:max(rightXs) + 5]

        left = cv2.resize(left, (120, 60))
        right = cv2.resize(right, (120, 60))

        if key == 32:
            cv2.imwrite(os.path.join('dataset/'+'train_imgs', 'eye_{}.png'.format(str(name))), left)
            cv2.imwrite(os.path.join('dataset/'+'train_imgs', 'eye_{}.png'.format(str(name))), right)
            print("Take eyeshot!")
            name += 2
        elif key == ord('q'):
            break

        cv2.imshow('All', image)

cap.release()
cv2.destroyAllWindows()
