from nn import *
import pickle
import numpy as np
import cv2
from utils import *

model = pickle.load(open("model.pickl", "rb"))



def prediction(X):
    return np.argmax(model.predict(normalize(X_test)), axis=1).reshape((-1,1))

def put_img(dst, src, pos):
    x,y = pos
    hd,wd = dst.shape[0:2]
    hs,ws = src.shape[0:2]
    src_depth = 1 if len(src.shape) == 2 else src.shape[2]
    dst_depth = 1 if len(dst.shape) == 2 else dst.shape[2]
    
    if src_depth < dst_depth:
        for i in range(3):
            dst[y:y+hs,x:x+ws,i] = src 
    elif src_depth == dst_depth:
        dst[y:y+hs,x:x+ws,:] = src 



y_pred = prediction(X_test)

print("accuracy: ", accuracy(y_test, y_pred))

nx, ny = 40, 20
w, h = (28,28)
x, y = (0,0)

fb = np.zeros((ny * h, nx * w, 3), dtype=np.uint8)

for j in range(ny):
    for i in range(nx):
        idx = np.random.randint(len(X_test))
        label = str(y_pred[idx][0])
        label_gt = str(y_test[idx][0])
        color = (0,255,0) if label == label_gt else (0,0,255)
        put_img(fb, X_test[idx].reshape(h,w), (x,y))
        cv2.putText(fb, label,(x,y+h-1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color)
        x += w
    x = 0
    y += h

cv2.imwrite("out.png", fb)

