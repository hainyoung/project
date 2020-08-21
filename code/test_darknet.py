from darknet import darknet
import cv2, numpy as np, sys
from ctypes import *

net = darknet.load_net(b"D:/python_module/darknet-master/build/darknet/x64/project/myyolov3.cfg", b"D:/python_module/darknet-master/build/darknet/x64/project/backup/myyolov3_final.weights", 0) 
meta = darknet.load_meta(b"D:/python_module/darknet-master/build/darknet/x64/project/my.data") 
cap = cv2.VideoCapture("D:/python_module/darknet-master/build/darknet/x64/project/26-4.mp4")
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('otter_out26-4.avi', fourcc, fps, (640,480))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
i = 0
while(cap.isOpened()):
    i += 1
    ret, image = cap.read()
    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    print(i)
    if not ret: 
        break 
    frame = darknet.nparray_to_image(image)
    r = darknet.detect_image(net, meta, frame, thresh=.5, hier_thresh=.5, nms=.45, debug= False)
    print(r)
    boxes = [] 

    for k in range(len(r)): 
        width = r[k][2][2] 
        height = r[k][2][3] 
        center_x = r[k][2][0] 
        center_y = r[k][2][1] 
        bottomLeft_x = center_x - (width / 2) 
        bottomLeft_y = center_y - (height / 2) 
        x, y, w, h = bottomLeft_x, bottomLeft_y, width, height
        mytexts = r[k][0]
        mythresh = r[k][1] 
        boxes.append((x, y, w, h, mytexts, mythresh))
    print(1)
    for k in range(len(boxes)): 
        x, y, w, h, texts, threshs = boxes[k] 
        top = max(0, np.floor(x + 0.5).astype(int)) 
        left = max(0, np.floor(y + 0.5).astype(int)) 
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int)) 
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int)) 
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2) 
        cv2.putText(image, texts.decode('utf-8') + '('+ str(threshs*100)[:5] + '%)', (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        # cv2.line(image, (top + int(w / 2), left), (top + int(w / 2), left + int(h)), (0,255,0), 3) 
        # cv2.line(image, (top, left + int(h / 2)), (top + int(w), left + int(h / 2)), (0,255,0), 3) 
        # cv2.circle(image, (top + int(w / 2), left + int(h / 2)), 2, tuple((0,0,255)), 5)

    cv2.imshow('frame', image)
    out.write(image)
    darknet.free_image(frame) ## darknet에서 쓰는 c언어 동적할당 해제해주는 함수(써주어야 메모리가 버틴다.)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
out.release()
cv2.destroyAllWindows()