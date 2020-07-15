import dlib, cv2

# dlib for Face Detection, Face Recognition
# cv2 : openCV(이미지 작업)


import numpy as np # 연산 작업을 위해
import matplotlib.pyplot as plt # 결과물을 그리기 위해
import matplotlib.patches as patches 
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./team/models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./team/models/dlib_face_recognition_resnet_model_v1.dat')

# 얼굴을 찾는 함수
def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0 :
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype = np.int)
    for k, d in enumerate(dets) :
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        # convert dlib shape to numpy array # shape : face's landmark
        for i in range(0, 68) :
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
    
    return rects, shapes, shapes_np

# 찾은 얼굴을 인코드 하는 함수
def encode_faces(img, shapes) :
    face_descriptors = []
    for shape in shapes :
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)


img_paths = {
    'neo' : './team/img/neo.jpg',
    'trinity' : './team/img/trinity.jpg',
    'morpheus' : './team/img/morpheus.jpg',
    'smith' : './team/img/smith.jpg'
}

# 계산한 결과를 저장할 변수
descs = {
    'neo' : None,
    'trinity' : None,
    'morpheus' : None,
    'smith' : None
}

for name, img_path in img_paths.items() :
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb) # RGB 체계로 바꾼 이미지의 랜드마크들을 가져온다(img_shapes 변수에 담음)
    descs[name] = encode_faces(img_rgb, img_shapes)[0] # 인코딩된 결과를 각 사람의 이름에 맞게 저장해준다

np.save('./team/img/descs.npy', descs)

print(descs)


# Compute Input
img_bgr = cv2.imread('./team/img/matrix5.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)

# Visulize Output
fig, ax = plt.subplots(1, figsize = (15, 15))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors) :

    found = False
    for name, saved_desc in descs.items() :
        dist = np.linalg.norm([desc] - saved_desc, axis = 1) # 유클리드 거리 구하는 코드

        if dist < 0.6 :
            found = True
            
            text = ax.text(rects[i][0][0], rects[i][0][1], name, color = 'b', fontsize = 40, fontweight = 'bold')
            text.set_path_effects([path_effects.Stroke(linewidth = 10, foreground = 'white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0], rects[i][1][1] - rects[i][0][1], rects[i][1][0] - rects[i][0][0],
                                     linewidth = 2, edgecolor = 'w', facecolor = 'none')
            ax.add_patch(rect)

            break
        
        if not found :
            ax.text(rects[i][0][0], rects[i][0][1], 'unknown', color = 'r', fontsize = 20, fontweight = 'bold')
            rect = patches.Rectangle(rects[i][0], rects[i][1][1] - rects[i][0][1], rects[i][1][0] - rects[i][0][0],
                                     linewidth = 2, edgecolor = 'r', facecolor = 'none')
            ax.add_patch(rect)
plt.show()