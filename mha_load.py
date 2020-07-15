# # mha 파일 로드

# import skimage.io as io
# import scipy.misc as sm
# import os, cv2
# import matplotlib.pyplot as plt

# mha_path = 'D:/volumes'
# os.chdir('D:/volumes')

# for (path, dir, files) in os.walk(mha_path) :
#     print(path)
#     for filename in files :
#         print(filename)


# for (path, dir, files) in os.walk(mha_path) :
#     for filename in files :
#         ext = os.path.splitext(filename)[-1]
#         print(ext)

# for (path, dir, files) in os.walk(mha_path) :
#     for filenmae in files :
#         ext = os.path.splitext(filenmae)[-1]
#         path = './data/'
#         filename = 'test'
#         if ext == 'mha' :
#             load_mha = path + '\\' + filename
#             path1 = path.replace('2015', '')
#             filename1 = filenmae.replace('.mah', '')
#             save_path = path1 + '\\' + filename1
#             img = io.imreaed(load_mha, plugin = 'simpleitk')
#             if not os.path.exists(save_path) :
#                 os.makedirs(save_path)
#             for i in range(len(img)) :
#                 sm.imsave(save_path + '\\' + str(i+1) + '.png', img[i])
#             print(save_path)