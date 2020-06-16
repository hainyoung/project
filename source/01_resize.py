from PIL import Image

import glob

from natsort import natsorted


# display image characteristics
imc_path = './img/close/ce1.jpg' 

imc = Image.open(imc_path) 

print('{}'.format(imc.format)) 
print('size : {}'.format(imc.size))
print('image mode : {}'.format(imc.mode))

# imc.show()

'''
JPEG
size : (852, 480)
image mode : RGB
'''

# close_eyes image resize

# empty lists
close_list = []    
resized_close = []  

# append images to list
for filename in natsorted(glob.glob('./img/close/*.jpg')) :
    print(filename) 
    imc = Image.open(filename) 
    close_list.append(imc)    

# append resized images to list
for imc in close_list :       
    imc = imc.resize((64, 64))
    resized_close.append(imc)  
    print('size : {}'.format(imc.size))

# save resized images to new folder
for (i, new) in enumerate(resized_close) :
    new.save ('{}{}{}'.format('./eyes/c_eyes/ce', i+1, '.jpg')) 
 

# open_eyes image resize

imo_path = './img/open/oe1.jpg'

imo = Image.open(imo_path)
print('{}'.format(imo.format))
print('size : {}'.format(imo.size))
print('image mode : {}'.format(imo.mode))

imo.show()



'''
JPEG
size : (299, 168)
image mode : RGB
'''

# empty lists
open_list = []
resized_open = []

# append images to list
for filename in natsorted(glob.glob('./img/open/*.jpg')) :
    print(filename)
    imo = Image.open(filename)
    open_list.append(imo)

# append resized images to list
for imo in open_list :
    imo = imo.resize((64, 64))
    resized_open.append(imo)
    print('size : {}'.format(imo.size))

# save resized images to new folder
for (i, new) in enumerate(resized_open) :
    new.save ('{}{}{}'.format('./eyes/o_eyes/oe', i+1, '.jpg'))
