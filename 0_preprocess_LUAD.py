import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image

palette = [0]*9
palette[0:3] = [255, 0, 0] # TUM 0 
palette[3:6] = [0,255,0] # STR 1
palette[6:9] = [255,255,255] # other



if __name__ == '__main__':
    
    root = 'F:/dataset/Med/HistoSeg/LUAD-HistoSeg'
    new_root = 'F:/dataset/Med/HistoSeg/LUAD-HistoSeg_dg'
    if os.path.exists(new_root) == False:
        os.mkdir(new_root)
    
    ## train ##
    new_train = os.path.join(new_root, 'train')
    if os.path.exists(new_train) == False:
        os.mkdir(new_train)
    for img in tqdm(os.listdir(os.path.join(root, 'train'))):
        label_str = img.split(']')[0].split('[')[-1]
        new_label_str = '['+label_str[0]+label_str[6]+']'
        origin_img = os.path.join(root, 'train', img)
        new_img = os.path.join(new_train, img.replace('['+label_str+']', new_label_str))
        shutil.copyfile(origin_img, new_img)

    ## val ##
    new_val = os.path.join(new_root, 'val')
    if os.path.exists(new_val) == False:
        os.mkdir(new_val)
        os.mkdir(os.path.join(new_val, 'img'))
        os.mkdir(os.path.join(new_val, 'mask'))
    for img in tqdm(os.listdir(os.path.join(root, 'val', 'img'))):
        origin_img = os.path.join(root, 'val', 'img', img)
        new_img = os.path.join(new_val, 'img', img)
        shutil.copyfile(origin_img, new_img)
    for mask in tqdm(os.listdir(os.path.join(root, 'val', 'mask'))):
        label = Image.open(os.path.join(root, 'val', 'mask', mask))
        new_label = np.array(label)
        new_label[new_label==1]=2
        new_label[new_label==2]=2
        new_label[new_label==3]=1
        new_label[new_label==4]=2
        visualimg  = Image.fromarray(new_label.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(new_val, 'mask', mask), format='PNG')

    ## test ##
    new_test = os.path.join(new_root, 'test')
    if os.path.exists(new_test) == False:
        os.mkdir(new_test)
        os.mkdir(os.path.join(new_test, 'img'))
        os.mkdir(os.path.join(new_test, 'mask'))
    for img in tqdm(os.listdir(os.path.join(root, 'test', 'img'))):
        origin_img = os.path.join(root, 'test', 'img', img)
        new_img = os.path.join(new_test, 'img', img)
        shutil.copyfile(origin_img, new_img)
    for mask in tqdm(os.listdir(os.path.join(root, 'test', 'mask'))):
        label = Image.open(os.path.join(root, 'test', 'mask', mask))
        new_label = np.array(label)
        new_label[new_label==3]=2
        new_label[new_label==4]=2
        visualimg  = Image.fromarray(new_label.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(new_test, 'mask', mask), format='PNG')