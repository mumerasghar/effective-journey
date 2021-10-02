import os
import time
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DIR = './Dataset/COCO/extracted/coco_training/'
CAP_FILE = './Dataset/COCO/captions.pickle'
IMG_NAME = './Dataset/COCO/img_name.pickle'
CAPTIONS = './Dataset/COCO/annotations/captions_train2014.json'

def copy_sub_data(a):

    _dest = './Dataset/COCO/extracted/coco_training/'
    _src = './Dataset/COCO/extracted/train2014/'
    import shutil
    a = a[0:40000]
    _atem = set(a)
    for i in _atem:
        _img = f"COCO_train2014_{''.join(['0' for i in range(12 - len(str(i)))])}{i}"
        shutil.copy(f'{_src}{_img}.jpg',_dest)    
        shutil.copy(f'{_src}{_img}.jpg.npy',_dest)   


def read_data():
    with open(CAPTIONS) as f:
        annotations = json.load(f)
        annotations = annotations['annotations']

    data = []
    for item in annotations:
        t = (item['image_id'], item['caption'].lower())
        data.append(t)

    data = pd.DataFrame(data, columns=['filename', 'captions'])

    a = data['filename'].values

    all_captions = []
    if os.path.isfile(CAP_FILE):
        print("found cached caption.pickle")
        with open(CAP_FILE, 'rb') as f:
            all_captions = pickle.load(f)
    else:
        print('formating captions')
        with open(CAP_FILE, 'wb') as f:
            for caption in data['captions'].astype(str):
                caption = '<start> ' + caption + ' <end>'
                all_captions.append(caption)
            pickle.dump(all_captions, f, protocol=pickle.HIGHEST_PROTOCOL)

    all_img_name = []
    if os.path.isfile(IMG_NAME):
        print('found cached img_name.pickle')
        with open(IMG_NAME, 'rb') as f:
            all_img_name = pickle.load(f)
    else:
        with open(IMG_NAME, 'wb') as f:
            for f_name in data['filename']:
                c_addr = f"COCO_train2014_{''.join(['0' for i in range(12 - len(str(f_name)))])}{f_name}.jpg"
                all_img_name.append(DIR + c_addr)

            pickle.dump(all_img_name, f, protocol=pickle.HIGHEST_PROTOCOL)

    return all_captions, all_img_name


ALL_CAPTIONS, ALL_IMG_NAME = read_data()


def data_limiter(num, captions, img_name_vector):
    t_cap = captions[:num]
    i_name_vec = img_name_vector[:num]
    t_cap, i_name_vec = shuffle(t_cap, i_name_vec, random_state=1)
    return t_cap, i_name_vec

print('delimiting data')
t_captions, t_img_name = data_limiter(40000, ALL_CAPTIONS, ALL_IMG_NAME)

