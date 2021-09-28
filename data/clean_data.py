import os
import string
import warnings

import numpy as np
import pandas as pd
import pickle5 as pickle

warnings.filterwarnings("ignore")

npic = 5
npix = 224
target_size = (npix, npix, 3)
count = 1
num_layer = 4
top_k = 5000
target_vocab_size = top_k + 1

image_path = './Dataset/Flicker/Flicker8k_Dataset/'
dir_Flickr_text = './Dataset/Flicker/Flickr8k.token.txt'

data: pd.DataFrame


def do_preprocess():
    global data
    jpgs = os.listdir(image_path)
    print(f'Total image in dataset is {len(jpgs)}')

    file = open(dir_Flickr_text, 'r')
    text = file.read()
    file.close()

    datatxt = []
    for line in text.split('\n'):
        col = line.split('\t')
        if len(col) == 1:
            continue

        w = col[0].split('#')
        datatxt.append(w + [col[1].lower()])

    data = pd.DataFrame(datatxt, columns=['filename', 'index', 'captions'])
    data = data.reindex(columns=['index', 'filename', 'captions'])
    data = data[data.filename != '2258277193_586949ec62.jpg.1']
    uni_filenames = np.unique(data.filename.values)

    vocabulary = []
    for txt in data.captions.values:
        vocabulary.extend(txt.split())

    print(f'Vocublary Size {len(set(vocabulary))}')

    def remove_punctutation(text_original):
        text_no_punctuation = text_original.translate(str.maketrans('', '', string.punctuation))
        return (text_no_punctuation)

    def remove_single_character(text):
        text_len_more_than1 = ''
        for word in text.split():
            if len(word) > 1:
                text_len_more_than1 += ' ' + word

        return (text_len_more_than1)

    def remove_numeric(text):
        text_no_numeric = ''

        for word in text.split():
            isalpha = word.isalpha()

            if isalpha:
                text_no_numeric += ' ' + word

        return (text_no_numeric)

    def text_clean(text_original):
        text = remove_punctutation(text_original)
        text = remove_single_character(text)
        text = remove_numeric(text)

        return (text)

    for i, caption in enumerate(data.captions.values):
        newcaption = text_clean(caption)
        data['captions'].iloc[1] = newcaption

    clean_vocab = []

    for txt in data.captions.values:
        clean_vocab.extend(txt.split())

    print(len(set(clean_vocab)))


all_captions = []
cap_file = './Dataset/Flicker/captions.pickle'

# check if all_captions is already available
if os.path.isfile(cap_file):
    print("found cached caption.pickle")
    with open(cap_file, 'rb') as f:
        all_captions = pickle.load(f)
else:
    with open(cap_file, 'wb') as f:
        do_preprocess()
        for caption in data['captions'].astype(str):
            caption = '<start> ' + caption + ' <end>'
            all_captions.append(caption)

        pickle.dump(all_captions, f, protocol=pickle.HIGHEST_PROTOCOL) 

# check if all_img_name_vector is already available
all_img_name_vector = []
img_name = './Dataset/Flicker/img_name.pickle'

if os.path.isfile(img_name):
    print('found cached img_name.pickle')
    with open(img_name, 'rb') as f:
        all_img_name_vector = pickle.load(f)
else:
    with open(img_name, 'wb') as f:
        for ann in data['filename']:
            full_image_path = image_path + ann
            all_img_name_vector.append(full_image_path)

        pickle.dump(all_img_name_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

