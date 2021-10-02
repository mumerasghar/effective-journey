import os
import json
import string
import pandas as pd
import pickle5 as pickle


class Flicker8K:
    def __init__(self, img_pth, txt_pth, cap_file, img_name):
        self.img_pth = img_pth
        self.txt_pth = txt_pth
        self.cap_file = cap_file
        self.img_name = img_name

    def pre_process(self):

        jpgs = os.listdir(self.img_pth)
        print(f'\t[+]Total image in dataset is {len(jpgs)}.')

        with open(self.txt_pth, 'r') as file:
            text = file.read()

        s_text = self.split_text(text)

        df = pd.DataFrame(s_text, columns=['filename', 'index', 'captions'])
        df = df.reindex(columns=['index', 'filename', 'captions'])
        df = df[df.filename != '2258277193_586949ec62.jpg.1']

        vocabulary = []
        for txt in df.captions.values:
            vocabulary.extend(txt.split())

        print(f'\t[+] Vocabulary Size {len(set(vocabulary))}.')

        for i, cap in enumerate(df.captions.values):
            new_cap = self.text_clean(cap)
            df['captions'].iloc[1] = new_cap

        return df

    def get_data(self):

        all_caps = []
        img_name = []

        data = self.pre_process()
        with open(self.cap_file, 'wb') as file:
            for caption in data['captions'].astype(str):
                caption = '<start> ' + caption + ' <end>'
                all_caps.append(caption)

            pickle.dump(all_caps, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.img_name, 'wb') as file:
            for ann in data['filename']:
                full_image_path = image_path + ann
                img_name.append(full_image_path)

            pickle.dump(img_name, file, protocol=pickle.HIGHEST_PROTOCOL)

        return all_caps, img_name

    @staticmethod
    def remove_numeric(text):
        text_no_numeric = ''
        for word in text.split():
            isalpha = word.isalpha()
            if isalpha:
                text_no_numeric += ' ' + word
        return text_no_numeric

    @staticmethod
    def remove_single_character(text):
        text_len_more_than1 = ''
        for word in text.split():
            if len(word) > 1:
                text_len_more_than1 += ' ' + word

        return text_len_more_than1

    @staticmethod
    def remove_punctuation(text_original):
        text_no_punctuation = text_original.translate(str.maketrans('', '', string.punctuation))
        return text_no_punctuation

    @staticmethod
    def text_clean(text_original):
        text = Flicker8K.remove_punctuation(text_original)
        text = Flicker8K.remove_single_character(text)
        text = Flicker8K.remove_numeric(text)

        return text

    @staticmethod
    def split_text(text):
        data_txt = []
        for line in text.split('\n'):
            col = line.split('\t')
            if len(col) == 1:
                continue
            w = col[0].split('#')
            data_txt.append(w + [col[1].lower()])

        return data_txt


class COCO:
    def __init__(self, img_pth, txt_pth, cap_file, img_name):
        self.img_pth = img_pth
        self.txt_pth = txt_pth
        self.cap_file = cap_file
        self.img_name = img_name

    @staticmethod
    def copy_sub_data(a):
        import shutil
        _src = './Dataset/COCO/extracted/train2014/'
        _dest = './Dataset/COCO/extracted/coco_training/'

        a = a[0:40000]
        _a_item = set(a)
        for i in _a_item:
            _img = f"COCO_train2014_{''.join(['0' for i in range(12 - len(str(i)))])}{i}"
            shutil.copy(f'{_src}{_img}.jpg', _dest)
            shutil.copy(f'{_src}{_img}.jpg.npy', _dest)

    def pre_process(self):
        with open(self.txt_pth) as file:
            annotations = json.load(file)
            annotations = annotations['annotations']

        df = []
        for item in annotations:
            t = (item['image_id'], item['caption'].lower())
            df.append(t)

        df = pd.DataFrame(df, columns=['filename', 'captions'])

        return df

    def get_data(self):

        all_caps = []
        img_name = []

        df = self.pre_process()

        print('\t[+] formatting captions')
        with open(self.cap_file, 'wb') as file:
            for cap in df['captions'].astype(str):
                cap = '<start> ' + cap + ' <end>'
                all_caps.append(cap)
            pickle.dump(all_caps, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.img_name, 'wb') as file:
            for f_name in df['filename']:
                _name = ''.join(['0' for _ in range(12 - len(str(f_name)))])
                c_adr = f"COCO_train2014_{_name}{f_name}.jpg"
                img_name.append(self.img_pth + c_adr)

            pickle.dump(img_name, file, protocol=pickle.HIGHEST_PROTOCOL)

        return all_caps, img_name


def Dataset(image_path, text_path, cap_file, img_name, dataset):
    if dataset == 'COCO':
        f_data = COCO(image_path, text_path, cap_file, img_name)
    else:
        f_data = Flicker8K(image_path, text_path, cap_file, img_name)

    if os.path.isfile(cap_file) and os.path.isfile(img_name):
        print("[+] found cached caption.pickle.")
        with open(cap_file, 'rb') as f:
            all_captions = pickle.load(f)

        print('[+] found cached img_name.pickle.')
        with open(img_name, 'rb') as f:
            all_img_name_vector = pickle.load(f)
    else:
        print('[+] creating cached caption.pickle and img_name.pickle.')
        all_captions, all_img_name_vector = f_data.get_data()

    return all_captions, all_img_name_vector

# if __name__ == '__main__':
#     all_captions = []
#     all_img_name_vector = []
#     image_path = '../Dataset/COCO/extracted/coco_training/'
#     text_path = '../Dataset/COCO/annotations/captions_train2014.json'
#     CAP_FILE = '../Dataset/COCO/captions.pickle'
#     IMG_NAME = '../Dataset/COCO/img_name.pickle'
#
#     f_data = COCO(image_path, text_path, CAP_FILE, IMG_NAME)
#     if os.path.isfile(CAP_FILE) and os.path.isfile(IMG_NAME):
#         print("found cached caption.pickle")
#         with open(CAP_FILE, 'rb') as f:
#             all_captions = pickle.load(f)
#
#         print('found cached img_name.pickle')
#         with open(IMG_NAME, 'rb') as f:
#             all_img_name_vector = pickle.load(f)
#
#     print('success')
