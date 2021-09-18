from train import *
from inference import evaluate

for idx, (img_tensor, cap, img_name, image) in enumerate(i_dataset):
    evaluate(img_tensor, img_name, cap, tokenizer, i2T_generator)
