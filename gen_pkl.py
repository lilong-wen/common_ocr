import cv2
import pickle as pkl
import numpy as np
from tqdm import tqdm
import argparse


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('value',
                        help="generate training data or testing data",
                        type=str,
                        choices=['train', 'test', 'val'])

    args = parser.parse_args()

    #print(args.value)
    image_path = './20K/images_processed/'
    checkout_file = './20K/formulas.txt'

    if args.value == 'train':
        label_file = './20K/train.txt'
        output_file = './train.pkl'
    elif args.value == 'test':
        label_file = './20K/test.txt'
        output_file = './test.pkl'
    else:
        label_file = './20K/val.txt'
        output_file = './val.pkl'


    features_dict = {}
    with open(label_file, 'r') as label_f:
        key_labels = label_f.readlines()
        total_lines = len(key_labels)
        for key_label in tqdm(key_labels):
            key_label = key_label.strip()
            image_file = image_path + key_label + ".png"
            image_data = cv2.imread(image_file, 0)
            image_data = np.expand_dims(image_data, axis=0)
            features_dict[key_label] = image_data

    print(f'load {total_lines} lines done')

    with open(output_file, 'wb') as output_f:
        pkl.dump(features_dict, output_f)

    print(f'dump to {output_file} file done')
