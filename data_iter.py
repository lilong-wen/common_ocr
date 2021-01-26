import numpy
import pickle as pkl
import sys
import torch


def dataIterator(feature_file,
                 label_file,
                 checkout_file,
                 dictionary,
                 maxlen,
                 maxImagesize,
                 batch_size):

    with open(feature_file, 'rb') as fp:
        features = pkl.load(fp)

    checkout_dict = {}
    with open(checkout_file, 'r') as checkout_f:
        while True:
            line = checkout_f.readline().strip()
            if not line:
                break
            key_checkout = line.split('\t')[0]
            value_checkout = line.split('\t')[1]
            checkout_dict[key_checkout] = value_checkout

    with open(label_file, 'r') as fp:
        key_labels = fp.readlines()

    len_label = len(key_labels)

    targets={}
    # map word to i nt with dictionary
    for key_label in key_labels:
        key_label = key_label.strip()
        value_label = checkout_dict[key_label].split()
        word_list=[]
        for word in value_label:
            if word in dictionary:
                word_list.append(dictionary[word])
            else:
                print(f'{key_label} : {word} is not in dictionary')
                sys.exit()
        targets[key_label]=word_list


    imageSize={}
    imagehigh={}
    imagewidth={}
    for uid,fea in features.items():
        imageSize[uid]=fea.shape[1]*fea.shape[2]
        imagehigh[uid]=fea.shape[1]
        imagewidth[uid]=fea.shape[2]

    imageSize= sorted(imageSize.items(), key=lambda d:d[1],reverse=True) # sorted by sentence length,  return a list with each triple element

    # I have no idea how to write a better method
    # needed_len = len(imageSize) - len(imageSize) % 6

    feature_total=[]
    label_total=[]

    # for uid,size in imageSize[:needed_len]:
    for uid,size in imageSize:
        fea=features[uid]
        lab=targets[uid]

        if len(lab)>maxlen:
            continue
            # print('sentence', uid, 'length bigger than', maxlen, 'ignore')

        elif size>maxImagesize:
            continue
            # print('image', uid, 'size bigger than', maxImagesize, 'ignore')

        else:
            feature_total.append(fea)
            label_total.append(lab)

    print(len(feature_total))
    if len(feature_total) % batch_size != 0:
        feature_total = feature_total[: -(len(feature_total) % batch_size)]
    len_ignore = len_label - len(feature_total)
    print('total ',len(feature_total), ' data loaded')
    print('ignore',len_ignore,'images')


    return feature_total,label_total
