import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data

#from Densenet_torchvision import densenet121
from encoder import densenet121_encoder
from attention_decoder import rnn_decoder

import random

#from config import *
from utils import load_dict
from data_iter import dataIterator
from dataset import custom_dataset

from custom_collate_fn import collate_fn

from train_function import train
from test_function import test

datasets=['./train.pkl','./20K/train.txt', './20K/formulas.txt']
valid_datasets=['./test.pkl', './20K/test.txt', './20K/formulas.txt']
dictionaries=['./dictionary.txt']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_AVALIABLE = True if torch.cuda.is_available() else False

epoches = 200
batch_size=6

teacher_forcing_ratio = 1

gpu = [0]

lr_rate = 0.0001
# flag to remember when to change the learning rate
flag = 0
# exprate
exprate = 0

worddicts, len_dict = load_dict(dictionaries[0])

#load train data and test data
maxlen=48
maxImagesize= 100000
train_dataset, train_label = dataIterator(
    datasets[0],
    datasets[1],
    datasets[2],
    worddicts,
    maxlen=maxlen,
    maxImagesize=maxImagesize)

test_dataset, test_label = dataIterator(
    valid_datasets[0],
    valid_datasets[1],
    valid_datasets[2],
    worddicts,
    maxlen=maxlen,
    maxImagesize=maxImagesize)

len_train_data = len(train_dataset)

image_train_dataset = custom_dataset(train_dataset, train_label)
image_test_dataset = custom_dataset(test_dataset, test_label)

# loader: input->(c,w,h) / output->(b,c+1, (h_pad; h_mask), (w_pad; w_mask))
train_loader = torch.utils.data.DataLoader(
    dataset = image_train_dataset,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers = 2
)

test_loader = torch.utils.data.DataLoader(
    dataset = image_test_dataset,
    batch_size = batch_size,
    collate_fn = collate_fn,
    num_workers = 2
)

# encoder: input->(b, c, h, w) / output->(b, 1024, h/16, w/16)
encoder = densenet121_encoder().to(device)
encoder_weight_pth = r'./model/densenet121-a639ec97.pth'
pretrained_dict = torch.load(encoder_weight_pth)
encoder_dict = encoder.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
encoder_dict.update(pretrained_dict)
encoder.load_state_dict(encoder_dict)

hidden_size = 256
attn_decoder = rnn_decoder(hidden_size, len_dict, dropout_p=0.5).to(device)

if CUDA_AVALIABLE:
    encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
    attn_decoder = torch.nn.DataParallel(attn_decoder, device_ids=gpu)

criterion = nn.NLLLoss()
# 220 == dict_len - 1
decoder_input_init = torch.LongTensor([220]*batch_size).to(device)
decoder_hidden_init = torch.randn(batch_size, 1, hidden_size).to(device)
nn.init.xavier_uniform_(decoder_hidden_init)

epoches = 1
for epoch in range(epoches):
    train(encoder, attn_decoder, train_loader, batch_size, hidden_size, gpu, len_train_data, epoch, criterion, decoder_input_init, decoder_hidden_init)
    test(encoder, attn_decoder, test_loader, batch_size, device)
