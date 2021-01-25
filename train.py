import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import argparse

import json

#from Densenet_torchvision import densenet121
from encoder import densenet121_encoder
# from Attention_RNN import AttnDecoderRNN as rnn_decoder
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=6, help='batch_size')
    parser.add_argument('--epoches', default=1, help='epoches')
    parser.add_argument('--teacher_forcing_ratio', default=1, help='teacher forcing ratio')
    parser.add_argument('--gpu', default=[0], help='gpu num list')
    parser.add_argument('--lr_rate', default=0.0001, help='learning rate')
    parser.add_argument('--flag', default=0, help='flag to remember when to change the learning rate')
    parser.add_argument('--exprate', default=0, help='exprate')
    parser.add_argument('--maxlen', default=48, help='maxlen')
    parser.add_argument('--maxImagesize', default=100000, help='maxImagesize')
    parser.add_argument('--hidden_size', default=256, help='hidden size')
    parser.add_argument('--len_train_data', default=0, help='len_train_data')

    opt = parser.parse_args()

    '''
    epoches = 200
    batch_size=6
    teacher_forcing_ratio = 1
    gpu = [0]
    lr_rate = 0.0001
    # flag to remember when to change the learning rate
    flag = 0
    # exprate
    exprate = 0
    maxlen=48
    maxImagesize= 100000
    '''

    worddicts, len_dict = load_dict(dictionaries[0])

    #load train data and test data
    train_dataset, train_label = dataIterator(
        datasets[0],
        datasets[1],
        datasets[2],
        worddicts,
        maxlen=opt.maxlen,
        maxImagesize=opt.maxImagesize)

    test_dataset, test_label = dataIterator(
        valid_datasets[0],
        valid_datasets[1],
        valid_datasets[2],
        worddicts,
        maxlen=opt.maxlen,
        maxImagesize=opt.maxImagesize)

    len_train_data = len(train_dataset)
    opt.len_train_data = len_train_data

    image_train_dataset = custom_dataset(train_dataset, train_label)
    image_test_dataset = custom_dataset(test_dataset, test_label)

    # loader: input->(c,w,h) / output->(b,c+1, (h_pad; h_mask), (w_pad; w_mask))
    train_loader = torch.utils.data.DataLoader(
        dataset = image_train_dataset,
        batch_size = opt.batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = 2
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = image_test_dataset,
        batch_size = opt.batch_size,
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

    attn_decoder = rnn_decoder(opt.hidden_size, len_dict, dropout_p=0.5).to(device)

    if CUDA_AVALIABLE:
        encoder = torch.nn.DataParallel(encoder, device_ids=opt.gpu)
        attn_decoder = torch.nn.DataParallel(attn_decoder, device_ids=opt.gpu)

    criterion = nn.NLLLoss()
    # 220 == dict_len - 1
    decoder_input_init = torch.LongTensor([220]*opt.batch_size).to(device)
    decoder_hidden_init = torch.randn(opt.batch_size, 1, opt.hidden_size).to(device)
    nn.init.xavier_uniform_(decoder_hidden_init)

    epoches = 1
    for epoch in range(epoches):
        '''
        train(encoder, attn_decoder, train_loader, batch_size, \
              hidden_size, gpu, len_train_data, epoch, criterion, \
              decoder_input_init, decoder_hidden_init)
        '''
        train(encoder, attn_decoder, train_loader, criterion, \
              decoder_input_init, decoder_hidden_init, epoch, opt)
        test(encoder, attn_decoder, test_loader, batch_size, device)
