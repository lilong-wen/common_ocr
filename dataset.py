import torch
import numpy as np


class custom_dataset(torch.utils.data.Dataset):


    def __init__(self, data, data_label):
        self.data = data
        self.data_label = data_label


    def __getitem__(self, index):
        train_item = torch.from_numpy(np.array(self.data[index]))
        label_item = torch.from_numpy(np.array(self.data_label[index])).type(torch.LongTensor)

        size = train_item.size()
        train_item = train_item.view(1,size[1],size[2])
        label_item = label_item.view(-1)
        return train_item, label_item


    def __len__(self):
        return len(self.data)
