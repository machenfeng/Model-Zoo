import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class jupyter_args(dict):
    def __init__(self, **kwargs):
        super(jupyter_args, self).__init__(**kwargs)
    
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = jupyter_args(value)
        return value
    
    
def dict_build(df, cols, use_char=True, save_dir=None):

    word = []
    for col in cols:
        word += df[col].tolist()
    word = list(set(word))

    if use_char is True:
        word = ['UKN'] + list(''.join(word))
    else:
        word = ['UKN'] + list(' '.join(word).split(' '))

    word = list(set(word))
    idx = range(1, len(word)+1)  # idx start from 1 in order to set word2idx['PAD'] = 0

    items = zip(word, idx)
    word2idx = dict((i, j) for i, j in items)
    word2idx['PAD'] = 0

    if save_dir is not None:
        output = open(save_dir, 'wb')
        pickle.dump(word2idx, output)
        output.close()

    return word2idx


def input_clip(x1, x2):
    assert x1.size() == x2.size()
    bs_range = range(x1.size(0))
    list1 = [x1[i].nonzero().nelement() for i in bs_range]
    list2 = [x2[i].nonzero().nelement() for i in bs_range]
    
    batch_max_length = max(list1 + list2)
    
    x1 = x1[:, :batch_max_length]
    x2 = x2[:, :batch_max_length]
    
    return x1, x2


class DataBuilder(data.Dataset):
    
    def __init__(self, df, x1_col, x2_col, max_length, idx_dict, use_char):
        
        self.df = df
        self.df.set_index(np.arange(self.df.shape[0]))
        
        self.x1 = self.df[x1_col].tolist()
        self.x2 = self.df[x2_col].tolist()
        self.label = self.df.label.tolist()
        
        self.max_length = max_length
        
        self.idx_dict = idx_dict
        self.use_char = use_char
        
    def __getitem__(self, index):
        
        #构造idx特征
        x1 = self.x1[index]
        x2 = self.x2[index]
        label = self.label[index]

        x1 = self.get_idx(x1, self.idx_dict, self.max_length, self.use_char)
        x1 = torch.LongTensor(x1)
        
        x2 = self.get_idx(x2, self.idx_dict, self.max_length, self.use_char)
        x2 = torch.LongTensor(x2)        
        
        label = torch.LongTensor([label])

        return x1, x2, label

    def __len__(self):
        return len(self.label)

    def get_idx(self, sentence, idx_dict, max_length, use_char):

        idxs = []
        
        if use_char:
            sentence = list(str(sentence))
        else:
            sentence = list(str(sentence).split(' '))
            
        pad_index = idx_dict['PAD']
        assert pad_index == 0
        
        for word in sentence:
            if word in idx_dict.keys():
                idx = idx_dict[word]
            else:
                idx = idx_dict['UKN']
            idxs.append(idx)

        L = len(idxs)
        if L >= max_length:
            idxs = idxs[:max_length]
        else:
            idxs = idxs + (max_length - L) * [pad_index]

        return idxs

