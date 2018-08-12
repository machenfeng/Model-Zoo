import time
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.utils.data import DataLoader

import utils
from utils import DataBuilder
from utils import jupyter_args

import de_att

logger_name = "mylog"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)

fh = logging.FileHandler("snli_train.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

train = pd.read_csv('data/train.csv')
train['len'] = train.apply(lambda x: max(len(x.x1.split(' ')), len(x.x2.split(' '))), axis=1)
train = train.sort_values('len')
logger.info('train size %d' % train.shape[0])

dev = pd.read_csv('data/dev.csv')
dev['len'] = dev.apply(lambda x: max(len(x.x1.split(' ')), len(x.x2.split(' '))), axis=1)
dev = dev.sort_values('len')
logger.info('dev size %d' % dev.shape[0])

tokenizer = open('data/word2idx.pkl', 'rb')
tokenizer = pickle.load(tokenizer)

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pretrained_weights = open('data/pretrained_weights_42b.pkl', 'rb')
pretrained_weights = pickle.load(pretrained_weights)
pretrained_weights = torch.FloatTensor(pretrained_weights)

args = jupyter_args(embed_dim=300, 
                    hidden_state=1024,
                    max_length=999,
                    num_classes=3,
                    batch_size=128,
                    pretrained_weights=pretrained_weights,
                    learning_rate=0.007,
                    max_grad_norm=5,
                    logger_interval=300,
                    epoch_num=50)

logger.info('embed_dim %d' % args['embed_dim'])
logger.info('hidden_state %d' % args['hidden_state'])
logger.info('max_length %d' % args['max_length'])
logger.info('num_classes %d' % args['num_classes'])
logger.info('batch_size %d' % args['batch_size'])
logger.info('learning_rate %g' %args['learning_rate'])
logger.info('max_grad_norm %d' %args['max_grad_norm'])
logger.info('logger_interval %d' %args['logger_interval'])
logger.info(' ')

trainbulider = DataBuilder(train, 'x1', 'x2', args.max_length, tokenizer, use_char=False)
trainloader = DataLoader(trainbulider, args.batch_size, shuffle=False)

devbulider = DataBuilder(dev, 'x1', 'x2', args.max_length, tokenizer, use_char=False)
devloader = DataLoader(devbulider, args.batch_size, shuffle=False)

model = de_att.model(args).to(device)
param = filter(lambda p: p.requires_grad, model.parameters())

lossfunc = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adagrad(param, args.learning_rate, weight_decay=1e-5)

for i in range(args.epoch_num):
    
    t1 = time.time()
    
    epoch_pred = []
    epoch_true = []
    epoch_loss = 0
    
    interval_pred = []
    interval_true = []
    interval_loss = 0
    
    model.train()
    for k, (x1, x2, label) in enumerate(trainloader):
        
        x1, x2 = utils.input_clip(x1, x2)
        x1, x2 = x1.to(device), x2.to(device)
        label = label.view(-1).to(device)

        optimizer.zero_grad()

        logits = model(x1, x2)
        loss = lossfunc(logits, label)
        loss.backward()
        
        ''' get param and grad norm '''
        grad_norm = 0.
        para_norm = 0.

        for m in model.modules():
            if isinstance(m, nn.Linear):
                grad_norm += m.weight.grad.data.norm() ** 2
                para_norm += m.weight.data.norm() ** 2
                if m.bias is True:
                    grad_norm += m.bias.grad.data.norm() ** 2
                    para_norm += m.bias.data.norm() ** 2

        grad_norm ** 0.5
        para_norm ** 0.5

        shrinkage = args.max_grad_norm / grad_norm
        
        if shrinkage < 1 :
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.weight.grad.data = m.weight.grad.data * shrinkage
                    if m.bias is True:
                        m.bias.grad.data = m.bias.grad.data * shrinkage
        
        optimizer.step()
        
        epoch_loss += loss.item()
        interval_loss += loss.item()
        
        predict = logits.max(1)[1]
        
        epoch_true += label.tolist()
        epoch_pred += predict.tolist()
        
        interval_true += label.tolist()
        interval_pred += predict.tolist()
        
        if (k + 1) % args.logger_interval == 0:
            t2 = time.time()
            time_cost = (t2 - t1)
            interval_acc = accuracy_score(interval_true, interval_pred)
            interval_loss /= args.logger_interval
            logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, para-norm %.3f, grad-norm %.3f, time_cost %.3fs,' %
                            (i + 1, k + 1, len(trainloader), interval_acc, interval_loss, para_norm, grad_norm, time_cost))
            t1 = time.time()
            interval_pred = []
            interval_true = []
            interval_loss = 0
        
    model.eval()
    
    train_acc = accuracy_score(epoch_true, epoch_pred)
    train_loss = epoch_loss / len(trainloader)
    
    dev_true, dev_pred, dev_loss = model.evaluate(model, devloader, lossfunc)
    dev_acc = accuracy_score(dev_true, dev_pred)
    
    logger.info('epoch %d, train_loss %.3f, train_acc %.3f' % (i + 1, train_loss, train_acc))
    logger.info('epoch %d, dev_loss %.3f, dec_acc %.3f' % (i + 1, dev_loss, dev_acc))

    model_path = 'model/de_att_epoch_%d.pkl' % (i + 1)
    torch.save(model.state_dict(), model_path) 
    logger.info('model saved')
        
torch.save(model.state_dict(), 'model/de_att_epoch.pkl') 
