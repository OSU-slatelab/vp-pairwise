import json
import random
import pdb
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import math
from collections import defaultdict
from sklearn.metrics import f1_score

def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True),
                         2, keepdim=True).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')

def get_keys(ind, data, num):
    idx = []
    xx = [len(t) for t in ind]
    for i,lst in enumerate(ind):
        random.shuffle(lst)
        idx.extend(lst[:num])
    return [data[i] for i in idx]

def get_cls_list(data,nclasses):
    full_data_labels = [json.loads(x)["label"] for x in data]
    cls_num_list = [0]*nclasses
    for i in full_data_labels:
        cls_num_list[i] += 1
    return cls_num_list

def split_dev(all_data, label_data, dev_ratio):
    if label_data is not None:
        full_data = label_data + all_data
    else:
        full_data = all_data
    label_data_dict = defaultdict(list)
    full_data_js = [json.loads(x) for x in full_data]
    data_val = []
    data_train = []
    for x in full_data_js:
        label_data_dict[x["label"]].append(x["text"])
    for label,texts in label_data_dict.items():
        alen = len(texts) * dev_ratio
        dev_len = math.floor(alen)
        dtexts = texts[:dev_len]
        ttexts = texts[dev_len:]
        data_val.extend([json.dumps({"label":label,"text":txt}) for txt in dtexts])
        data_train.extend([json.dumps({"label":label,"text":txt}) for txt in ttexts])
    return data_train, data_val

def split2folds(data, num_folds = 5):
    data_per_fold = math.floor(1. * len(data) / num_folds)
    folds = []
    for i in range(num_folds):
        folds.append(data[data_per_fold * i:(i+1) * data_per_fold])
    rem_data = data[num_folds * data_per_fold:]
    rem_len = len(rem_data)
    for i in range(rem_len):
        folds[i].append(rem_data[i])
    return folds

class QEval(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.quintiles = self.split_list(self.get_quintiles(), 5)

    def split_list(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def get_quintiles(self):
        full_data_labels = [json.loads(x)["label"] for x in self.train_data]
        lbl_dist = defaultdict(int)
        for lbl in full_data_labels:
            lbl_dist[lbl] += 1
        lbl_dist_tup = [(lbl, freq) for lbl, freq in lbl_dist.items()]
        tdist_sorted = sorted(lbl_dist_tup, key=lambda tup: tup[1])
        lbl_sorted = [x[0] for x in tdist_sorted]
        test_data_labels = [json.loads(x)["label"] for x in self.test_data]
        trim_sorted = [x for x in lbl_sorted if x in test_data_labels]

        return trim_sorted
    
    def evaluate(self, y_pred, y_true):
        evl_acc = []
        evl_f1 = []
        for q in self.quintiles:
            yt,yp=[],[]
            for i,y in enumerate(y_true):
                if y in q:
                    yt.append(y)
                    yp.append(y_pred[i])
            evl_acc.append((np.asarray(yt)==np.asarray(yp)).astype(int).mean())
            evl_f1.append(f1_score(yt,yp,list(set(yt)),average='macro'))
        return evl_acc, evl_f1


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin_pos, margin_neg):
        super(ContrastiveLoss, self).__init__()
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = torch.norm(F.normalize(output1,dim=1)-F.normalize(output2,dim=1), dim=1)
        losses = 1.0 * (target.float() * torch.pow(torch.clamp(distances - self.margin_pos, min=0.0), 2) +
                                  (1 + -1 * target).float() * torch.pow(torch.clamp(self.margin_neg - distances, min=0.0), 2))
        return losses.mean() if size_average else losses.sum()


class VarBeta(object):
    def __init__(self,cls_num_list,scale=15,bmax=15,bmin=0.5):
        self.bmax = bmax
        self.bmin = bmin
        self.cls_num_list = cls_num_list
        self.quints = self.quintiles()

    def split_list(self, a, n):
        k, m = divmod(len(a), n)
        x = [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
        return x

    def quintiles(self):
        class_sort = sorted(range(len(self.cls_num_list)), key=lambda k: self.cls_num_list[k])
        quints = self.split_list(class_sort, 5)
        return quints

    def sample(self,target1,target2):
        betas = []
        c1 = [x for x in target1.cpu().tolist()]
        c2 = [x for x in target2.cpu().tolist()]
        tups = [x for x in zip(c1,c2)]
        for t in tups:
            if t[0] in self.quints[0] or t[1] in self.quints[0]:
                betas.append(self.bmax) #20
            elif t[0] in self.quints[1] or t[1] in self.quints[1]:
                betas.append(5.0) #5.0
            elif t[0] in self.quints[2] or t[1] in self.quints[2]:
                betas.append(1.0)
            elif t[0] in self.quints[3] or t[1] in self.quints[3]:
                betas.append(0.5) #0.5
            elif t[0] in self.quints[4] or t[1] in self.quints[4]:
                betas.append(self.bmin) #0.05 0.1-bert
    
        return betas

class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = list()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindev-data', type=str, default='traindev.json',
                        help='location of the training data, should be a json file from vp-tokenizer')
    parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a json file from vp-tokenizer')
    parser.add_argument('--valid-data', type=str, default='',
                        help='location of the validation data, should be a json file from vp-tokenizer')
    parser.add_argument('--label-data', type=str, default='',
                        help='location of the label map (int -> sentence) in json format')
    parser.add_argument('--dev-ratio', type=float, default=0.1,
                        help='fraction of train-dev data to use for dev')
    parser.add_argument('--validation', action='store_true',
                        help='use k-fold cross validation')
    parser.add_argument('--validation-log', type=str, default='rnn.attn.kfold.log',
                        help='location to store kfold logs')
    parser.add_argument('--lamb', type=float, default=0.25,
                        help='weight of 1-nearest-neighbor scores during inference')
    parser.add_argument('--encoder-type', type=str, default='rnn',
                        help='')
    parser.add_argument('--prebert-path', type=str, default='',
                        help='path to pretrained bert model')
    parser.add_argument('--bert-pooling', type=str, default='mean',
                        help='mean pool the bert outputs, otherwise use [CLS] representation')
    parser.add_argument('--ploss_wt', type=float, default=0.25,
                        help='weight of contrastive loss during training')
    parser.add_argument('--beta-max', type=float, default=15.0,
                        help='max alpha value to use for the beta distribution')
    parser.add_argument('--beta-min', type=float, default=0.5,
                        help='min alpha value to use for the beta distribution')
    parser.add_argument('--num-filters', type=int, default=300,
                        help='number of filters to use for 2D convolution')
    parser.add_argument('--filter-ht', type=int, default=300,
                        help='height of each filter')
    parser.add_argument('--n-gram', nargs="*", type=int, default=[1,2,3],
                        help='width of the filter')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers in BiLSTM')
    parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--attention-hops', type=int, default=16,
                        help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--lr', type=float, default=.00004,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--stage2', type=int, default=20,
                        help='number of epochs to run in boosting stage')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--dictionary', type=str, default='dict.json',
                        help='path to save the dictionary, for faster corpus loading')
    parser.add_argument('--word-vector', type=str, default='',
                        help='path for pre-trained word vectors (e.g. GloVe), should be a PyTorch model.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--test-bsize', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--nclasses', type=int, default=348,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--penalization-coeff', type=float, default=0, 
                        help='the attention orthogonality penalization coefficient')
    parser.add_argument('--margin-pos', type=float, default=1.0,
                        help='margin pos for contrastive loss')
    parser.add_argument('--margin-neg', type=float, default=1.0,
                        help='margin neg for contrastive loss')
    parser.add_argument('--num-pos', type=int, default=100000,
                        help='number of positive pairs to sample')
    parser.add_argument('--samp-freq', type=int, default=1,
                        help='frequency of sampling')
    parser.add_argument('--rsamp', action='store_true',
                        help='whether to resample pairs')
    parser.add_argument('--num-keys', type=int, default=10,
                        help='keys for evaluation')
    
    return parser

