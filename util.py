import json
import random
import pdb
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
from collections import defaultdict

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        #b = x * torch.log(x)
        b = -1.0 * b.sum(dim=1)
        return b.mean()

class EucLoss(nn.Module):
    def __init__(self, euc_mar=0.0):
        super(EucLoss, self).__init__()
        self.margin = euc_mar
        
    def forward(self, x, target):
        #distances = torch.norm(F.normalize(x, dim=1)-F.normalize(target, dim=1), dim=1)
        distances = torch.norm(x - target, dim=1)
        losses = torch.clamp(distances - self.margin, min=0.0)
        return losses.mean()

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

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

    def forward(self, output1, output2, target, size_average=True, norm=True, mpos=None, mneg=None):
        if norm:
            distances = torch.norm(F.normalize(output1,dim=1)-F.normalize(output2,dim=1), dim=1)
        else:
            distances = torch.norm(output1-output2, dim=1)
        #distances = (output2 - output1).pow(2).sum(1)  # squared distances
        #losses = 0.5 * (target.float() * F.relu(distances - self.margin_pos +
        #                (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        if mpos is None and mneg is None:
            losses = 1.0 * (target.float() * torch.pow(torch.clamp(distances - self.margin_pos, min=0.0), 2) +
                                      (1 + -1 * target).float() * torch.pow(torch.clamp(self.margin_neg - distances, min=0.0), 2))
        elif mneg is None:
            losses = 1.0 * (target.float() * torch.pow(torch.clamp(distances - mpos, min=0.0), 2) +
                                      (1 + -1 * target).float() * torch.pow(torch.clamp(self.margin_neg - distances, min=0.0), 2))
        elif mpos is None:
            losses = 1.0 * (target.float() * torch.pow(torch.clamp(distances - self.margin_pos, min=0.0), 2) +
                                      (1 + -1 * target).float() * torch.pow(torch.clamp(mneg - distances, min=0.0), 2))
        else:
            losses = 1.0 * (target.float() * torch.pow(torch.clamp(distances - mpos, min=0.0), 2) +
                                      (1 + -1 * target).float() * torch.pow(torch.clamp(mneg - distances, min=0.0), 2))
        return losses.mean() if size_average else losses.sum()

class MarginContrastiveLoss(nn.Module):

    def __init__(self, cls_num_list, m_pmax, m_pmin, m_nmax, m_nmin, scale, device):
        super(MarginContrastiveLoss, self).__init__()
        self.m_pmax = m_pmax
        self.m_pmin = m_pmin
        self.m_nmax = m_nmax
        self.m_nmin = m_nmin
        self.scale = scale
        self.device = device
        self.cls_num_list = cls_num_list

    def get_margins(self,target1,target2=None):
        if target2==None:
            count = [self.cls_num_list[x]+1 for x in target1.tolist()]
            count = torch.tensor(count, dtype=torch.float)
            mp = self.m_pmax - (self.m_pmax - self.m_pmin) * torch.exp(-1.0 * (count - 2.0)/self.scale)
            return mp.to(self.device)
        count1 = [self.cls_num_list[x] for x in target1.tolist()]
        count2 = [self.cls_num_list[x] for x in target2.tolist()]
        count = [min(x) for x in zip(count1,count2)]
        count = torch.tensor(count, dtype=torch.float)
        #mp = self.m_pmax - (self.m_pmax - self.m_pmin) * torch.exp(-1.0 * (count - 2.0)/self.scale)
        #mn = self.m_nmin + (self.m_nmax - self.m_nmin) * torch.exp(-1.0 * (count - 1.0)/self.scale)
        mp = self.m_pmin + ((self.m_pmax - self.m_pmin)/np.power((max(self.cls_num_list)-2),0.25))*torch.pow(count-2,0.25)
        mn = self.m_nmax - ((self.m_nmax - self.m_nmin)/np.power((max(self.cls_num_list)-1),0.25))*torch.pow(count-1,0.25)
        mp[mp!=mp]=0
        assert True not in torch.isnan(mp).tolist()
        assert True not in torch.isnan(mn).tolist()
        return mp.to(self.device), mn.to(self.device)
        
    def forward(self, output1, output2, target, target1, target2, size_average=True,norm=True):
        if norm:
            distances = torch.norm(F.normalize(output1,dim=1)-F.normalize(output2,dim=1), dim=1)
        else:
            distances = torch.norm(output1-output2, dim=1)
        mp, mn = self.get_margins(target1,target2)
        losses = 1.0 * (target.float() * torch.pow(torch.clamp(distances - mp, min=0.0), 2) +
                                      (1 + -1 * target).float() * torch.pow(torch.clamp(mn - distances, min=0.0), 2))
        return losses.mean() if size_average else losses.sum()

class VarBeta(object):
    def __init__(self,cls_num_list,scale=15,bmax=15,bmin=0.5):
        self.bmax = bmax
        self.bmin = bmin
        self.cls_num_list = cls_num_list
        self.scale = 15
        self.quints = self.quintiles()

    def split_list(self, a, n):
        k, m = divmod(len(a), n)
        x = [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
        return x #(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def quintiles(self):
        class_sort = sorted(range(len(self.cls_num_list)), key=lambda k: self.cls_num_list[k])
        quints = self.split_list(class_sort, 5)
        return quints

    def sample(self,target1,target2):
        #count1 = [self.cls_num_list[x] for x in target1.cpu().tolist()]
        #count2 = [self.cls_num_list[x] for x in target2.cpu().tolist()]
        #count = [min(x) for x in zip(count1,count2)]
        #count = torch.tensor(count, dtype=torch.float)
        #betas = self.bmin + (self.bmax - self.bmin) * torch.exp(-1.0 * (count - 1.0)/self.scale)
        #betas = []
        #for cn in count:
        #    if cn < 50:
        #        betas.append(1.0)
        #    elif cn >= 50 and cn < 100:
        #        betas.append(5.0)
        #    else:
        #        betas.append(15.0)
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
    
        return betas #betas.tolist() betas

class DataTrans(nn.Module):
    def __init__(self,dropout=0.00):
        super(DataTrans, self).__init__()
        self.l1 = nn.Linear(2,2)
        self.l2 = nn.Linear(2,2)
        self.l3 = nn.Linear(2,1)
        self.drop = nn.Dropout(dropout)
        
    def forward(self,input):
        return self.l3(self.drop(torch.tanh(self.l2(self.drop(torch.tanh(self.l1(self.drop(input))))))))
    
class QInfo(object):
    def __init__(self,train_data,hl_type='none'):
        self.label2quint = self.get_quint(train_data)
        if hl_type == 1:
            self.hl_coeff = [1.0,0.5,-0.001,-0.5,-1.0]
        elif hl_type == 2:
            self.hl_coeff = [0.2,0.1,-0.001,-0.1,-0.2]
        elif hl_type == 3:
            self.hl_coeff = [0.1,0.05,-0.001,-0.05,-0.1]
        else:
            self.hl_coeff = [1.0]*5
            

    def split_list(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def get_quint(self,data):
        full_data_labels = [json.loads(x)["label"] for x in data]
        lbl_dist = defaultdict(int)
        for lbl in full_data_labels:
            lbl_dist[lbl]+=1
        lbl_dist_tup = [(lbl,freq) for lbl,freq in lbl_dist.items()]
        tdist_sorted = sorted(lbl_dist_tup, key=lambda tup: tup[1]) 
        lbl_sorted = [x[0] for x in tdist_sorted]
        quints = self.split_list(lbl_sorted,5)
        l2q = {}
        for i,q in enumerate(quints):
            for lbl in q:
                l2q[lbl] = i
        return l2q

    def tgt2quint(self,target1,target2):
        q1 = [self.label2quint[x] for x in target1.cpu().tolist()]
        q2 = [self.label2quint[x] for x in target2.cpu().tolist()]
        qf = [[x[0],x[1]] for x in zip(q1,q2)]
        #hc = [self.hl_coeff[i] for i in qf]
        qf = torch.tensor(qf,dtype=torch.long)    
        #hc = torch.tensor(hc,dtype=torch.float)
        return qf

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
    parser.add_argument('--pooling', type=str, default='all',
                        help='pooling strategy; choices: [all, mean, max]')
    parser.add_argument('--att-pooling', type=str, default='mean',
                        help='pooling strategy for dpattention; choices: [max, agg, mean]')
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
    parser.add_argument('--reserved', type=int, default=0,
                        help='number of representation heads to reserve for boosting')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--nfc', type=int, default=512,
                        help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--ncat', type=int, default=8,
                        help='number of categories for each random variable in the sparse intermediate representation')
    parser.add_argument('--penalty', type=str, default='overlap',
                        help='attention penalty; options: overlap, uncover')
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
    parser.add_argument('--save', type=str, default='saved_models/reproduced_res.pt',
                        help='path to save the final model')
    parser.add_argument('--dictionary', type=str, default='dict.json',
                        help='path to save the dictionary, for faster corpus loading')
    parser.add_argument('--word-vector', type=str, default='',
                        help='path for pre-trained word vectors (e.g. GloVe), should be a PyTorch model.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--test-bsize', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--shuffle', action='store_true',
                        help='re-shuffle training data at every epoch')
    parser.add_argument('--nclasses', type=int, default=348,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--penalization-coeff', type=float, default=0, 
                        help='the attention orthogonality penalization coefficient')
    parser.add_argument('--entropy-coeff', type=float, default=0)
    parser.add_argument('--eval-on-test', action='store_true',
                        help='evaluate test set after training')
    parser.add_argument('--shared', action='store_true',
                        help='shared weights for each hop')
    parser.add_argument('--attention-type', type=str, default='self')
    parser.add_argument('--margin-pos', type=float, default=1.0,
                        help='margin pos for contrastive loss')
    parser.add_argument('--margin-neg', type=float, default=1.0,
                        help='margin neg for contrastive loss')
    parser.add_argument('--pairwise', action='store_true',
                        help='whether to do pairwise training')
    parser.add_argument('--num-pos', type=int, default=100000,
                        help='number of positive pairs to sample')
    parser.add_argument('--samp-freq', type=int, default=1,
                        help='frequency of sampling')
    parser.add_argument('--rpos', action='store_true',
                        help='whether to resample positives')
    parser.add_argument('--rneg', action='store_true',
                        help='whether to resample negatives')
    parser.add_argument('--nneighbors', type=int, default=1,
                        help='value of k for knn')
    parser.add_argument('--norm-dist', action='store_true',
                        help='whether to normalize representations')
    parser.add_argument('--num-keys', type=int, default=10,
                        help='keys for evaluation')
    
    return parser

