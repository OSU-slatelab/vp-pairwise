import numpy as np
import pdb
from time import sleep,time
import sys
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import BertTokenizer
from collections import defaultdict, OrderedDict
import itertools
import random
import json

FAC1 = 3
FAC2 = 2

def shuffle_pairs(p,l):
    x = list(zip(p,l))
    random.shuffle(x)
    ps,ls = [list(t) for t in zip(*x)]
    return ps, ls

def get_random_samples(a, b, n):
    n_prod = len(a) * len(b)
    indices = random.sample(range(n_prod), n)
    return [(a[idx % len(a)], b[idx // len(a)]) for idx in indices]

def package_pair(data, idx_pair, label, dictionary, bert=False, is_train=True):
    
    def package1(data,dictionary,is_train=True):
        data = [json.loads(x) for x in data]
        dat = [[dictionary.word2idx[y] for y in x['text']] for x in data]
        maxlen = 0
        for item in dat:
            maxlen = max(maxlen, len(item))
        targets = [x['label'] for x in data]
        maxlen = min(maxlen, 500)
        for i in range(len(data)):
            if maxlen < len(dat[i]):
                dat[i] = dat[i][:maxlen]
            else:
                for j in range(maxlen - len(dat[i])):
                    dat[i].append(dictionary.word2idx['<pad>'])
        with torch.set_grad_enabled(is_train):
            dat = torch.tensor(dat, dtype=torch.long)
            targets = torch.tensor(targets, dtype=torch.long)
        return dat.t(),targets

    def package2(data,dictionary,is_train=True):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        data = [json.loads(x) for x in data]
        dat = tokenizer([" ".join(x['text']) for x in data], return_tensors="pt", padding=True, truncation=True)
        targets = [x['label'] for x in data]
        targets = torch.tensor(targets, dtype=torch.long)
        return dat,targets

    idx1,idx2 = [list(t) for t in zip(*idx_pair)] 
    pdata1 = [data[x] for x in idx1]
    pdata2 = [data[x] for x in idx2]
    if not bert:
        data1,target1 = package1(pdata1, dictionary, is_train)
        data2,target2 = package1(pdata2, dictionary, is_train)
    else:
        data1,target1 = package2(pdata1, dictionary, is_train)
        data2,target2 = package2(pdata2, dictionary, is_train)
    return data1, data2

def bt_on_device(tokenizer, device, bert=False):
    if not bert:
        return tokenizer.to(device)
    t2 = {}
    for key, val in tokenizer.items():
        t2[key] = val.to(device)
    return t2

class PairedData:
    def __init__(self, nclasses, data, num_pos=50000):
        self.pidx_pairs = []
        self.ci = []
        self.class_indices = []
        jsdata = [json.loads(x) for x in data]
        self.data = data
        self.nclasses = nclasses
        self.max_pts = 3000
        self.num_pos = num_pos
        for i in range(nclasses):
            idx_i = [ii for ii,x in enumerate(jsdata) if x['label'] == i]
            if len(idx_i) > 0:
                self.ci.append(idx_i)
        self.sample_pairs()

    def sample_pairs(self):
        self.pidx_pairs = []
        self.class_indices = []
        for ipts in self.ci:
            random.shuffle(ipts)
            ppts = ipts[0:self.max_pts]
            self.class_indices.append(ppts)
            if len(ppts) <= 1:
                continue
            pairs = list(itertools.combinations(ppts,r=2))
            random.shuffle(pairs)
            self.pidx_pairs.append(pairs)
        rnclasses = len(self.pidx_pairs)
        negcls = len(self.class_indices)
        avail = sum([len(x) for x in self.pidx_pairs])
        if self.num_pos > avail:
            num_pos = avail
            self.fac1 = 3.0
            self.fac2 = 1.0
        else:
            num_pos = self.num_pos
            self.fac1 = 3.0
            self.fac2 = 2.0
        num_pos = min(num_pos,sum([len(x) for x in self.pidx_pairs]))
        self.fpos = self.positive_pairs(rnclasses, num_pos)
        self.fneg = self.negative_pairs(negcls, int(self.fac1*num_pos))
        random.shuffle(self.fpos)
        random.shuffle(self.fneg)

    def sample_pairs2(self, num_neg=2):
        class_weight = []
        for ipts in self.ci:
            class_weight.append(1./len(ipts))
        total = sum(class_weight)
        class_weight = [1.*x/total for x in class_weight]
        classes = list(range(len(self.ci)))
        self.fpos = []
        self.fneg = []
        spos = {}
        sneg = {}
        for cls, ipts in enumerate(self.ci):
            for i, idx in enumerate(ipts):
                pool = ipts[:]
                pool.remove(idx)
                for j in range(50):
                    pos_c = random.choice(pool)
                    tup = (min(idx,pos_c),max(idx,pos_c))
                    if str(tup) not in spos:
                        self.fpos.append(tup)
                        spos[str(tup)] = True
                        break
                pool = classes[:]
                class_weight_copy = class_weight[:]
                pool.remove(cls)
                del class_weight_copy[cls]
                class_rewt = [1.*x/sum(class_weight_copy) for x in class_weight_copy]
                for j in range(num_neg):
                    for k in range(20):
                        neg_class = np.random.choice(pool, 1, p=class_rewt)[0]
                        neg_c = random.choice(self.ci[neg_class])
                        tup = (min(idx,neg_c),max(idx,neg_c))
                        if str(tup) not in sneg:
                            self.fneg.append(tup)
                            sneg[str(tup)] = True
                            break
        random.shuffle(self.fpos)
        random.shuffle(self.fneg)
        
    def positive_pairs(self,nclasses,num):
        proportions = []
        total = sum([len(x) for x in self.ci])
        for i, lst in enumerate(self.ci):
            proportions.append(1. * len(lst)/total)
        pos_idx_pairs = []
        st = [0]*nclasses
        #fi = [num // nclasses]*nclasses
        fi = []
        for por in proportions:
            fi.append(int(por * num))
        flag = True
        while flag:
            for i in range(nclasses):
                pos_idx_pairs.extend(self.pidx_pairs[i][st[i]:fi[i]])
                if len(pos_idx_pairs) >= num:
                    flag = False
                    break
                st[i] = fi[i]
                fi[i] = st[i]+1                
        return pos_idx_pairs

    def negative_pairs(self,nclasses,num):
        bin_pairs = list(itertools.combinations(range(nclasses),r=2)) 
        bin_sizes = [len(self.class_indices[b1])*len(self.class_indices[b2]) for b1,b2 in bin_pairs]
        q,r = num // len(bin_pairs), num % len(bin_pairs)

        num_per_bin = [q]*len(bin_pairs)
        for i in range(r):
            num_per_bin[i]+=1
        x=[min(num_per_bin[i],bs) for i,bs in enumerate(bin_sizes)]
        diff = num - sum(x)
        flag=True
        while flag and diff > 0:
            for i in range(len(x)):
                if bin_sizes[i] > x[i]:
                    x[i] += 1
                if sum(x) >= num:
                    flag=False
                    break

        neg_idx_pairs = []
        for i,(b1,b2) in enumerate(bin_pairs):
            bin1 = self.class_indices[b1]
            bin2 = self.class_indices[b2]
            neg_idx_pairs.extend(get_random_samples(bin1, bin2, x[i]))

        return neg_idx_pairs

    def resample(self, full, num, model, dictionary, device, bert=False, bsz=32, typ='neg'):
        model.eval()
        cpu = torch.device('cpu')
        dists = []
        if typ == 'pos':
            print('Postive examples\n')
        else:
            print('Negative examples\n')
        for batch, i in enumerate(range(0, len(full), bsz)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(full), bsz))))  
            last = min(len(full), i+bsz)
            intoks = full[i:last]
            with torch.no_grad():
                hidden = None
                data1, data2 = package_pair(self.data, intoks, [0]*bsz, dictionary, bert=bert, is_train=False)
                data1, data2 = bt_on_device(data1, device, bert=bert), bt_on_device(data2, device, bert=bert)
                if not bert:
                    hidden = model.init_hidden(data1.size(1))
                _, r1, r2, _, _ = model.forward(data1, data2, hidden)
                dist = torch.norm(F.normalize(r1,dim=1)-F.normalize(r2,dim=1), dim=1)
                dists.append(dist.to(cpu))
            del data1, data2, hidden, r1, r2
            torch.cuda.empty_cache()

            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        dist_mat = torch.cat(dists)
        lar = (typ == 'pos')
        _, min_idx = torch.topk(dist_mat,num,dim=0,largest=lar)
        midx = min_idx.tolist()
        midx2 = [x for i,x in enumerate(midx)]
        return [full[x] for x in midx2]

    def sample(self, epoch, model, dictionary, device, bert=False, rsamp=True, bsz=32):
        npos = int(len(self.fpos) / self.fac2)
        nneg = int(len(self.fneg) / (self.fac1 * self.fac2/2))
        if epoch == 1 or not rsamp:
            neg = self.fneg[:nneg]
            pos = self.fpos[:npos]
        else:
            pos = self.resample(self.fpos, npos, model, dictionary, device, bert, bsz, 'pos')
            neg = self.resample(self.fneg, nneg, model, dictionary, device, bert, bsz, 'neg')
            npos = len(pos)
            nneg = len(neg)
        plbl = [1]*npos
        nlbl = [0]*nneg
        dt = pos+neg
        lbl = plbl+nlbl
        print(f'Data size = {len(lbl)}')
        return shuffle_pairs(dt,lbl)
