from __future__ import print_function
from models import *

from util import *

import faiss
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer
from models import *
import json
import time
from time import sleep
import sys
import random
import os

from sklearn.metrics import f1_score 

def get_bucket(epoch, a=2, b=3):
    start = 0
    while True:
        for i in range(int(a)):
            start+=1
            if start == epoch:
                return True
        for i in range(int(b)):
            start+=1
            if start == epoch:
                return False
        

def one_hot(tgt,nclasses):
    otgt = torch.zeros(tgt.size(0),nclasses)
    otgt.scatter_(1,tgt.unsqueeze(1),1)
    return otgt

def accuracy(ytrue,ypred):
    lbl = list(set(ytrue))
    lbl_map = {}
    for i,l in enumerate(lbl):
        lbl_map[l] = i
    acc = [0]*len(lbl)
    gt = [0]*len(lbl)
    for i,tr in enumerate(ytrue):
        if tr == ypred[i]:
           acc[lbl_map[tr]] += 1
        gt[lbl_map[tr]] += 1
    facc = [acc[i]/ttl for i,ttl in enumerate(gt)]
    return facc

def most_common(lst):
    return max(set(lst), key=lst.count)

def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True),
                         2, keepdim=True).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')

class Evaluator:
    def __init__(self,model_s=None,model_p=None,model_e=None,key_data=None,key_lbl=None,lamb=0.6,lamb2=0.6,dictionary=None,device=None):
        self.device = device
        self.dictionary = dictionary
        if model_p is not None:
            model = model_p
        elif model_e is not None:
            model = model_e
        else:
            model = None
        if model is not None:
            if isinstance(model.pair_encoder, PairEncoder) or isinstance(model.pair_encoder, PairEncoderConv):
                #pass
                #hidden = model.init_hidden(key_data.size(1)) #
                #self.r2_ = model.single(key_data,hidden) #
                #self.key_data = key_data #
                bsz = 15000
                r2_ = []
                for batch, i in enumerate(range(0, len(key_data), bsz)):
                    last = min(len(key_data), i+bsz)
                    intoks = key_data[i:last]
                    key, _ = self.spackage_rnn(intoks)
                    key = key.to(self.device)
                    with torch.no_grad():
                        hidden = model.init_hidden(key.size(1))
                        x, _ = model.single(key,hidden)
                        r2_.append(x) 
                self.r2_ = torch.cat(r2_, dim=0).cpu()
                self.key_data = key_data                

            elif isinstance(model.pair_encoder, BertEncoder):
                #pass
                assert isinstance(key_data, list)
                keys = []
                bsz = 2000
                for batch, i in enumerate(range(0, len(key_data), bsz)):
                    j=(batch+1) / (len(list(range(0, len(key_data), bsz))))
                    last = min(len(key_data), i+bsz)
                    intoks = key_data[i:last]
                    data, _ = self.spackage_bert(intoks)
                    key = self.bt_on_device(data)
                    with torch.no_grad():
                        r2,_ = model.single(key,None)
                        keys.append(F.normalize(r2, dim=1).cpu())
                keys = torch.cat(keys, dim=0).numpy()
                self.key_data = keys  
            #keys = []
            #bsz = 2000
            #for batch, i in enumerate(range(0, len(key_data), bsz)):
            #    j=(batch+1) / (len(list(range(0, len(key_data), bsz))))
            #    last = min(len(key_data), i+bsz)
            #    intoks = key_data[i:last]
            #    if isinstance(model.pair_encoder, PairEncoder):
            #        key, _ = self.spackage_rnn(intoks)
            #        key = key.to(self.device)
            #        hidden = model.init_hidden(key.size(1))
            #    elif isinstance(model.pair_encoder, BertEncoder):
            #        key, _ = self.spackage_bert(intoks)
            #        key = self.bt_on_device(key)
            #        hidden = None
            #    else:
            #        raise Exception(f'Not Implemented {type(model.pair_encoder)}')
            #    with torch.no_grad():
            #        r2 = model.single(key,hidden)
            #        keys.append(F.normalize(r2, dim=1).cpu())
            #keys = torch.cat(keys, dim=0).numpy()
            #self.key_data = keys
            self.key_lbl = key_lbl
        self.model_s = model_s
        self.model_p = model_p
        self.model_e = model_e
        self.lamb = lamb
        self.lamb2 = lamb2

    def bt_on_device(self, tokenizer):
        t2 = {}
        for key, val in tokenizer.items():
            t2[key] = val.to(self.device)
        return t2

    def spackage_bert(self, data):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        data = [json.loads(x) for x in data]
        dat = tokenizer([" ".join(x['text']) for x in data], return_tensors="pt", padding=True, truncation=True)
        targets = [x['label'] for x in data]
        targets = torch.tensor(targets, dtype=torch.long)
        return dat, targets 

    def spackage_rnn(self,data,is_train=False):
        data = [json.loads(x) for x in data]
        dat = [[self.dictionary.word2idx[y] for y in x['text']] for x in data]
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
                    dat[i].append(self.dictionary.word2idx['<pad>'])
        with torch.set_grad_enabled(is_train):
            dat = torch.tensor(dat, dtype=torch.long)
            targets = torch.tensor(targets, dtype=torch.long)
        return dat.t(), targets

    def majority_vote(self,tens,nclasses):
        tensl = tens.cpu().tolist()
        y = []
        z = []
        for t in tensl:
            x = [1]*nclasses
            maxm = -np.inf
            for i in t:
                x[i]+=1
                if x[i] > maxm:
                    maxm=x[i]
                    idx=i
            z.append(idx)
            y.append(x) 
        return torch.tensor(z,dtype=torch.long),torch.tensor(y,dtype=torch.float)
        
    def get_prediction(self,idx_pred,val_pred,key_lbl):
        ipl = idx_pred.tolist()
        #vpl = val_pred.tolist()
        kll = key_lbl.tolist()
        lpl = []
        preds = [kll[x[0]] for x in ipl]
        return torch.tensor(preds,dtype=torch.long)

    def get_scores(self,idx_pred,val_pred,key_lbl):
        predicted_labels = key_lbl[idx_pred]
        x=[]
        for pl in predicted_labels:
            _,b=np.unique(pl, return_index=True)
            x.append(b)
        x = torch.from_numpy(np.asarray(x))
        scores = torch.gather(val_pred.cpu(),1,x)
        return scores

    def spredict(self,data):
        hidden = self.model_s.init_hidden(data.size(1))
        output, _, _ = self.model_s.forward(data, hidden)
        scores = output.view(data.size(1), -1)
        prediction = torch.max(scores, 1)[1]
        return scores.cpu(), prediction.cpu()

    def ppredict(self,data):
        hidden = self.model_p.init_hidden(data.size(1))
        r1 = self.model_p.single(data,hidden)
        r1,r2 = self.model_p.pair(r1,self.r2_,data,self.key_data)
        dists = torch.norm(F.normalize(r1,dim=2)-F.normalize(r2,dim=2), dim=2)
        topk_vals, topk_inds = torch.sort(dists,dim=1)
        prediction = self.get_prediction(topk_inds, topk_vals, self.key_lbl)
        scores = self.get_scores(topk_inds, topk_vals, self.key_lbl)
        scores = 1./scores
        return scores, prediction

    def epredict(self,data):
        with torch.no_grad(): 
            hidden = self.model_e.init_hidden(data.size(1))
            r1 = self.model_e.single(data,hidden)
            r1_ = torch.mean(r1,dim=1,keepdim=False)
            scores_s = self.model_e.classify_direct(r1_).view(data.size(1),-1).cpu()
            prediction_s = torch.max(scores_s, 1)[1].cpu()
            r1,r2 = self.model_e.pair(r1,self.r2_,data,self.key_data)       
        dists = torch.norm(F.normalize(r1,dim=2)-F.normalize(r2,dim=2), dim=2)
        topk_vals, topk_inds = torch.sort(dists,dim=1)
        prediction_p = self.get_prediction(topk_inds, topk_vals, self.key_lbl)
        scores_p = self.get_scores(topk_inds, topk_vals, self.key_lbl)
        scores_p = 1./(1e-6 + scores_p)
        #del hidden, r1, r1_, r2
        #torch.cuda.empty_cache()
        return scores_s,scores_p,prediction_s,prediction_p

    def use_faiss(self, queries, keys, dim):
        k = keys.shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(keys)
        D, I = index.search(queries, k)        
        return torch.sqrt(torch.from_numpy(D)), torch.from_numpy(I)

    def epredict_bert(self, data, bsz=15000):
        with torch.no_grad():
            r1 = self.model_e.single(data,None)
            ### for comb ###
            scores_s = None #self.model_e.classify_direct(r1).view(data['input_ids'].size(0),-1).cpu() 
            prediction_s = None #torch.max(scores_s, 1)[1].cpu()
            ###############
            r1 = F.normalize(r1, dim=1).cpu()
            r2 = torch.from_numpy(self.key_data)
            r1_, r2_ = self.model_e.pair(r1, r2)
            dists = torch.norm(F.normalize(r1_,dim=2)-F.normalize(r2_,dim=2), dim=2)
            topk_vals, topk_inds = torch.sort(dists,dim=1)
            prediction_p = self.get_prediction(topk_inds, topk_vals, self.key_lbl)
            scores_p = self.get_scores(topk_inds, topk_vals, self.key_lbl)
            scores_p = 1./(1e-6+scores_p)
            return scores_s,scores_p,prediction_s,prediction_p
       #     dists = []
       #     for batch, i in enumerate(range(0, len(self.key_data), bsz)):
       #         j=(batch+1) / (len(list(range(0, len(self.key_data), bsz))))
       #         last = min(len(self.key_data), i+bsz)
       #         intoks = self.key_data[i:last]
       #         data, _ = self.spackage(intoks)
       #         key_data = self.bt_on_device(data)
       #         r2 = self.model_e.single(key_data,None)    
       #         r1_, r2_ = self.model_e.pair(r1, r2, data, key_data)
       #         dists.append(torch.norm(F.normalize(r1_,dim=2)-F.normalize(r2_,dim=2), dim=2).cpu())
       #         #prediction_p = self.use_faiss(r1.cpu(), r2.cpu(), r1.size(1))   
       #dists = torch.cat(dists, dim = 1)
       # topk_vals, topk_inds = torch.sort(dists,dim=1)
       # prediction_p = self.get_prediction(topk_inds, topk_vals, self.key_lbl)
       # scores_p = self.get_scores(topk_inds, topk_vals, self.key_lbl)
       # scores_p = 1./scores_p
       # return scores_s,scores_p,prediction_s,prediction_p

    def epredict_batch(self,data,bsz=15000):
        dists = []
        scores_s2, prediction_s2, scores_p, prediction_p = None, None, None, None
        with torch.no_grad():
            hidden = self.model_e.init_hidden(data.size(1))
            r1,_ = self.model_e.single(data,hidden)
            if isinstance(self.model_e.pair_encoder, PairEncoder):
                r1_ = torch.mean(r1,dim=1,keepdim=False)
            else:
                r1_ = r1
            scores_s = self.model_e.classify_direct(r1_).view(data.size(1),-1).cpu()
            prediction_s = torch.max(scores_s, 1)[1]
            #scores_s2,_  = self.model_e.classify_direct2(data,hidden)
            #scores_s2 = scores_s2.view(data.size(1),-1).cpu()
            #scores_s2 = self.model_e.classify_direct2(r1_).view(data.size(1),-1).cpu()
            #prediction_s2 = torch.max(scores_s2, 1)[1]
        key_length = self.r2_.size(0)
        for batch, i in enumerate(range(0, key_length, bsz)):
            last = min(key_length, i+bsz)
            r2 = self.r2_[i:last,:].to(self.device)
            r1_, r2_ = self.model_e.pair(r1, r2, data, None)
            dists.append(torch.norm(F.normalize(r1_,dim=2)-F.normalize(r2_,dim=2), dim=2).cpu())
        dists = torch.cat(dists, dim = 1)
        topk_vals, topk_inds = torch.sort(dists,dim=1)
        prediction_p = self.get_prediction(topk_inds, topk_vals, self.key_lbl)
        scores_p = self.get_scores(topk_inds, topk_vals, self.key_lbl)
        scores_p = 1./(1e-6+scores_p)
        return scores_s,scores_p,prediction_s,prediction_p, scores_s2, prediction_s2

    def epredict_faiss(self, data, bert=False):
        scores_s2, prediction_s2, scores_p, prediction_p = None, None, None, None
        with torch.no_grad():
            if bert:
                r1,_ = self.model_e.single(data,None)
                scores_s = self.model_e.classify_direct(r1).view(data['input_ids'].size(0),-1).cpu() 
                prediction_s = torch.max(scores_s, 1)[1]
                #scores_s2 = self.model_e.classify_direct2(r1).view(data['input_ids'].size(0),-1).cpu()
            else:
                hidden = self.model_e.init_hidden(data.size(1))
                r1,_ = self.model_e.single(data,hidden)
                scores_s = self.model_e.classify_direct(r1).view(data.size(1),-1).cpu()
            dim = r1.size(1)
            queries_faiss = F.normalize(r1, dim=1).cpu().numpy()
        topk_vals, topk_inds = self.use_faiss(queries_faiss, self.key_data, dim)
        prediction_p = self.get_prediction(topk_inds, None, self.key_lbl)
        scores_p = self.get_scores(topk_inds, topk_vals, self.key_lbl)
        scores_p = 1./(1e-6+scores_p)
        return scores_s, scores_p, prediction_s, prediction_p, scores_s2, prediction_s2

    def predict(self,data,bert=False,mode='comb'):
        if self.model_s:
            scores_s,prediction_s = self.spredict(data)
        if self.model_p:
            scores_p,prediction_p = self.ppredict(data)
        if self.model_e:
            if not bert:
                scores_s,scores_p,prediction_s,prediction_p,scores_s2,prediction_s2 = self.epredict_batch(data)
                #scores_s,scores_p,prediction_s,prediction_p = self.epredict(data)
            else:
                scores_s,scores_p,prediction_s,prediction_p,scores_s2,prediction_s2 = self.epredict_faiss(data, bert=True)
        if mode=='single':
            return prediction_s
        elif mode=='pair':
            return prediction_p
        elif mode=='comb':
            scores_sn = (scores_s - scores_s.mean(dim=1,keepdim=True))/scores_s.std(dim=1,keepdim=True)
            scores_pn = (scores_p - scores_p.mean(dim=1,keepdim=True))/scores_p.std(dim=1,keepdim=True)
            #scores_sn2 = (scores_s2 - scores_s2.mean(dim=1,keepdim=True))/scores_s2.std(dim=1,keepdim=True)
            scores_comb = (1. - self.lamb) * scores_sn + self.lamb * scores_pn
            #scores_comb2 = (1. - self.lamb2) * scores_sn2 + self.lamb2 * scores_comb
            prediction_soft = torch.max(scores_comb, 1)[1]
            return prediction_soft


class Trainer:
    def __init__(self,data,dictionary,device,args,criterions=None,criterionp=None,optimizer=None):
        self.data = data
        self.dictionary = dictionary
        self.criterions = criterions
        self.criterionp = criterionp
        self.optimizer = optimizer
        self.device = device
        self.args = args 
        self.I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
        for i in range(args.batch_size):
            for j in range(args.attention_hops):
                self.I.data[i][j][j] = 1

    def update_opt(self,optimizer):
        self.optimizer = optimizer

    def update_crit(self,crs=None,crp=None):
        if crs:
            self.criterions = crs
        if crp:
            self.criterionp = crp

    def data_shuffle(self):
        random.shuffle(self.data)

    def spackage(self,data,is_train=True):
        data = [json.loads(x) for x in data]
        dat = [[self.dictionary.word2idx[y] for y in x['text']] for x in data]
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
                    dat[i].append(self.dictionary.word2idx['<pad>'])
        with torch.set_grad_enabled(is_train):
            dat = torch.tensor(dat, dtype=torch.long)
            targets = torch.tensor(targets, dtype=torch.long)
        return dat.t(), targets        

    def opt_step(self,model,loss,retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        nn.utils.clip_grad_norm(model.parameters(), self.args.clip)
        self.optimizer.step()

    def attn_reg1(self,attention,loss):
        I = self.I.to(self.device)
        p_coeff = self.args.penalization_coeff
        for att in attention:
            attT = torch.transpose(att, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(att, attT) - I[:att.size(0)])
            loss += p_coeff*extra_loss
        return loss

    def attn_reg2(self,attention,loss):
        e_coeff = self.args.entropy_coeff
        extra_loss = -1*(attention*torch.log(attention)).sum(dim=1).mean()
        loss += e_coeff*extra_loss
        return loss 

    def evaluate(self,model,keys,data_val,knum=1,bsz=32,test=False):
        model.eval()  # turn on the eval() switch to disable dropout
        total_loss = 0
        total_correct = 0
        y_pred = []
        y_true = []
        lbls = []
        for batch, i in enumerate(range(0, len(data_val), bsz)):
            last = min(len(data_val), i+bsz)
            intoks = data_val[i:last]
            data, targets = self.spackage(intoks, is_train=False)
            data, targets = data.to(self.device), targets.to(self.device)
            hidden = model.init_hidden(data.size(1))
            output, attention, intermediate = model.forward(data, hidden)
            output_flat = output.view(data.size(1), -1)
            total_loss += self.criterions(output_flat, targets).item()
            prediction = torch.max(output_flat, 1)[1]
            total_correct += torch.sum((prediction == targets).float()).item()
            y_pred.extend(prediction.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
        avg_batch_loss = total_loss / (len(data_val) // bsz)
        acc = total_correct / len(data_val)
        macro_f1 = f1_score(y_true,y_pred,list(set(y_true)),average='macro')
        return avg_batch_loss, acc, macro_f1, y_pred, y_true 

    def forward(self,i,model,sdata,bsz=32,is_train=True):
        last = min(len(sdata), i+bsz)
        intoks = sdata[i:last]
        data, targets = self.spackage(intoks, is_train=is_train)
        data, targets = data.to(self.device), targets.to(self.device) #data --> [seq_len, bsz]
        hidden = model.init_hidden(data.size(1)) #hidden --> [num_dir*num_layer,bsz,nhid]
        output, attention, intermediate = model.forward(data, hidden)
        loss = self.criterions(output.view(data.size(1), -1), targets)
        if attention is not None:  # add penalization term
            loss = self.attn_reg1(attention,loss)
        return loss

    def epoch(self,ep,model,rpos=True,rneg=True):
        model.train()
        total_loss = 0
        total_pure_loss = 0  # without the penalization term
        for batch, i in enumerate(range(0, len(self.data), self.args.batch_size)):
            loss = self.forward(i,model,self.data,self.args.batch_size,is_train=True)
            self.opt_step(model,loss)
            total_loss += loss.item()
        return total_loss/ ((len(self.data) // self.args.batch_size)), model 

class PairTrainer(Trainer):
    def __init__(self,data,data_pair,vdata_pair,dictionary,device,args,criterions=None,criterionp=None,optimizer=None):
        super(PairTrainer,self).__init__(data,dictionary,device,args,criterions,criterionp,optimizer)
        self.data_pair = data_pair
        self.vdata_pair = vdata_pair

    def data_shuffle(self):
        pass

    def ppackage(self, idx_pair, label, is_train=True):

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

        idx1,idx2 = [list(t) for t in zip(*idx_pair)]
        pdata1 = [self.data[x] for x in idx1]
        pdata2 = [self.data[x] for x in idx2]
        data1,target1 = package1(pdata1, self.dictionary, is_train)
        data2,target2 = package1(pdata2, self.dictionary, is_train)
        targets = torch.tensor(label, dtype=torch.long)
        return data1, data2, targets, target1, target2

    def majority_vote(self,idx_pred,val_pred,key_lbl):
        ipl = idx_pred.tolist()
        vpl = val_pred.tolist()
        kll = key_lbl.tolist()
        lpl = []
        for i,x in enumerate(ipl):
            cont = []
            cont_ = []
            for j,idx in enumerate(x):
                cont_.append(kll[idx])
                if vpl[i][j] <= self.args.margin_pos:
                    cont.append(kll[idx])
            if not cont:
                lpl.append(most_common(cont_))
            else:
                lpl.append(most_common(cont))
        #lpl = [most_common([kll[i] for i in x]) for x in ipl]
        return torch.tensor(lpl,dtype=torch.long)

    def forward_pair(self,i,model,pdata,plabel,bsz=32,is_train=True):
        last = min(len(pdata), i+bsz)
        inpair = pdata[i:last]
        inlbl = plabel[i:last]
        data1, data2, targets, target1, target2 = self.ppackage(inpair, inlbl, is_train)
        data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
        hidden = model.init_hidden(data1.size(1))
        r1, r2, attention1, attention2 = model.forward(data1, data2, hidden)
        mpos,mneg = None,None
        if self.args.var_margin:
            loss = self.criterionp(r1, r2, targets, target1, target2, True, self.args.norm_dist) 
        else:
            loss = self.criterionp(r1, r2, targets, True, self.args.norm_dist, mpos, mneg)

        if attention1 is not None:  # add penalization term
            loss = self.attn_reg1(attention1,loss)
        if attention2 is not None:
            loss = self.attn_reg2(attention2,loss)
        return loss

    def evaluate1(self,model,keys,data_val,knum=1,bsz=32):
        model.eval()
        total_correct=0
        key_data, key_lbl = self.spackage(keys,False)
        margin = self.criterionp.get_margins(key_lbl)
        margin = margin.unsqueeze(0)
        key_data = key_data.to(self.device)
        hidden = model.init_hidden(key_data.size(1))
        r2_ = model.single(key_data,hidden)
        y_pred,y_true = [],[]
        print("Validation")
        for batch, i in enumerate(range(0, len(data_val), bsz)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(data_val), bsz))))
            
            last = min(len(data_val), i+bsz)
            intoks = data_val[i:last]
            data,targets = self.spackage(intoks, is_train=False)
            raw1 = data.to(self.device)
            hidden = model.init_hidden(data.size(1))
            r1 = model.single(raw1,hidden)
            r1,r2 = model.pair(r1,r2_,raw1,key_data)
            margin_ = torch.cat([margin for i in range(r1.size(0))],dim=0)
            if self.args.norm_dist:
                try:
                    dists = torch.norm(F.normalize(r1,dim=2)-F.normalize(r2,dim=2), dim=2) #- margin_ #32,348
                except:
                    pdb.set_trace()
            else:
                dists = torch.norm(r1-r2, dim=2) #32,348
            topk_vals, topk_inds = torch.topk(dists,knum,dim=1,largest=False)
            prediction = self.majority_vote(topk_inds, topk_vals, key_lbl)
            total_correct += torch.sum((prediction == targets).float()).item()
            y_pred.extend(prediction.tolist())
            y_true.extend(targets.tolist())
            del raw1, hidden, r1, r2, dists
            torch.cuda.empty_cache()

            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        acc = total_correct / len(data_val)
        macro_f1 = f1_score(y_true,y_pred,list(set(y_true)),average='macro')
        return acc, macro_f1, y_pred, y_true

    def evaluate2(self,model,bsz=32):
        model.eval()
        total_loss = 0
        mvp, mvn, pdata_ = self.vdata_pair.sample(1, model, self.dictionary, self.device, self.args.norm_dist, rpos=False, rneg=False, bsz=self.args.batch_size)
        pdata,plabel = pdata_[0],pdata_[1]
        print('Validation')
        for batch, i in enumerate(range(0, len(pdata), bsz)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(pdata), bsz))))

            loss = self.forward_pair(i,model,pdata,plabel,bsz, is_train=False)
            total_loss += loss.item()

            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        return total_loss/ ((len(pdata) // bsz))

    def evaluate(self,model,keys,data_val,knum=1,bsz=32,test=True):
        if test:
            acc, macro_f1, y_pred, y_true = self.evaluate1(model,keys,data_val,knum,bsz)
            return None, acc, macro_f1, y_pred, y_true
        else:
            total_loss = self.evaluate2(model,bsz)
            return total_loss, None, None, None, None

    def epoch(self,ep,model,rpos=True,rneg=True):
        total_loss = 0
        total_pure_loss = 0
        if ep == 1:
            mvp, mvn, pdata_ = self.data_pair.sample(ep, model, self.dictionary, self.device, self.args.norm_dist, rpos, rneg, self.args.batch_size)
            self.pdata,self.plabel = pdata_[0],pdata_[1]
        else:
            if ep % self.args.samp_freq == 0:
                print('Resampling training data')
                mvp, mvn, pdata_ = self.data_pair.sample(ep, model, self.dictionary, self.device, self.args.norm_dist, rpos, rneg, self.args.batch_size)
                print('Done')
                self.pdata,self.plabel = pdata_[0],pdata_[1]
        print("Epoch")
        model.train()
        for batch, i in enumerate(range(0, len(self.pdata), self.args.batch_size)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(self.pdata), self.args.batch_size))))

            loss = self.forward_pair(i,model,self.pdata,self.plabel,self.args.batch_size,is_train=False)
            self.opt_step(model,loss)
            total_loss += loss.item()

            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        return total_loss/ ((len(self.pdata) // self.args.batch_size)), model 

class CombTrainer(PairTrainer):
    def __init__(self,data,data_pair,vdata_pair,dictionary,device,args,criterions=None,criterionp=None,optimizer=None,cls_num_list=None,beta_max=None,beta_min=None):
        super(CombTrainer,self).__init__(data,data_pair,vdata_pair,dictionary,device,args,criterions,criterionp,optimizer)
        self.var_beta = VarBeta(cls_num_list,bmax=beta_max,bmin=beta_min)
        self.criterion = nn.CrossEntropyLoss()
        self.data4train = copy.deepcopy(self.data)

    def data_shuffle(self):
        random.shuffle(self.data4train)

    def forward_ens(self,i,model,pdata,plabel,bsz=32,is_train=True):
        last = min(len(pdata), i+bsz)
        inpair = pdata[i:last]
        inlbl = plabel[i:last]
        data1, data2, targets, target1, target2 = self.ppackage(inpair, inlbl, is_train)
        data1, data2, targets, target1, target2 = data1.to(self.device), data2.to(self.device), targets.to(self.device),target1.to(self.device), target2.to(self.device)
        otarget1 = one_hot(target1.cpu(),self.args.nclasses).to(self.device)
        otarget2 = one_hot(target2.cpu(),self.args.nclasses).to(self.device)
        beta_list = self.var_beta.sample(target1, target2) 
        la = torch.from_numpy(np.random.beta(beta_list,beta_list)).to(self.device)
        la = la.unsqueeze(1).float()
        target_n = la * otarget1 + (1. - la) * otarget2
        hidden = model.init_hidden(data1.size(1))
        pred, r1, r2, attention1, _ = model(data1, data2, hidden, la=la)
        loss_p = self.args.ploss_wt * self.criterionp(r1, r2, targets, True, self.args.norm_dist, None, None)
        loss_s = (1. - self.args.ploss_wt) * self.criterions(F.log_softmax(pred.view(data1.size(1),-1),dim=1),target_n)
        loss = loss_s + loss_p

        if self.args.advp:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            pred_, r1_, r2_, _, _ = model(data1, data2, hidden, la=la, perturb=self.args.epsilon)
            loss_p_ = self.args.ploss_wt * self.criterionp(r1_, r2_, targets, True, self.args.norm_dist, None, None)
            loss_s_ = self.args.sloss_wt * self.criterions(F.log_softmax(pred_.view(data1.size(1),-1),dim=1),target_n)
            loss_p_total = (1. - self.args.adv_wt) * loss_p + self.args.adv_wt * loss_p_
            loss_s_total = (1. - self.args.adv_wt) * loss_s + self.args.adv_wt * loss_s_
        else:
            loss_p_total = loss_p
            loss_s_total = loss_s
        loss_total = loss_p_total + loss_s_total
            
        if attention1 is not None:  # add penalization term
            loss_total = self.attn_reg1(attention1,loss_total)
        #if attention2 is not None:
        #    loss = self.attn_reg2(attention2,loss)
        return loss_total,loss_p_total,loss_s_total

    def forward_dent(self, i, model, data, bsz=32, is_train = True):
        last = min(len(data), i+bsz)
        intoks = data[i:last]
        data, targets = self.spackage(intoks, is_train=is_train)
        data, targets = data.to(self.device), targets.to(self.device) #data --> [seq_len, bsz]
        #otarget = one_hot(targets.cpu(),self.args.nclasses).to(self.device)
        hidden = model.init_hidden(data.size(1)) #hidden --> [num_dir*num_layer,bsz,nhid]
        r, attn = model.single(data, hidden)
        r_ = torch.mean(r,dim=1,keepdim=False)
        output = model.classify_direct2(r_)
        #output = model.classify_direct(r_)
        #attn = None
        loss = self.criterion(output.view(data.size(1), -1), targets)
        if attn is not None:  # add penalization term
            loss = self.attn_reg1(attn,loss)
        #loss = self.criterions(F.log_softmax(output.view(data.size(1),-1),dim=1),otarget)
        return loss

    def evaluate1(self,keys,data_val,model_s=None,model_p=None,model_e=None,knum=1,bsz=32):
        if model_s:
            model_s.eval()
        if model_p:
            model_p.eval()
        if model_e:
            model_e.eval()
        total_correct=0
        key_data, key_lbl = self.spackage(keys,False)
        key_data = keys
        #key_data = key_data.to(self.device)
        evaluator = Evaluator(model_s=model_s,model_p=model_p,model_e=model_e,
                                key_data=key_data,key_lbl=key_lbl,lamb=self.args.lamb,lamb2=self.args.lamb2,dictionary=self.dictionary,device=self.device)
        y_pred,y_true = [],[]
        print("Validation")
        for batch, i in enumerate(range(0, len(data_val), bsz)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(data_val), bsz))))

            last = min(len(data_val), i+bsz)
            intoks = data_val[i:last]
            data,targets = self.spackage(intoks, is_train=False)
            raw1 = data.to(self.device)
            prediction = evaluator.predict(raw1,mode=self.args.mode)
            total_correct += torch.sum((prediction == targets).float()).item()
            y_pred.extend(prediction.tolist())
            y_true.extend(targets.tolist())

            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        acc = total_correct / len(data_val)
        macro_f1 = f1_score(y_true,y_pred,list(set(y_true)),average='macro')
        return acc, macro_f1, y_pred, y_true


    def evaluate(self,keys,data_val,model_s=None,model_p=None,model_e=None,knum=1,bsz=32,test=True):
        acc, macro_f1, y_pred, y_true = self.evaluate1(keys,data_val,model_s=model_s,model_p=model_p,model_e=model_e,knum=knum,bsz=bsz)
        return None, acc, macro_f1, y_pred, y_true

    def epoch_ent(self,ep,model,rpos=True,rneg=True):
        total_loss = 0
        lp = 0
        ls = 0
        le = 0
        ladv = 0
        total_pure_loss = 0
        if ep == 1:
            mvp, mvn, pdata_ = self.data_pair.sample(ep, model, self.dictionary, self.device, False, self.args.norm_dist, rpos, rneg, self.args.batch_size)
            self.pdata,self.plabel = pdata_[0],pdata_[1]
        else:
            if ep % self.args.samp_freq == 0:
                print('Resampling training data')
                mvp, mvn, pdata_ = self.data_pair.sample(ep, model, self.dictionary, self.device, False, self.args.norm_dist, rpos, rneg, self.args.batch_size)
                print('Done')
                self.pdata,self.plabel = pdata_[0],pdata_[1]
        print("Epoch")
        model.train()
        for batch, i in enumerate(range(0, len(self.pdata), self.args.batch_size)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(self.pdata), self.args.batch_size))))
            loss,loss_p,loss_s = self.forward_ens(i,model,self.pdata,self.plabel,self.args.batch_size,is_train=True)
            self.opt_step(model,loss)
            total_loss += loss.item()
            lp += loss_p.item()
            ls += loss_s.item()
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        return total_loss/ ((len(self.pdata) // self.args.batch_size)), lp/((len(self.pdata) // self.args.batch_size)), ls/((len(self.pdata) // self.args.batch_size)), model 
        
    def epoch_dent(self, ep, model):
        model.train()
        total_loss = 0
        for batch, i in enumerate(range(0, len(self.data4train), self.args.batch_size)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(self.data4train), self.args.batch_size))))
            loss = self.forward_dent(i, model, self.data4train, self.args.batch_size, is_train = True)
            self.opt_step(model, loss)
            total_loss += loss.item()
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        return total_loss/ ((len(self.data) // self.args.batch_size)), 0, 0, model

    def epoch(self,ep,model,rpos=True,rneg=True):
        pairw = get_bucket(ep, a=self.args.ae, b=self.args.be)
        if False:
            random.shuffle(self.data4train)
            total_loss,loss1,loss2,model = self.epoch_dent(ep, model)
        else:
            total_loss,loss1,loss2,model = self.epoch_ent(ep, model, rpos, rneg)
        return total_loss, loss1, loss2, model


class BertCombTrainer(PairTrainer):
    def __init__(self,data,data_pair,vdata_pair,dictionary,device,args,criterions=None,criterionp=None,optimizer=None,cls_num_list=None,beta_max=None,beta_min=None):
        super(BertCombTrainer,self).__init__(data,data_pair,vdata_pair,dictionary,device,args,criterions,criterionp,optimizer)
        self.var_beta = VarBeta(cls_num_list,bmax=beta_max,bmin=beta_min)
        self.criterion = nn.CrossEntropyLoss()
        self.data4train = copy.deepcopy(self.data)

    def data_shuffle(self):
        random.shuffle(self.data)

    def bt_on_device(self, tokenizer):
        t2 = {}
        for key, val in tokenizer.items():
            t2[key] = val.to(self.device)
        return t2

    def spackage(self, data):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = [json.loads(x) for x in data]
        dat = tokenizer([" ".join(x['text']) for x in data], return_tensors="pt", padding=True, truncation=True)
        targets = [x['label'] for x in data]
        targets = torch.tensor(targets, dtype=torch.long)
        return dat, targets 

    def ppackage(self, idx_pair, label):

        def package1(data):
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            data = [json.loads(x) for x in data]
            dat = tokenizer([" ".join(x['text']) for x in data], return_tensors="pt", padding=True, truncation=True)
            targets = [x['label'] for x in data]
            targets = torch.tensor(targets, dtype=torch.long)
            return dat, targets

        idx1,idx2 = [list(t) for t in zip(*idx_pair)]
        pdata1 = [self.data[x] for x in idx1]
        pdata2 = [self.data[x] for x in idx2]
        data1, target1 = package1(pdata1)
        data2, target2 = package1(pdata2)
        targets = torch.tensor(label, dtype=torch.long)
        return data1, data2, targets, target1, target2

    def bertpackage(self, data, is_train = True):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = [json.loads(x) for x in data]
        dat = tokenizer([" ".join(x['text']) for x in data], return_tensors="pt", padding=True, truncation=True)
        targets = [x['label'] for x in data]
        with torch.set_grad_enabled(is_train):
            targets = torch.tensor(targets, dtype=torch.long)
        return dat, targets

    def forward_ens(self,i,model,pdata,plabel,bsz=32,is_train=True):
        last = min(len(pdata), i+bsz)
        inpair = pdata[i:last]
        inlbl = plabel[i:last]
        data1, data2, targets, target1, target2 = self.ppackage(inpair, inlbl)
        data1, data2 = self.bt_on_device(data1), self.bt_on_device(data2)
        targets, target1, target2 = targets.to(self.device), target1.to(self.device), target2.to(self.device)
        otarget1 = one_hot(target1.cpu(),self.args.nclasses).to(self.device)
        otarget2 = one_hot(target2.cpu(),self.args.nclasses).to(self.device)
        beta_list = self.var_beta.sample(target1, target2) 
        la = torch.from_numpy(np.random.beta(beta_list,beta_list)).to(self.device)
        la = la.unsqueeze(1).float()
        target_n = la * otarget1 + (1. - la) * otarget2
        hidden = None
        pred, r1, r2, _, _ = model(data1, data2, hidden, la=la)
        loss_p = self.args.ploss_wt * self.criterionp(r1, r2, targets, True, self.args.norm_dist, None, None)
        loss_s = (1. - self.args.ploss_wt) * self.criterions(F.log_softmax(pred.view(data1['input_ids'].size(0),-1),dim=1),target_n)
        loss = loss_s + loss_p

        if self.args.advp:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            pred_, r1_, r2_, _, _ = model(data1, data2, hidden, la=la, perturb=self.args.epsilon)
            loss_p_ = self.args.ploss_wt * self.criterionp(r1_, r2_, targets, True, self.args.norm_dist, None, None)
            loss_s_ = self.args.sloss_wt * self.criterions(F.log_softmax(pred_.view(data1['input_ids'].size(0),-1),dim=1),target_n)
            loss_p_total = (1. - self.args.adv_wt) * loss_p + self.args.adv_wt * loss_p_
            loss_s_total = (1. - self.args.adv_wt) * loss_s + self.args.adv_wt * loss_s_
        else:
            loss_p_total = loss_p
            loss_s_total = loss_s
        loss_total = loss_p_total + loss_s_total
            
        #if attention1 is not None:  # add penalization term
        #    loss = self.attn_reg1(attention1,loss)
        #if attention2 is not None:
        #    loss = self.attn_reg2(attention2,loss)
        return loss_total,loss_p_total,loss_s_total

    def forward_dent(self, i, model, data, bsz=32, is_train = True):
        last = min(len(data), i+bsz)
        intoks = data[i:last]
        data, targets = self.bertpackage(intoks, is_train=is_train)
        data = self.bt_on_device(data)
        targets = targets.to(self.device) #data --> [seq_len, bsz]
        #otarget = one_hot(targets.cpu(),self.args.nclasses).to(self.device)
        hidden = None
        r,_ = model.single(data, hidden)
        output = model.classify_direct2(r)
        loss = self.criterion(output.view(data['input_ids'].size(0), -1), targets)
        return loss

    def evaluate1(self,keys,data_val,model_s=None,model_p=None,model_e=None,knum=1,bsz=32):
        if model_s:
            model_s.eval()
        if model_p:
            model_p.eval()
        if model_e:
            model_e.eval()
        total_correct=0
        _, key_lbl = self.spackage(keys)
        evaluator = Evaluator(model_s=model_s,model_p=model_p,model_e=model_e,
                                key_data=keys,key_lbl=key_lbl,lamb=self.args.lamb,device=self.device)
        y_pred,y_true = [],[]
        begin = time.time()
        print("Validation")
        for batch, i in enumerate(range(0, len(data_val), bsz)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(data_val), bsz))))

            last = min(len(data_val), i+bsz)
            intoks = data_val[i:last]
            data, targets = self.spackage(intoks)
            raw1 = self.bt_on_device(data)
            prediction = evaluator.predict(raw1,bert=True,mode=self.args.mode)
            total_correct += torch.sum((prediction == targets).float()).item()
            y_pred.extend(prediction.tolist())
            y_true.extend(targets.tolist())

            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        print(f'time for inference = {1. * (time.time() - begin)/len(data_val)}')
        acc = total_correct / len(data_val)
        macro_f1 = f1_score(y_true,y_pred,list(set(y_true)),average='macro')
        return acc, macro_f1, y_pred, y_true


    def evaluate(self,keys,data_val,model_s=None,model_p=None,model_e=None,knum=1,bsz=32,test=True):
        acc, macro_f1, y_pred, y_true = self.evaluate1(keys,data_val,model_s=model_s,model_p=model_p,model_e=model_e,knum=knum,bsz=bsz)
        return None, acc, macro_f1, y_pred, y_true

    def epoch_ent(self,ep,model,rpos=True,rneg=True):
        total_loss = 0
        lp = 0
        ls = 0
        le = 0
        ladv = 0
        total_pure_loss = 0
        if ep == 1:
            mvp, mvn, pdata_ = self.data_pair.sample(ep, model, self.dictionary, self.device, True, self.args.norm_dist, rpos, rneg, self.args.batch_size)
            self.pdata,self.plabel = pdata_[0],pdata_[1]
        else:
            if ep % self.args.samp_freq == 0:
                print('Resampling training data')
                mvp, mvn, pdata_ = self.data_pair.sample(ep, model, self.dictionary, self.device, True, self.args.norm_dist, rpos, rneg, self.args.batch_size)
                print('Done')
                self.pdata,self.plabel = pdata_[0],pdata_[1]
        print("Epoch")
        model.train()
        for batch, i in enumerate(range(0, len(self.pdata), self.args.batch_size)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(self.pdata), self.args.batch_size))))
            loss,loss_p,loss_s = self.forward_ens(i,model,self.pdata,self.plabel,self.args.batch_size,is_train=True)
            self.opt_step(model,loss)
            total_loss += loss.item()
            lp += loss_p.item()
            ls += loss_s.item()
            del loss, loss_p, loss_s
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        return total_loss/ ((len(self.pdata) // self.args.batch_size)), lp/((len(self.pdata) // self.args.batch_size)), ls/((len(self.pdata) // self.args.batch_size)), model 
        
    def epoch_dent(self, ep, model):
        model.train()
        total_loss = 0
        for batch, i in enumerate(range(0, len(self.data4train), self.args.batch_size)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(self.data4train), self.args.batch_size))))
            loss = self.forward_dent(i, model, self.data4train, self.args.batch_size, is_train = True)
            self.opt_step(model, loss)
            total_loss += loss.item()
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        return total_loss/ ((len(self.data) // self.args.batch_size)), 0, 0, model
    
    def epoch(self,ep,model,rpos=True,rneg=True):
        pairw = get_bucket(ep, a=self.args.ae, b=self.args.be)
        if False:
            random.shuffle(self.data4train)
            total_loss,loss1,loss2,model = self.epoch_dent(ep, model)
        else:
            total_loss,loss1,loss2,model = self.epoch_ent(ep, model, rpos, rneg)
        return total_loss, loss1, loss2, model

def log(start_time, size, total_loss, total_pure_loss, batch, args):
    elapsed = time.time() - start_time
    total_batches = size // args.batch_size
    batch_time = elapsed * 1000 / args.log_interval
    batch_loss = total_loss / args.log_interval
    pure_batch_loss = total_pure_loss / args.log_interval
    print('| {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'
          .format(batch, total_batches, batch_time,
                  batch_loss, pure_batch_loss))

def save(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model, f)
        f.close()
    

if __name__ == '__main__':
    # parse the arguments
    parser = get_base_parser()
    parser.add_argument('--train-data', type=str, default='',
                        help='location of the training data, should be a json file')
    parser.add_argument('--val-data', type=str, default='',
                        help='location of the development data, should be a json file')
    parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a json file')
    parser.add_argument('--test-model', type=str, default='',
                        help='path to load model to test from')
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
            print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
            print('-' * 84)
            exit(0)
    else:
        model = torch.load(args.test_model)
        model = model.to(device)

    if args.eval_on_test:
        data_val = open(args.test_data).readlines()
        evaluate_start_time = time.time()
        test_loss, acc = evaluate(model, data_val, dictionary, criterion, device, args)
        print('-' * 84)
        fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
        print('-' * 84)
    exit(0)
