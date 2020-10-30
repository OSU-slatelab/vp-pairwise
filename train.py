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
import json
import time
from time import sleep
import sys
import random
import os

from sklearn.metrics import f1_score 

def one_hot(tgt,nclasses):
    otgt = torch.zeros(tgt.size(0),nclasses)
    otgt.scatter_(1,tgt.unsqueeze(1),1)
    return otgt

class Evaluator:
    def __init__(self,model,key_data=None,key_lbl=None,lamb=0.6,dictionary=None,device=None):
        self.device = device
        self.dictionary = dictionary
        self.model = model
        if isinstance(model.pair_encoder, BertEncoder):
            assert isinstance(key_data, list)
            keys = []
            bsz = 1000
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
        else:
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
        self.key_lbl = key_lbl
        self.lamb = lamb

    def bt_on_device(self, tokenizer):
        t2 = {}
        for key, val in tokenizer.items():
            t2[key] = val.to(self.device)
        return t2

    def spackage_bert(self, data):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

    def get_prediction(self,idx_pred,val_pred,key_lbl):
        ipl = idx_pred.tolist()
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

    def use_faiss(self, queries, keys, dim):
        k = keys.shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(keys)
        D, I = index.search(queries, k)        
        return torch.sqrt(torch.from_numpy(D)), torch.from_numpy(I)

    def epredict_batch(self,data,bsz=15000):
        dists = []
        scores_p = None
        with torch.no_grad():
            hidden = self.model.init_hidden(data.size(1))
            r1,_ = self.model.single(data,hidden)
            if isinstance(self.model.pair_encoder, RnnEncoder):
                r1_ = torch.mean(r1,dim=1,keepdim=False)
            else:
                r1_ = r1
            scores_s = self.model.classify_direct(r1_).view(data.size(1),-1).cpu()
        key_length = self.r2_.size(0)
        for batch, i in enumerate(range(0, key_length, bsz)):
            last = min(key_length, i+bsz)
            r2 = self.r2_[i:last,:].to(self.device)
            r1_, r2_ = self.model.pair(r1, r2, data, None)
            dists.append(torch.norm(F.normalize(r1_,dim=2)-F.normalize(r2_,dim=2), dim=2).cpu())
        dists = torch.cat(dists, dim = 1)
        topk_vals, topk_inds = torch.sort(dists,dim=1)
        scores_p = self.get_scores(topk_inds, topk_vals, self.key_lbl)
        scores_p = 1./(1e-6+scores_p)
        return scores_s,scores_p

    def epredict_faiss(self, data, bert=False):
        scores_p = None
        with torch.no_grad():
            if bert:
                r1,_ = self.model.single(data,None)
                scores_s = self.model.classify_direct(r1).view(data['input_ids'].size(0),-1).cpu() 
            else:
                hidden = self.model.init_hidden(data.size(1))
                r1,_ = self.model.single(data,hidden)
                scores_s = self.model.classify_direct(r1).view(data.size(1),-1).cpu()
            dim = r1.size(1)
            queries_faiss = F.normalize(r1, dim=1).cpu().numpy()
        topk_vals, topk_inds = self.use_faiss(queries_faiss, self.key_data, dim)
        scores_p = self.get_scores(topk_inds, topk_vals, self.key_lbl)
        scores_p = 1./(1e-6+scores_p)
        return scores_s, scores_p

    def predict(self,data,bert=False):
        if not bert:
            scores_s,scores_p = self.epredict_batch(data)
        else:
            scores_s,scores_p = self.epredict_faiss(data, bert=True)
        scores_sn = (scores_s - scores_s.mean(dim=1,keepdim=True))/scores_s.std(dim=1,keepdim=True)
        scores_pn = (scores_p - scores_p.mean(dim=1,keepdim=True))/scores_p.std(dim=1,keepdim=True)
        scores_comb = (1. - self.lamb) * scores_sn + self.lamb * scores_pn
        prediction_soft = torch.max(scores_comb, 1)[1]
        return prediction_soft


class Trainer:
    def __init__(self,data,data_pair,dictionary,device,args,criterions=None,criterionp=None,optimizer=None,cls_num_list=None,beta_max=None,beta_min=None, bert=False):
        self.data_pair = data_pair
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
        self.var_beta = VarBeta(cls_num_list,bmax=beta_max,bmin=beta_min)
        self.bert = bert

    def update_opt(self,optimizer):
        self.optimizer = optimizer

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

    def bt_on_device(self, tokenizer):
        return tokenizer.to(self.device)

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

    def forward_ens(self,i,model,pdata,plabel,bsz=32,is_train=True):
        last = min(len(pdata), i+bsz)
        inpair = pdata[i:last]
        inlbl = plabel[i:last]
        data1, data2, targets, target1, target2 = self.ppackage(inpair, inlbl, is_train)
        data1, data2 = self.bt_on_device(data1), self.bt_on_device(data2)
        targets, target1, target2 = targets.to(self.device), target1.to(self.device), target2.to(self.device)
        otarget1 = one_hot(target1.cpu(),self.args.nclasses).to(self.device)
        otarget2 = one_hot(target2.cpu(),self.args.nclasses).to(self.device)
        beta_list = self.var_beta.sample(target1, target2) 
        la = torch.from_numpy(np.random.beta(beta_list,beta_list)).to(self.device)
        la = la.unsqueeze(1).float()
        target_n = la * otarget1 + (1. - la) * otarget2
        hidden = None
        if not self.bert:
            hidden = model.init_hidden(data1.size(1))
        pred, r1, r2, attention1, _ = model(data1, data2, hidden, la=la)
        loss_p = self.args.ploss_wt * self.criterionp(r1, r2, targets)
        loss_s = (1. - self.args.ploss_wt) * self.criterions(F.log_softmax(pred,dim=1),target_n)
        loss = loss_s + loss_p

        if attention1 is not None:  # add penalization term
            loss = self.attn_reg1(attention1,loss)
        return loss,loss_p,loss_s

    def evaluate1(self,keys,data_val,model,bsz=32):
        model.eval()
        total_correct=0
        key_data, key_lbl = self.spackage(keys,False)
        key_data = keys
        evaluator = Evaluator(model,key_data=key_data,key_lbl=key_lbl,lamb=self.args.lamb,dictionary=self.dictionary,device=self.device)
        y_pred,y_true = [],[]
        print("Validation")
        for batch, i in enumerate(range(0, len(data_val), bsz)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(data_val), bsz))))

            last = min(len(data_val), i+bsz)
            intoks = data_val[i:last]
            data,targets = self.spackage(intoks, is_train=False)
            raw1 = data.to(self.device)
            prediction = evaluator.predict(raw1)
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


    def evaluate(self,keys,data_val,model,bsz=32):
        acc, macro_f1, y_pred, y_true = self.evaluate1(keys,data_val,model,bsz=bsz)
        return acc, macro_f1, y_pred, y_true

    def epoch_ent(self,ep,model,rsamp=True):
        total_loss = 0
        lp = 0
        ls = 0
        if ep == 1:
            pdata_ = self.data_pair.sample(ep, model, self.dictionary, self.device, self.bert, rsamp, self.args.batch_size)
            self.pdata,self.plabel = pdata_[0],pdata_[1]
        else:
            if ep % self.args.samp_freq == 0:
                print('Resampling training data')
                pdata_ = self.data_pair.sample(ep, model, self.dictionary, self.device, self.bert, rsamp, self.args.batch_size)
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
        
    def epoch(self,ep,model,rsamp=True):
        total_loss,loss1,loss2,model = self.epoch_ent(ep, model, rsamp)
        return total_loss, loss1, loss2, model


class BertTrainer(Trainer):
    def __init__(self,data,data_pair,dictionary,device,args,criterions=None,criterionp=None,optimizer=None,cls_num_list=None,beta_max=None,beta_min=None, bert=True):
        super(BertTrainer,self).__init__(data,data_pair,dictionary,device,args,criterions,criterionp,optimizer,cls_num_list,beta_max,beta_min,bert)

    def bt_on_device(self, tokenizer):
        t2 = {}
        for key, val in tokenizer.items():
            t2[key] = val.to(self.device)
        return t2

    def spackage(self, data, is_train=True):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = [json.loads(x) for x in data]
        dat = tokenizer([" ".join(x['text']) for x in data], return_tensors="pt", padding=True, truncation=True)
        targets = [x['label'] for x in data]
        targets = torch.tensor(targets, dtype=torch.long)
        return dat, targets 

    def ppackage(self, idx_pair, label, is_train=True):

        def package1(data):
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

    def evaluate1(self,keys,data_val,model,knum=1,bsz=32):
        model.eval()
        total_correct=0
        _, key_lbl = self.spackage(keys)
        evaluator = Evaluator(model,key_data=keys,key_lbl=key_lbl,lamb=self.args.lamb,device=self.device)
        y_pred,y_true = [],[]
        print("Validation")
        for batch, i in enumerate(range(0, len(data_val), bsz)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(data_val), bsz))))

            last = min(len(data_val), i+bsz)
            intoks = data_val[i:last]
            data, targets = self.spackage(intoks)
            raw1 = self.bt_on_device(data)
            prediction = evaluator.predict(raw1,bert=True)
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
