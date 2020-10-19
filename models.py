import numpy as np
import pdb
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import BertModel, DistilBertModel

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def toggle_layers(model,freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze


class Embed(nn.Module):
    def __init__(self,ntoken, dictionary, ninp, device, word_freq=None, word_vector=None):
        super(Embed, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.dictionary = dictionary
        self.device = device
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(word_vector):
            print('Loading word vectors from', word_vector)
            vectors = torch.load(word_vector)
            assert vectors[3] >= ninp
            vocab = vectors[1]
            vectors = vectors[2]
            loaded_cnt = 0
            unseen_cnt = 0
            freq = [0]*len(self.dictionary.word2idx)
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    to_add = torch.zeros_like(vectors[0]).uniform_(-0.25,0.25)
                    print("uncached word: " + word)
                    unseen_cnt += 1
                    #print(to_add)
                else:
                    loaded_id = vocab[word]
                    to_add = vectors[loaded_id][:ninp]
                    loaded_cnt += 1
                real_id = self.dictionary.word2idx[word]
                self.encoder.weight.data[real_id] = to_add
                freq[real_id] = word_freq[word]
            self.freq = F.normalize(torch.tensor(freq,dtype=torch.float).unsqueeze(1), dim=0, p=1).to(device)
            #emb_wts = self.encoder.weight.data
            #mean = (emb_wts * self.freq.view(-1,1)).sum(dim=0)
            #std = torch.sqrt(1e-6 + (torch.pow(emb_wts - mean.view(1,-1), 2) * self.freq.view(-1,1)).sum(dim=0))
            #self.encoder.weight.data = (emb_wts - mean.view(1,-1))/std.view(1,-1)
            print('%d words from external word vectors loaded, %d unseen' % (loaded_cnt, unseen_cnt))

    def get_mean_std(self):
        emb_wts = self.encoder.weight.data
        mean = (emb_wts * self.freq.view(-1,1)).sum(dim=0)
        std = torch.sqrt(1e-6 + (torch.pow(emb_wts - mean.view(1,-1), 2) * self.freq.view(-1,1)).sum(dim=0))
        return mean, std

    def forward(self,input,norm=False):
        if norm:
            return F.normalize(self.encoder(input), dim=2)
            m, s = self.get_mean_std()
            return (self.encoder(input) - m.view(1,1,-1)) / s.view(1,1,-1)
        return self.encoder(input)

class RNN(nn.Module):
    def __init__(self, inp_size, nhid, nlayers, batch_first=False, bidirectional=True):
        super(RNN, self).__init__()
        self.nlayers = nlayers
        self.nhid = nhid
        self.rnn = nn.GRU(inp_size, nhid, nlayers, bidirectional=bidirectional, batch_first=batch_first)
        if bidirectional == True:
            self.ndir = 2
        else:
            self.ndir = 1

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return torch.zeros(self.nlayers * self.ndir, bsz, self.nhid, dtype=weight.dtype,
                            layout=weight.layout, device=weight.device)

    def fwd_all(self, input, hidden):
        out_rnn = self.rnn(input, hidden)
        return out_rnn

    def forward(self, input, hidden): #input --> (seq_len, bsize, inp_size)
        out_rnn = self.rnn(input, hidden)[0]
        return out_rnn # (seq_len, bsize, 2*nhid)

class BaseAttn(nn.Module):
    def __init__(self, attention_hops, inp_size, dictionary, dropout, pooling, freeze_pair=False):
        super(BaseAttn, self).__init__()
        if freeze_pair:
            self.trans1 = nn.Linear(inp_size,inp_size)
            self.trans2 = nn.Linear(inp_size,inp_size)
        self.IPM = nn.Linear(inp_size,inp_size,bias=False)
        self.aggregate = nn.Linear(inp_size*2,inp_size)
        self.aggregator1 = nn.Linear(attention_hops * inp_size, inp_size)
        self.aggregator2 = nn.Linear(inp_size, inp_size)
        self.dictionary = dictionary
        self.drop = nn.Dropout(dropout)
        self.pooling = pooling
        self.heads = attention_hops
        self.freeze_pair = freeze_pair

    def masked(self, inp, num, hops=False):
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, seq_len]
        transformed_inp = transformed_inp.view(inp.size()[1], 1, inp.size()[0])  # [bsz, 1, seq_len]
        if not hops:
            concatenated_inp = [transformed_inp for i in range(num)]
        else:
            concatenated_inp = [transformed_inp for i in range(self.heads)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, num, seq_len]
        mask = (concatenated_inp == self.dictionary.word2idx['<pad>']).float()
        return mask    

    def combine_hops(self,r1,r2,temp=1): 
        adist = torch.softmax(torch.tanh((r1*self.IPM(r2)).sum(2).unsqueeze(1))/temp,dim=2)
        r1 = torch.bmm(adist,r1).squeeze(1)
        r2 = torch.bmm(adist,r2).squeeze(1)
        return r1, r2, adist.squeeze(1)
        #return r1.mean(dim=1, keepdim=False), r2.mean(dim=1, keepdim=False), None

    def single(self,input,raw):
        return input

    def pcomb(self,r1,r2,temp=1):
        assert len(r1.size()) == len(r2.size())
        if len(r1.size()) == 3: #  # r1 --> (32,8,_); r2 --> (348,8,_)
            r2_ = self.IPM(r2)
            dist=torch.softmax(torch.tanh(torch.matmul(r1.unsqueeze(1).unsqueeze(3),r2_.unsqueeze(3)).squeeze(-1).squeeze(-1))/temp,dim=2) #(32,348,8)
            r1 = torch.bmm(dist,r1) # (32,348,_)
            r2 = torch.bmm(dist.transpose(0,1),r2).transpose(0,1) # (32,348,_)
            return r1,r2
        elif len(r1.size()) == 4: #r1 -> (32,348,8,_) r2 -> (348,32,8,_)
            if r1.size(0) != r2.size(0):
                r2 = r2.transpose(0,1)
            r2_ = self.IPM(r2)
            dist = torch.softmax(torch.tanh(torch.matmul(r1.unsqueeze(3),r2_.unsqueeze(4)).squeeze(4).transpose(2,3))/temp,dim=3) # 32,348,1,8
            r1 = torch.matmul(dist,r1).squeeze(-2) # (32,348,_)
            r2 = torch.matmul(dist,r2).squeeze(-2) # (32,348,_)
            return r1,r2
        else:
            raise Exception(f'Unexpected number of dimensions for combining ({len(r1.size())})')

    #def pcomb(self,r1,r2,temp=1):
    #    r1 = torch.cat([r1.unsqueeze(1) for i in range(r2.size(0))], dim=1)
    #    r2 = torch.cat([r2.unsqueeze(0) for i in range(r1.size(0))], dim=0)
    #    return r1, r2

class NoAttn(nn.Module):
    def __init__(self):
        super(NoAttn, self).__init__()
        pass

    def forward(self,input1,input2,raw1,raw2):
        r1 = torch.mean(input1,dim=0)
        r2 = torch.mean(input2,dim=0)
        return r1,r2,None,None

    def single(self,input,raw):
        return torch.mean(input,dim=0),None

    def pair(self,r1,r2,raw1,raw2):
        r1 = torch.cat([r1.unsqueeze(1) for i in range(r2.size(0))],dim=1)
        r2 = torch.cat([r2.unsqueeze(0) for i in range(r1.size(0))],dim=0)
        return r1,r2
    
class CrossSDP(BaseAttn):
    def __init__(self, attention_hops, inp_size, attention_unit, dictionary, dropout, pooling, shared=False):
        super(CrossSDP, self).__init__(attention_hops, inp_size, dictionary, dropout, pooling)
        self.Wq = nn.ModuleList()
        self.Wk = nn.ModuleList()
        if shared:
            for i in range(attention_hops):
                wt = nn.Linear(inp_size,attention_unit,bias=False)
                self.Wq.append(wt)
                self.Wk.append(wt)
        else:
            for i in range(attention_hops):
                self.Wq.append(nn.Linear(inp_size,attention_unit,bias=False))
                self.Wk.append(nn.Linear(inp_size,attention_unit,bias=False))
        self.aggregate = nn.Linear(inp_size*2,inp_size)

    def attn_layer(self,query,key,mask,head_num,pairwise=False):
        query = query.transpose(0,1)
        key = key.transpose(0,1)
        Q = self.Wq[head_num](self.drop(query))
        K = self.Wk[head_num](self.drop(key))
        if not pairwise:
            attn = torch.bmm(Q,K.transpose(1,2)) / np.sqrt(K.size(2))
            attn = attn - 10000*mask
            attn = F.softmax(attn,2)
            out = torch.bmm(attn,key)
            return out
        else:
            attn = torch.matmul(Q.unsqueeze(1),K.transpose(1,2)) / np.sqrt(K.size(2)) #348, 32, 67, 57 mask-> 32,67,57
            mask = torch.cat([mask.unsqueeze(0) for i in range(query.size(0))],dim=0)
            attn = attn - 10000*mask
            attn = F.softmax(attn,3)
            out = torch.matmul(attn,key)
            return out

    def forward(self,input1,input2,raw1,raw2):
        outs1 = []
        outs2 = []
        mask1 = self.masked(raw1, raw2.size(0))
        mask2 = self.masked(raw2, raw1.size(0))
        for i,_ in enumerate(self.Wq):
            out1 = self.attn_layer(input1,input2,mask2,i)
            out2 = self.attn_layer(input2,input1,mask1,i)
            if self.pooling=='max':
                outs1.append(torch.max(out1,dim=1,keepdim=True)[0])
                outs2.append(torch.max(out2,dim=1,keepdim=True)[0])
            elif self.pooling=='mean':
                outs1.append(torch.mean(out1,dim=1,keepdim=True))
                outs2.append(torch.mean(out2,dim=1,keepdim=True))
            elif self.pooling=='agg':
                outs1.append(torch.mean(torch.tanh(self.aggregate(self.drop(torch.cat([input1.transpose(0,1),out1],dim=2)))),dim=1,keepdim=True))
                outs2.append(torch.mean(torch.tanh(self.aggregate(self.drop(torch.cat([input2.transpose(0,1),out2],dim=2)))),dim=1,keepdim=True))
        r1, r2 = torch.cat(outs1,1), torch.cat(outs2,1)
        r1, r2, adist = self.combine_hops(r1,r2)
        return r1, r2, None, adist 

    def pair(self, r1, r2, raw1, raw2):
        outs1 = []
        outs2 = []
        mask1 = self.masked(raw1, raw2.size(0))
        mask2 = self.masked(raw2, raw1.size(0))
        for i,_ in enumerate(self.Wq):
            out1 = self.attn_layer(r1,r2,mask2,i,True) #32,348,57,600
            out2 = self.attn_layer(r2,r1,mask1,i,True) #348,32,67,600
            if self.pooling=='max':
                outs1.append(torch.max(out1,dim=2,keepdim=True)[0])
                outs2.append(torch.max(out2,dim=2,keepdim=True)[0])
            elif self.pooling=='mean':
                outs1.append(torch.mean(out1,dim=2,keepdim=True))
                outs2.append(torch.mean(out2,dim=2,keepdim=True))
            elif self.pooling=='agg':
                input2 = torch.cat([r2.transpose(0,1).unsqueeze(1) for i in range(out2.size(1))],dim=1)
                input1 = torch.cat([r1.transpose(0,1).unsqueeze(1) for i in range(out1.size(1))],dim=1)
                outs1.append(torch.mean(torch.tanh(self.aggregate(self.drop(torch.cat([input1,out1],dim=3)))),dim=2,keepdim=True))
                outs2.append(torch.mean(torch.tanh(self.aggregate(self.drop(torch.cat([input2,out2],dim=3)))),dim=2,keepdim=True))
        r1, r2 = torch.cat(outs1,2), torch.cat(outs2,2)
        r1, r2 = self.pcomb(r1,r2)
        return r1,r2

class CrossSelf(BaseAttn):
    def __init__(self, attention_hops, inp_size, attention_unit, dictionary, dropout, pooling, freeze_pair=False):
        super(CrossSelf, self).__init__(attention_hops, inp_size, dictionary, dropout, pooling, freeze_pair=freeze_pair)
        self.ws1 = nn.Linear(inp_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.freeze_pair = freeze_pair

    def fwd(self,input,raw):
        inp = torch.transpose(input, 0, 1).contiguous()
        size = inp.size()  # [bsz, seq_len, inp_size]
        compressed_embeddings = inp.view(-1, size[2])  # [bsz*seq_len, inp_size]
        hbar = torch.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*seq_len, attention-unit]
        alphas = self.ws2(self.drop(hbar)).view(size[0], size[1], -1)  # [bsz, seq_len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, seq_len]
        mask = self.masked(raw,True)
        penalized_alphas = alphas + -10000*mask
        alphas = F.softmax(penalized_alphas.view(-1, size[1]),1)  # [bsz*hop, seq_len]
        alphas = alphas.view(size[0], self.heads, size[1])  # [bsz, hop, seq_len]
        out_agg, attention = torch.bmm(alphas, inp), alphas # [bsz, hop, inp_size], [bsz, hop, seq_len]
        return out_agg, attention
        
    def forward(self, input1, input2, raw1, raw2):
        r1, a1 = self.fwd(input1, raw1)
        r2, a2 = self.fwd(input2, raw2)
        r1, r2, adist = self.combine_hops(r1, r2)
        return r1, r2, [a1, a2], adist   

    def single(self, input, raw):
        r, attn = self.fwd(input, raw)
        return r, attn
        #r, attn = self.fwd(input, raw)
        #r = r.mean(dim = 1, keepdim=False)
        #return r, attn

    def pair(self, r1, r2, raw1, raw2): # r1 --> (32,8,_); r2 --> (348,8,_)
        return self.pcomb(r1,r2)

class AttnPool(BaseAttn):
    def __init__(self, attention_hops, inp_size, attention_unit, dictionary, dropout, pooling):
        super(AttnPool, self).__init__(attention_hops, inp_size, dictionary, dropout, pooling)
        self.UM = nn.ModuleList()
        for i in range(attention_hops):
            self.UM.append(nn.Linear(inp_size, inp_size, bias=False))
        
    def forward(self, input1, input2, raw1, raw2):
        mask1 = self.masked(raw1, raw2.size(0))
        mask2 = self.masked(raw2, raw1.size(0))
        a1, a2 = [], []
        for i,U in enumerate(self.UM):
            G = torch.tanh(torch.bmm(input1.transpose(0,1),U(self.drop(input2)).permute(1,2,0)))
            Gp = G.transpose(1,2)
            G = G - 10000*mask2
            Gp = Gp - 10000*mask1
            a1.append(F.softmax(torch.max(G,dim=2,keepdim=True)[0].transpose(1,2),dim=2))
            a2.append(F.softmax(torch.max(Gp,dim=2,keepdim=True)[0].transpose(1,2),dim=2))
        a1,a2=torch.cat(a1,1),torch.cat(a2,1)                  
        r1 = torch.bmm(a1,input1.transpose(0,1))
        r2 = torch.bmm(a2,input2.transpose(0,1))
        r1, r2, adist = self.combine_hops(r1, r2)
        return r1, r2, [a1, a2], adist

    def pair(self, r1, r2, raw1, raw2):
        mask1 = self.masked(raw1, raw2.size(0)) # 32,67,57
        mask2 = self.masked(raw2, raw1.size(0)) # 348,57,67
        mask1 = torch.cat([mask1.unsqueeze(1) for i in range(r2.size(1))],dim=1)
        mask2 = torch.cat([mask2.unsqueeze(0) for i in range(r1.size(1))],dim=0)
        a1, a2 = [], []
        for i,U in enumerate(self.UM):
            G = torch.tanh(torch.matmul(r1.transpose(0,1).unsqueeze(1),U(self.drop(r2)).permute(1,2,0))) #32,348,57,67
            Gp = G.transpose(2,3)
            G = G - 10000*mask2
            Gp = Gp - 10000*mask1
            a1.append(F.softmax(torch.max(G,dim=3,keepdim=True)[0].transpose(2,3),dim=3))
            a2.append(F.softmax(torch.max(Gp,dim=3,keepdim=True)[0].transpose(2,3),dim=3))
        a1,a2=torch.cat(a1,2),torch.cat(a2,2)
        r1,r2=torch.matmul(a1,r1.transpose(0,1).unsqueeze(1)), torch.matmul(a2,r2.transpose(0,1).unsqueeze(0))
        r1, r2 = self.pcomb(r1,r2)
        return r1,r2

class AttnDec(BaseAttn):
    def __init__(self, attention_hops, inp_size, attention_unit, dictionary, dropout, pooling):
        super(AttnDec, self).__init__(attention_hops, inp_size, dictionary, dropout, pooling)
        self.attend = nn.Linear(inp_size,inp_size)
        self.compare = nn.Linear(2*inp_size,inp_size)

    def forward(self, input1, input2, raw1, raw2):
        mask1 = self.masked(raw1,raw2.size(0))
        mask2 = self.masked(raw2,raw1.size(0))
        A = F.relu(self.attend(self.drop(input1)))
        B = F.relu(self.attend(self.drop(input2)))
        An = torch.bmm(A.transpose(0,1),B.permute(1,2,0)) 
        Bn = An.transpose(1,2)
        An = An - 10000*mask2
        Bn = Bn - 10000*mask1
        beta = torch.bmm(F.softmax(An,dim=2),input2.transpose(0,1))
        alpha = torch.bmm(F.softmax(Bn,dim=2),input1.transpose(0,1))
        r1 = torch.sum(F.relu(self.compare(self.drop(torch.cat([input1.transpose(0,1),beta],dim=2)))),dim=1)
        r2 = torch.sum(F.relu(self.compare(self.drop(torch.cat([input2.transpose(0,1),alpha],dim=2)))),dim=1)
        return r1, r2, None, None

    def single(self, input, raw):
        r = F.relu(self.attend(self.drop(input)))
        return r

    def pair(self, r1, r2, raw1, raw2):
        mask1 = self.masked(raw1, raw2.size(0)) # 32,67,57
        mask2 = self.masked(raw2, raw1.size(0)) # 348,57,67
        mask1 = torch.cat([mask1.unsqueeze(1) for i in range(r2.size(1))],dim=1)
        mask2 = torch.cat([mask2.unsqueeze(0) for i in range(r1.size(1))],dim=0)
        An = torch.tanh(torch.matmul(r1.transpose(0,1).unsqueeze(1),r2.permute(1,2,0))) #32,348,57,67
        Bn = An.transpose(2,3)
        An = An - 10000*mask2
        Bn = Bn - 10000*mask1
        beta = torch.matmul(F.softmax(An,dim=3),r2.transpose(0,1)) #32, 348, 57, 600
        alpha = torch.matmul(F.softmax(Bn,dim=3).transpose(0,1),r1.transpose(0,1)).transpose(0,1) #32, 348, 67, 600
        input2 = torch.cat([r2.transpose(0,1).unsqueeze(0) for i in range(alpha.size(0))],dim=0)
        input1 = torch.cat([r1.transpose(0,1).unsqueeze(1) for i in range(beta.size(1))],dim=1)
        r1 = torch.sum(torch.relu(self.compare(self.drop(torch.cat([input2,alpha],dim=3)))),dim=2)
        r2 = torch.sum(torch.relu(self.compare(self.drop(torch.cat([input1,beta],dim=3)))),dim=2)
        return r1, r2

class FCClassifier(nn.Module):
    def __init__(self,inp_size, nfc, nclasses, dropout):
        super(FCClassifier, self).__init__()
        self.fc = nn.Linear(inp_size, nfc)
        self.pred = nn.Linear(nfc, nclasses)
        self.drop = nn.Dropout(dropout)

    def forward(self, input): #input --> (bsz, inp_size)
        fc = torch.tanh(self.fc(self.drop(input))) # [bsz, nfc]
        pred = self.pred(self.drop(fc)) # [bsz, ncls]
        return pred

class Classifier(nn.Module):
    def __init__(self,inp_size,nclasses,dropout):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(inp_size,inp_size)
        self.l2 = nn.Linear(inp_size,inp_size)
        self.cl = nn.Linear(inp_size,nclasses)
        self.drop = nn.Dropout(dropout)
        
    def forward(self,input):
        return self.cl(self.drop(torch.tanh(self.l2(self.drop(torch.tanh(self.l1(self.drop(input))))))))

class Classifiercnn(nn.Module):
    def __init__(self,inp_size, nclasses, dropout):
        super(Classifiercnn, self).__init__()
        self.fc = nn.Linear(inp_size, inp_size)
        self.pred1 = nn.Linear(inp_size, nclasses)
        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        fc = torch.relu(self.fc(self.drop(input)))
        pred = self.pred1(self.drop(fc))
        return pred

#class Classifier_big(nn.Module):
#    def __init__(self,inp_size,nhops,nclasses,dictionary,dropout):
#        super(Classifier, self).__init__()
#        self.rnn = RNN(inp_size, 300, 1)
#        self.attention = CrossSelf(nhops, 600, 350,
#                                       dictionary, dropout, False, freeze_pair=False)
#        self.l1 = nn.Linear(nhops * 600, 300)
#        self.l2 = nn.Linear(300, nclasses)
#        self.drop = nn.Dropout(dropout)
#        
#    def forward(self,input,input_raw):
#        hidden = model.init_hidden(input.size(0))
#        rout = self.rnn(self.drop(input),hidden)
#        r,_ = self.attention.single(rout,input_raw)
#        rf = r.view(r.size(0),-1)
#        return self.cl(self.drop(torch.tanh(self.l2(self.drop(torch.tanh(self.l1(self.drop(input))))))))

class PairEncoder(nn.Module):
    def __init__(self,config):
        super(PairEncoder,self).__init__()
        self.emb = Embed(config['ntoken'], config['dictionary'], config['ninp'], config['device'], word_freq=config['word-freq'], word_vector=config['word-vector'])
        self.norm = config['norm-emb']
        self.advp = config['advp']
        self.rnn = RNN(config['ninp'], config['nhid'], config['nlayers'])
        self.drop = nn.Dropout(config['dropout'])
        if config['attention-type'] == 'self':
            self.attention = CrossSelf(config['attention-hops'], 2*config['nhid'], config['attention-unit'],
                                       config['dictionary'], config['dropout'], config['att-pooling'], freeze_pair=False)
        else:
            self.attention = NoAttn()  
        self.dictionary = config['dictionary']
        self.pdrop = nn.Dropout(config['pdropout'])

    def init_hidden(self,bsz):
        return self.rnn.init_hidden(bsz)
    
    def get_perturb(self):
        assert self.advp
        return grads['eout1'], grads['eout2']

    def forward(self,input1,input2,hidden,evaluate=False,perturb=None):
        eout1, eout2 = self.emb(input1, norm=self.norm), self.emb(input2, norm=self.norm)
        if self.advp and perturb is None and not evaluate:
            eout1.register_hook(save_grad('eout1'))
            eout2.register_hook(save_grad('eout2'))
        if perturb is not None:
            p1, p2 = self.get_perturb()
            mask1 = 1. - (input1 == self.dictionary.word2idx['<pad>']).float()
            mask2 = 1. - (input2 == self.dictionary.word2idx['<pad>']).float()
            p1 = mask1.unsqueeze(2) * p1
            p2 = mask2.unsqueeze(2) * p2
            p1 = perturb * (p1 / (1e-12 + torch.sqrt(torch.sum(p1**2,dim=(0,2))).view(1,-1,1))) #perturb * F.normalize(p1, dim=2) perturb * (p1 / torch.norm(p1))
            p2 = perturb * (p2 / (1e-12 + torch.sqrt(torch.sum(p2**2,dim=(0,2))).view(1,-1,1))) #perturb * F.normalize(p2, dim=2) perturb * (p2 / torch.norm(p2))
            #eout1 = eout1 + p1
            #eout2 = eout2 + p2
            rout1, rout2 = self.rnn(self.pdrop(eout1)+p1,hidden), self.rnn(self.pdrop(eout2)+p2,hidden)
        else:
            rout1, rout2 = self.rnn(self.drop(eout1),hidden), self.rnn(self.drop(eout2),hidden)
        r1, r2, attention, adist = self.attention(rout1, rout2, input1, input2)
        return r1, r2, attention, adist

    def single(self,input,hidden):
        eout = self.emb(input, norm=self.norm)
        rout = self.rnn(self.drop(eout),hidden)
        r,attn = self.attention.single(rout,input)
        return r, [attn]

    def classify_direct2(self,input,hidden):
        eout = self.emb(input, norm=self.norm)
        rout = self.rnn(self.drop(eout),hidden)
        return rout

    def pair(self,r1,r2,raw1,raw2):
        return self.attention.pair(r1,r2,raw1,raw2) # (32,348,_); (32,348,_)

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(config['prebert-path'], output_hidden_states = True) #DistilBertModel.from_pretrained(config['prebert-path'], output_hidden_states = True) #BertModel.from_pretrained(config['prebert-path'], output_hidden_states = True)
        self.norm = config['norm-emb']
        self.advp = config['advp']
        self.drop = nn.Dropout(config['dropout'])
        self.bert_pooling = config['bert-pooling']

    def init_hidden(self,bsz):
        return None 

    def pool(self, r, mask = None):
        if self.bert_pooling == 'cls':
            return r[:, 0, :]
        assert mask is not None
        lens = mask.sum(dim = 1, keepdim = True)
        r_ = (r * mask.unsqueeze(2)).sum(dim = 1) / lens      
        return r_

    def get_perturb(self):
        assert self.advp
        return grads['eout1'], grads['eout2']

    def forward(self, input1, input2, hidden, evaluate = False, perturb = None): # input --> instance of bert tokenizer
        emb = self.bert_model.get_input_embeddings()
        eout1, eout2 = emb(input1["input_ids"]), emb(input2["input_ids"])       
        mask1 = input1["attention_mask"]
        mask2 = input2["attention_mask"]
        if self.advp and perturb is None and not evaluate:
            eout1.register_hook(save_grad('eout1'))
            eout2.register_hook(save_grad('eout2'))
        if perturb is not None:
            p1, p2 = self.get_perturb()
            p1 = mask1.unsqueeze(2) * p1
            p2 = mask2.unsqueeze(2) * p2
            p1 = perturb * (p1 / (1e-12 + torch.sqrt(torch.sum(p1**2,dim=(0,2))).view(-1,1,1))) #perturb * F.normalize(p1, dim=2) perturb * (p1 / torch.norm(p1))
            p2 = perturb * (p2 / (1e-12 + torch.sqrt(torch.sum(p2**2,dim=(0,2))).view(-1,1,1))) #perturb * F.normalize(p2, dim=2) perturb * (p2 / torch.norm(p2))
            outp1 = self.bert_model(inputs_embeds = eout1 + p1)
            outp2 = self.bert_model(inputs_embeds = eout2 + p2)
            last1 = outp1[0]
            last2 = outp2[0]
        else:
            out1 = self.bert_model(inputs_embeds = eout1)
            out2 = self.bert_model(inputs_embeds = eout2)
            last1 = out1[0]
            last2 = out2[0]
        r1 = self.pool(last1, mask1)
        r2 = self.pool(last2, mask2)
        return r1, r2, None, None
        
    def single(self, input, hidden):
        out = self.bert_model(**input)
        last = out[0]
        mask = input["attention_mask"]
        r = self.pool(last, mask)
        return r, None

    def pair(self, r1, r2, raw1=None, raw2=None): # r1 --> (32, 768); r2 --> (348, 768)
        r1 = torch.cat([r1.unsqueeze(1) for i in range(r2.size(0))], dim=1)
        r2 = torch.cat([r2.unsqueeze(0) for i in range(r1.size(0))], dim=0)
        return r1, r2

class CNN_wt(nn.Module):
    def __init__(self, num_filters, filter_height, filter_wts):
        super(CNN_wt,self).__init__()
        self.num_filters = num_filters
        self.filter_wts = filter_wts
        self.conv = nn.ModuleList([nn.Conv2d(1,num_filters,(filter_height,i),stride=1) for i in filter_wts])

    def forward(self,input, hidden=None): #input --> (bsize, hops, inp_size) (seq_len, bsize, inp_size)
        left = torch.zeros(1, input.size(1), input.size(2))
        right = torch.zeros(1, input.size(1), input.size(2))
        input = torch.cat([left.cuda(), input, right.cuda()], dim=0)
        inp = input.permute(1,2,0).unsqueeze(1) # (bsize, hops, inp_size) --> (bsize, 1, inp_size, hops)
        try:
            cout = [c(inp) for c in self.conv] 
        except:
            pdb.set_trace()
        out_cnn = torch.cat([F.max_pool2d(c,(1,c.size(3))).squeeze(3) for c in cout],1).squeeze(2)   
        return out_cnn

class PairEncoderConv(nn.Module):
    def __init__(self,config):
        super(PairEncoderConv,self).__init__()
        self.emb = Embed(config['ntoken'], config['dictionary'], config['ninp'], config['device'], word_freq=config['word-freq'], word_vector=config['word-vector'])
        self.cnn = CNN_wt(config['num-filters'], config['filter-ht'], config['n-gram'])
        self.bottleneck = nn.Linear(900,600)
        self.drop = nn.Dropout(config['dropout'])
        self.dictionary = config['dictionary']
        self.norm = False

    def init_hidden(self,bsz):
        pass
    
    def forward(self,input1,input2,hidden=None,evaluate=False,perturb=None):
        eout1, eout2 = self.emb(input1, norm=self.norm), self.emb(input2, norm=self.norm)
        rout1, rout2 = self.cnn(self.drop(eout1),hidden), self.cnn(self.drop(eout2),hidden)
        r1, r2, attention, adist = torch.tanh(self.bottleneck(self.drop(rout1))), torch.tanh(self.bottleneck(self.drop(rout2))), None, None
        return r1, r2, attention, adist

    def single(self,input,hidden=None):
        eout = self.emb(input, norm=self.norm)
        rout = self.cnn(self.drop(eout),hidden)
        r = torch.tanh(self.bottleneck(self.drop(rout)))
        return r, None

    def pair(self, r1, r2, raw1=None, raw2=None): # r1 --> (32, 768); r2 --> (348, 768)
        r1 = torch.cat([r1.unsqueeze(1) for i in range(r2.size(0))], dim=1)
        r2 = torch.cat([r2.unsqueeze(0) for i in range(r1.size(0))], dim=0)
        return r1, r2

class EEModel(nn.Module):
    def __init__(self,config):
        super(EEModel,self).__init__()
        if config['encoder-type'] == 'bert':
            self.pair_encoder = BertEncoder(config)
            self.classifier = Classifier(768, config['nclasses'], config['dropout'])
            self.classifier2 = Classifier(768, config['nclasses'], config['dropout'])
            self.enc_type = 'bert'
        elif config['encoder-type'] == 'rnn':
            self.pair_encoder = PairEncoder(config)
            self.classifier = Classifier(2*config['nhid'], config['nclasses'], config['dropout'])
            self.attention = CrossSelf(config['attention-hops'], 2*config['nhid'], config['attention-unit'],
                                   config['dictionary'], config['dropout'], config['att-pooling'], freeze_pair=False)
            #self.classifier2 = Classifier(2*config['nhid']*config['attention-hops'], config['nclasses'], config['dropout'])
            self.classifier2 = Classifier(2*config['nhid'], config['nclasses'], config['dropout'])
            self.enc_type = 'rnn'
        else:
            self.pair_encoder = PairEncoderConv(config)
            self.classifier = Classifiercnn(600, config['nclasses'], config['dropout'])
            self.enc_type = 'cnn'
        self.drop = nn.Dropout(config['dropout'])
        self.device = config['device']

    def init_hidden(self,bsz):
        return self.pair_encoder.init_hidden(bsz)

    def get_perturb(self):
        return self.pair_encoder.get_perturb()

    def single(self,input,hidden):
        return self.pair_encoder.single(input,hidden)

    def pair(self,r1,r2,raw1=None,raw2=None):
        return self.pair_encoder.pair(r1,r2,raw1,raw2)

    def combine(self,input1,input2,hidden,la=0.5,evaluate=False,perturb=None):
        r1, r2, attention, adist = self.pair_encoder(input1,input2,hidden,evaluate=evaluate,perturb=perturb)
        rcomb = la * r1 + (1. - la) * r2
        return rcomb, r1, r2, attention, adist

    def forward(self,input1,input2,hidden,la=0.5,evaluate=False,perturb=None):
        rcomb, r1, r2, attention, adist = self.combine(input1,input2,hidden,la=la,evaluate=evaluate,perturb=perturb)
        out = self.classifier(rcomb)
        return out, r1, r2, attention, adist

    def classify_direct(self,input):
        return self.classifier(input)

    def classify_direct2(self,input, hidden=None):
        #emb_out = self.pair_encoder.emb(input)
        #rnn_out = self.pair_encoder.rnn(self.drop(emb_out),hidden)
        #r, attn = self.attention.single(rnn_out,input)
        #return self.classifier2(r.view(r.size(0), -1)), [attn]
        return self.classifier2(input)
