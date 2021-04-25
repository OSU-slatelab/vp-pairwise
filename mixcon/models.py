import numpy as np
import pdb
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import BertModel

class Embed(nn.Module):
    def __init__(self, ntoken, dictionary, ninp, device, word_freq=None, word_vector=None):
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
                if word_freq:
                    freq[real_id] = word_freq[word]
            print('%d words from external word vectors loaded, %d unseen' % (loaded_cnt, unseen_cnt))

    def forward(self,input):
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
        cout = [c(inp) for c in self.conv] 
        out_cnn = torch.cat([F.max_pool2d(c,(1,c.size(3))).squeeze(3) for c in cout],1).squeeze(2)   
        return out_cnn

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
    
class BaseAttn(nn.Module):
    def __init__(self, attention_hops, inp_size, dictionary, dropout):
        super(BaseAttn, self).__init__()
        self.IPM = nn.Linear(inp_size,inp_size,bias=False)
        self.dictionary = dictionary
        self.drop = nn.Dropout(dropout)
        self.heads = attention_hops

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

    def single(self,input,raw):
        return input

    def pcomb(self,r1,r2,temp=1):
        assert len(r1.size()) == len(r2.size())
        if len(r1.size()) == 3:
            r2_ = self.IPM(r2)
            dist=torch.softmax(torch.tanh(torch.matmul(r1.unsqueeze(1).unsqueeze(3),r2_.unsqueeze(3)).squeeze(-1).squeeze(-1))/temp,dim=2)
            r1 = torch.bmm(dist,r1)
            r2 = torch.bmm(dist.transpose(0,1),r2).transpose(0,1)
            return r1,r2
        elif len(r1.size()) == 4:
            if r1.size(0) != r2.size(0):
                r2 = r2.transpose(0,1)
            r2_ = self.IPM(r2)
            dist = torch.softmax(torch.tanh(torch.matmul(r1.unsqueeze(3),r2_.unsqueeze(4)).squeeze(4).transpose(2,3))/temp,dim=3)
            r1 = torch.matmul(dist,r1).squeeze(-2)
            r2 = torch.matmul(dist,r2).squeeze(-2)
            return r1,r2
        else:
            raise Exception(f'Unexpected number of dimensions for combining ({len(r1.size())})')

class CrossSelf(BaseAttn):
    def __init__(self, attention_hops, inp_size, attention_unit, dictionary, dropout):
        super(CrossSelf, self).__init__(attention_hops, inp_size, dictionary, dropout)
        self.ws1 = nn.Linear(inp_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)

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

    def pair(self, r1, r2, raw1, raw2):
        return self.pcomb(r1,r2)

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


class RnnEncoder(nn.Module):
    def __init__(self,config):
        super(RnnEncoder,self).__init__()
        self.emb = Embed(config['ntoken'], config['dictionary'], config['ninp'], config['device'], word_vector=config['word-vector'])
        self.rnn = RNN(config['ninp'], config['nhid'], config['nlayers'])
        self.drop = nn.Dropout(config['dropout'])
        self.attention = CrossSelf(config['attention-hops'], 2*config['nhid'], config['attention-unit'],
                                       config['dictionary'], config['dropout'])
        self.dictionary = config['dictionary']

    def init_hidden(self,bsz):
        return self.rnn.init_hidden(bsz)
    
    def forward(self,input1,input2,hidden):
        eout1, eout2 = self.emb(input1), self.emb(input2)
        rout1, rout2 = self.rnn(self.drop(eout1),hidden), self.rnn(self.drop(eout2),hidden)
        r1, r2, attention, adist = self.attention(rout1, rout2, input1, input2)
        return r1, r2, attention, adist

    def single(self,input,hidden):
        eout = self.emb(input)
        rout = self.rnn(self.drop(eout),hidden)
        r,attn = self.attention.single(rout,input)
        return r, [attn]

    def pair(self,r1,r2,raw1,raw2):
        return self.attention.pair(r1,r2,raw1,raw2) # (32,348,_); (32,348,_)

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(config['prebert-path'], output_hidden_states = True)
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

    def forward(self, input1, input2, hidden): # input --> instance of bert tokenizer
        emb = self.bert_model.get_input_embeddings()
        eout1, eout2 = emb(input1["input_ids"]), emb(input2["input_ids"])       
        mask1 = input1["attention_mask"]
        mask2 = input2["attention_mask"]
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


class CnnEncoder(nn.Module):
    def __init__(self,config):
        super(CnnEncoder,self).__init__()
        self.emb = Embed(config['ntoken'], config['dictionary'], config['ninp'], config['device'], word_vector=config['word-vector'])
        self.cnn = CNN_wt(config['num-filters'], config['filter-ht'], config['n-gram'])
        self.bottleneck = nn.Linear(900,600)
        self.drop = nn.Dropout(config['dropout'])
        self.dictionary = config['dictionary']

    def init_hidden(self,bsz):
        pass
    
    def forward(self,input1,input2,hidden=None):
        eout1, eout2 = self.emb(input1), self.emb(input2)
        rout1, rout2 = self.cnn(self.drop(eout1),hidden), self.cnn(self.drop(eout2),hidden)
        r1, r2, attention, adist = torch.tanh(self.bottleneck(self.drop(rout1))), torch.tanh(self.bottleneck(self.drop(rout2))), None, None
        return r1, r2, attention, adist

    def single(self,input,hidden=None):
        eout = self.emb(input)
        rout = self.cnn(self.drop(eout),hidden)
        r = torch.tanh(self.bottleneck(self.drop(rout)))
        return r, None

    def pair(self, r1, r2, raw1=None, raw2=None): # r1 --> (32, 768); r2 --> (348, 768)
        r1 = torch.cat([r1.unsqueeze(1) for i in range(r2.size(0))], dim=1)
        r2 = torch.cat([r2.unsqueeze(0) for i in range(r1.size(0))], dim=0)
        return r1, r2

class PairModel(nn.Module):
    def __init__(self,config):
        super(PairModel,self).__init__()
        if config['encoder-type'] == 'bert':
            self.pair_encoder = BertEncoder(config)
            self.classifier = Classifier(768, config['nclasses'], config['dropout'])
        elif config['encoder-type'] == 'rnn':
            self.pair_encoder = RnnEncoder(config)
            self.classifier = Classifier(2*config['nhid'], config['nclasses'], config['dropout'])
            self.attention = CrossSelf(config['attention-hops'], 2*config['nhid'], config['attention-unit'],
                                   config['dictionary'], config['dropout'])
        else:
            self.pair_encoder = CnnEncoder(config)
            self.classifier = Classifiercnn(600, config['nclasses'], config['dropout'])
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

    def combine(self,input1,input2,hidden,la=0.5):
        r1, r2, attention, adist = self.pair_encoder(input1,input2,hidden)
        rcomb = la * r1 + (1. - la) * r2
        return rcomb, r1, r2, attention, adist

    def forward(self,input1,input2,hidden,la=0.5):
        rcomb, r1, r2, attention, adist = self.combine(input1,input2,hidden,la=la)
        out = self.classifier(rcomb)
        return out, r1, r2, attention, adist

    def classify_direct(self,input):
        return self.classifier(input)

