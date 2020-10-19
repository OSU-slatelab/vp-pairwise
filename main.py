from train import *
from models import *
from util import Dictionary, get_base_parser, ContrastiveLoss
from collections import defaultdict
from sklearn.metrics import f1_score
import json
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import time
import random
import os
import copy
from math import floor
from pairdata import *

def get_keys(ind, data, num):
    idx = []
    xx = [len(t) for t in ind]
    for i,lst in enumerate(ind):
        random.shuffle(lst)
        idx.extend(lst[:num])
    return [data[i] for i in idx]

def split_dev(all_data, label_data, dev_ratio):
    # data should be coming in shuffled; all we have to do is
    # split the right proportion off the end and add labels
    # CNN splits dev off the front, so we'll do that here
    dev_len = floor(len(all_data) * dev_ratio)
    data_val = all_data[:dev_len]
    data_train = all_data[dev_len:]
    data_train += label_data
    return data_train, data_val
    
def get_word_freq(data):
    d = defaultdict(int)
    data_js = [json.loads(x) for x in data]
    for x in data_js:
        for word in x["text"]:
            d[word] += 1
    return d    

def split_dev2(all_data, label_data, dev_ratio):
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
        dev_len = floor(alen)
        dtexts = texts[:dev_len]
        ttexts = texts[dev_len:]
        data_val.extend([json.dumps({"label":label,"text":txt}) for txt in dtexts])
        data_train.extend([json.dumps({"label":label,"text":txt}) for txt in ttexts])
    return data_train, data_val

def split2folds(data, num_folds=5):
    data_per_fold = math.floor(1. * len(data) / num_folds)
    folds = []
    for i in range(num_folds):
        folds.append(data[data_per_fold*i:(i+1)*data_per_fold])
    rem_data = data[num_folds*data_per_fold:]
    rem_len = len(rem_data)
    for i in range(rem_len):
        folds[i].append(rem_data[i])
    return folds


def class_quintiles(train_data,test_data):
    full_data_labels = [json.loads(x)["label"] for x in train_data]
    lbl_dist = defaultdict(int)
    for lbl in full_data_labels:
        lbl_dist[lbl]+=1
    lbl_dist_tup = [(lbl,freq) for lbl,freq in lbl_dist.items()]
    tdist_sorted = sorted(lbl_dist_tup, key=lambda tup: tup[1]) 
    lbl_sorted = [x[0] for x in tdist_sorted]
    test_data_labels = [json.loads(x)["label"] for x in test_data]
    trim_sorted = [x for x in lbl_sorted if x in test_data_labels]
    
    return trim_sorted

def qunit_eval(y_pred,y_true,quintiles,tag='acc'):
    evl=[]
    for q in quintiles:
        yt,yp=[],[]
        for i,y in enumerate(y_true):
            if y in q:
                yt.append(y)
                yp.append(y_pred[i])
        if tag == 'acc':
            evl.append((np.asarray(yt)==np.asarray(yp)).astype(int).mean())
        else:
            evl.append(f1_score(yt,yp,list(set(yt)),average='macro'))
    return evl

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def eval_quintiles(label_sort, label_f, test_data):
    test_data_labels = list(set([json.loads(x)["label"] for x in test_data]))
    lbl2idx = dict([(l,i) for i,l in enumerate(test_data_labels)])
    quints = split_list(label_sort,5)
    f1_perq = []
    for x in quints:
        idxs = [lbl2idx[i] for i in x if i in test_data_labels]
        f1_perq.append(np.mean(label_f[idxs]))
    return f1_perq

def count_corrects(label_list, correct_list):
    class_counts = {}
    class_corrects = {}
    for i in range(len(label_list)):
        curr_lbl = label_list[i]
        if curr_lbl in class_counts:
            class_counts[curr_lbl] += 1
        else:
            class_counts[curr_lbl] = 1
            class_corrects[curr_lbl] = 0
        if correct_list[i]:
            class_corrects[curr_lbl] += 1
    return class_counts, class_corrects

def adam_eval(y_pred,labels):
    corrects = (np.asarray(labels)==np.asarray(y_pred))
    class_counts, class_corrects = count_corrects(labels, corrects)         
    sorted_freqs = sorted([(count, lbl) for (lbl, count) in class_counts.items()])
    quint_break = len(labels) / 5.0
    cum_sum = 0
    quint_sums = []
    quint_corrects = []
    k = 0
    for j in range(5):
        quint_sums.append(0)
        quint_corrects.append(0)
        while cum_sum < ((j+1)*quint_break):
            cum_sum += sorted_freqs[k][0]
            lbl = sorted_freqs[k][1]
            quint_sums[j] += class_counts[lbl]
            quint_corrects[j] += class_corrects[lbl]
            k += 1
    pdb.set_trace()
    quint_accs = [float(a)/float(b) for a,b in zip(quint_corrects, quint_sums)]
    return quint_accs

def get_cls_list(data,nclasses):
    full_data_labels = [json.loads(x)["label"] for x in data]
    cls_num_list = [0]*nclasses
    for i in full_data_labels:
        cls_num_list[i] += 1
    return cls_num_list

if __name__ == "__main__":
    # parse the arguments
    parser = get_base_parser()
    parser.add_argument('--traindev-data', type=str, default='traindev.json',
                        help='location of the training data, should be a json file from vp-tokenizer')
    parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a json file from vp-tokenizer')
    parser.add_argument('--valid-data', type=str, default='',
                        help='location of the validation data, should be a json file from vp-tokenizer')
    parser.add_argument('--label-data', type=str, default='',
                        help='location of the label map (int -> sentence) in json format')
    parser.add_argument('--dev-ratio', type=float, default=0.1, help='fraction of train-dev data to use for dev')
    parser.add_argument('--kfold', action='store_true', help='use k-fold cross validation')
    parser.add_argument('--kfold-log', type=str, default='rnn.attn.kfold.log',
                        help='location to store kfold logs')
    parser.add_argument('--lamb', type=float, default=0.6,
                        help='')
    parser.add_argument('--lamb2', type=float, default=0.6,
                        help='')
    parser.add_argument('--pdrop', type=float, default=0.5,
                        help='')
    parser.add_argument('--mode', type=str, default='comb',
                        help='')
    parser.add_argument('--encoder-type', type=str, default='rnn',
                        help='')
    parser.add_argument('--prebert-path', type=str, default='',
                        help='')
    parser.add_argument('--bert-pooling', type=str, default='mean',
                        help='')
    parser.add_argument('--sloss_wt', type=float, default=1.0,
                        help='')
    parser.add_argument('--ploss_wt', type=float, default=1.0,
                        help='')
    parser.add_argument('--beta-max', type=float, default=15.0,
                        help='')
    parser.add_argument('--beta-min', type=float, default=0.5,
                        help='')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='')
    parser.add_argument('--adv-wt', type=float, default=0.5,
                        help='')
    parser.add_argument('--ae', type=int, default=2,
                        help='')
    parser.add_argument('--be', type=int, default=4,
                        help='')
    parser.add_argument('--dent-wt', type=float, default=0.1,
                        help='')
    parser.add_argument('--norm-emb', action='store_true',
                        help='')
    parser.add_argument('--advp', action='store_true',
                        help='')

    args = parser.parse_args()
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda")
    
    # Set the random seed manually for reproducibility.
    if True: #args.kfold:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed) # ignored if not --cuda
        random.seed(args.seed)
        np.random.seed(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


    # Load Dictionary
    assert os.path.exists(args.traindev_data)
    #assert os.path.exists(args.test_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)
    n_token = len(dictionary)
    criterionp = ContrastiveLoss(args.margin_pos, args.margin_neg)
    criterions = nn.KLDivLoss(reduction='batchmean') #[nn.KLDivLoss(reduction='batchmean'),nn.CrossEntropyLoss()]

    print('Begin to load data.')
    traindev_data = open(args.traindev_data).readlines()
    if args.valid_data != "":
        data_validation = open(args.valid_data).readlines()
        val_sep = True
    else:
        val_sep = False
        data_validation = None
    if args.test_data == "":
        data_test = None
    else:
        data_test = open(args.test_data).readlines()
    if args.label_data == "":
        label_data = None
    else:
        label_data = open(args.label_data).readlines()
    #data_test = open(args.test_data).readlines()
    #label_data = open(args.label_data).readlines()
    #logfile = open(args.out_log, "w")

    kbest_acc = []
    kbest_f1 = []
    data_folds = split2folds(traindev_data)
    if args.kfold:
        if not val_sep:
            numf = len(data_folds)
            random.shuffle(traindev_data)
            data_folds = split2folds(traindev_data)
        else:
            numf = 1
            data_train,data_val = split_dev2(traindev_data, label_data, 0.1)
            data_test = data_validation
            cls_num_list = get_cls_list(data_train,args.nclasses)
            word_freq = get_word_freq(data_train)
            print('Preparing pairs')
            data_pair = PairedData(args.nclasses,data_train,args.num_pos)
            data_key = get_keys(data_pair.ci, data_train, args.num_keys)
            print('Done')
    else:
        numf = 1
        data_train,data_val = split_dev2(traindev_data, label_data, 0.1)
        cls_num_list = get_cls_list(data_train,args.nclasses)
        word_freq = get_word_freq(data_train)
        print('Preparing pairs')
        data_pair = PairedData(args.nclasses,copy.deepcopy(data_train),args.num_pos)
        data_key = get_keys(data_pair.ci, data_train, args.num_keys)
        print('Done')
    flag = True
    for nfold in range(int(numf)):
        if args.kfold and not val_sep:
            if flag:
                len_val, len_test = len(data_folds[nfold]) // 2, len(data_folds[nfold]) // 2 + len(data_folds[nfold]) % 2
                data_val, data_test = data_folds[nfold][0:len_val], data_folds[nfold][len_val:len_val+len_test]
                data_train = []
                for i,fold in enumerate(data_folds):
                    if i != nfold:
                        data_train.extend(fold)
                data_train += label_data
                cls_num_list = get_cls_list(data_train,args.nclasses)
                word_freq = get_word_freq(data_train)
                print('Preparing pairs')
                data_pair = PairedData(args.nclasses,copy.deepcopy(data_train),args.num_pos)
                data_key = get_keys(data_pair.ci, data_train, args.num_keys)
                print('Done')
            else:
                data_val, data_test = data_test, data_val
#            flag = not flag
        model = EEModel({'ntoken':n_token,
                         'dictionary':dictionary,
                         'ninp':args.emsize,
                         'word-vector':args.word_vector,
                         'nhid':args.nhid,
                         'nlayers':args.nlayers,
                         'nfc':args.nfc,
                         'nclasses':args.nclasses,
                         'attention-type':args.attention_type,
                         'attention-hops':args.attention_hops,
                         'attention-unit':args.attention_unit,
                         'att-pooling':args.att_pooling,
                         'dropout':args.dropout,
                         'pdropout':args.pdrop,
                         'device':device,
                         'advp':args.advp,
                         'norm-emb':args.norm_emb,
                         'word-freq':word_freq,
                         'encoder-type':args.encoder_type,
                         'prebert-path':args.prebert_path,
                         'bert-pooling':args.bert_pooling,
                         'num-filters':args.num_filters,
                         'filter-ht':args.filter_ht,
                         'n-gram':args.n_gram
                        })
        #model = Encoder({'ntoken':n_token,
        #                 'dictionary':dictionary,
        #                 'ninp':args.emsize,
        #                 'word-vector':args.word_vector,
        #                 'nhid':args.nhid,
        #                 'nlayers':args.nlayers,
        #                 'nfc':args.nfc,
        #                 'nclasses':args.nclasses,
        #                 'attention-type':args.attention_type,
        #                 'attention-hops':args.attention_hops,
        #                 'attention-unit':args.attention_unit,
        #                 'att-pooling':args.att_pooling,
        #                 'shared':args.shared,
        #                 'dropout':args.dropout,
        #                 'freeze-direct':args.freeze_direct,
        #                 'freeze-pair':args.freeze_pair,
        #                 'stos':args.stos,
        #                 'st-rnn':args.st_rnn,
        #                 'st-step':args.st_step,
        #                 'st-hidc':args.st_hidc,
        #                 'st-hida':args.st_hida,
        #                 'st-attn':args.st_attn,
        #                 'mlbl-loss':args.mlbl_loss,
        #                 'tmix':args.tmix
        #                })
        model = model.to(device)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        elif args.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95)
        else:
            raise Exception('For other optimizers, please add it yourself. '
                            'supported ones are: SGD and Adam.')             
        best_f1 = None
        best_model = None
        if args.encoder_type == 'bert':
            trainer = BertCombTrainer(copy.deepcopy(data_train),data_pair,None,dictionary,device,args,criterions=criterions,criterionp=criterionp,optimizer=optimizer,cls_num_list=cls_num_list,beta_max=args.beta_max,beta_min=args.beta_min)
        else:
            trainer = CombTrainer(copy.deepcopy(data_train),data_pair,None,dictionary,device,args,criterions=criterions,criterionp=criterionp,optimizer=optimizer,cls_num_list=cls_num_list,beta_max=args.beta_max,beta_min=args.beta_min)
        for epoch in range(args.epochs):
            #trainer.data_shuffle()
            torch.cuda.empty_cache()
            train_loss, loss_p, loss_s, model = trainer.epoch(epoch+1, model, args.rpos, args.rneg)
            #train_loss, loss_p, loss_s = 0, 0, 0
            _,acc,f1,_,_ = trainer.evaluate(data_key, data_val, model_e=model, knum=args.nneighbors, bsz=args.test_bsize)
            print('-' * 110)
            print(f'| fold {nfold+1} stage 1 epoch {epoch+1} | total loss {train_loss:.8f} | single loss {loss_s:.8f} | pair loss {loss_p:.8f} | valid f1. {f1:.4f}')
            print('-' * 110)
            if not best_f1 or f1 > best_f1:
        #        if not args.kfold: save(model, args.save[:-3]+'_'+str(args.seed)+'.best_f1.pt')
                best_f1 = f1
                best_model = copy.deepcopy(model)
        best_boost_f1 = best_f1
        if args.stage2 > 0:
            model = best_model
            model = model.to(device)
            #model.flatten_parameters()
            if args.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=args.lr*0.25, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
            elif args.optimizer == 'Adadelta':
                optimizer = optim.Adadelta(model.parameters(), lr=args.lr*0.25, rho=0.95)
            trainer.update_opt(optimizer)
            for epoch in range(args.stage2):
                #trainer.data_shuffle()
                torch.cuda.empty_cache()
                train_loss, loss_p, loss_s, model = trainer.epoch(epoch+1, model, args.rpos, args.rneg)
                _,acc,f1,_,_ = trainer.evaluate(data_key, data_val, model_e=model, knum=args.nneighbors, bsz=args.test_bsize)
                print('-' * 110)
                print(f'| fold {nfold+1} stage 2 epoch {epoch+1} | train loss (total) {train_loss:.8f} | single loss {loss_s:.8f} | pair loss {loss_p:.8f} | valid f1. {f1:.4f}')
                print('-' * 110)
                # Save the model, if the validation loss is the best we've seen so far.
                if not best_boost_f1 or f1 > best_boost_f1:
         #           if not args.kfold: save(model, args.save[:-3]+'_'+str(args.seed)+'.best_acc.pt')
                    best_boost_f1 = f1
                    best_model = copy.deepcopy(model)

        #print(f'| best valid f1. for fold {nfold+1} {best_boost_f1:.4f} |')
       
        best_model = best_model.to(device)
        class_ind = []
        if label_data is not None:
            traindev_data = traindev_data + label_data
        jsdata = [json.loads(x) for x in traindev_data]
        for i in range(args.nclasses):
            idx_i = [ii for ii,x in enumerate(jsdata) if x['label'] == i]
            class_ind.append(idx_i)
        data_key2 = get_keys(class_ind, traindev_data, args.num_keys)
        torch.cuda.empty_cache()
        _, test_acc, macf1, y_pred, y_true = trainer.evaluate(data_key2, data_test, model_e=best_model, knum=args.nneighbors, bsz=args.test_bsize)
        print('-' * 92)
        print(f'| test acc. {test_acc:.4f} | test macro. F1 {macf1:.4f} |')
        print('-' * 92)
        if not args.kfold:
            label_freq = class_quintiles(data_train,data_test) 
            quints = split_list(label_freq,5)
            acc_quin = qunit_eval(y_pred,y_true,quints,tag='acc')     
            quints = split_list(label_freq,5)
            f1_quin = qunit_eval(y_pred,y_true,quints,tag='f1')
            #adams = adam_eval(y_pred,y_true)
            print(f'f1 per quintile = {f1_quin}\n')
            print(f'accuracy per quintile = {acc_quin}\n')
            #print(f'accuracy adams = {adams}\n')
        kbest_acc.append(test_acc)
        kbest_f1.append(macf1)
        del model, best_model
        torch.cuda.empty_cache()

    if args.kfold:
        with open(args.kfold_log,'a') as f:
            nl = '\n'
            tab = '\t'
            f.write(f'{nl}{nl}{args}{nl}{tab}{kbest_acc}{tab}{np.mean(kbest_acc)}{nl}{tab}{kbest_f1}{tab}{np.mean(kbest_f1)}')
    #logfile.close()
    exit(0)
    
