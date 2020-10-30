from train import *
from models import *
from util import *
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


if __name__ == "__main__":
    # parse the arguments
    parser = get_base_parser()

    args = parser.parse_args()
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda")
    
    # Set the random seed manually for reproducibility.
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
    criterions = nn.KLDivLoss(reduction='batchmean')

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

    kbest_acc = []
    kbest_f1 = []
    data_folds = split2folds(traindev_data)
    if args.validation:
        if not val_sep:
            numf = len(data_folds)
            random.shuffle(traindev_data)
            data_folds = split2folds(traindev_data)
        else:
            numf = 1
            data_train,data_val = split_dev(traindev_data, label_data, 0.1)
            data_test = data_validation
            cls_num_list = get_cls_list(data_train,args.nclasses)
            print('Preparing pairs')
            data_pair = PairedData(args.nclasses,data_train,args.num_pos)
            data_key = get_keys(data_pair.ci, data_train, args.num_keys)
            print('Done')
    else:
        numf = 1
        data_train,data_val = split_dev(traindev_data, label_data, 0.1)
        cls_num_list = get_cls_list(data_train,args.nclasses)
        print('Preparing pairs')
        data_pair = PairedData(args.nclasses,copy.deepcopy(data_train),args.num_pos)
        data_key = get_keys(data_pair.ci, data_train, args.num_keys)
        print('Done')
    flag = True
    for nfold in range(int(numf)):
        if args.validation and not val_sep:
            if flag:
                len_val, len_test = len(data_folds[nfold]) // 2, len(data_folds[nfold]) // 2 + len(data_folds[nfold]) % 2
                data_val, data_test = data_folds[nfold][0:len_val], data_folds[nfold][len_val:len_val+len_test]
                data_train = []
                for i,fold in enumerate(data_folds):
                    if i != nfold:
                        data_train.extend(fold)
                data_train += label_data
                cls_num_list = get_cls_list(data_train,args.nclasses)
                print('Preparing pairs')
                data_pair = PairedData(args.nclasses,copy.deepcopy(data_train),args.num_pos)
                data_key = get_keys(data_pair.ci, data_train, args.num_keys)
                print('Done')
            else:
                data_val, data_test = data_test, data_val
            flag = not flag
        model = PairModel({'ntoken':n_token,
                         'dictionary':dictionary,
                         'ninp':args.emsize,
                         'word-vector':args.word_vector,
                         'nhid':args.nhid,
                         'nlayers':args.nlayers,
                         'nclasses':args.nclasses,
                         'attention-hops':args.attention_hops,
                         'attention-unit':args.attention_unit,
                         'dropout':args.dropout,
                         'device':device,
                         'encoder-type':args.encoder_type,
                         'prebert-path':args.prebert_path,
                         'bert-pooling':args.bert_pooling,
                         'num-filters':args.num_filters,
                         'filter-ht':args.filter_ht,
                         'n-gram':args.n_gram
                        })
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
            trainer = BertTrainer(copy.deepcopy(data_train),data_pair,dictionary,device,args,criterions=criterions,criterionp=criterionp,optimizer=optimizer,cls_num_list=cls_num_list,beta_max=args.beta_max,beta_min=args.beta_min)
        else:
            trainer = Trainer(copy.deepcopy(data_train),data_pair,dictionary,device,args,criterions=criterions,criterionp=criterionp,optimizer=optimizer,cls_num_list=cls_num_list,beta_max=args.beta_max,beta_min=args.beta_min)
        for epoch in range(args.epochs):
            torch.cuda.empty_cache()
            #train_loss, loss_p, loss_s, model = trainer.epoch(epoch+1, model, args.rsamp)
            train_loss, loss_p, loss_s = 0, 0, 0
            acc,f1,_,_ = trainer.evaluate(data_key, data_val, model, bsz=args.test_bsize)
            print('-' * 110)
            print(f'| fold {nfold+1} stage 1 epoch {epoch+1} | total loss {train_loss:.8f} | single loss {loss_s:.8f} | pair loss {loss_p:.8f} | valid f1. {f1:.4f}')
            print('-' * 110)
            if not best_f1 or f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model)
        best_boost_f1 = best_f1
        if args.stage2 > 0:
            model = best_model
            model = model.to(device)
            if args.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=args.lr*0.25, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
            elif args.optimizer == 'Adadelta':
                optimizer = optim.Adadelta(model.parameters(), lr=args.lr*0.25, rho=0.95)
            trainer.update_opt(optimizer)
            for epoch in range(args.stage2):
                torch.cuda.empty_cache()
                train_loss, loss_p, loss_s, model = trainer.epoch(epoch+1, model, args.rsamp)
                acc,f1,_,_ = trainer.evaluate(data_key, data_val, model, bsz=args.test_bsize)
                print('-' * 110)
                print(f'| fold {nfold+1} stage 2 epoch {epoch+1} | train loss (total) {train_loss:.8f} | single loss {loss_s:.8f} | pair loss {loss_p:.8f} | valid f1. {f1:.4f}')
                print('-' * 110)
                if not best_boost_f1 or f1 > best_boost_f1:
                    best_boost_f1 = f1
                    best_model = copy.deepcopy(model)

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
        test_acc, macf1, y_pred, y_true = trainer.evaluate(data_key2, data_test, model, bsz=args.test_bsize)
        print('-' * 92)
        print(f'| test acc. {test_acc:.4f} | test macro. F1 {macf1:.4f} |')
        print('-' * 92)
        if not args.validation:
            qeval  = QEval(data_train, data_test)
            per_quin_acc, per_quin_f1 = qeval.evaluate(y_pred, y_true)
            print(f'f1 per quintile = {per_quin_f1}\n')
            print(f'accuracy per quintile = {per_quin_acc}\n')
        kbest_acc.append(test_acc)
        kbest_f1.append(macf1)
        del model, best_model
        torch.cuda.empty_cache()

    if args.validation:
        with open(args.validation_log,'a') as f:
            nl = '\n'
            tab = '\t'
            f.write(f'{nl}{nl}{args}{nl}{tab}{kbest_acc}{tab}{np.mean(kbest_acc)}{nl}{tab}{kbest_f1}{tab}{np.mean(kbest_f1)}')
    exit(0)
    
