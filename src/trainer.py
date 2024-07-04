import os
import csv
import time
import config
import random
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader 
from sklearn.model_selection import KFold
from utils  import *
from network import *
from Dataset import *
from torchvision import models,transforms
from PIL import Image
from sklearn.metrics import *
from datetime import datetime
from tqdm import tqdm

#2値分類
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

class Subset_label(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def pick_label(self,idx):
        return self.dataset.pick_label(self.indices[idx])

def write_LogHeader(log_path):
    #CSVファイルのヘッダー記述
    with open(log_path + "/log.csv",'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['model_name','lr','seed','sampler','beta','gamma','phase','epoch','loss','roc-auc','pr-auc','f1','TN','FN','FP','TP'])

def write_Scores(log_path,result_list):
    #リストを1行書き加える。
    with open(log_path + "/log.csv",'a',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result_list)

def plot_Loss(dir_path,lossList,lr,preprocess):
    for phase in ['learning','valid']:
        epochs = [x+1 for x in list(range(len(lossList[phase])))]
        plt.plot(epochs,lossList[phase],label= phase+'_lr=' + '{:.1e}'.format(lr) + '_' + preprocess)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(dir_path,'Loss_') + '{:%y%m%d-%H:%M}'.format(datetime.now()) + '.png')

def calc_class_count(dataset,n_class):
    #データセットからラベルの値を一つずつ取り出して、クラス数を計算する
    class_count = [0] * n_class 
    l = len(dataset)
    print(l)

    start_time = time.time()
    for i in range(l):
        if all(torch.argmax(dataset.pick_label(i)) == torch.Tensor([1])):
            label = 1
        else :
            label = 0
        class_count[label] += 1

    end_time = time.time()
    print(class_count)

    return class_count

def calc_class_inverse_weight(dataset,n_class):
    #逆数の場合の重み付け
    class_count = calc_class_count(dataset,n_class)
    class_weight = [torch.Tensor([n**(-1) * sum(class_count)]) for n in class_count]

    print(class_weight)
    return torch.Tensor(class_weight)

def calc_class_weight(dataset,n_class,beta):
    #Class Balanced Lossによる重み付け
    #class_count = calc_class_count(dataset,n_class)
    if beta == -1:
        return calc_class_inverse_weight(dataset,n_class)
    elif beta == 0:
        class_weight = [torch.Tensor([1]) for n in range(config.n_class)]
        print(class_weight)
        return torch.Tensor(class_weight)
    else:
        class_weight = [torch.Tensor([(1-beta)/(1-beta**n)*sum(class_count)]) for n in class_count]
        print(class_weight)
        return torch.Tensor(class_weight)


class Trainer():
    def __init__(self, c):
        self.dataloaders = {}
        self.prms = []
        self.search = c
        self.n_seeds = len(c['seed'])
        self.n_splits = 5
        self.loss = {}
        self.now = '{:%y%m%d%H%M}'.format(datetime.now())
        self.log_path = '/home/takinami/exfoliation/log/' + str(self.now)
        os.makedirs(self.log_path, exist_ok=True)
        write_LogHeader(self.log_path)

    def run(self):
        #実行時間計測
        start = time.time()

        #評価に使う変数の初期化
        for phase in ['learning','valid']:
            self.loss[phase] = 0

        for n_iter,(c,param) in enumerate(iterate(self.search)):
            start_time = time.time()
            random.seed(c['seed'])
            torch.manual_seed(c['seed'])
            print('Parameter :',c)
            self.c = c

            self.net = make_model(self.c['model_name'],1)
            #self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
            self.optimizer = optim.Adam(params=self.net.parameters(),lr=self.c['lr'])
            #self.net = nn.DataParallel(self.net)
            #訓練、検証に分けてデータ分割
            if os.path.exists(config.normal_pkl):
                with open(config.normal_pkl,mode="rb") as f:
                    self.dataset = pickle.load(f)
            else :
                self.dataset = load_dataset()
                with open(config.normal_pkl,mode="wb") as f:
                    pickle.dump(self.dataset,f)
            
            #訓練、検証に分けてデータ分割
#            self.dataset = load_dataset()
            kf = KFold(n_splits=self.n_splits,shuffle=True,random_state=0)
            self.dataset_train_id = kf.split(self.dataset['train']) if not self.c['evaluate'] else  [(self.dataset['train'],[])]

            lossList = {}
            for phase in ['learning','valid']:
                if self.c['cv'] == 0:
                    lossList[phase] = [[] for x in range(1)]
                else:
                    lossList[phase] = [[] for x in range(self.n_splits)]

            #learning_id_index , valid_id_index = kf.split(train_id_index,y).__next__() 1つだけ取り出したいとき
            
            for a,(learning_id,valid_id) in enumerate(self.dataset_train_id):
                # 事前学習なしモデルの場合
                if not self.c['pretrained']:
                    self.net.apply(init_weights)

                self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
                if self.c['evaluate']:
                    learning_dataset = self.dataset['train']
                if not self.c['evaluate']:
                    learning_dataset = Subset_label(self.dataset['train'],learning_id)
                    valid_dataset = Subset_label(self.dataset['train'],valid_id)
                self.class_weight = calc_class_weight(learning_dataset,config.n_class,beta=self.c['beta'])
                self.criterion = Focal_MultiLabel_Loss(gamma=self.c['gamma'],weights=self.class_weight.to(device))

                #Dataloaderの種類を指定
                if self.c['sampler'] == 'normal':
                    self.dataloaders['learning'] = DataLoader(learning_dataset,self.c['bs'],num_workers=os.cpu_count()//2,shuffle=True,persistent_workers=True)
                elif self.c['sampler'] == 'over':
                    self.dataloaders['learning'] = BinaryOverSampler(learning_dataset,self.c['bs']//2)
                elif self.c['sampler'] == 'under':
                    self.dataloaders['learning'] = BinaryUnderSampler(learning_dataset,self.c['bs']//2)

                end_time = time.time()

                print("実行時間 :",end_time-start_time)

                if not self.c['evaluate']:
                    #検証データに対するSamplerは普通のDataLoaderと同じ
                    self.dataloaders['valid'] = DataLoader(valid_dataset,self.c['bs'],
                    shuffle=True,num_workers=os.cpu_count()//2,persistent_workers=True)
                    
                for epoch in range(1, self.c['n_epoch']+1):
                    learningauc,learningloss,learningprecision,learningrecall \
                        = self.execute_epoch(epoch, 'learning')
                    #平均計算用にauc.lossを保存。
                    lossList['learning'][a].append(learningloss)

                    if not self.c['evaluate']:
                        valid_pr_auc,validloss,validprecision,validrecall \
                            = self.execute_epoch(epoch, 'valid')


                        lossList['valid'][a].append(validloss)

                        #valid_pr_aucを蓄えておいてあとでベスト10を出力
                        temp = valid_pr_auc,epoch,self.c
                        self.prms.append(temp)

                    #乱数シード×CV数で平均を取るときのために残しておく。
                    if epoch == self.c['n_epoch']:
                        self.loss['learning'] += learningloss
                        if not self.c['evaluate']:
                            self.loss['valid'] += validloss
                
                #n_epoch後の処理
                save_process_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))                
                #CVパラメーターで交差検証を行うかどうかをコントロールできるようにする。
                if not self.c['cv']:
                    break

            #分割交差検証後の処理
            #lossList['learning'] = list(np.mean(lossList['learning'],axis=0))
            #if not self.c['evaluate']:
            #    lossList['valid'] = list(np.mean(lossList['valid'],axis=0))

            #if self.c['cv']:
            #    plot_Loss(os.path.join(config.LOG_DIR_PATH,'images'),lossList,self.c['lr'],self.c['preprocess'])

                #for phase in ['learning','valid']:
                #    epochs = list(range(1,len(lossList[phase])))
                #    plt.plot(epochs,lossList[Phase],label='loss')
                #    plt.xlabel('Epochs')
                #    plt.ylabel(phase + 'Loss')
                #    plt.legend()
                #    plt.savefig(path)


            #n_iter後、シード数×CV数で平均を取る。もっといい方法がありそう。
            if not self.c['evaluate']:
                if not ((n_iter+1)%self.n_seeds):
                    temp = self.n_seeds * self.n_splits
                    for phase in ['learning','valid']:
                        self.loss[phase] /= temp
                        #Scoreの初期化。
                        self.loss[phase] = 0

        #パラメータiter後の処理。
        elapsed_time = time.time() - start
        print(f"実行時間 : {elapsed_time:.01f}")
        # 訓練後、モデルを、'(実行回数)_(モデル名)_(学習epoch).pth' という名前で保存。
        try : 
            model_name = self.search['model_name']
            n_ep = self.search['n_epoch']
            n_ex = 0
            with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'r') as f:
                n_ex = len(f.readlines())
            with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.now,n_ex,model_name,n_ep])
            save_path = '{:0=2}'.format(n_ex)+ '_' + model_name + '_' + '{:0=3}'.format(n_ep)+'ep.pth'
            model_save_path = os.path.join(config.MODEL_DIR_PATH,save_path)
            torch.save(self.net.state_dict(),model_save_path)

        except FileNotFoundError:
            with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Time','n_ex','Model_name','n_ep'])

    #1epochごとの処理
    def execute_epoch(self, epoch, phase):
        preds, labels,total_loss= [], [],0
        if phase == 'learning':
            self.net.train()
        else:
            self.net.eval()

        for inputs_,labels_,_ in tqdm(self.dataloaders[phase]):

            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)

            #Samplerを使うときの処理
            #if phase == 'learning' and ((self.c['sampler'] == 'over') or (self.c['sampler'] == 'under')):
                #inputs_ = inputs_.unsqueeze(1)
                #labels_ = labels_.unsqueeze(1)


            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'learning'):
                outputs_ = self.net(inputs_).to(device)
                #softmax = nn.Softmax(dim=1)
                #outputs_ = softmax(outputs_)
                #outputs__ = outputs_.unsqueeze(1)
                loss = self.criterion(outputs_, labels_)
                total_loss += loss.item()

                if phase == 'learning':
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
            
            #softmax = nn.Softmax(dim=1)
            #outputs_ = softmax(outputs_)

            preds += [outputs_.detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]
            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        labels = np.argmax(labels,axis=1)

        try:
            roc_auc = roc_auc_score(labels, preds[:,1])
        except:
            roc_auc = 0
        
        #labels = np.argmax(labels,axis=1)

        precisions, recalls, thresholds = precision_recall_curve(labels, preds[:,1])
        try:
            pr_auc = auc(recalls, precisions)
        except:
            pr_auc = 0

        temp_class = np.arange(config.n_class)
        preds_scores = np.dot(preds,temp_class)

        preds_scores[preds_scores < 0.5] = 0
        preds_scores[preds_scores >= 0.5] = 1

        print(preds_scores)

        total_loss /= len(preds)
        recall = recall_score(labels,preds_scores)
        precision = precision_score(labels,preds_scores)
        
        f1 = f1_score(labels,preds_scores)
        confusion_Matrix = confusion_matrix(labels,preds_scores)
        try:
            TN = confusion_Matrix[0][0]
        except:
            TN = 0
        try:
            FN = confusion_Matrix[0][1]
        except:
            FN = 0
        try:
            FP = confusion_Matrix[1][0]
        except:
            FP = 0
        try:
            TP = confusion_Matrix[1][1]
        except:
            TP = 0


        print(
            f'epoch: {epoch} phase: {phase} loss: {total_loss:.3f} au-roc: {roc_auc:.3f} au-prc: {pr_auc:.3f} f1:{f1:.3f} TN:{TN} FN:{FN} FP:{FP} TP:{TP}')

        #ここにpr_auc,f1,tn,fn,tp,fpを追加
        result_list = [self.c['model_name'],self.c['lr'],self.c['seed'],self.c['sampler'],self.c['beta'],self.c['gamma'],phase,epoch,total_loss,roc_auc,pr_auc,f1,TN,FN,FP,TP]

        write_Scores(self.log_path,result_list)
        
        #with open(self.log_path + "/log.csv",'a') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(result_list)


        return pr_auc,total_loss,recall,precision