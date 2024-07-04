import os
import sys
import time
import copy
import random
import config
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import seaborn as sns
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils  import *
from network import *
from Dataset import *
from tqdm import tqdm
from sklearn.metrics import *
from datetime import datetime
from PIL import Image
#from gradcam import GradCAM
#from gradcam.utils import visualize_cam
from torchvision.utils import make_grid, save_image
import matplotlib


c = {
    'model_name': 'ViT','seed': 0, 'bs': 32
}

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Evaluater():
    def __init__(self,c):
        self.dataloaders = {}
        self.c = c
        self.c['n_per_unit'] = 1

        #now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        #コマンドラインからのを処理する部分。
        args = len(sys.argv)
        with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv')) as f:
            lines = [s.strip() for s in f.readlines()]
        if args < 2 :
            target_data = lines[-1].split(',')
        elif isint(sys.argv[1]):
            if int(sys.argv[1])<1:
                print('Use the first data')
                target_data = lines[-1].split(',')
            else:
                try:
                    target_data = lines[int(sys.argv[1])].split(',')
                except IndexError:
                    print('It does not exit. Use the first data')
                    target_data = lines[-1].split(',') 
        else:
            target_data = lines[-1].split(',')

        #上の情報からモデルを作る部分。
        self.n_ex = '{:0=2}'.format(int(target_data[1]))
        self.c['model_name'] = target_data[2]
        self.c['n_epoch'] = '{:0=3}'.format(int(target_data[3]))
        temp = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep.pth'
        model_path = os.path.join(config.MODEL_DIR_PATH,temp)

        #モデルの作成。
        self.net = make_model(self.c['model_name'],self.c['n_per_unit'])
        self.net.load_state_dict(torch.load(model_path,map_location=device))
        self.criterion = nn.BCEWithLogitsLoss()

    def run(self):
        #各々の部位を整数に変換
        #テストデータセットの用意。
        #訓練、検証に分けてデータ分割
        if os.path.exists(config.normal_pkl):
            with open(config.normal_pkl,mode="rb") as f:
                self.dataset = pickle.load(f)
        else :
            self.dataset = load_dataset()
            with open(config.normal_pkl,mode="wb") as f:
                pickle.dump(self.dataset,f)

#        test_id_index,_ = calc_kfold_criterion('test')
#        test_index,_ = calc_dataset_index(test_id_index,[],'test',self.c['n_per_unit'])
#        test_dataset = Subset(self.dataset['test'],test_index)
        self.dataloaders['test'] = DataLoader(self.dataset['test'],self.c['bs'],
                    shuffle=False,num_workers=os.cpu_count())
        preds,labels,paths,total_loss,accuracy= [],[],[],0,0
        right,notright = 0,0

        #GradCAM
        #target_layer = self.net.net.features[-1]
        #cam = GradCAM(self.net,target_layer)
        self.net.eval()

        for inputs_, labels_,paths_ in tqdm(self.dataloaders['test']):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)


            torch.set_grad_enabled(True)
            outputs_ = self.net(inputs_)
            loss = self.criterion(outputs_, labels_)

            #total_loss += loss.item()
            #mask,_ = cam(inputs_[None])
            #heatmap,result = visualize_cam(mask,inputs_)
            #gray_scale_cam = gray_scale_cam[0, :]
            #visualization = show_cam_on_image(inputs_[0],gray_scale_cam)

            preds += [outputs_.detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]
            paths += paths_

            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)


        r_cnt = 0
        tmp = 0
        cnt = 0
        fig,ax = plt.subplots(4,4,figsize=(16,16))
        
        #水平画像だけが対象のとき
        
#        if self.c['n_per_unit'] == 1:
#            for i,(pred,ans,path) in enumerate(zip(preds,labels,paths)):
#                pred = 0 if sigmoid(pred)<0.5 else 1
#                #GradCAMで判断根拠を図示
#                if pred !=int(ans[0]):
#                    img = Image.open(path).convert('L')
#                   torch_img = transforms.Compose([
#                    transforms.Resize((224, 224)),
#                   transforms.ToTensor()
#                    ])(img).to(device)
#                    normed_torch_img = transforms.Normalize((0.5, ),
#                                                            (0.5, ))(torch_img)[None]
#                    mask, _ = cam(normed_torch_img)
#                    heatmap, result = visualize_cam(mask, torch_img)
#                    heatmap = transforms.ToPILImage()(heatmap)
#                    result = transforms.ToPILImage()(result)
#                    ax[(cnt%16)//4][cnt%4].imshow(result,cmap='gray')
#                    ax[(cnt%16)//4][cnt%4].set_title('Predict:' + str(pred) + '  Answer:' + str(ans[0]))
#                    ax[(cnt%16)//4][cnt%4].title.set_size(20)
#                    if cnt%16==15:
#                        fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','gradcam' + str(cnt//16) + '.png'))
#                        fig,ax = plt.subplots(4,4,figsize=(16,16))                   
#                    tmp = i
#                    cnt += 1
#
#            if cnt%16 != 15:
#                while cnt%16:
#                    ax[(cnt%16)//4][cnt%4].axis('off')
#                    cnt += 1
#                fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','gradcam' + str(cnt//16) + '.png'))                
#
#            fig,ax = plt.subplots(4,4,figsize=(16,16))
#            
#            tmp = 0
#            cnt = 0
#            for i,(pred,ans,path) in enumerate(zip(preds,labels,paths)):
                #不正解画像の予測と答えを図示
#                im = Image.open(path).convert('L')
#                pred = 0 if sigmoid(pred)<0.5 else 1
#                if pred !=int(ans[0]):
#                    ax[(cnt%16)//4][cnt%4].imshow(im,cmap='gray')
#                    ax[(cnt%16)//4][cnt%4].set_title('Predict:' + str(pred) + '  Answer:' + str(ans[0]))
#                    ax[(cnt%16)//4][cnt%4].title.set_size(20)
#                    if cnt%16==15:
#                        fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','exp' + str(cnt//16) + '.png'))
#                        fig,ax = plt.subplots(4,4,figsize=(16,16))
#                    tmp = i
#                    cnt += 1
#
#            if cnt%16 != 15:
#                while cnt%16:
#                    ax[(cnt%16)//4][cnt%4].axis('off')
#                    cnt += 1
#                fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','exp' + str(cnt//16) + '.png'))

        #回転画像も対象のとき
#        else:
#            tmp_l = []
#            for i,(pred,ans,path) in enumerate(zip(preds,labels,paths)):
                #GradCAMで判断根拠を図示
#                pred = 0 if sigmoid(pred)<0.5 else 1
#                if True:
#                    img = Image.open(path).convert('L')
#                    torch_img = transforms.Compose([
#                    transforms.Resize((224, 224)),
#                    transforms.ToTensor()
#                    ])(img).to(device)
#                    normed_torch_img = transforms.Normalize((0.5, ),
#                                                            (0.5, ))(torch_img)[None]
#                    mask, _ = cam(normed_torch_img)
#                    heatmap, result = visualize_cam(mask, torch_img)
#                    heatmap = transforms.ToPILImage()(heatmap)
#                    result = transforms.ToPILImage()(result)
#                    ax[(cnt%16)//4][cnt%4].imshow(result,cmap='gray')
#                    ax[(cnt%16)//4][cnt%4].set_title('Predict:' + str(pred) + '  Answer:' + str(ans[0]))
#                    ax[(cnt%16)//4][cnt%4].title.set_size(20)
#                    if cnt%16==15:
#                        fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','gradcam_spin' + str(cnt//16) + '.png'))
#                        fig,ax = plt.subplots(4,4,figsize=(16,16))                   
#                    tmp = i
#                    cnt += 1
#
#            if cnt%16 != 15:
#                while cnt%16:
#                    ax[(cnt%16)//4][cnt%4].axis('off')
#                    cnt += 1
#                fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','gradcam_spin' + str(cnt//16) + '.png')) 
                #回転画像の中で答えが全て一致しているものを計算
#                if (not (i%16)) and (i//16)>0:
#                    print(tmp_l)
#                    print(tmp_l.count(tmp_l[0]))
#                    if tmp_l.count(tmp_l[0]) == 16:
#                        r_cnt += 1
#                    tmp_l = []
#                pred = 0 if sigmoid(pred)<0.5 else 1
#                tmp_l.append(pred)
                #pred = pred==pred[0].sum()
                #print(pred)
                #im = Image.open(path).convert('L')
                #pred = 0 if sigmoid(pred)<0.5 else 1

                #ax[i//4][i%4].imshow(im,cmap='gray')
                #ax[i//4][i%4].set_title('Predict:' + str(pred) + '  Answer:' + str(ans[0]))
                #ax[i//4][i%4].title.set_size(20)
                #fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','exp.png'))
#            print(r_cnt/(len(self.dataset['test'])/16))
#            print(r_cnt)




        #fig,ax = plt.subplots(4,4,figsize=(16,16))
        #for i,(pred,ans,path) in enumerate(zip(preds[:16],labels[:16],paths[:16])):
        #    im = Image.open(path).convert('L')
        #    pred = 0 if sigmoid(pred)<0.5 else 1

        #    ax[i//4][i%4].imshow(im,cmap='gray')
        #    ax[i//4][i%4].set_title('Predict:' + str(pred) + '  Answer:' + str(ans[0]))
        #    ax[i//4][i%4].title.set_size(20)
        #    fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','exp.png'))
        #    print(path)

        total_loss /= len(preds)


        #Auc値の計算

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

        print("au-roc:", roc_auc)
        print("au-prc:",pr_auc)
        print("recall:",recall)
        print("f1:",f1)
        

        fig,ax = plt.subplots()
        #plt.plot(fpr,tpr,label = 'ROC curve (area = %.3f'%auc)
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_roc_curve1.png'
        fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))

        preds_origin = copy.deepcopy(preds)
        labels_origin = copy.deepcopy(labels)

        for threshold in [0.5]:

            #出力をもとに分類
            #temp_class = np.arange(4)
            #preds = np.dot(preds,temp_class)
            preds[preds < threshold] = 0
            preds[preds >= threshold] = 1
            right = 0

            #labels = np.argmax(labels,axis=1)

            #labels[labels != 1] = 0
            #labels[labels == 1] = 1

            #混同行列を作り、ヒートマップで可視化。
            cm = confusion_matrix(labels,preds_scores)
            fig,ax = plt.subplots()

            #import matplotlib
            #from matplotlib.font_manager import FontProperties
            #font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            #font_prop = FontProperties(fname=font_path)
            #matplotlib.rcParams["font.family"] = font_prop.get_name()
            sns.set(font_scale=1.8)
            sns.heatmap(cm,square=True,cbar=True,annot=True,cmap='Blues',fmt='d')
            #ax.set_ylabel('答え',fontsize=18)
            ax.set_xticklabels([0,1],fontsize=20)
            ax.set_yticklabels([0,1],fontsize=20)
            #ax.set_xlabel('モデルの予測',fontsize=18)
            #ax.set_title('混同行列',fontsize=20)
            fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_hm.png'

            plt.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))

            #right += (preds == labels).sum()
            #notright += len(preds) - (preds == labels).sum()
            #accuracy = right / len(self.dataset['test'])
            #recall = recall_score(labels,preds)
            #precision = precision_score(labels,preds)
            #print('threshold:',threshold)
            #print('accuracy :',accuracy)
            #print('auc :',auc)


        #評価値の棒グラフを作って保存。
        #fig,ax = plt.subplots()
        #ax.bar(['Acc','Auc','Recall','Precision'],[accuracy,auc,recall,precision],width=0.4,tick_label=['Accuracy','Auc','Recall','Precision'],align='center')
        #ax.grid(True)
        #plt.yticks(np.linspace(0,1,21))
        #fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_graph.png'
        #fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))


if __name__ == '__main__':
    evaluater = Evaluater(c)
    evaluater.run()
