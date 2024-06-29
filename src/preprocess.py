import os
import datetime
import numpy as np
import config
from utils import isint
from PIL import Image,ImageDraw
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
plt.rcParams["axes.facecolor"] = 'white'

#画像のパスを生成
def make_img_paths():
    img_paths = []
    image_list_file = config.train_info_list

    with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(',')
                if isint(items[1]):
                    img_path = items[1] + '_' + items[2] + '_' + '{:0=3}'.format(0) + '.jpg'
                    img_paths.append(img_path)
    return img_paths
    

#画像の作成
def make_imgs(img_paths):
    # imgs = []
    # 画像のファイル名も保存したいため、dictを使用
    root=config.data_root
    imgs = {}
    for path in img_paths:
        img_path = os.path.join(root,path)
        img = Image.open(img_path).convert('L')
        imgs[path] = img
    return imgs

#画像の上下に黒色の矩形領域を追加
# imgs : 入力画像のリスト (一つ一つの画像はPIL Image) 
# 返り値 : 前処理後の画像リスト
def Add_BlackRects(imgs):
    preprocessed_imgs = {}
    for img_path,img in imgs.items():
        #一つ一つの画像に対して、新しい新規作成画像に対して黒色の矩形を追加
        Width = img.width
        Height = img.height
        Size = max(Width,Height)

        preprocessed_img = Image.new('L',(Size,Size))
        AddSize = (max(Width,Height) - min(Width,Height)) // 2

        draw = ImageDraw.Draw(preprocessed_img)

        #上の矩形の部分に黒色の長方形領域を追加
        #(1) : 素朴な実装
        #for x in range(Width):
        #    for y in range(AddSize):
        #        preprocessed_img.putpixel((x,y),0) 
        #(2) : 機能をフル活用
        draw.rectangle((0,0,Width,AddSize),fill=0)

        
        #次の矩形の部分に元の画像領域を追加
        #(1) 素朴な実装
        #for x in range(Width):
        #    for y in range(AddSize,AddSize+Height):
        #        preprocessed_img.putpixel((x,y),img.getpixel((x,y-AddSize)))

        #(2) 画像の貼り付け
        preprocessed_img.paste(img,(0,AddSize+1))

        #下の矩形の部分に長方形領域を追加
        #for x in range(Width):
        #    for y in range(AddSize+Height,2*AddSize+Height):
        #        preprocessed_img.putpixel((x,y),0)

        #(2) : 機能をフル活用
        draw.rectangle((0,AddSize+Height+1,Width,Width),fill=0)

        preprocessed_imgs[img_path] = preprocessed_img
        print(len(preprocessed_imgs))
        
    return preprocessed_imgs

#画像の左右から矩形領域を除く
# imgs : 入力画像のリスト
def Trim_LeftRight(imgs):
    preprocessed_imgs = {}
    for img_path,img in imgs.items():
        Width = img.width
        Height = img.height
        Size = min(Width,Height)
        Offset = Width-Height

        if Offset%2:
            offsetLeftx = Offset//2
            offsetRightx = Width-Offset//2-1
        else:
            offsetLeftx = Offset//2
            offsetRightx = Width-Offset//2

        #PILの切り取り機能
        preprocessed_img = img.crop((offsetLeftx,0,offsetRightx,Height))
        #preprocessed_img = img.crop()
        #preprocessed_img = Image.new('L',(Size,Size))

        #手動で切り取る場合
        #for x in range(Size):
        #    for y in range(Size):
        #        preprocessed_img.putpixel((x,y),img.getpixel((x,y)))

        preprocessed_imgs[img_path] = preprocessed_img

    return preprocessed_imgs


def Crop_SquareFromCenter(imgs,alpha):
    #前処理3 「画像の中心から、Size (px) x Size (px)の正方形領域を切り抜く」
    #ひとまず水平断面画像だけ
	# Size = Height * αとし、αの値を0.5-1.0まで0.1刻みで試す

    preprocessed_imgs = {}
    for img_path,img in imgs.items():
        Width = img.width
        Height = img.height
        Size = Height * alpha
        print(Height,Width)
        Centerx = Width//2
        Centery = Height//2

        #PILの切り取り機能
        Leftx = Centerx - Size//2
        Lefty = Centery - Size//2
        Rightx = Centerx + Size//2
        Righty = Centery + Size//2
        preprocessed_img = img.crop((Leftx,Lefty,Rightx,Righty))
        preprocessed_imgs[img_path] = preprocessed_img

    return preprocessed_imgs

# imgs : 入力画像のリスト
def Detect_LensByHist(imgs):

    preprocessed_imgs = {}
    for img_path,img in imgs.items():
        Width = img.width
        Height = img.height
        # 画像データから画素値を計算 (縦方向の和)
        Sum_pixel_ylist = []
        for x in range(Width):
            Sum_pixel_y = 0
            for y in range(Height):
                Sum_pixel_y += img.getpixel((x,y))
            Sum_pixel_ylist.append(Sum_pixel_y)

        #取得した縦方向の和からヒストグラムを作成
        fig, ax1 = plt.subplots()
        n, bins, patches = ax1.hist(Sum_pixel_ylist, alpha=0.7, label='Frequency')
        Sum_percent = np.add.accumulate(n) / n.sum()
        print(Sum_percent)
        print(bins)

    return preprocessed_imgs



    # 画像データから画素値を計算 (横方向の和)
    np.random.seed(19990811)
    x, y = np.random.rand(2, 100) * 4

    #上で求めた画像位置ごとに画素値を算出しヒストグラムを作成。
    hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

    # これはそのまま使えそう
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # ここは直さないとダメそう。
    #dx = dy = 0.5
    #dz = hist.ravel()
    #fig = plt.figure(figsize=(5,5))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    #plt.savefig("3d_histogram.jpg",dpi=120)
    #plt.show()
    # フォルダを作成 > 保存

def recovery(img):
    AddSize = (1929 - 1688)//2
    Width = 1929
    Height = 1688
    img = img.crop((0,AddSize,Width,Height+AddSize))
    return img


#画像を保存する関数
def Save_imgs(imgs,path_name):

    #imgsはkey : 画像パス、val : 実際の画像 (PIL Image型)のdict
    for img_path,img in imgs.items():
        img.save(img_path)

if __name__ == '__main__':
    #画像パスを入力 > 画像を作成 > 前処理の実施 > 行った前処理ごとにフォルダを作成
    img_paths = glob('C:/Users/syunt/medicaldata/image/OCT/train/NORMAL/*.jpeg')
 #'../medicaldata/images/CASIA2_16/*_000.jpg'
    imgs = make_imgs(img_paths)

    #前処理ごとに画像を作成・保存
    #前処理1 : 上下に黒の矩形領域を増やして正方形にする。
    preprocessed_imgs = Add_BlackRects(imgs)
    Save_imgs(preprocessed_imgs,'Add_BlackRects')

    #前処理を行う。

    #前処理2 : 左右を同じ幅だけトリミングして正方形領域を繰り抜く。
    #preprocessed_imgs = Trim_LeftRight(imgs)
    #Save_imgs(preprocessed_imgs,'Trim_LeftRight')

    #前処理3 : 画像中心からSize * α だけトリミング
    # α = 1のときは上と同じ？
    #for alpha in [0.5,0.6,0.7,0.8,0.9,1.0]:
    #    preprocessed_imgs = Crop_SquareFromCenter(imgs,alpha)
    #   Save_imgs(preprocessed_imgs,'Crop_SquareFromCenter_' + str(alpha))

    #前処理4 : ヒストグラムを作り、真ん中の値で中心を決める。
    #preprocessed_imgs = Detect_LensByHist(imgs)
    #Save_imgs(preprocessed_imgs,'Detect_LensByHist')

    #preprocessed_imgs = Detect_LensByHough(imgs)
    #Save_imgs(preprocessed_imgs,'LensByHough')