import torchvision.models as models
import torch.nn as nn
import torch
import config
import timm

device = torch.device('cuda:0')

class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=1)
        self.net.features[0] = nn.Conv2d(3,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg16_bn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg16_bn()
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=config.n_class)
        self.net.features[0] = nn.Conv2d(3,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg19_bn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg19_bn()
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=config.n_class)
        self.net.features[0] = nn.Conv2d(3,64,3,padding=1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.net(x)
        x = self.softmax(x)
        return x

class Resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet34()
        self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=512,out_features=config.n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18()
        self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=512,out_features=config.n_class)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x
    
class EfficientNetV2(nn.Module):
    def __init__(self,num_classes=config.n_class):
        super().__init__()
        self.net = timm.create_model('efficientnetv2_m', num_classes=config.n_class) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x
    
class ViT(nn.Module):
    def __init__(self,num_classes=config.n_class):
        super().__init__()
        model_name = 'vit_base_patch16_224'
        self.net = timm.create_model(model_name,pretrained=True,num_classes=config.n_class)
        self.softmax = nn.Softmax(dim=1)


        # # 重みを更新するパラメータを選択する
        # update_param_names = ['head.weight','head.bias']
        # for name, param in self.net.named_parameters():
        #     if name in update_param_names:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x
    
class ViT21k_FF(nn.Module):
    def __init__(self,num_classes=config.n_class):
        super().__init__()
        model_name = 'vit_base_patch16_224_in21k'
        self.net = timm.create_model(model_name,pretrained=True,num_classes=config.n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self,num_classes=config.n_class):
        super().__init__()
        model_name = 'swin_base_patch4_window7_224'
        self.net =  model = timm.create_model(model_name, pretrained=True,num_classes=config.n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x

class DeiT(nn.Module):
    def __init__(self,num_classes=config.n_class):
        super().__init__()
        model_name =  'deit_base_distilled_patch16_224'
        self.net =  model = timm.create_model(model_name, pretrained=True,num_classes=config.n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x

class ConViT(nn.Module):
    def __init__(self,num_classes=config.n_class):
        super().__init__()
        model_name =  'convit_base'
        self.net =  model = timm.create_model(model_name, pretrained=True,num_classes=config.n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x

class DeiT3(nn.Module):
    def __init__(self,num_classes=config.n_class):
        super().__init__()
        model_name =  'deit3_base_patch16_224'
        self.net =  model = timm.create_model(model_name, pretrained=True,num_classes=config.n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x

    
#ViT追加とpretrainアリの場合はモデルの重み初期化、なしの場合はモデルの重み初期化なしにするためのパラメータ追加


def make_model(name,n_per_unit):
    if name == 'Vgg16':
        net = Vgg16().to(device)
    elif name == 'Vgg16_bn':
        net = Vgg16_bn().to(device)
    elif name == 'Vgg19_bn':
        net = Vgg19_bn().to(device)
    elif name == 'ResNet18':
        net = Resnet18().to(device)
    elif name == 'ResNet34':
        net = Resnet34().to(device)
    elif name == 'EfficientNetV2':
        net = EfficientNetV2().to(device)
    elif name =='ViT':
        net = ViT().to(device)
    elif name == 'ViT_21k':
        net = ViT21k_FF().to(device)
    elif name == 'DeiT':
        net = DeiT().to(device)
    elif name == 'Swin':
        net = SwinTransformer().to(device)
    elif name == 'ConViT':
        net = ConViT().to(device)

    return net