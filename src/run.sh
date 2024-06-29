#!/bin/bash
python ./src/run.py cv=0 evaluate=1  pretrained=1 epoch=50 model_name=ViT lr=5e-4 beta=0 gamma=0 sampler=normal
#python ./src/run.py cv=1 evaluate=0  pretrained=0 epoch=50 model_name=ResNet18 lr=5e-4 beta=0 gamma=0 sampler=normal
#python ./src/run.py cv=0 evaluate=1  pretrained=0 epoch=15 model_name=Vgg19_bn lr=5e-4 beta=0 gamma=0 sampler=normal
#python ./src/run.py cv=0 evaluate=1  pretrained=0 epoch=15 model_name=EfficientNetV2 lr=5e-4 beta=0 gamma=0 sampler=normal
