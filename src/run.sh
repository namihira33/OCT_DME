#!/bin/bash

for lr in 3e-6 5e-6 3e-5
    do
        for beta in 0 -1
            do
               for gamma in 0 1.0
            do
                for sampler in over normal
                do
                    python ./src/run.py cv=1 evaluate=0  pretrained=1 epoch=100 model_name=Vgg19_bn lr=$lr beta=$beta gamma=$gamma sampler=$sampler 
                done
        done
    done
done

for lr in 3e-6 5e-6 3e-5
    do
        for beta in 0 -1
            do
               for gamma in 0 1.0
            do
                for sampler in over normal
                do
                    python ./src/run.py cv=1 evaluate=0  pretrained=1 epoch=100 model_name=ViT_21k lr=$lr beta=$beta gamma=$gamma sampler=$sampler
                done
            done
        done
    done