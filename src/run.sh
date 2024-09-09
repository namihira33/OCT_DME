#!/bin/bash
    for lr in 3e-5
    do
        for beta in -1
            do
               for gamma in 1.0
            do
                for sampler in normal
                do
                    python3 ./src/run.py cv=0 evaluate=1  pretrained=1 epoch=21 model_name=DeiT lr=$lr beta=$beta gamma=$gamma sampler=$sampler
                done
            done
        done
    done

    for lr in 3e-5
    do
        for beta in -1
            do
               for gamma in 1.0
            do
                for sampler in normal
                do
                    python3 ./src/run.py cv=0 evaluate=1  pretrained=1 epoch=13 model_name=ConViT lr=$lr beta=$beta gamma=$gamma sampler=$sampler
                done
            done
        done
    done


    for lr in 3e-5
    do
        for beta in -1
            do
               for gamma in 0
            do
                for sampler in normal
                do
                    python3 ./src/run.py cv=0 evaluate=1  pretrained=1 epoch=89 model_name=Swin lr=$lr beta=$beta gamma=$gamma sampler=$sampler
                done
            done
        done
    done