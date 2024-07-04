from trainer import Trainer
import numpy as np
import sys

c = {'model_name': ['Vgg16_bn'],'seed':[0], 'bs': 8,'lr':[5e-4]}


if __name__ =='__main__':
    args = len(sys.argv)
    if args > 1:
        print(sys.argv)
        # c['model_name'] = sys.argv[1]
        c['cv'] = int(sys.argv[1].split('=')[1])
        c['evaluate'] = int(sys.argv[2].split('=')[1])
        c['pretrained'] = int(sys.argv[3].split('=')[1])
        c['n_epoch'] = int(sys.argv[4].split('=')[1])
        c['model_name'] = sys.argv[5].split('=')[1]
        c['lr'] = float(sys.argv[6].split('=')[1])
        c['beta'] = float(sys.argv[7].split('=')[1])
        c['gamma'] = float(sys.argv[8].split('=')[1])
        c['sampler'] = sys.argv[9].split('=')[1]

    trainer = Trainer(c)
    trainer.run()