import sys
import argparse
from model_ce import HE2RNA
from os import path
import os
import ntpath
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader

from wsi_data import load_labels, AggregatedDataset, TCGAFolder, TCGAFolder_coords, \
    H5Dataset, patient_split, match_patient_split, \
    patient_kfold, match_patient_kfold

def topkmean(x):
    return np.mean(np.sort(x)[::-1][:100])

def bottomkmean(x):
    return np.mean(np.sort(x)[:100])

def lastsaved(wdir, fold, pattern = 'weight'):
    file=None
    wdir = wdir + '/model_' + str(fold)
    if  os.path.isdir(wdir):
        files = os.listdir(wdir)
        paths = [os.path.join(wdir, basename) for basename in files if basename.startswith(pattern)]
        file = max(paths, key=os.path.getctime)
    return file 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str, default=".")
    parser.add_argument('--outdir', type=str, default=".")
    parser.add_argument('--inputfile', type=str, default=".")
    parser.add_argument('--featuredir', type=str, default=".")
    parser.add_argument('--nClass', type=int, default=2)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=0)

    args = parser.parse_args()
    
    dataset = TCGAFolder_coords(args.inputfile, args.featuredir, 2051)
                                                   
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=0)
    
    for fold in range(args.n_splits):
        file = lastsaved(args.saved_model, fold)
        print(file)
        if file:
            outfile1 = args.outdir + '/pred.cv' + str(fold) + '.csv'
            if not os.path.isfile(outfile1):
                weights= torch.load(file, map_location='cpu')
                model = HE2RNA(input_dim=weights['conv0.weight'].shape[1],
                   output_dim=weights['conv2.weight'].shape[0],
                   layers=[512, 512],
                   nonlin=nn.ReLU(),
                   ks=[100],
                   dropout=0.25,
                   device=args.device)
                model.load_state_dict(weights)
                model.eval()

                colnames=['x', 'y']
                [colnames.append('pred'+str(i)) for i in range(int(args.nClass))] 
                
                Pred = pd.DataFrame(columns = np.hstack([colnames, 'image', 'label', 'sets']))
                Pred.to_csv(outfile1, index=False)
                
                for x, y, path, i in dataloader:
                    print(path, i.numpy()[0])
                    coord = x[:, 1:3, :]
                    x = x[:, 3:, :]
                    pred_logit = model.conv(x.float().to(model.device))
                    pred = nn.Softmax(dim=1)(pred_logit)
                    pred = pred.detach().cpu().numpy()

                    df = pd.DataFrame(data=np.concatenate((coord[0], pred[0])).T, columns = colnames)
                    df['image'] = ntpath.basename(path[0])
                    df['label'] = y[0].numpy()
                    
                    df['sets'] = 'test'
                    df.to_csv(outfile1, mode='a', header=False, index=False)

if __name__ == '__main__':
    main()

