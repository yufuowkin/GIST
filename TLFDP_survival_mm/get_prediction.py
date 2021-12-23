import sys
import argparse
from model import HE2RNA
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

from wsi_data import TCGAFolder, TCGAFolder_coords, patient_kfold

def topkmean(x):
    return np.mean(np.sort(x)[::-1][:np.min(100, len(x))])

def bottomkmean(x):
    return np.mean(np.sort(x)[:np.min(100, len(x))])

def lastsaved(wdir, pattern = 'weight'):
    file=None
    if  os.path.isdir(wdir):
        files = os.listdir(wdir)
        paths = [os.path.join(wdir, basename) for basename in files if basename.startswith(pattern)]
        file = max(paths, key=os.path.getctime)
    return file 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str, default=".")
    parser.add_argument('--model_name', type=str, default="HE2RNA")
    parser.add_argument('--model_mm', type=bool, default=False)
    parser.add_argument('--outdir', type=str, default=".")
    parser.add_argument('--infofile', type=str, default=".")
    parser.add_argument('--featuredir', type=str, default=".")
    parser.add_argument('--nClass', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--nFeature', type=int, default=2048)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=0)

    args = parser.parse_args()
    
    dataset = TCGAFolder_coords(args.infofile, 
                                args.featuredir, 
                                args.nFeature+3, 
                                args.model_mm)
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=0)
                            
    for repeat in range(args.repeats):
        for fold in range(args.n_splits):
            file = lastsaved(args.saved_model + 'model_' + str(repeat) + '_' + str(fold))
            print(file)
            if file:
                outfile1 = args.outdir + '/pred.repeat' + str(repeat) + '_cv' + str(fold) + '.csv'
                
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

                    colnames = ['pred'+str(i) for i in range(int(args.nClass))] 

                    Pred = pd.DataFrame(columns =  ['x', 'y', 'score-X', 'image', 'label', 'sets'])    
                    Pred.to_csv(outfile1, index=False) # per tile
                    
                    for x, time, event, path, i in dataloader:
                        sets = 'test'
                        
                        print(path, i.numpy()[0])
                        coord = x[:, 1:3, :]
                        x = x[:, 3:, :]
                            
                        pred = model.conv(x.float().to(model.device)).detach().cpu().numpy()
                            
                        df = pd.DataFrame(data=np.concatenate((coord[0], pred[0])).T)
                        df['image'] = ntpath.basename(path[0])
                        df['time'] = time[0].numpy()
                        df['event'] = event[0].numpy()
                        df['sets'] = sets
                        df.to_csv(outfile1, mode='a', header=False, index=False)

if __name__ == '__main__':
    main()