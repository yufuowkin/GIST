"""
HE2RNA: Train a model to predict gene expression on TCGA slides, either on a single train/valid/test split or in cross-validation 
Copyright (C) 2020  Owkin Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# +
import os
import configparser
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import copy as cp
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch import optim
from sklearn.metrics import roc_auc_score
from wsi_data import load_labels, AggregatedDataset, TCGAFolder, \
    H5Dataset, patient_split, match_patient_split, \
    patient_kfold, match_patient_kfold
from model_ce import HE2RNA, fit, predict
from utils import aggregated_metrics, aggregated_metrics_2, compute_metrics

import torch.multiprocessing as mp

class Experiment(object):
    """An class that uses a config file to setup and run a gene expression
    prediction experiment.

    Args:
        configfile (str): Path to the configuration file.
    """

    def __init__(self, args):
        self.outdir = args.outdir
        self.use_saved_model = args.use_saved_model
        self.p_value = 't_test'
        self.lr = float(args.lr)
        self.optimizer = args.optimizer
        self.weight_decay = float(args.weight_decay)
        self.featuredir = args.featuredir
        self.cellcountsfile = args.cellcountsfile
        self.projects  = args.projects.split(",")
        self.lr_decay_gamma = float(args.lr_decay_gamma)
        self.nFeature = int(args.nFeature) # 2048 or 2049
        self.n_folds = int(args.n_folds)
        self.rs = float(args.rs)
        self.nClass = int(args.nClass)
        self.masks=args.masks
        
        model_params = {}
        model_params['layers'] = [512,512]
        model_params['dropout'] = 0.25
        model_params['ks'] = [100] # always use 100 randomly selected tiles per slide
        model_params['nonlin'] = nn.ReLU()
        model_params['device'] = args.device
        model_params['max_epochs'] = int(args.max_epochs)
        model_params['patience'] = int(args.patience)
        model_params['batch_size'] = int(args.batch_size)
        model_params['num_workers'] = int(args.num_workers)
        
        if args.class_weights:
            model_params['class_weights'] = [float(i) for i in args.class_weights.split(',')]
        else:
            model_params['class_weights'] = np.ones(self.nClass)
        
        if args.class_weights_loader:
            model_params['class_weights_loader'] = [float(i) for i in args.class_weights_loader.split(',')]
        else:
            model_params['class_weights_loader'] = False
            
        self.model_params = model_params
        
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)



    def _setup_optimization(self, model):
        optim_params = {'params': model.parameters(),
                        'lr': self.lr,
                        'weight_decay': self.weight_decay}
        if self.optimizer == 'sgd':
            return optim.SGD(**optim_params)
        elif self.optimizer == 'adam':
            return optim.Adam(**optim_params)

    def _setup_lr_decay(self, optimizer):
        return optim.lr_scheduler.ExponentialLR(optimizer, 
                                         gamma=self.lr_decay_gamma,
                                         last_epoch= -1)

    def _build_dataset(self):
        print(self.cellcountsfile, self.featuredir)
        dataset = TCGAFolder(self.cellcountsfile, self.featuredir, 2048, self.masks)
        return dataset

    

    def cross_validation(self, n_folds=5, random_state=0, logdir='exp'):
        """N-fold cross-validation.

        Args:
            n (int): Number of folds
            random_state (int): Random seed used for splitting the data.
            logdir (str): Path for TensoboardX.

        Returns:
            pandas DataFrame: The metrics per gene and per fold.
        """

        logdir = os.path.join(self.outdir, "log")
        
        dataset = self._build_dataset()
        evalset = self._build_dataset()
        
        input_dim = self.nFeature
        output_dim = self.nClass
        
        train_idx, valid_idx, test_idx = patient_kfold(
                    dataset, n_splits=n_folds, valid_size=0.1,
                    random_state=random_state, outdir=self.outdir)
        
        report = {'cells': list(dataset.cells)}
        n_samples = {project: [] for project in dataset.projects}
        print("input dim: " + str(input_dim) + ", output dim: " +str(output_dim))
        for k in range(n_folds):
            train_set = Subset(dataset, train_idx[k])
            train_set_labels = dataset.labels[train_idx[k]]

            test_set = Subset(evalset, test_idx[k])
            print("training size: " + str(len(train_set)))
            print("testing size: " + str(len(test_set)))
                
            if len(valid_idx) > 0:
                valid_set = Subset(evalset, valid_idx[k])
                valid_projects = dataset.projects[valid_idx[k]]
                valid_projects = valid_projects.astype(
                    'category').cat.codes.values.astype('int64')
                print("validation size: " +  str(len(valid_set)))
            
            else:
                valid_set = None
                valid_projects = None

            test_projects = dataset.projects[test_idx[k]].apply(
                lambda x: x.replace('_', '-')).values

            # Initialize the model and define optimizer
            training_params = self.model_params
            model = HE2RNA(input_dim=input_dim,
                               output_dim=output_dim,
                               layers=training_params['layers'],
                               nonlin=training_params['nonlin'],
                               ks=training_params['ks'],
                               dropout=training_params['dropout'],
                               device=training_params['device'])
            if self.use_saved_model:
                weights = torch.load(os.path.join(self.use_saved_model), 
                                     map_location = torch.device(training_params['device']))
                model.load_state_dict(weights)
    
            optimizer = self._setup_optimization(model)
            scheduler = self._setup_lr_decay(optimizer)
            
            # Train model
            preds, labels = fit(model,
                                train_set,
                                train_set_labels,
                                valid_set,
                                valid_projects,
                                test_set=test_set,
                                params=training_params,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                logdir=logdir,
                                path=os.path.join(
                                    self.outdir,
                                    'model_' + str(k)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_folds", help="number of folds for 'cross_validation'",
        default=5)
    parser.add_argument(
        "--rs", help="random state",
        default=0)
    parser.add_argument(
        "--lr", help="learning rate",
        default=3e-5)
    parser.add_argument(
        "--lr_decay_gamma", help="learning rate decay expo",
        default=0.9)
    parser.add_argument(
        "--outdir", help="learning rate",
        default='./result')
    parser.add_argument(
        "--cellcountsfile", help="cell counts input file",
        default='./transcripts.csv')
    parser.add_argument(
        "--featuredir", help="features directory",
        default='./feature.h5')
    parser.add_argument(
        "--nFeature", help="",
        default=2048)
    parser.add_argument(
        "--masks", help="",
        default=False)
    parser.add_argument(
        "--nClass", help="",
        default=2)
    parser.add_argument(
        "--class_weights", help="weight for each class for loss",
        default=False)
    parser.add_argument(
        "--class_weights_loader", help="weight for each class for data loading",
        default=False)
    parser.add_argument(
        "--use_saved_model", help="train from saved model, path to the saved weights",
        default=None)
    parser.add_argument(
        "--max_epochs", help="max_epochs",
        default=100)
    parser.add_argument(
        "--patience", help="early stop",
        default=50)
    parser.add_argument(
        "--batch_size", help="max_epochs",
        default=32)
    parser.add_argument(
        "--num_workers", help="max_epochs",
        default=0)
    parser.add_argument(
        "--optimizer", help="max_epochs",
        default='adam')
    parser.add_argument(
        "--weight_decay", help="max_epochs",
        default=0)
    parser.add_argument(
        "--device", help="gpu",
        default="cude:0")
    parser.add_argument(
        "--projects", help="",
        default="TCGA-BRCA")

    args = parser.parse_args()

    exp = Experiment(args)
    report = exp.cross_validation(
                n_folds=int(args.n_folds),
                random_state=int(args.rs))
    
if __name__ == '__main__':
    main()
