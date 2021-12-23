"""
HE2RNA: Arrange data and labels into pytorch datasets
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
import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, TensorDataset, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Compose
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold


def make_dataset(dir, file_list, times, events):
    """Associate file names and labels"""
    images = []
    dir = os.path.expanduser(dir)

    for fname, time, event in zip(file_list, times, events):
        path = os.path.join(dir, fname)
        if os.path.exists(path):
            item = (path, time, event)
            images.append(item)
        else:
            print(path)

    return images

def make_dataset_mm(dir, file_list, times, events, tumorsizes, tumorlocs):
    """Associate file names and labels"""
    images = []
    dir = os.path.expanduser(dir)
    for fname, time, event, tumorsize, tumorloc in zip(file_list, times, events, tumorsizes, tumorlocs):
        path = os.path.join(dir, fname)
        if os.path.exists(path):
            item = (path, time, event, tumorsize, tumorloc)
            images.append(item)
        else:
            print(path)

    return images


class ToTensor(object):
    """A simple transformation on numpy array to obtain torch-friendly tensors.
    """
    def __init__(self, dim=2048, n_tiles=8000):
        self.n_tiles = n_tiles
        self.dim = dim
    def __call__(self, sample):
        x = torch.from_numpy(sample).float()
        if x.shape[0] > self.n_tiles:
            x = x[: self.n_tiles]
        elif x.shape[0] < self.n_tiles:
            x = torch.cat((x, torch.zeros((self.n_tiles - x.shape[0], self.dim+3))))
        return x.t()

class ToTensor_all(object):
    """A simple transformation on numpy array to obtain torch-friendly tensors.
    """
    def __call__(self, sample):
        x = torch.from_numpy(sample).float()
        return x.t()


class RemoveCoordinates(object):
    """Remove tile levels and coordinates."""
    def __call__(self, sample):
        return sample[3:]


class TCGAFolder(Dataset):
    """A class similar to torchvision.FolderDataset for dealing with npy files
    of one or several TCGA project(s).

    Args
        genes (list): List of Ensembl IDs of genes to be used as targets.
        patients (list): list of patient IDs to perform patient split.
        projectname (str or None): Project.ID
        file_list (list): list of paths to .npy files containing tiled slides.
        labels (list or np.array): the associated gene expression values.
        transform (callable): Preprocessing of the data.
        target_transform (callable): Preprocessing of the targets.
    """
    def __init__(self, infofile, featuredir, dim, masks, model_mm):
        self.featuredir = featuredir
        self.info = pd.read_csv(infofile)
        self.dim = dim
        self.patients = self.info['case'].values
        self.times = self.info['time'].values
        self.events = self.info['event'].values
        self.projects = self.info['project']
        self.transform = Compose([ToTensor(dim=self.dim), RemoveCoordinates()])
        self.masks = masks
        self.model_mm = model_mm

        if self.model_mm:
            self.samples = make_dataset_mm(self.featuredir, 
                                            self.info['image'].values, 
                                            self.info['time'].values,
                                            self.info['event'].values,
                                            self.info['tumorsize'].values,
                                            self.info['tumorloc'].values)
        else:
            self.samples = make_dataset(self.featuredir, 
                                        self.info['image'].values, 
                                        self.info['time'].values,
                                        self.info['event'].values)
            

    def __getitem__(self, index):
        if self.model_mm:
            path, time, event, tumorsize, tumorloc = self.samples[index]
            sample = np.load(path)
            sample = np.hstack((sample, 
                                tumorsize*np.ones((sample.shape[0], 1)),
                                tumorloc*np.ones((sample.shape[0], 1))))
        else:
            path, time, event = self.samples[index]
            sample = np.load(path)
        
        if self.masks:
            mask = np.load(path + '.mask.npy')
            if mask.shape[0]!=sample.shape[0]:
                print("mask of wrong size")
                sys.exit(1)
            sample = sample[np.where(mask == True)[0], :]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, time, event

    def __len__(self):
        return len(self.samples)

    

class TCGAFolder_coords(Dataset):
    """A class similar to torchvision.FolderDataset for dealing with npy files
    of one or several TCGA project(s).

    Args
        genes (list): List of Ensembl IDs of genes to be used as targets.
        patients (list): list of patient IDs to perform patient split.
        projectname (str or None): Project.ID
        file_list (list): list of paths to .npy files containing tiled slides.
        labels (list or np.array): the associated gene expression values.
        transform (callable): Preprocessing of the data.
        target_transform (callable): Preprocessing of the targets.
    """
    def __init__(self, infofile, featuredir, dim, model_mm):
        self.featuredir = featuredir
        self.info = pd.read_csv(infofile)
        self.dim = dim
        self.patients = self.info['case'].values
        self.times = self.info['time'].values
        self.events = self.info['event'].values
        self.projects = self.info['project']
        self.transform = ToTensor_all()
        self.model_mm = model_mm
        
        if self.model_mm:
            self.samples = make_dataset_mm(self.featuredir, 
                                            self.info['image'].values, 
                                            self.info['time'].values,
                                            self.info['event'].values,
                                            self.info['tumorsize'].values,
                                            self.info['tumorloc'].values)
        else:
            self.samples = make_dataset(self.featuredir, 
                                        self.info['image'].values, 
                                        self.info['time'].values,
                                        self.info['event'].values,)
    
    def __getitem__(self, index):
        if self.model_mm:
            path, time, event, tumorsize, tumorloc = self.samples[index]
            sample = np.load(path)
            sample = np.hstack((sample, 
                                tumorsize*np.ones((sample.shape[0], 1)),
                                tumorloc*np.ones((sample.shape[0], 1))))
        else:
            path, time, event = self.samples[index]
            sample = np.load(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, time, event, path, index

    def __len__(self):
        return len(self.samples)


def patient_kfold(dataset, n_splits=5, random_state=0, valid_size=0.1, outdir="."):
    """Perform cross-validation with patient split.
    """
    indices = np.arange(len(dataset))
    print(len(dataset))
    
    events = dataset.events
    patients = dataset.patients

    patients_unique, pos = np.unique(patients, return_index=True)
    events = events[pos]

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    ind = skf.split(patients_unique, events)

    train_idx = []
    valid_idx = []
    test_idx = []

    for k, (ind_train, ind_test) in enumerate(ind):
        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]
        print(len(indices))
        test_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                       patients_test[np.newaxis], axis=1)])
        np.savetxt(outdir + "/test_pats_repeat_" + str(random_state) + "_fold_" + str(k) + ".csv",  
                    patients_test, delimiter=',', fmt="%s")
        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=0)
            valid_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                            patients_valid[np.newaxis], axis=1)])

        train_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        patients_train[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx
