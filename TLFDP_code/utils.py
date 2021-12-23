"""
HE2RNA: Computation of correlations and p-values
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
import numpy as np
import pickle as pkl
import scipy
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import chi2
import pandas as pd 
from sklearn.metrics import r2_score, roc_curve, auc
import re
import matplotlib.pyplot as plt

def cleanTable(tmp):
    characters_to_remove = ' ()\''
    pattern = "[" + characters_to_remove + "]"

    re.sub(pattern, "", tmp['Unnamed: 0'][0])
    tmp['image'] = [re.sub(pattern, '', i.split(',')[0]) for i in tmp['Unnamed: 0']]
    tmp['label'] = [re.sub(pattern, '', i.split(',')[1]) for i in tmp['Unnamed: 0']]
    tmp['label'] = tmp['label'].astype('int')
    tmp['sets'] = [re.sub(pattern, '', i.split(',')[2]) for i in tmp['Unnamed: 0']]
    tmp = tmp.drop(columns='Unnamed: 0')
    return tmp


def getMetric(tmp, measure = 'topkmean', plot=True, cutoff=None):
    tv_t = tmp.loc[tmp['sets']=='train']['label']
    pred_t = tmp.loc[tmp['sets']=='train'][measure]

    tv_v = tmp.loc[tmp['sets']=='valid']['label']
    pred_v = tmp.loc[tmp['sets']=='valid'][measure]
    
    if plot:
        fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, figsize=(30,10))
    if len(tv_t) >= 1:
        #r2
        r2_train = r2_score(tv_t, pred_t)
        #rho
        rho_train= stats.pearsonr(tv_t, pred_t)
    
        tv_t_cat = np.zeros(len(pred_t))
        tv_t_cat[tv_t > cutoff] = 1
   
        fpr_t, tpr_t, thresholds_t = roc_curve(tv_t_cat, pred_t, pos_label=1)
        auc_train = auc(fpr_t, tpr_t)
        if plot:
            ax1.plot(tv_t, pred_t, 'o')
            ax1.set(xlabel='true score', ylabel='predicted score')
            ax1.set_title('train')
            ax1.annotate("R2: " + str(round(r2_train,2)) + ", rho: " + str(rho_train[0]), (0, np.min(pred_t)))

            ax3.plot(fpr_t, tpr_t, 'r', label="AUC train: "+ str(round(auc(fpr_t, tpr_t), 2)))
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else: 
        r2_train=None
        rho_train=[None, None]
        auc_train =None
            
    if len(tv_v) >= 1:
        r2_valid = r2_score(tv_v, pred_v)
    
        rho_valid= stats.pearsonr(tv_v, pred_v)
        
        tv_v_cat = np.zeros(len(pred_v))
        tv_v_cat[tv_v > cutoff] = 1

        fpr_v, tpr_v, thresholds_v = roc_curve(tv_v_cat, pred_v, pos_label=1)
        auc_valid = auc(fpr_v, tpr_v)
        if plot:    
            ax2.plot(tv_v, pred_v, 'o')
            ax2.set(xlabel='true score', ylabel='predicted score')
            ax2.set_title('valid')
            ax2.annotate("R2: " + str(round(r2_valid,2)) + ", rho: " + str(rho_valid[0]), (0, np.min(pred_v)))
        
            ax3.plot(fpr_v, tpr_v, 'b', label="AUC valid:" + str(round(auc(fpr_v, tpr_v), 2)))
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else: 
        r2_valid=None
        rho_valid=[None, None]
        auc_valid =None
        
    return pd.DataFrame({"r2_train": [r2_train],
            "r2_valid": [r2_valid],
            "rho_train": [rho_train[0]],
            "rho_pv_train": [rho_train[1]],
            "rho_valid": [rho_valid[0]],
            "rho_pv_valid": [rho_valid[1]],
            "auc_train": [auc_train],
            "auc_valid": [auc_valid]})


def corr(pred, label, i):
    return stats.pearsonr(
        label[:, i],
        pred[:, i]) # return rho and pv

def compute_metrics(label, pred):
    res = Parallel(n_jobs=16)(
        delayed(corr)(pred, label, i) for i in range(label.shape[1])
    )
    return res

def cv_rho(rhos, Ns, pvs):
    rhos = rhos[~pd.isnull(pvs)].astype(float)
    Ns = Ns[~pd.isnull(pvs)].astype(float)
    pvs = pvs[~pd.isnull(pvs)].astype(float)

    z=0.5*np.log((1+rhos)/(1-rhos))
    s = 1.06/(Ns-3)
    s0 = 1/np.sum(1/s)
    z0 = np.sum(z/s)*s0
    
    rho_l = np.tanh(z0-1.96*np.sqrt(s0))
    rho = np.tanh(z0)
    rho_h = np.tanh(z0+1.96*np.sqrt(s0))
    pv_cv = 1 - chi2.cdf(-2*np.sum(np.log(pvs)), df = 2*len(pvs))
    
    return [rho_l, rho, rho_h, pv_cv]


def aggregated_metrics_2(df, project, df_null=None, ns=None):
    col_rhos = [col for col in df.columns if col.startswith('rho_' + project)]
    col_pvs = [col for col in df.columns if col.startswith('pv_' + project)]
    col_Ns = [col for col in df.columns if col.startswith('Nsample_' + project)]
    
    CV_rhos = df.apply(lambda x: cv_rho(x[col_rhos].values.astype(float), 
                                        x[col_Ns].values.astype(int), 
                                        x[col_pvs].values.astype(float)), axis = 1, result_type='expand')
    CV_rhos = CV_rhos.rename(columns={0: "rho_l", 1: "rho", 2: "rho_h",  3: "pv_cv"})
    CV_rhos['gene'] = df['gene']
    CV_rhos = CV_rhos[['gene', 'rho_l', 'rho', 'rho_h', 'pv_cv']]
    CV_rhos['p_value_corrected_hs'] = multipletests(CV_rhos['pv_cv'].values, method='hs')[1]
    CV_rhos['p_value_corrected_bh'] = multipletests(CV_rhos['pv_cv'].values, method='fdr_bh')[1]
    return CV_rhos


def aggregated_metrics(df, project, available_genes, df_null=None, ns=None):

    #available_genes = pkl.load(open('genes_with_nonzero_median_per_project.pkl', 'rb'))
    df_p = df[['gene'] + [col for col in df.columns if col.startswith('correlation_' + project)]]
    #df_p = df_p.loc[df_p['gene'].isin(available_genes[project])].fillna(0)
    df_p['mean_correlation'] = df_p[[col for col in df_p.columns if col.startswith('correlation_')]].mean(axis=1)

    if df_null is not None:
        null_hypothesis = df_null.loc[
            df_null['gene'].isin(available_genes[project]),
            [col for col in df_null.columns if col.startswith('correlation_' + project)]].mean(axis=1)
        null_hypothesis = null_hypothesis[null_hypothesis.notnull()]

        def compute_empirical_p_value(x):
            p_value = (null_hypothesis >= x).sum() / null_hypothesis.count()
            return p_value

        df_p['p_value'] = df_p['mean_correlation'].apply(compute_empirical_p_value)
    
    elif ns is not None:
        def compute_cross_val_t_test(r):
            p = 0
            for n in ns:
                dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
                p += dist.rvs(size=10000000) / len(ns)
            p_value = np.mean(p >= r)
            return p_value

        df_p['p_value'] = df_p['mean_correlation'].apply(compute_cross_val_t_test)

    _, pvalues_corrected, _ , _ = multipletests(df_p['p_value'].values, method='hs')
    df_p['p_value_corrected_hs'] = pvalues_corrected
    _, pvalues_corrected, _ , _ = multipletests(df_p['p_value'].values, method='fdr_bh')
    df_p['p_value_corrected_bh'] = pvalues_corrected

    return df_p[['gene', 'mean_correlation', 'p_value', 'p_value_corrected_hs', 'p_value_corrected_bh']]
