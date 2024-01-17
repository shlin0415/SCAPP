# -*- coding: utf-8 -*-
# Some references:
# D2l. https://zh-v2.d2l.ai/
# Pandas. https://pandas.pydata.org/docs/index.html
# PyTorch. https://pytorch.org/
# Scikit-learn. https://scikit-learn.org/stable/

import sys
import itertools
import statistics
import shutil
import random
import math
from functools import reduce
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import seaborn as sns
import gc
import os
import psutil
import inspect
import re
import pickle
import torch
import torchvision.models
import datetime
from torch import nn
from d2l import torch as d2l
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import time
from IPython.core.interactiveshell import InteractiveShell 
from torchvision import transforms  
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

InteractiveShell.ast_node_interactivity = "all"

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print(current_time)

gc.collect()

available_memory = psutil.virtual_memory().available

available_memory_gb = available_memory / (1024 * 1024 * 1024)

print(f"available_memory: {available_memory_gb} GB")

used_memory = psutil.virtual_memory().used

used_memory_gb = used_memory / (1024 * 1024 * 1024)

print(f"used_memory: {used_memory_gb} GB")


class PCmaster_anno_0_Error(Exception):
    pass

class trainDataset(Dataset):  
    def __init__(self,train_data,train_label):
        self.Data = torch.from_numpy(np.float32(np.array(train_data)))
        self.Data = torch.reshape(self.Data, (train_data.shape[0], 224, 224))
        self.Data = torch.unsqueeze(self.Data, 1)
        self.Data = torch.cat((self.Data, self.Data, self.Data), dim=1)
        self.Label = torch.from_numpy(np.int64(np.array(train_label)).reshape(1,-1)[0])
    def __getitem__(self, index):
        x = self.Data[index]
        y = self.Label[index]
        return x, y  
    def __len__(self):
        return len(self.Data)
            
class validDataset(Dataset):  
    def __init__(self,valid_data,valid_label):
        self.Data = torch.from_numpy(np.float32(np.array(valid_data)))
        self.Data = torch.reshape(self.Data, (valid_data.shape[0], 224, 224))
        self.Data = torch.unsqueeze(self.Data, 1)
        self.Data = torch.cat((self.Data, self.Data, self.Data), dim=1)
        self.Label = torch.from_numpy(np.int64(np.array(valid_label)).reshape(1,-1)[0])
    def __getitem__(self, index):
        x = self.Data[index]
        y = self.Label[index]
        return x, y  
    def __len__(self):
        return len(self.Data)

class CustomDataset(Dataset):  
    def __init__(self, root_dir, transform=None):  
        self.root_dir = root_dir  
        self.classes = sorted(os.listdir(root_dir))  
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  
        self.samples = self._make_dataset()  
        self.transform = transforms.Compose([  
            transforms.Resize((224, 224)),  # 224x224  
            transforms.ToTensor()  # to tensor
        ])   
  
    def _make_dataset(self):  
        samples = []  
        for class_name in self.classes:  
            class_dir = os.path.join(self.root_dir, class_name)  
            if not os.path.isdir(class_dir):  
                continue  
            for file_name in os.listdir(class_dir):  
                file_path = os.path.join(class_dir, file_name)  
                samples.append((file_path, self.class_to_idx[class_name]))  
        return samples  
  
    def __len__(self):  
        return len(self.samples)  
  
    def __getitem__(self, idx):  
        file_path, label = self.samples[idx]  
        image = Image.open(file_path)  
        if self.transform:  
            image = self.transform(image)  
        return image, label  

class PCmaster_anno_0():
    result_file = None
    shape = None
    
    obj = None 
    
    species = None
    
    organ = None
    dd = None  # dense dataframe, pandas object
    genes = None
    cells = None
    cellnamelabel = None
    
    objs = []
    
    batch_species = []
    
    batch_organs = []
    markergenepd_head5 = None
    markergenepd_head10 = None
    markergenepd_head15 = None
    markergenepd_head20 = None
    markergenepd_head = None
    
    marker_grade_1 = None
    anno_re_using_rank_1_markers = None
    anno_re_using_rank_1and2_markers = None
    dir_gene_species = None
    dir_gene_organ_celltype = None
    dir_gene_score = None
    anno_re_using_marker_scores = None
    
    # deep learning
    ref_obj = None
    test_obj = None
    num_types = None
    mapping_1 = None
    mapping_2 = None
    model_weight = None
    accuracys = None
    precisions = None
    recalls = None
    predicted_results = None
    dl_cellnames = None
    original_celltypes = None
    train_accuracy = None
    train_macro_f1_score = None

    def memory_collect_0(self):
        gc.collect()

        available_memory = psutil.virtual_memory().available

        available_memory_gb = available_memory / (1024 * 1024 * 1024)
        
        print(f"max_memory: {available_memory_gb} GB")
        
        used_memory = psutil.virtual_memory().used
        
        used_memory_gb = used_memory / (1024 * 1024 * 1024)
        
        print(f"used_memory: {used_memory_gb} GB")

    def set_result_file_0(self, result_file=None):
        if result_file is not None:
            self.result_file = result_file
        else:
            raise PCmaster_anno_0_Error("The result_file is None.")

    def set_default_0(self, verbosity=None, dpi=None, facecolor=None):
        # default settings of scanpy
        if verbosity is None:
            verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
        if dpi is None:
            dpi = 80
        if facecolor is None:
            facecolor = 'white'
        sc.settings.verbosity = verbosity  # verbosity: errors (0), warnings (1), info (2), hints (3)
        sc.logging.print_header()
        sc.settings.set_figure_params(dpi=dpi, facecolor=facecolor)

    def data_input_0(self, filepath=None, species=None, organ=None, organs=None):
        if filepath is not None:
            if filepath[-6:] == "tsv.gz":    
                self.obj = sc.read_umi_tools(filepath)
            elif filepath[-6:] == "mtx.gz":
                self.obj = sc.read_10x_mtx(filepath[:-13], var_names='gene_symbols', cache=True)
            elif 'filtered_feature_bc_matrix' in filepath:    
                self.obj = sc.read_10x_mtx(filepath, var_names='gene_symbols', cache=True)
            else:
                self.obj = sc.read_10x_mtx(filepath, var_names='gene_symbols', cache=True)
                
        if organs is not None:
            self.batch_organs = organs
        if species is not None or organ is not None:
            self.species = species
            self.organ = organ
        if self.obj is not None:
            self.genes = self.obj.var_names
            self.cells = self.obj.obs_names
        else:
            raise PCmaster_anno_0_Error("The expression matrix is not successfully loaded.")

    def batch_data_input_0(self, filepaths=None, species=None, organs=None):
        if filepaths is None:
            raise PCmaster_anno_0_Error("The filepaths are None.")
        for i in filepaths:
            if i[-6:] == "tsv.gz":
                self.objs.append(sc.read_umi_tools(i))
            if i[-6:] == "mtx.gz":
                self.objs.append(sc.read_10x_mtx(i[:-13], var_names='gene_symbols', cache=True))
            tmpstr1 = 'filtered_feature_bc_matrix'
            if tmpstr1 in i:
                self.objs.append(sc.read_10x_mtx(i, var_names='gene_symbols', cache=True))
        if species is not None:
            for i in species:
                self.batch_species.append(i)
        if organs is not None:
            for i in organs:
                self.batch_organs.append(i)
        if self.objs is None:
            raise PCmaster_anno_0_Error("The expression matrices are not successfully loaded.")

    def data_input_0_from_anndata(self, anndata=None, species=None, organ=None, organs=None):
        if anndata is not None:
            self.obj = anndata
        else:
            raise PCmaster_anno_0_Error("The anndata is None.")
        if organs is not None:
            self.batch_organs = organs
        if species is not None or organ is not None:
            self.species = species
            self.organ = organ
        if self.obj is not None:
            self.genes = self.obj.var_names
            self.cells = self.obj.obs_names
        else:
            raise PCmaster_anno_0_Error("The expression matrix is not successfully loaded.")
            
    def batch_data_input_0_from_anndata(self, anndatas=None, species=None, organs=None):
        if anndatas is not None:
            for i in anndatas:
                self.objs.append(i)
        else:
            raise PCmaster_anno_0_Error("The anndatas are None.")
        if species is not None:
            for i in species:
                self.batch_species.append(i)
        if organs is not None:
            for i in organs:
                self.batch_organs.append(i)
        if self.objs is None:
            raise PCmaster_anno_0_Error("The expression matrices are not successfully loaded.")

    def concat_0(self):
        batch_categories = []
        if self.objs is None:
            raise PCmaster_anno_0_Error("The expression matrices are not successfully loaded.")
        for i in range(len(self.objs)):
            batch_categories.append("data_number_" + str(i))
        if self.obj is not None:
            print("Warning: self.obj is not None and will be changed.")
        self.obj = self.objs[0].concatenate(self.objs[1:], batch_categories=batch_categories)

    def batch_data_input_0_and_concat_0(self, filepaths=None, species=None, organs=None):
        pass

    def get_dense_0(self):
        if self.obj is None:
            raise PCmaster_anno_0_Error("The expression matrix is not successfully loaded.")
        self.dd = pd.DataFrame(self.obj.X.todense(), index=self.obj.obs_names, columns=self.obj.var_names)

    # Pre-processing, filtering correlation, using basic parameters of scanpy.
    # Added the function of filtering by absolute median difference MAD.
    # MAD = the median of absolute values of all values after the median of Xi-X.
    # The median of # 1, 2, and 3 is 2. If you subtract 2 from each item, you will get -1,0,1. After taking the absolute value, sort it. If it is 0,1,1, then MAD = 1.
    # This function is to keep the data whose values are within the range of+-mad _ coefficient * mad.
    # In general, it doesn't matter if you don't need to filter through the absolute median difference MAD.
    def filter_0(self, n_top=None, num_min_genes=None, num_max_genes=None, num_min_cells=None,
                 max_percent_of_specific_genes=None,
                 genelist=None, mad_coeffcient=None):
        # tmp = self.obj.copy()
        if n_top is None:
            n_top = 20
        if num_min_genes is None:
            num_min_genes = 200
        if num_min_cells is None:
            num_min_cells = 3
        # The following parameter is used to filter specific gene sets, such as mitochondrial genes and so on.
        if max_percent_of_specific_genes is None:
            max_percent_of_specific_genes = 0.05
        self.genes = self.obj.var.index
        
        mad_min_genes = 0
        mad_max_genes = 0
        mad_min_counts = 0
        mad_max_counts = 0
        mad_max_pct = 0
        
        print("\n")
        print("Before filter")
        print("\n")
        sc.pl.highest_expr_genes(self.obj, n_top, )
        sc.pp.filter_cells(self.obj, min_genes=-1)
        sc.pp.filter_genes(self.obj, min_cells=-1)
        if genelist is None:
            self.obj.var['sp'] = self.obj.var_names.str.startswith('meaninglessmeaningless-')
        else:
            genelist = genelist.copy()
            tmp = []
            for i in self.genes:
                if i in genelist:
                    tmp.append(True)
                else:
                    tmp.append(False)
            # Here, add a column of my own identification of genes in the var of scanpy object. For example, if it is a mitochondrial gene, it will be true, and if it is not, it will be false.
            self.obj.var['sp'] = np.asarray(tmp)
        sc.pp.calculate_qc_metrics(self.obj, qc_vars=['sp'], percent_top=None, log1p=False, inplace=True)
        sc.pl.violin(self.obj, ['n_genes_by_counts'])
        sc.pl.violin(self.obj, ['total_counts'])
        sc.pl.violin(self.obj, ['pct_counts_sp'])
        sc.pl.scatter(self.obj, x='total_counts', y='pct_counts_sp')
        sc.pl.scatter(self.obj, x='total_counts', y='n_genes_by_counts')
        if mad_coeffcient is not None:
            mad_min_genes = self.obj.obs.n_genes_by_counts.median() \
                            - mad_coeffcient * (
                                    self.obj.obs.n_genes_by_counts - self.obj.obs.n_genes_by_counts.median()).abs().median()
            mad_max_genes = self.obj.obs.n_genes_by_counts.median() \
                            + mad_coeffcient * (
                                    self.obj.obs.n_genes_by_counts - self.obj.obs.n_genes_by_counts.median()).abs().median()
            mad_min_counts = self.obj.obs.total_counts.median() \
                             - mad_coeffcient * (
                                     self.obj.obs.total_counts - self.obj.obs.total_counts.median()).abs().median()
            mad_max_counts = self.obj.obs.total_counts.median() \
                             + mad_coeffcient * (
                                     self.obj.obs.total_counts - self.obj.obs.total_counts.median()).abs().median()
            # mad_max_pct = self.obj.obs.pct_counts_sp.median() \
            #     + mad_coeffcient*(self.obj.obs.pct_counts_sp - self.obj.obs.pct_counts_sp.median()).abs().median()
        total_counts_describe_before = self.obj.obs.total_counts.describe()
        n_genes_by_counts_describe_before = self.obj.obs.n_genes_by_counts.describe()
        pct_counts_sp_describe_before = self.obj.obs.pct_counts_sp.describe()
        # self.obj = tmp
        # del tmp
        print("\n")
        print("After filter")
        print("\n")
        
        # filter
        if num_min_genes is not None:
            sc.pp.filter_cells(self.obj, min_genes=num_min_genes)
        if num_min_cells is not None:
            sc.pp.filter_genes(self.obj, min_cells=num_min_cells)
        if num_max_genes is not None:
            self.obj = self.obj[self.obj.obs.n_genes_by_counts < num_max_genes, :]
        if max_percent_of_specific_genes is not None:
            self.obj = self.obj[self.obj.obs.pct_counts_sp < max_percent_of_specific_genes, :]
        if mad_coeffcient is not None:
            self.obj = self.obj[self.obj.obs.n_genes_by_counts < mad_max_genes, :]
            self.obj = self.obj[self.obj.obs.n_genes_by_counts > mad_min_genes, :]
            self.obj = self.obj[self.obj.obs.total_counts < mad_max_counts, :]
            self.obj = self.obj[self.obj.obs.total_counts > mad_min_counts, :]
            # if genelist !=None:
            #     self.obj = self.obj[self.obj.obs.pct_counts_sp < mad_max_pct, :]
        
        # show
        sc.pl.violin(self.obj, ['n_genes_by_counts'])
        sc.pl.violin(self.obj, ['total_counts'])
        sc.pl.violin(self.obj, ['pct_counts_sp'])
        sc.pl.scatter(self.obj, x='total_counts', y='pct_counts_sp')
        sc.pl.scatter(self.obj, x='total_counts', y='n_genes_by_counts')
        total_counts_describe_after = self.obj.obs.total_counts.describe()
        n_genes_by_counts_describe_after = self.obj.obs.n_genes_by_counts.describe()
        pct_counts_sp_describe_after = self.obj.obs.pct_counts_sp.describe()
        print("\n")
        print("Before filter")
        print("\n")
        print(total_counts_describe_before)
        print(n_genes_by_counts_describe_before)
        print(pct_counts_sp_describe_before)
        print("\n")
        print("After filter")
        print("\n")
        print(total_counts_describe_after)
        print(n_genes_by_counts_describe_after)
        print(pct_counts_sp_describe_after)

    # Normalization, screening hypervariable genes, etc., are roughly completed with scanpy's own functions.
    def norm_log_hvg_filter_regress_scale_0(self, norm_target_sum=None, hvg_min_mean=None, hvg_max_mean=None,
                               hvg_min_dispersions=None, n_top_hvg_genes=None,
                               regresslist=None, scale_max_value=None):
        if norm_target_sum is None:
            norm_target_sum = 1e4
        if hvg_min_mean is None:
            hvg_min_mean = 0.0125
        if hvg_max_mean is None:
            hvg_max_mean = 3
        if hvg_min_dispersions is None:
            hvg_min_dispersions = 0.5
        if regresslist is None:
            regresslist = ['total_counts']
        if scale_max_value is None:
            scale_max_value = 10

        if norm_target_sum is not None:
            sc.pp.normalize_total(self.obj, target_sum=norm_target_sum)
            sc.pp.log1p(self.obj)
        if hvg_min_mean is not None and hvg_max_mean is not None and hvg_min_dispersions is not None:
            if n_top_hvg_genes is not None:
                sc.pp.highly_variable_genes(self.obj, min_mean=hvg_min_mean, max_mean=hvg_max_mean,
                                   min_disp=hvg_min_dispersions,n_top_genes=n_top_hvg_genes,)
            else:
                sc.pp.highly_variable_genes(self.obj, min_mean=hvg_min_mean, max_mean=hvg_max_mean,
                                   min_disp=hvg_min_dispersions,)
            sc.pl.highly_variable_genes(self.obj)
            self.obj.raw = self.obj
            self.obj = self.obj[:, self.obj.var.highly_variable]
        if regresslist is not None:
            regresslist = regresslist.copy()
            sc.pp.regress_out(self.obj, regresslist)
        if scale_max_value is not None:
            sc.pp.scale(self.obj, max_value=scale_max_value)

    # Carry out PCA to screen the differentially expressed gene DEGs.
    def pca_cluster_markerSelect_0(self, pca_svd_algorithm=None, # 'randomized','auto','full' # is ok too
                            pca_genelist=None,
                            default_cluster_method=None,leiden_resolution=None,louvain_resolution=None,
                            cluster_n_neighbors=None, cluster_n_pcs=None, use_rep=None, perform_umap=None,
                            umap_genelist=None, perform_paga=None,
                            markerSelect_method=None,  # 'wilcoxon','logreg' # is ok too
                            markerSelect_n_genes=None, show_n=None):
        if pca_svd_algorithm is None:
            pca_svd_algorithm = 'arpack'
        if default_cluster_method is None:
            default_cluster_method = 'leiden'
        if leiden_resolution is None:
            leiden_resolution = 1.0
        if louvain_resolution is None:
            louvain_resolution = 1.0
        if cluster_n_neighbors is None:
            cluster_n_neighbors = 10
        if cluster_n_pcs is None:
            cluster_n_pcs = 40
        if perform_umap is None:
            perform_umap = True
        if perform_paga is None:
            perform_paga = False
        if markerSelect_method is None:
            markerSelect_method = 't-test'  # 'wilcoxon','logreg' # is ok too
        if markerSelect_n_genes is None:
            markerSelect_n_genes = 25
        if show_n is None:
            show_n = 25

        if pca_svd_algorithm is not None:
            sc.tl.pca(self.obj, svd_solver=pca_svd_algorithm)
            if pca_genelist is not None:
                pca_genelist = pca_genelist.copy()
                sc.pl.pca(self.obj, color=pca_genelist)
            sc.pl.pca_variance_ratio(self.obj, log=True)
        if cluster_n_neighbors is not None and cluster_n_pcs is not None and use_rep is None:
            sc.pp.neighbors(self.obj, n_neighbors=cluster_n_neighbors, n_pcs=cluster_n_pcs)
        
        # harmony for batch effect correction
        if cluster_n_neighbors is not None and cluster_n_pcs is not None and use_rep == "X_pca_harmony":
            sce.pp.harmony_integrate(self.obj, 'batch')
            sc.pp.neighbors(self.obj, n_neighbors=cluster_n_neighbors, n_pcs=cluster_n_pcs, use_rep="X_pca_harmony")
        if perform_umap:
            sc.tl.umap(self.obj)
            sc.tl.leiden(self.obj,resolution = leiden_resolution)
            if default_cluster_method == 'louvain':
                sc.tl.louvain(self.obj,resolution = louvain_resolution)
            if default_cluster_method == 'louvain':
                tmplist = ['louvain']
            else:
                tmplist = ['leiden']
            if umap_genelist is not None:
                umap_genelist = umap_genelist.copy()
                for i in umap_genelist:
                    tmplist.append(i)
            sc.pl.umap(self.obj, color=tmplist)
        
        # paga
        if perform_paga:
            sc.tl.paga(self.obj)
            sc.pl.paga(self.obj)  # remove `plot=False` if you want to see the coarse-grained graph
            sc.tl.umap(self.obj, init_pos='paga')
            if default_cluster_method == 'louvain':
                tmplist = ['louvain']
            else:
                tmplist = ['leiden']
            if use_rep == "X_pca_harmony" and default_cluster_method == 'louvain':
                tmplist = ['louvain', 'batch']
            if use_rep == "X_pca_harmony" and default_cluster_method == 'leiden':
                tmplist = ['leiden', 'batch']
            if umap_genelist is not None:
                umap_genelist = umap_genelist.copy()
                for i in umap_genelist:
                    tmplist.append(i)
            for i in tmplist:
                sc.pl.umap(self.obj, color=i)
        if markerSelect_method is not None:
            if default_cluster_method == 'leiden':
                sc.tl.rank_genes_groups(self.obj, 'leiden', method=markerSelect_method)
            else:
                sc.tl.rank_genes_groups(self.obj, 'louvain', method=markerSelect_method)
            sc.pl.rank_genes_groups(self.obj, n_genes=markerSelect_n_genes, sharey=False)
        print("\n")
        print("The below is the top marker gene list for each cluster.\n\
            You can change the annotation result according to this.")
        print("\n")
        print("The list of top marker genes \n col : clusters")
        print("\n")
        print(pd.DataFrame(self.obj.uns['rank_genes_groups']['names']).head(show_n))
        # print( pd.DataFrame(
        #     {group + '_' + key: self.obj.uns['rank_genes_groups'][key][group]
        #     for group in self.obj.uns['rank_genes_groups']['names'].dtype.names \
        # for key in ['names', 'pvals']}).head(show_n) )
        # result = self.obj.uns['rank_genes_groups']
        # groups = result['names'].dtype.names
        # print(result)
        # print("\n")
        # print("row : marker_genes ; col : n , names , p , pvals")
        # print("\n")
        # pd.DataFrame(
        #     {group + '_' + key[:1]: result[key][group]
        #     for group in groups for key in ['names', 'pvals']}).head(show_n)
        result = self.obj.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        
        # save DEGs and p-values
        self.markergenepd_head5 = pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
             for group in groups for key in ['names', 'pvals']}).head(5)
        self.markergenepd_head10 = pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
             for group in groups for key in ['names', 'pvals']}).head(10)
        self.markergenepd_head15 = pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
             for group in groups for key in ['names', 'pvals']}).head(15)
        self.markergenepd_head20 = pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
             for group in groups for key in ['names', 'pvals']}).head(20)
        print("\n")
        print("Top marker genes and their p-values of each cluster: ")
        print("\n")
        if show_n is not None:
            self.markergenepd_head = pd.DataFrame(
                {group + '_' + key[:1]: result[key][group]
                 for group in groups for key in ['names', 'pvals']}).head(show_n)
            print(self.markergenepd_head)
        else:
            print(self.markergenepd_head20)
        # print(self.markergenepd_head5)
        # print(self.markergenepd_head10)
        # print(self.markergenepd_head15)
        # print(self.markergenepd_head20)

    def markergenepd_make_0(self):
        if self.obj is None:
            raise PCmaster_anno_0_Error("The expression matrix is not successfully loaded.")
        result = self.obj.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        self.markergenepd_head5 = pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
             for group in groups for key in ['names', 'pvals']}).head(5)
        self.markergenepd_head10 = pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
             for group in groups for key in ['names', 'pvals']}).head(10)
        self.markergenepd_head15 = pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
             for group in groups for key in ['names', 'pvals']}).head(15)
        self.markergenepd_head20 = pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
             for group in groups for key in ['names', 'pvals']}).head(20)
        # print(self.markergenepd_head5)
        # print(self.markergenepd_head10)
        # print(self.markergenepd_head15)
        # print(self.markergenepd_head20)
        
    def print_organ_types(self):
        ref_txt = open('./plant_marker_gene_list.txt', 'r', encoding='utf-8')
        tmpset_for_print = set()
        for line in ref_txt:
                # if len(line.split('\t')) < 5:
                #     print(line)
                if len(line.split('\t')) >= 5:
                    species = line.split('\t')[0]
                    organ = line.split('\t')[1]
                    celltype = line.split('\t')[2]
                    gene = line.split('\t')[3]
                    if organ not in tmpset_for_print:
                        tmpset_for_print.add(organ)
        tmpset_for_print = sorted(tmpset_for_print)
        for i in tmpset_for_print:
            print(i)
        
    # for generate dict{gene-others} in ./plant_marker_gene_list.txt
    def make_marker_gene_list_0(self,marker_select_flag = None,marker_gene_list = None):
        # organ = None
        # batch_organs = []
        if (marker_select_flag is None) or (marker_select_flag != '#1'):
            marker_select_flag = '#1 and #2'
        ref_txt = None
        if marker_gene_list is None:
            ref_txt = open('./plant_marker_gene_list.txt', 'r', encoding='utf-8')
        else:
            ref_txt = open(marker_gene_list, 'r', encoding='utf-8')
        
        # Here, a filter is carried out first. If it is a single organ type, only the marker gene of that organ type will be left, and others will be deleted.
        ref1 = []
        for line in ref_txt:
            if len(line.split('\t')) >= 5:
                if line.split('\t')[4] == 'Marker #1\n' or line.split('\t')[4] == 'Cell marker #1\n' \
                or line.split('\t')[4] == 'Marker #2\n' or line.split('\t')[4] == 'Cell marker #2\n':
                    the_organ = line.split('\t')[1]
                    if self.organ is not None:
                        if self.organ.upper() in the_organ.upper() or self.organ.upper() == 'ALL':
                            ref1.append(line)
                    elif len(self.batch_organs) != 0:
                        for i in self.batch_organs:
                            if i.upper() in the_organ.upper() or i.upper() == 'ALL':
                                ref1.append(line)
                    else:
                        ref1.append(line)
        
        # The following are the steps to make two dictionary for comparing DEG with known marker.
        dir_gene_species = {}
        dir_gene_organ_celltype = {}
        count = 0
        if marker_select_flag == '#1 and #2':
            forbidden_set = set()
            set_rank1 = set()
            for line in ref1:
                # print(line.split('\t'))
                if len(line.split('\t')) >= 5:
                    if line.split('\t')[4] == 'Marker #1\n' or line.split('\t')[4] == 'Cell marker #1\n':
                        species = line.split('\t')[0]
                        organ = line.split('\t')[1]
                        celltype = line.split('\t')[2]
                        gene = line.split('\t')[3]
                        # print(species)
                        if gene not in forbidden_set:
                            if gene in dir_gene_organ_celltype:
                                # The recognition of'/'indicates that the gene points to at least three different cell types.
                                if '/' in dir_gene_organ_celltype[gene]:
                                    # Delete all the information previously left by this gene.
                                    del dir_gene_organ_celltype[gene]
                                    del dir_gene_species[gene]
                                    # Blacken the gene
                                    forbidden_set.add(gene)
                                else:
                                    tmpinfo_1 = 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                                    # If the same item is added, it will be skipped and no changes will be made.
                                    if dir_gene_organ_celltype[gene] == tmpinfo_1:
                                        pass
                                    # Only, the same gene is allowed to point to at most two cell types.
                                    else:
                                        dir_gene_organ_celltype[gene] = dir_gene_organ_celltype[gene] + \
                                        ' / ' + 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                                        dir_gene_species[gene] = species
                            else:
                                # first add
                                dir_gene_organ_celltype[gene] = 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                                dir_gene_species[gene] = species
                                # record marker#1 gene
                                set_rank1.add(gene)
                        # print(organ)
                        # print(celltype)
                        # print(gene)
                count = count + 1
            for line in ref1:
                # print(line.split('\t'))
                if len(line.split('\t')) < 5:
                    print(line)
                if len(line.split('\t')) >= 5 and ( line.split('\t')[4] == 'Marker #2\n' or \
                                                   line.split('\t')[4] == 'Cell marker #2\n' ):
                    species = line.split('\t')[0]
                    organ = line.split('\t')[1]
                    celltype = line.split('\t')[2]
                    gene = line.split('\t')[3]
                    # organ_yesno = 1
                    # if self.organ is not None:
                    #     if self.organ.upper() not in organ.upper() and self.organ.upper() != 'ALL':
                    #         organ_yesno = 0
                    # elif self.batch_organs is not None:
                    #     tmp_organs = []
                    #     for tmp_organ in self.batch_organs:
                    #         tmp_organs.append(tmp_organ.upper())
                    #     if 'ALL' in tmp_organs:
                    #         organ_yesno = 1
                    #     elif organ.upper() in tmp_organs:
                    #         organ_yesno = 1
                    #     else:
                    #         organ_yesno = 0
                    # if organ_yesno == 0:
                    #     continue
                    # print(species)
                    
                    # marker#1 and marker#2 will not interact each other
                    if (gene not in forbidden_set) and (gene not in set_rank1):
                        if gene in dir_gene_organ_celltype:
                            if '/' in dir_gene_organ_celltype[gene]:
                                del dir_gene_organ_celltype[gene]
                                del dir_gene_species[gene]
                                forbidden_set.add(gene)
                            else:
                                tmpinfo_1 = 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                                if dir_gene_organ_celltype[gene] == tmpinfo_1:
                                    pass
                                else:
                                    dir_gene_organ_celltype[gene] = dir_gene_organ_celltype[gene] + \
                                    ' / ' + 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                                    dir_gene_species[gene] = species
                        else:
                            dir_gene_organ_celltype[gene] = 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                            dir_gene_species[gene] = species
                    # print(organ)
                    # print(celltype)
                    # print(gene)
                count = count + 1
            self.marker_grade_1 = set_rank1
        else:
            # the below only process for marker#1
            forbidden_set = set()
            for line in ref1:
                # print(line.split('\t'))
                if len(line.split('\t')) < 5:
                    print(line)
                if len(line.split('\t')) >= 5:
                    if line.split('\t')[4] == 'Marker #1\n' or line.split('\t')[4] == 'Cell marker #1\n':
                        species = line.split('\t')[0]
                        organ = line.split('\t')[1]
                        celltype = line.split('\t')[2]
                        gene = line.split('\t')[3]
                        # print(species)
                        if gene not in forbidden_set:
                            if gene in dir_gene_organ_celltype:
                                if '/' in dir_gene_organ_celltype[gene]:
                                    del dir_gene_organ_celltype[gene]
                                    del dir_gene_species[gene]
                                    forbidden_set.add(gene)
                                else:
                                    tmpinfo_1 = 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                                    if dir_gene_organ_celltype[gene] == tmpinfo_1:
                                        pass
                                    else:
                                        dir_gene_organ_celltype[gene] = dir_gene_organ_celltype[gene] + \
                                        ' / ' + 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                                        dir_gene_species[gene] = species
                            else:
                                dir_gene_organ_celltype[gene] = 'organ: ' + organ + '  ' + 'celltype: ' + celltype
                                dir_gene_species[gene] = species
                        # print(organ)
                        # print(celltype)
                        # print(gene)
                count = count + 1
        return dir_gene_species, dir_gene_organ_celltype

    def annotation_with_marker_gene_list_0(self,marker_gene_list = None,marker_select_flag = None,
                                           marker_grade_1_coefficient = None):
        # Update the unchanged default situation, which is the weight of marker#1. Only when marker#1 and marker#2 are mixed,
        # and one gene is marker#1 gene and only corresponds to one cell type.
        if marker_grade_1_coefficient is None:
            marker_grade_1_coefficient = 1
        
        # get dictionary
        self.dir_gene_species, self.dir_gene_organ_celltype = self.make_marker_gene_list_0(marker_select_flag = '#1 and #2',
                                                         marker_gene_list = marker_gene_list)
        dir_gene_species, dir_gene_organ_celltype = self.make_marker_gene_list_0(marker_select_flag = '#1 and #2',
                                                         marker_gene_list = marker_gene_list)
        count_1 = 0
        count_2 = 0
        anno_re = []

        # use self.markergenepd_head20 as a temporary carrier, it will be changed back later.
        tmpmarkergenepd = self.markergenepd_head20.copy()
        if self.markergenepd_head is not None:
            self.markergenepd_head20 = self.markergenepd_head
        for i in range(0, int(self.markergenepd_head20.shape[1] / 2)):
            tmp = {}
            # According to the original organization form of DEG table, DEG is extracted independently for each cluster, 
            # and then compared with the known marker table one by one.
            for j in self.markergenepd_head20[str(i) + '_n'].values:
                j = str(j).strip()
                # print(j)
                # break
                if j in dir_gene_species:
                    organ_celltype = dir_gene_organ_celltype[j]
                    organ_celltype_1 = None
                    organ_celltype_2 = None
                    if '/' in organ_celltype:
                        organ_celltype_1 = organ_celltype.split('/')[0].strip()
                        organ_celltype_2 = organ_celltype.split('/')[1].strip()
                        
                    if organ_celltype in tmp:
                        if j in self.marker_grade_1 and ('/' in organ_celltype) is False:
                            tmp[organ_celltype] = tmp[organ_celltype] + 1*marker_grade_1_coefficient
                        else:
                            tmp[organ_celltype] = tmp[organ_celltype] + 1
                        if organ_celltype_1 is not None and organ_celltype_1 in tmp:
                            tmp[organ_celltype_1] = tmp[organ_celltype_1] + 1
                        if organ_celltype_2 is not None and organ_celltype_2 in tmp:
                            tmp[organ_celltype_2] = tmp[organ_celltype_2] + 1
                    else:
                        if j in self.marker_grade_1 and ('/' in organ_celltype) is False:
                            tmp[organ_celltype] = 1*marker_grade_1_coefficient
                        else:
                            tmp[organ_celltype] = 1
                        if organ_celltype_1 is not None and (organ_celltype_1 in tmp) is False:
                            tmp[organ_celltype_1] = 1
                        if organ_celltype_2 is not None and (organ_celltype_2 in tmp) is False:
                            tmp[organ_celltype_2] = 1
            if tmp != {}:
                anno_re.append(max(tmp, key=tmp.get))
            else:
                anno_re.append('Unknown')
        self.anno_re_using_rank_1and2_markers = anno_re

        # for i in anno_re:
        #     print(i)
        # print(len(anno_re))
        anno_re_1and2 = anno_re

        dir_gene_species, dir_gene_organ_celltype = self.make_marker_gene_list_0(marker_select_flag = '#1',
                                                                                 marker_gene_list = marker_gene_list)

        count_1 = 0
        count_2 = 0
        anno_re = []
        for i in range(0, int(self.markergenepd_head20.shape[1] / 2)):
            tmp = {}
            for j in self.markergenepd_head20[str(i) + '_n'].values:
                j = str(j).strip()
                # print(j)
                # break
                if j in dir_gene_species:
                    organ_celltype = dir_gene_organ_celltype[j]
                    organ_celltype_1 = None
                    organ_celltype_2 = None
                    if '/' in organ_celltype:
                        organ_celltype_1 = organ_celltype.split('/')[0].strip()
                        organ_celltype_2 = organ_celltype.split('/')[1].strip()
                        
                    if organ_celltype in tmp:
                        tmp[organ_celltype] = tmp[organ_celltype] + 1
                        if organ_celltype_1 is not None and organ_celltype_1 in tmp:
                            tmp[organ_celltype_1] = tmp[organ_celltype_1] + 1
                        if organ_celltype_2 is not None and organ_celltype_2 in tmp:
                            tmp[organ_celltype_2] = tmp[organ_celltype_2] + 1
                    else:
                        tmp[organ_celltype] = 1
                        if organ_celltype_1 is not None and (organ_celltype_1 in tmp) is False:
                            tmp[organ_celltype_1] = 1
                        if organ_celltype_2 is not None and (organ_celltype_2 in tmp) is False:
                            tmp[organ_celltype_2] = 1
            if tmp != {}:
                anno_re.append(max(tmp, key=tmp.get))
            else:
                anno_re.append('Unknown')
        self.anno_re_using_rank_1_markers = anno_re

        # for i in anno_re:
        #     print(i)
        # print(len(anno_re))
        anno_re_1 = anno_re

        # print("The below is the top marker gene list for each cluster.\n\
        #     You can change the annotation result according to this.")
        # print(pd.DataFrame(self.obj.uns['rank_genes_groups']['names']).head(20))

        tmplist = []
        for i in self.obj.obs['leiden']:
            tmplist.append(self.anno_re_using_rank_1and2_markers[int(i)])
        # print(tmplist)
        # print(len(self.anno_re_using_rank_1and2_markers))
        # add annotation results
        self.obj.obs['anno_re_using_rank_1and2_markers'] = tmplist.copy()

        for i, j in enumerate(anno_re_1and2):
            print('cluster ' + str(i) + ': ' + str(j))
        sc.pl.umap(self.obj, color='anno_re_using_rank_1and2_markers')

        tmplist = []
        for i in self.obj.obs['leiden']:
            tmplist.append(self.anno_re_using_rank_1_markers[int(i)])
        # print(tmplist)
        # print(len(self.anno_re_using_rank_1_markers))
        # add annotation results
        self.obj.obs['anno_re_using_rank_1_markers'] = tmplist.copy()
        for i, j in enumerate(anno_re_1):
            print('cluster ' + str(i) + ': ' + str(j))
        sc.pl.umap(self.obj, color='anno_re_using_rank_1_markers')

        self.markergenepd_head20 = tmpmarkergenepd.copy()
        del tmpmarkergenepd
        self.memory_collect_0()
        
    def get_dir_gene_score(self,marker_gene_list = None):
        self.dir_gene_score = {}
        if marker_gene_list is None:
            ref_txt = open('./plant_marker_gene_list.txt', 'r', encoding='utf-8')
        else:
            ref_txt = open(marker_gene_list, 'r', encoding='utf-8')
            
        ref1 = []
        for line in ref_txt:
            if len(line.split('\t')) == 6:
                ref1.append(line.strip())
        for line in ref1:
            gene = line.split('\t')[3].strip()
            score = float(line.split('\t')[5].strip())
            self.dir_gene_score[gene] = score
        
    def annotation_with_marker_gene_scores(self,the_least_score = None):
        if the_least_score is None:
            the_least_score = 0.0
        count_1 = 0
        count_2 = 0
        anno_re = []

        tmpmarkergenepd = self.markergenepd_head20.copy()
        if self.markergenepd_head is not None:
            self.markergenepd_head20 = self.markergenepd_head
        for i in range(0, int(self.markergenepd_head20.shape[1] / 2)):
            tmp = {}
            for j in self.markergenepd_head20[str(i) + '_n'].values:
                j = str(j).strip()
                # print(j)
                # break
                if j in self.dir_gene_species and j in self.dir_gene_score and float(self.dir_gene_score[j])>=the_least_score:
                    organ_celltype = self.dir_gene_organ_celltype[j]
                    organ_celltype_1 = None
                    organ_celltype_2 = None
                    if '/' in organ_celltype:
                        organ_celltype_1 = organ_celltype.split('/')[0].strip()
                        organ_celltype_2 = organ_celltype.split('/')[1].strip()
                        
                    if organ_celltype in tmp:
                        tmp[organ_celltype] = tmp[organ_celltype] + 1.0*float(self.dir_gene_score[j])
                        if organ_celltype_1 is not None and organ_celltype_1 in tmp:
                            tmp[organ_celltype_1] = tmp[organ_celltype_1] + 1.0*float(self.dir_gene_score[j])
                        if organ_celltype_2 is not None and organ_celltype_2 in tmp:
                            tmp[organ_celltype_2] = tmp[organ_celltype_2] + 1.0*float(self.dir_gene_score[j])
                    else:
                        tmp[organ_celltype] = 1.0*float(self.dir_gene_score[j])
                        if organ_celltype_1 is not None and (organ_celltype_1 in tmp) is False:
                            tmp[organ_celltype_1] = 1.0*float(self.dir_gene_score[j])
                        if organ_celltype_2 is not None and (organ_celltype_2 in tmp) is False:
                            tmp[organ_celltype_2] = 1.0*float(self.dir_gene_score[j])
            if tmp != {}:
                anno_re.append(max(tmp, key=tmp.get))
            else:
                anno_re.append('Unknown')
            print(i)
            for tmp_key in tmp:
                print(tmp_key)
                print(tmp[tmp_key])
        self.anno_re_using_marker_scores = anno_re
        
        tmplist = []
        for i in self.obj.obs['leiden']:
            tmplist.append(self.anno_re_using_marker_scores[int(i)])
        # print(tmplist)
        # print(len(self.anno_re_using_rank_1and2_markers))
        self.obj.obs['anno_re_using_marker_scores'] = tmplist.copy()

        for i, j in enumerate(anno_re):
            print('cluster ' + str(i) + ': ' + str(j))
        sc.pl.umap(self.obj, color='anno_re_using_marker_scores')

        self.markergenepd_head20 = tmpmarkergenepd.copy()
        del tmpmarkergenepd
        self.memory_collect_0()

    # integrated function
    def to_clusters_0(self, verbosity=None, dpi=None, facecolor=None, # set_default_0
            anndata=None,filepath=None, species=None, organ=None, organs=None, # data_input_0
            n_top=None, num_min_genes=None, num_max_genes=None, num_min_cells=None, # filter_0
                max_percent_of_specific_genes=None,
                genelist=None, mad_coeffcient=None,
            norm_target_sum=None, hvg_min_mean=None, hvg_max_mean=None,  # norm_log_hvg_filter_regress_scale_0
                hvg_min_dispersions=None, regresslist=None, scale_max_value=None,n_top_hvg_genes=None,
            pca_svd_algorithm=None, pca_genelist=None, # pca_cluster_markerSelect_0
                default_cluster_method=None,leiden_resolution=None,louvain_resolution=None,
                cluster_n_neighbors=None, cluster_n_pcs=None, use_rep=None, perform_umap=None,
                umap_genelist=None, perform_paga=None,
                markerSelect_method=None,  # 'wilcoxon','logreg' # is ok too
                markerSelect_n_genes=None, show_n=None):
        self.set_default_0(verbosity=verbosity, dpi=dpi, facecolor=facecolor)
        if filepath is not None:
            self.data_input_0(filepath=filepath, species=species, organ=organ, organs=organs)
        if anndata is not None:
            self.data_input_0_from_anndata(anndata=anndata, species=species, organ=organ, organs=organs)
        self.filter_0(n_top=n_top, num_min_genes=num_min_genes, 
            num_max_genes=num_max_genes, num_min_cells=num_min_cells,
            max_percent_of_specific_genes=max_percent_of_specific_genes,
            genelist=genelist, mad_coeffcient=mad_coeffcient)
        self.norm_log_hvg_filter_regress_scale_0(norm_target_sum=norm_target_sum, hvg_min_mean=hvg_min_mean, 
            hvg_max_mean=hvg_max_mean, hvg_min_dispersions=hvg_min_dispersions, n_top_hvg_genes=n_top_hvg_genes,
            regresslist=regresslist, scale_max_value=scale_max_value)
        self.pca_cluster_markerSelect_0(pca_svd_algorithm=pca_svd_algorithm, pca_genelist=pca_genelist,
            default_cluster_method=default_cluster_method,
            leiden_resolution=leiden_resolution,louvain_resolution=louvain_resolution,
            cluster_n_neighbors=cluster_n_neighbors, cluster_n_pcs=cluster_n_pcs, use_rep=use_rep, 
            perform_umap=perform_umap,umap_genelist=umap_genelist, perform_paga=perform_paga,
            markerSelect_method=markerSelect_method,  # 'wilcoxon','logreg' # is ok too
            markerSelect_n_genes=markerSelect_n_genes, show_n=show_n)

    # integrated function
    def batch_to_clusters_0(self, verbosity=None, dpi=None, facecolor=None, # set_default_0
            anndatas=None,filepaths=None, species=None, organs=None, # batch_data_input_0
            n_top=None, num_min_genes=None, num_max_genes=None, num_min_cells=None, # filter_0
                max_percent_of_specific_genes=None,
                genelist=None, mad_coeffcient=None,
            norm_target_sum=None, hvg_min_mean=None, hvg_max_mean=None,  # norm_log_hvg_filter_regress_scale_0
                hvg_min_dispersions=None, regresslist=None, scale_max_value=None,n_top_hvg_genes=None,
            pca_svd_algorithm=None, pca_genelist=None, # pca_cluster_markerSelect_0
                default_cluster_method=None,leiden_resolution=None,louvain_resolution=None,
                cluster_n_neighbors=None, cluster_n_pcs=None, use_rep=None, perform_umap=None,
                umap_genelist=None, perform_paga=None,
                markerSelect_method=None,  # 'wilcoxon','logreg' # is ok too
                markerSelect_n_genes=None, show_n=None):
        self.set_default_0(verbosity=verbosity, dpi=dpi, facecolor=facecolor)
        if filepaths is not None:
            self.batch_data_input_0(filepaths=filepaths, species=species, organs=organs)
        if anndatas is not None:
            self.batch_data_input_0_from_anndata(anndatas=anndatas, species=species, organs=organs)
        self.concat_0()
        self.filter_0(n_top=n_top, num_min_genes=num_min_genes, 
            num_max_genes=num_max_genes, num_min_cells=num_min_cells,
            max_percent_of_specific_genes=max_percent_of_specific_genes,
            genelist=genelist, mad_coeffcient=mad_coeffcient)
        self.norm_log_hvg_filter_regress_scale_0(norm_target_sum=norm_target_sum, hvg_min_mean=hvg_min_mean, 
            hvg_max_mean=hvg_max_mean, hvg_min_dispersions=hvg_min_dispersions, n_top_hvg_genes=n_top_hvg_genes,
            regresslist=regresslist, scale_max_value=scale_max_value)
        self.pca_cluster_markerSelect_0(pca_svd_algorithm=pca_svd_algorithm, pca_genelist=pca_genelist,
            default_cluster_method=default_cluster_method,
            leiden_resolution=leiden_resolution,louvain_resolution=louvain_resolution,
            cluster_n_neighbors=cluster_n_neighbors, cluster_n_pcs=cluster_n_pcs, use_rep=use_rep, 
            perform_umap=perform_umap,umap_genelist=umap_genelist, perform_paga=perform_paga,
            markerSelect_method=markerSelect_method,  # 'wilcoxon','logreg' # is ok too
            markerSelect_n_genes=markerSelect_n_genes, show_n=show_n)

    # integrated function
    def to_annotation_0(self, verbosity=None, dpi=None, facecolor=None, # set_default_0
            species=None,anndata=None,filepath=None,organ=None,anndatas=None,filepaths=None,organs=None, 
                        # batch_data_input_0
            n_top=None, num_min_genes=None, num_max_genes=None, num_min_cells=None, # filter_0
                max_percent_of_specific_genes=None,
                genelist=None, mad_coeffcient=None,
            norm_target_sum=None, hvg_min_mean=None, hvg_max_mean=None,  # norm_log_hvg_filter_regress_scale_0
                hvg_min_dispersions=None, regresslist=None, scale_max_value=None,n_top_hvg_genes=None,
            pca_svd_algorithm=None, pca_genelist=None, # pca_cluster_markerSelect_0
                default_cluster_method=None,leiden_resolution=None,louvain_resolution=None,
                cluster_n_neighbors=None, cluster_n_pcs=None, use_rep=None, perform_umap=None,
                umap_genelist=None, perform_paga=None,
                markerSelect_method=None,  # 'wilcoxon','logreg' # is ok too
                markerSelect_n_genes=None, show_n=None,
            marker_select_flag = None,marker_grade_1_coefficient = None,marker_gene_list = None,
            the_least_score = None):
        if filepath is not None:
            self.to_clusters_0(verbosity=verbosity, dpi=dpi, facecolor=facecolor, # set_default_0
                anndata=anndata,filepath=filepath, species=species, organ=organ, organs=organs, # data_input_0
                n_top=n_top, num_min_genes=num_min_genes, num_max_genes=num_max_genes, # filter_0
                    num_min_cells=num_min_cells, 
                    max_percent_of_specific_genes=max_percent_of_specific_genes,
                    genelist=genelist, mad_coeffcient=mad_coeffcient,
                norm_target_sum=norm_target_sum, hvg_min_mean=hvg_min_mean, hvg_max_mean=hvg_max_mean,  
                               # norm_log_hvg_filter_regress_scale_0
                    hvg_min_dispersions=hvg_min_dispersions, regresslist=regresslist, scale_max_value=scale_max_value,
                               n_top_hvg_genes=n_top_hvg_genes,
                pca_svd_algorithm=pca_svd_algorithm, pca_genelist=pca_genelist, # pca_cluster_markerSelect_0
                    default_cluster_method=default_cluster_method,
                    leiden_resolution=leiden_resolution,louvain_resolution=louvain_resolution,
                    cluster_n_neighbors=cluster_n_neighbors, cluster_n_pcs=cluster_n_pcs, 
                    use_rep=use_rep, perform_umap=perform_umap,
                    umap_genelist=umap_genelist, perform_paga=perform_paga,
                    markerSelect_method=markerSelect_method,  # 'wilcoxon','logreg' # is ok too
                    markerSelect_n_genes=markerSelect_n_genes, show_n=show_n)
        elif anndata is not None:
            self.to_clusters_0(verbosity=verbosity, dpi=dpi, facecolor=facecolor, # set_default_0
                anndata=anndata,filepath=filepath, species=species, organ=organ, organs=organs, # batch_data_input_0
                n_top=n_top, num_min_genes=num_min_genes, num_max_genes=num_max_genes, # filter_0
                    num_min_cells=num_min_cells, 
                    max_percent_of_specific_genes=max_percent_of_specific_genes,
                    genelist=genelist, mad_coeffcient=mad_coeffcient,
                norm_target_sum=norm_target_sum, hvg_min_mean=hvg_min_mean, hvg_max_mean=hvg_max_mean,  
                               # norm_log_hvg_filter_regress_scale_0
                    hvg_min_dispersions=hvg_min_dispersions, regresslist=regresslist, scale_max_value=scale_max_value,
                               n_top_hvg_genes=n_top_hvg_genes,
                pca_svd_algorithm=pca_svd_algorithm, pca_genelist=pca_genelist, # pca_cluster_markerSelect_0
                    default_cluster_method=default_cluster_method,
                    leiden_resolution=leiden_resolution,louvain_resolution=louvain_resolution,
                    cluster_n_neighbors=cluster_n_neighbors, cluster_n_pcs=cluster_n_pcs, 
                    use_rep=use_rep, perform_umap=perform_umap,
                    umap_genelist=umap_genelist, perform_paga=perform_paga,
                    markerSelect_method=markerSelect_method,  # 'wilcoxon','logreg' # is ok too
                    markerSelect_n_genes=markerSelect_n_genes, show_n=show_n)
        elif filepaths is not None:
            if use_rep is None:
                use_rep="X_pca_harmony"
            self.batch_to_clusters_0(verbosity=verbosity, dpi=dpi, facecolor=facecolor, # set_default_0
                anndatas=anndatas,filepaths=filepaths, species=species, organs=organs, # batch_data_input_0
                n_top=n_top, num_min_genes=num_min_genes, num_max_genes=num_max_genes, # filter_0
                    num_min_cells=num_min_cells, 
                    max_percent_of_specific_genes=max_percent_of_specific_genes,
                    genelist=genelist, mad_coeffcient=mad_coeffcient,
                norm_target_sum=norm_target_sum, hvg_min_mean=hvg_min_mean, hvg_max_mean=hvg_max_mean,  
                                     # norm_log_hvg_filter_regress_scale_0
                    hvg_min_dispersions=hvg_min_dispersions, regresslist=regresslist, scale_max_value=scale_max_value,
                                     n_top_hvg_genes=n_top_hvg_genes,
                pca_svd_algorithm=pca_svd_algorithm, pca_genelist=pca_genelist, # pca_cluster_markerSelect_0
                    default_cluster_method=default_cluster_method,
                    leiden_resolution=leiden_resolution,louvain_resolution=louvain_resolution,
                    cluster_n_neighbors=cluster_n_neighbors, cluster_n_pcs=cluster_n_pcs, 
                    use_rep=use_rep, perform_umap=perform_umap,
                    umap_genelist=umap_genelist, perform_paga=perform_paga,
                    markerSelect_method=markerSelect_method,  # 'wilcoxon','logreg' # is ok too
                    markerSelect_n_genes=markerSelect_n_genes, show_n=show_n)
        elif anndatas is not None:
            if use_rep is None:
                use_rep="X_pca_harmony"
            self.batch_to_clusters_0(verbosity=verbosity, dpi=dpi, facecolor=facecolor, # set_default_0
                anndatas=anndatas,filepaths=filepaths, species=species, organs=organs, # batch_data_input_0
                n_top=n_top, num_min_genes=num_min_genes, num_max_genes=num_max_genes, # filter_0
                    num_min_cells=num_min_cells, 
                    max_percent_of_specific_genes=max_percent_of_specific_genes,
                    genelist=genelist, mad_coeffcient=mad_coeffcient,
                norm_target_sum=norm_target_sum, hvg_min_mean=hvg_min_mean, hvg_max_mean=hvg_max_mean,  
                                     # norm_log_hvg_filter_regress_scale_0
                    hvg_min_dispersions=hvg_min_dispersions, regresslist=regresslist, scale_max_value=scale_max_value,
                                     n_top_hvg_genes=n_top_hvg_genes,
                pca_svd_algorithm=pca_svd_algorithm, pca_genelist=pca_genelist, # pca_cluster_markerSelect_0
                    default_cluster_method=default_cluster_method,
                    leiden_resolution=leiden_resolution,louvain_resolution=louvain_resolution,
                    cluster_n_neighbors=cluster_n_neighbors, cluster_n_pcs=cluster_n_pcs, 
                    use_rep=use_rep, perform_umap=perform_umap,
                    umap_genelist=umap_genelist, perform_paga=perform_paga,
                    markerSelect_method=markerSelect_method,  # 'wilcoxon','logreg' # is ok too
                    markerSelect_n_genes=markerSelect_n_genes, show_n=show_n)
        elif self.obj is not None:
            self.to_clusters_0(verbosity=verbosity, dpi=dpi, facecolor=facecolor, # set_default_0
                filepath=filepath, species=species, organ=organ, organs=organs, # data_input_0
                n_top=n_top, num_min_genes=num_min_genes, num_max_genes=num_max_genes, # filter_0
                    num_min_cells=num_min_cells, 
                    max_percent_of_specific_genes=max_percent_of_specific_genes,
                    genelist=genelist, mad_coeffcient=mad_coeffcient,
                norm_target_sum=norm_target_sum, hvg_min_mean=hvg_min_mean, hvg_max_mean=hvg_max_mean,  
                               # norm_log_hvg_filter_regress_scale_0
                    hvg_min_dispersions=hvg_min_dispersions, regresslist=regresslist, scale_max_value=scale_max_value,
                               n_top_hvg_genes=n_top_hvg_genes,
                pca_svd_algorithm=pca_svd_algorithm, pca_genelist=pca_genelist, # pca_cluster_markerSelect_0
                    default_cluster_method=default_cluster_method,
                    leiden_resolution=leiden_resolution,louvain_resolution=louvain_resolution,
                    cluster_n_neighbors=cluster_n_neighbors, cluster_n_pcs=cluster_n_pcs, 
                    use_rep=use_rep, perform_umap=perform_umap,
                    umap_genelist=umap_genelist, perform_paga=perform_paga,
                    markerSelect_method=markerSelect_method,  # 'wilcoxon','logreg' # is ok too
                    markerSelect_n_genes=markerSelect_n_genes, show_n=show_n)
        else:
            raise PCmaster_anno_0_Error("No input.")
        if self.obj is not None:
            self.markergenepd_make_0()
            self.annotation_with_marker_gene_list_0(marker_select_flag = marker_select_flag,
                                                    marker_grade_1_coefficient = marker_grade_1_coefficient,
                                                    marker_gene_list = marker_gene_list)
            self.get_dir_gene_score(marker_gene_list = marker_gene_list)
            self.annotation_with_marker_gene_scores(the_least_score = the_least_score)

    # not used
    def batch_data_input_for_CRA004082_PRJCA004855_0(self,
                                                     filepaths=None, species=None,
                                                     organs=None, labelpath=None):
        if labelpath is not None:
            self.cellnamelabel = pd.read_csv(labelpath)
        for i in filepaths:
            if i[-6:] == "tsv.gz":
                tmp = sc.read_umi_tools(i)
                print(tmp.obs_names)
                cellnamelisttmp = []
                for j in tmp.obs_names:
                    j = j[0:-2] + '-' + i.split('/')[7].replace('-', '').replace('_', '')
                    # print(i)
                    # break
                    cellnamelisttmp.append(j)

                tmp.obs_names = cellnamelisttmp
                print(tmp.obs_names)
                tmp_any_tmp_list = []
                label_set = set()
                tflist = []
                for i in self.cellnamelabel['cellname']:
                    label_set.add(str(i))
                for i in tmp.obs_names:
                    tmp_any_tmp_list.append(str(i))
                for i in tmp_any_tmp_list:
                    if i in label_set:
                        tflist.append(1)
                    else:
                        tflist.append(0)

                ls = {'cellname': tmp_any_tmp_list,
                      'tf': tflist}
                tfpd = pd.DataFrame(ls)
                tfpd.describe()

                tmp.obs['tf'] = np.array(tfpd['tf']).tolist()
                tmp = tmp[tmp.obs.tf == 1, :]
                print(tmp.obs_names)

                # self.objs.append( tmp )
                del tmp
                del cellnamelisttmp
                del tmp_any_tmp_list
                del label_set
                del tflist
                del ls
                del tfpd
                gc.collect()
                info = psutil.virtual_memory()
                print(psutil.Process(os.getpid()).memory_info().rss)
                print(info.total)
                print(info.percent)
                print(psutil.cpu_count())
            if i[-6:] == "mtx.gz":
                tmp = sc.read_10x_mtx(i[:-13], var_names='gene_symbols', cache=False)
                print(tmp.obs_names)
                cellnamelisttmp = []
                for j in tmp.obs_names:
                    j = j[0:-2] + '-' + i.split('/')[7].replace('-', '').replace('_', '')
                    # print(i)
                    # break
                    cellnamelisttmp.append(j)

                tmp.obs_names = cellnamelisttmp
                print(tmp.obs_names)
                tmp_any_tmp_list = []
                label_set = set()
                tflist = []
                for i in self.cellnamelabel['cellname']:
                    label_set.add(str(i))
                for i in tmp.obs_names:
                    tmp_any_tmp_list.append(str(i))
                for i in tmp_any_tmp_list:
                    if i in label_set:
                        tflist.append(1)
                    else:
                        tflist.append(0)

                ls = {'cellname': tmp_any_tmp_list,
                      'tf': tflist}
                tfpd = pd.DataFrame(ls)
                tfpd.describe()

                tmp.obs['tf'] = np.array(tfpd['tf']).tolist()
                tmp = tmp[tmp.obs.tf == 1, :]
                print(tmp.obs_names)

                # self.objs.append( tmp )
                del tmp
                del cellnamelisttmp
                del tmp_any_tmp_list
                del label_set
                del tflist
                del ls
                del tfpd
                gc.collect()
                info = psutil.virtual_memory()
                print(psutil.Process(os.getpid()).memory_info().rss)
                print(info.total)
                print(info.percent)
                print(psutil.cpu_count())
        if species is not None:
            for i in species:
                self.batch_species.append(i)
        if organs is not None:
            for i in organs:
                self.batch_organs.append(i)

    # not used
    def process_for_CRA004082_PRJCA004855_0(self, filepaths=None, use_rep="X_pca_harmony",
                                            labelpath=None):
        # if labelpath is not None:
        #     self.cellnamelabel = pd.read_csv(labelpath)
        self.set_default_0()
        self.batch_data_input_for_CRA004082_PRJCA004855_0(filepaths, labelpath)
        self.concat_0()

        self.objs = []
        gc.collect()
        info = psutil.virtual_memory()
        print(psutil.Process(os.getpid()).memory_info().rss)
        print(info.total)
        print(info.percent)
        print(psutil.cpu_count())

        self.filter_0()
        self.norm_log_hvg_filter_regress_scale_0()
        self.pca_cluster_markerSelect_0(use_rep=use_rep)
        
    # input ref and test, annotation by deep learning model
    def auto_anno_mldl_train_0(self,ref_obj=None,test_obj=None,
                               random_sample_num=1000,
                               gpu_code = None,
                               learning_rate = None,epochs = None,
                               batch_size = None,dropout = None,
                               last_linear_n = None,num_workers = None,
                               model_weight_save_path = None,
                               mapping_1_save_path = None,
                               mapping_2_save_path = None,
                               num_types_save_path = None):
        # which GPU
        if gpu_code is None:
            gpu_code = 0
        self.accuracys = []
        self.precisions = []
        self.recalls = []
        lr = self.set_the_value(input_value = learning_rate,default = 0.01)
        num_epochs = self.set_the_value(input_value = epochs,default = 10)
        the_batch_size = self.set_the_value(input_value = batch_size,default = 16)
        the_num_workers = self.set_the_value(input_value = num_workers,default = 4)
        the_dropout = self.set_the_value(input_value = dropout,default = 0.3)
        the_last_linear_n = self.set_the_value(input_value = last_linear_n,default = 1000)
        print('lr='+str(lr))
        print('num_epochs='+str(num_epochs))
        print('the_batch_size='+str(the_batch_size))
        print('the_num_workers='+str(the_num_workers))
        print('the_dropout='+str(the_dropout))
        print('the_last_linear_n='+str(the_last_linear_n))

        try:   
            df,mapping_1,mapping_2 = self.auto_annotation_with_deep_learning_transfer_adata_to_df_0(adata_input = ref_obj)  
        except Exception:   
            df,mapping_1,mapping_2 = self.auto_annotation_with_deep_learning_transfer_adata_to_df_without_todense_0(adata_input = ref_obj) 
        
        # displays the number of categories and the number of each category.
        category_counts = df['celltype'].value_counts()
        print("the number of categories: ", len(category_counts))
        print("the number of each category: ")
        print(category_counts)
        print(mapping_1)
        print(mapping_2)
        
        category_counts = df['celltype'].value_counts()
        train_data = pd.DataFrame()
        valid_data = pd.DataFrame()
        test_data = pd.DataFrame()
        # for each celltype, catch random_sample_num cells
        for category in category_counts.index:
            category_data = df[df['celltype'] == category]
            print(category)
            print(category_data.shape[0])
            # print(type(category_data.shape[0]))
            category_data = category_data.sample(n=min(random_sample_num,int(category_data.shape[0])), random_state=1, replace=True)
            # from sklearn.model_selection import train_test_split
            train, temp = train_test_split(category_data, train_size=0.8, random_state=1)
            valid, test = train_test_split(temp, train_size=0.5, random_state=1)
            train_data = pd.concat([train_data, train])
            valid_data = pd.concat([valid_data, valid])
            test_data = pd.concat([test_data, test])
        category_counts = train_data['celltype'].value_counts()
        train_data_balanced = pd.DataFrame()
        for category in category_counts.index:
            category_data = train_data[train_data['celltype'] == category]
            print(category)
            print(category_data.shape[0])
            # print(type(category_data.shape[0]))
            category_data = category_data.sample(n=random_sample_num, random_state=1, replace=True)
            train_data_balanced = pd.concat([train_data_balanced, category_data])
        train_data = train_data_balanced
            
        train = train_data
        valid = valid_data
        test = test_data
        
        # divide data and label
        # +0.001
        train_data = train.iloc[:, :-2]
        train_label = train.iloc[:, -1:]
        train_data = train_data + 0.001
        valid_data = valid.iloc[:, :-2]
        valid_label = valid.iloc[:, -1:]
        valid_data = valid_data + 0.001
        test_data = test.iloc[:, :-2]
        test_label = test.iloc[:, -1:]
        test_data = test_data + 0.001
            
        import torch
            
        traindata = trainDataset(train_data,train_label)
        validdata = validDataset(valid_data,valid_label)
        train_sampler = RandomSampler(traindata)
        train_iter = DataLoader(traindata,batch_size=the_batch_size,shuffle=False,sampler=train_sampler,
                                num_workers=the_num_workers)
        valid_sampler = RandomSampler(validdata)
        valid_iter = DataLoader(validdata,batch_size=the_batch_size,shuffle=False,sampler=valid_sampler,
                                num_workers=the_num_workers)
        
        num_types = train_label.nunique().values[0]

        import torchvision.models
        
        class PscResNet(nn.Module):
            def __init__(self):
                super(PscResNet, self).__init__()
                # change "pretrained=False" to "weights=None" may be better
                self.resnet18 = torchvision.models.resnet18(weights=None)   
                self.pscMLP = nn.Sequential(
                                nn.ReLU(),
                                nn.Dropout(the_dropout),
                                nn.Linear(the_last_linear_n, num_types))
                
            def forward(self, x):
                x = self.resnet18(x)
                x = self.pscMLP(x)
                return x
        
        pscResNet1 = PscResNet()
        # load pth
        pretrained_dict = torch.load('resnet18.pth')
        model_dict = pscResNet1.resnet18.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        pscResNet1.resnet18.load_state_dict(model_dict)
        
        import torch
        torch.cuda.init()
                    
        # GPU
        def evaluate_accuracy_gpu(net, data_iter, device=None): 
            if isinstance(net, nn.Module):
                net.eval()
                if not device:
                    device = next(iter(net.parameters())).device
            metric = d2l.Accumulator(2)
            with torch.no_grad():
                for X, y in data_iter:
                    if isinstance(X, list):
                        X = [x.to(device) for x in X]
                    else:
                        X = X.to(device)
                    y = y.to(device)
                    metric.add(d2l.accuracy(net(X), y), y.numel())
            return metric[0] / metric[1]
        
        # GPU
        def train_with_GPU(net, train_iter, valid_iter, num_epochs, lr, device):
            def init_weights(m):
                if type(m) == nn.Linear or type(m) == nn.Conv2d:
                    nn.init.xavier_uniform_(m.weight)
            net.apply(init_weights)
            print('training on', device)
            net.to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0.00001)
            loss = nn.CrossEntropyLoss()
            animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],ylim=[0,1],
                                    legend=['train loss', 'train acc', 'valid acc'])
            timer, num_batches = d2l.Timer(), len(train_iter)
            for epoch in range(num_epochs):
                metric = d2l.Accumulator(3)
                net.train()
                for i, (X, y) in enumerate(train_iter):
                    timer.start()
                    optimizer.zero_grad()
                    X, y = X.to(device), y.to(device)
                    y_hat = net(X)
                    l = loss(y_hat, y)
                    l.backward()
                    optimizer.step()
                    scheduler.step()
                    with torch.no_grad():
                        metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                    timer.stop()
                    train_l = metric[0] / metric[2]
                    train_acc = metric[1] / metric[2]
                    if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                        animator.add(epoch + (i + 1) / num_batches,
                                     (train_l, train_acc, None))
                valid_acc = evaluate_accuracy_gpu(net, valid_iter)
                animator.add(epoch + 1, (None, None, valid_acc))
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
                  f'valid acc {valid_acc:.3f}')
            print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
                  f'on {str(device)}')
        train_with_GPU(pscResNet1, train_iter, valid_iter, num_epochs, lr, d2l.try_all_gpus()[gpu_code])
        self.model_weight = pscResNet1.state_dict()
        self.mapping_1 = mapping_1
        self.mapping_2 = mapping_2
        self.num_types = num_types
        if model_weight_save_path is not None and \
                mapping_1_save_path is not None and \
                mapping_2_save_path is not None and \
                num_types_save_path is not None:
            torch.save(pscResNet1.state_dict(), model_weight_save_path)
            tmp_txt = open(mapping_1_save_path, "w", encoding='utf-8')
            for key, value in mapping_1.items():
                tmp_txt.write(str(key) + ": " + str(value) + "\n")
            tmp_txt.close()
            tmp_txt = open(mapping_2_save_path, "w", encoding='utf-8')
            for key, value in mapping_2.items():
                tmp_txt.write(str(key) + ": " + str(value) + "\n")
            tmp_txt.close()
            tmp_txt = open(num_types_save_path, "w", encoding='utf-8')
            tmp_txt.write(str(num_types) + "\n")
            tmp_txt.close()
        else:
            now = datetime.datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            print(current_time)
            torch.save(pscResNet1.state_dict(), 'model_weight_'+current_time.replace(':','-').replace(' ','-')+'.pth')
            tmp_txt = open('model_'+current_time.replace(':','-').replace(' ','-')+"_mapping_1.txt", "w", encoding='utf-8')
            for key, value in mapping_1.items():
                tmp_txt.write(str(key) + ": " + str(value) + "\n")
            tmp_txt.close()
            tmp_txt = open('model_'+current_time.replace(':','-').replace(' ','-')+"_mapping_2.txt", "w", encoding='utf-8')
            for key, value in mapping_2.items():
                tmp_txt.write(str(key) + ": " + str(value) + "\n")
            tmp_txt.close()
            tmp_txt = open('model_'+current_time.replace(':','-').replace(' ','-')+"_num_types.txt", "w", encoding='utf-8')
            tmp_txt.write(str(num_types) + "\n")
            tmp_txt.close()
        
        # test_data
        tmp_shape = test_data.shape[0]
        test_data = torch.from_numpy(np.float32(np.array(test_data)))
        test_data = torch.reshape(test_data, (tmp_shape, 224, 224))
        test_data = torch.unsqueeze(test_data, 1)
        test_data = torch.cat((test_data, test_data, test_data), dim=1)
        test_label = torch.from_numpy(np.int64(np.array(test_label)).reshape(1,-1)[0])
        
        pscResNet1 = pscResNet1.to('cpu')
        pscResNet1.eval()
        
        # close
        with torch.no_grad():
            outputs = pscResNet1(test_data)
        
        _, predicted = torch.max(outputs, 1)
        
        # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # to NumPy
        predicted = predicted.numpy()
        test_label = test_label.numpy()
        
        # accuracy = accuracy_score(test_label, predicted)
        # 
        # precision = precision_score(test_label, predicted, average='macro', zero_division=1)
        # 
        # recall = recall_score(test_label, predicted, average='macro', zero_division=1)
        # 
        # print('accuracy')
        # print(accuracy)
        # print('precision')
        # print(precision)
        # print('recall')
        # print(recall)
        # print('macro f1 score')
        # print( 2*precision*recall / (precision+recall) )
        
        # accuracy
        accuracy = accuracy_score(test_label, predicted)  
        
        # macro precision 
        macro_precision = precision_score(test_label, predicted, average='macro', zero_division=1)  
        
        # macro recall  
        macro_recall = recall_score(test_label, predicted, average='macro', zero_division=1)  
        
        # macro F1 score
        macro_f1_score = f1_score(test_label, predicted, average='macro', zero_division=1)  
        
        print("Accuracy:", accuracy)  
        print("Macro Precision:", macro_precision)  
        print("Macro Recall:", macro_recall)  
        print("Macro F1 Score:", macro_f1_score)  
        self.train_accuracy = accuracy
        self.train_macro_f1_score = macro_f1_score
        
    def auto_anno_mldl_predict_0(self,ref_obj=None,test_obj=None,gpu_code = None,
                                 model_weight_load_path = None,
                                 dropout = None,
                                 mapping_1_load_path = None,
                                 mapping_2_load_path = None,
                                 num_types_load_path = None,
                                 predicted_results_save_path = None):
        the_dropout = self.set_the_value(input_value = dropout,default = 0.3)
        if self.model_weight is None and model_weight_load_path is None:
            raise PCmaster_anno_0_Error('No model weight! \n self.model_weight is None and model_weight_load_path is None!')
        if self.model_weight is not None and model_weight_load_path is not None:
            raise PCmaster_anno_0_Error('Only one is needed! \n self.model_weight and model_weight_load_path both exist!')
        if self.model_weight is None:
            self.model_weight = torch.load(model_weight_load_path)
            print('load model weight from '+ model_weight_load_path)
        if self.mapping_1 is None:
            self.mapping_1 = self.auto_annotation_with_deep_learning_load_mapping_0(mapping_load_path = mapping_1_load_path)
            print('load mapping_1 from '+mapping_1_load_path)
        if self.mapping_2 is None:
            self.mapping_2 = self.auto_annotation_with_deep_learning_load_mapping_0(mapping_load_path = mapping_2_load_path)
            print('load mapping_2 from '+mapping_2_load_path)
        if self.num_types is None:
            self.num_types = self.auto_annotation_with_deep_learning_load_num_types_0(num_types_load_path = num_types_load_path)
            print('load num_types from '+num_types_load_path)
        # print(self.model_weight)
        print(self.mapping_1)
        print(self.mapping_2)
        print(self.num_types)
        aaa_num_types = self.num_types
        
        import torchvision.models

        class PscResNet(nn.Module):
            def __init__(self):
                super(PscResNet, self).__init__()
                self.resnet18 = torchvision.models.resnet18(pretrained=False)   
                self.pscMLP = nn.Sequential(
                                nn.ReLU(),
                                nn.Dropout(the_dropout),
                                nn.Linear(1000, aaa_num_types))
                
            def forward(self, x):
                x = self.resnet18(x)
                x = self.pscMLP(x)
                return x
        
        pscResNet1 = PscResNet()
        pretrained_dict = self.model_weight
        model_dict = pscResNet1.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        pscResNet1.load_state_dict(model_dict)
        
        try:   
            df,_,_ = self.auto_annotation_with_deep_learning_transfer_adata_to_df_0(adata_input = test_obj)  
        except Exception:   
            df,_,_ = self.auto_annotation_with_deep_learning_transfer_adata_to_df_without_todense_0(adata_input = test_obj) 
        
        df_label = df.iloc[:, -1:]
        df_label_full = df.iloc[:, -2:]
        df = df.iloc[:, :-2]
        
        import math
        
        batch_size = 32 
        total_size = df.shape[0]
        total_batches = math.ceil(total_size / batch_size)
        
        net_try = pscResNet1
        net_try = net_try.to('cpu')
        net_try.eval()
        
        with torch.no_grad():
            predicted_results = []
            for i in range(total_batches):
                start = i * batch_size
                end = min(start + batch_size, total_size)
                batch_data = df[start:end]
                batch_label = df_label[start:end]
                batch_data = torch.from_numpy(np.float32(np.array(batch_data)))
                batch_data = torch.reshape(batch_data, (batch_data.shape[0], 224, 224))
                batch_data = torch.unsqueeze(batch_data, 1)
                batch_data = torch.cat((batch_data, batch_data, batch_data), dim=1)
                batch_label = torch.from_numpy(np.int64(np.array(batch_label)).reshape(1,-1)[0])
                outputs = net_try(batch_data)
                _, predicted = torch.max(outputs, 1)
                predicted_results.append(predicted.numpy())
        
        predicted_results = np.concatenate(predicted_results)
        dl_cellnames = df_label_full['cellname'].tolist()
        original_celltypes = df_label_full['celltype'].tolist()
        # print(predicted_results)
        print(len(predicted_results))
        # print(dl_cellnames)
        print(len(dl_cellnames))
        # print(original_celltypes)
        print(len(original_celltypes))
        
        list2 = predicted_results.tolist()
        
        reverse_mapping_2 = {v: k for k, v in self.mapping_2.items()}
        
        list2_0_mapped = [reverse_mapping_2[value] for value in list2]
        
        list2_0 = [next(key for key, value in self.mapping_1.items() if value == mapped_value) for mapped_value in list2_0_mapped]
        
        # print(list2_0)
        
        predicted_results_true = list2_0
        
        self.predicted_results = predicted_results_true
        self.dl_cellnames = dl_cellnames
        self.original_celltypes = original_celltypes
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(current_time)
        tmp_txt = open('model_'+current_time.replace(':','-').replace(' ','-')+"_predicted_results.txt", "w", encoding='utf-8')
        for list_item in self.predicted_results:
            tmp_txt.write(str(list_item) + "\n")
        tmp_txt.close()
        tmp_txt = open('model_'+current_time.replace(':','-').replace(' ','-')+"_dl_cellnames.txt", "w", encoding='utf-8')
        for list_item in self.dl_cellnames:
            tmp_txt.write(str(list_item) + "\n")
        tmp_txt.close()
        tmp_txt = open('model_'+current_time.replace(':','-').replace(' ','-')+"_original_celltypes.txt", "w", encoding='utf-8')
        for list_item in self.original_celltypes:
            tmp_txt.write(str(list_item) + "\n")
        tmp_txt.close()
        
    def auto_annotation_with_deep_learning_0(self,original_obj=None,ref_test_label=None,
                                           ref_test_label_for_ref=None,ref_test_label_for_test=None,
                                           random_sample_num=1000,gpu_code = None,
                                           learning_rate = None,epochs = None,
                                           batch_size = None,dropout = None,
                                           last_linear_n = None,num_workers = None,
                                           model_weight_save_path = None,
                                           mapping_1_save_path = None,
                                           mapping_2_save_path = None,
                                           num_types_save_path = None,
                                           model_weight_load_path = None,
                                           mapping_1_load_path = None,
                                           mapping_2_load_path = None,
                                           num_types_load_path = None,
                                           predicted_results_save_path = None,
                                           run_model = None):
        if run_model is None:
            run_model = 'train_and_predict'
        if model_weight_load_path is not None:
            run_model = 'predict'
        if ref_test_label is None:
            ref_test_label = 'ref_test_label'
        if ref_test_label_for_ref is None:
            ref_test_label_for_ref = 'ref'
        if ref_test_label_for_test is None:
            ref_test_label_for_test = 'test'
        ref_obj,test_obj = self.auto_annotation_with_deep_learning_split_dataset_0(original_obj=original_obj,ref_test_label=ref_test_label,
                                                                            ref_test_label_for_ref=ref_test_label_for_ref,
                                                                            ref_test_label_for_test=ref_test_label_for_test)
        if run_model == 'train':
            self.auto_anno_mldl_train_0(ref_obj=ref_obj,test_obj=test_obj,random_sample_num=random_sample_num,gpu_code = gpu_code,
                                        learning_rate = learning_rate,epochs = epochs,
                                        batch_size = batch_size,dropout = dropout,
                                        last_linear_n = last_linear_n,num_workers = num_workers,
                                        model_weight_save_path = model_weight_save_path,
                                        mapping_1_save_path = mapping_1_save_path,
                                        mapping_2_save_path = mapping_2_save_path,
                                        num_types_save_path = num_types_save_path)
        if run_model == 'train_and_predict':
            self.auto_anno_mldl_train_0(ref_obj=ref_obj,test_obj=test_obj,random_sample_num=random_sample_num,gpu_code = gpu_code,
                                        learning_rate = learning_rate,epochs = epochs,
                                        batch_size = batch_size,dropout = dropout,
                                        last_linear_n = last_linear_n,num_workers = num_workers,
                                        model_weight_save_path = model_weight_save_path,
                                        mapping_1_save_path = mapping_1_save_path,
                                        mapping_2_save_path = mapping_2_save_path,
                                        num_types_save_path = num_types_save_path)
            self.auto_anno_mldl_predict_0(ref_obj=ref_obj,test_obj=test_obj,gpu_code = gpu_code,
                                          model_weight_load_path = model_weight_load_path,
                                          dropout = dropout,
                                          mapping_1_load_path = mapping_1_load_path,
                                          mapping_2_load_path = mapping_2_load_path,
                                          num_types_load_path = num_types_load_path,
                                          predicted_results_save_path = predicted_results_save_path)
        if run_model == 'predict':
            self.auto_anno_mldl_predict_0(ref_obj=ref_obj,test_obj=test_obj,gpu_code = gpu_code,
                                          model_weight_load_path = model_weight_load_path,
                                          dropout = dropout,
                                          mapping_1_load_path = mapping_1_load_path,
                                          mapping_2_load_path = mapping_2_load_path,
                                          num_types_load_path = num_types_load_path,
                                          predicted_results_save_path = predicted_results_save_path)
        
    def auto_annotation_with_deep_learning_split_dataset_0(self,original_obj=None,ref_test_label=None,
                                                         ref_test_label_for_ref=None,ref_test_label_for_test=None):
        ref_obj = original_obj[original_obj.obs[ref_test_label] == ref_test_label_for_ref]
        test_obj = original_obj[original_obj.obs[ref_test_label] == ref_test_label_for_test]
        return ref_obj,test_obj
    
    def auto_annotation_with_deep_learning_load_mapping_0(self,mapping_load_path = None):
        tmp_txt = open(mapping_load_path, "r", encoding='utf-8')
        tmp_dict = {}
        for line in tmp_txt:
            tmp_dict[int(line.strip().split(": ")[0])] = line.strip().split(": ")[1]
        return tmp_dict
    
    def auto_annotation_with_deep_learning_load_num_types_0(self,num_types_load_path = None):
        tmp_txt = open(num_types_load_path, "r", encoding='utf-8')
        tmp_num_types = None
        for line in tmp_txt:
            tmp_num_types = line.strip()
        tmp_num_types = int(tmp_num_types)
        return tmp_num_types
    
    def set_the_value(self,input_value = None,default = None):
        if input_value is None:
            return default
        else:
            return input_value

    def auto_annotation_with_deep_learning_transfer_adata_to_df_for_all_0(self,adata_input = None,the_length = None):
        if the_length is None:
            the_length = 224
        try:   
            df,mapping_1,mapping_2 = self.auto_annotation_with_deep_learning_transfer_adata_to_df_0(
                adata_input = adata_input,the_length = the_length,
            )  
        except Exception:   
            df,mapping_1,mapping_2 = self.auto_annotation_with_deep_learning_transfer_adata_to_df_without_todense_0(
                adata_input = adata_input,the_length = the_length,
            )
        return df,mapping_1,mapping_2
        
    def auto_annotation_with_deep_learning_transfer_adata_to_df_0(self,adata_input = None,the_length = None):
        # normalize_total
        # sc.pp.normalize_total(adata_input, target_sum=1e4)
        if the_length is None:
            the_length = 224
        adata_input.obs['cellname'] = adata_input.obs_names
        
        tmp_pcm = adata_input
        data=tmp_pcm.X.todense()
        pcm_pd=pd.DataFrame(data,index=tmp_pcm.obs_names,columns=tmp_pcm.var_names)
        pcm_pd.loc[:,'cellname']=pcm_pd.index
        df = pcm_pd
        df.drop_duplicates(subset='cellname', keep='first', inplace=True)
        
        duplicate_rows = df.duplicated(subset='cellname', keep=False)
        df = df[~duplicate_rows]
        pcm_pd = df
        # print('pcm')
        # print(pcm_pd)
        # print('\n')
        
        label=pd.concat([adata_input.obs['cellname'], adata_input.obs['celltype']], axis=1)
        df = label
        num_categories = df['celltype'].nunique()
        mapping_1 = {category: i for i, category in enumerate(df['celltype'].unique())}
        # print('mapping_1')
        # print(mapping)
        # print('\n')
        df['celltype'] = df['celltype'].map(mapping_1)
        duplicate_rows = df.duplicated(subset='cellname', keep=False)
        df = df[~duplicate_rows]
        # print('df')
        # print(df)
        # print('\n')
        label = df
        
        merged = pd.merge(pcm_pd, label, on='cellname', how='inner')
        # print("merged")
        # print(merged)
        df = merged
        num_categories = df['celltype'].nunique()
        mapping_2 = {category: i for i, category in enumerate(df['celltype'].unique())}
        # print('mapping_2')
        # print(mapping)
        # print('\n')
        df['celltype'] = df['celltype'].map(mapping_2)
        
        # Create an all-zero DataFrame with the same number of rows as the original DataFrame, and the number of columns is the number of target columns to be expanded.
        # ResNet is used by default here. To change the data shape to 224*224=50176.
        # Here +2 is because the columns of celltype and cellname should be removed.
        # Because others' ResNet has trained tens of thousands of epoch, we use others' weights, and the convergence speed will be much faster than training from scratch.
        # The newly added line will be filled with 0 because it is meaningless, and then it will become 0.0001.
        # Here, making every value in the whole world not 0 takes into account the statement that data of 0 is not conducive to updating parameters.
        target_columns = the_length*the_length-df.shape[1]+2
        zeros_df = pd.DataFrame(np.zeros((df.shape[0], target_columns)), columns=[f'new_col_{i}' for i in range(target_columns)])
        
        expanded_df = pd.concat([df, zeros_df], axis=1)
        
        # print(expanded_df.shape) 
        
        df = expanded_df
        # print(df)
        columns = df.columns.tolist()
        
        columns.remove('cellname')
        columns.append('cellname')
        
        columns.remove('celltype')
        columns.append('celltype')
        
        # reorder
        df = df[columns]
        
        return df,mapping_1,mapping_2

    def auto_annotation_with_deep_learning_transfer_adata_to_df_without_todense_0(self,adata_input = None,the_length = None):
        # sc.pp.normalize_total(adata_input, target_sum=1e4)
        if the_length is None:
            the_length = 224
        
        adata_input.obs['cellname'] = adata_input.obs_names
        
        tmp_pcm = adata_input
        data=tmp_pcm.X
        pcm_pd=pd.DataFrame(data,index=tmp_pcm.obs_names,columns=tmp_pcm.var_names)
        pcm_pd.loc[:,'cellname']=pcm_pd.index
        df = pcm_pd
        df.drop_duplicates(subset='cellname', keep='first', inplace=True)
        
        duplicate_rows = df.duplicated(subset='cellname', keep=False)
        df = df[~duplicate_rows]
        pcm_pd = df
        # print('pcm')
        # print(pcm_pd)
        # print('\n')
        
        label=pd.concat([adata_input.obs['cellname'], adata_input.obs['celltype']], axis=1)
        df = label
        num_categories = df['celltype'].nunique()
        mapping_1 = {category: i for i, category in enumerate(df['celltype'].unique())}
        # print('mapping_1')
        # print(mapping)
        # print('\n')
        df['celltype'] = df['celltype'].map(mapping_1)
        duplicate_rows = df.duplicated(subset='cellname', keep=False)
        df = df[~duplicate_rows]
        # print('df')
        # print(df)
        # print('\n')
        label = df
        
        merged = pd.merge(pcm_pd, label, on='cellname', how='inner')
        # print("merged")
        # print(merged)
        df = merged
        num_categories = df['celltype'].nunique()
        mapping_2 = {category: i for i, category in enumerate(df['celltype'].unique())}
        # print('mapping_2')
        # print(mapping)
        # print('\n')
        df['celltype'] = df['celltype'].map(mapping_2)

        target_columns = the_length*the_length-df.shape[1]+2
        zeros_df = pd.DataFrame(np.zeros((df.shape[0], target_columns)), columns=[f'new_col_{i}' for i in range(target_columns)])
        
        expanded_df = pd.concat([df, zeros_df], axis=1)
        
        # print(expanded_df.shape)  
        
        df = expanded_df
        # print(df)
        columns = df.columns.tolist()
        
        columns.remove('cellname')
        columns.append('cellname')
        
        columns.remove('celltype')
        columns.append('celltype')
        
        df = df[columns]
        
        return df,mapping_1,mapping_2

    def save_df_to_images(self,root_dir = None,df = None,mapping_1 = None,mapping_2 = None,the_length = None):
        if the_length is None:
            the_length = 224
        
        start_time = time.time()
        if root_dir is None or df is None or mapping_1 is None or mapping_2 is None:
            raise PCmaster_anno_0_Error("root_dir == None or df == None or mapping_1 == None or mapping_2 == None.")
        # root_dir = './data_to_image_test/'
        reverse_mapping_1 = {v: k for k, v in mapping_1.items()}  
        reverse_mapping_2 = {v: k for k, v in mapping_2.items()} 
        # Read the DataFrame containing gene expression data  
        # df = pd.read_csv('your_dataframe.csv')  
        the_max = df.iloc[:, :the_length*the_length].values.max()
        print(the_max)
        # Create a MinMaxScaler object  
        # scaler = MinMaxScaler(feature_range=(0, 255))  
        print('ok')
        # Iterate over each cell  
        for index in range(df.shape[0]):
            row = df.iloc[index]
            # Extract the gene expression data  
            gene_expression = row[:the_length*the_length].values  
            # print(gene_expression)
            # the_max = gene_expression.max()
            # gene_expression = gene_expression*255.0/the_max
            
            gene_expression = gene_expression.reshape(1,-1)
            # print(gene_expression)
            # break

            # Scale the gene_expression to have values between 0 and 255  
            # scaled_gene_expression = scaler.fit_transform(gene_expression)  
            scaled_gene_expression = gene_expression  
            
            # Reshape the gene expression data into a 224x224 matrix  
            matrix = scaled_gene_expression.reshape(the_length, the_length)  
            scaled_matrix = matrix
            
            # Convert the scaled matrix to uint8 data type  
            scaled_matrix = scaled_matrix.astype(np.float32)  
            
            # Copy and expand the matrix into a 224x224x3 matrix  
            tmp_matrix = np.repeat(scaled_matrix[:, :, np.newaxis], 3, axis=2)  
            
            # Create an image object  
            image = Image.fromarray(tmp_matrix)  
            
            # Get the celltype and cellname of the cell  
            cell_type = row[the_length*the_length+1]
            tmp_re_1 = reverse_mapping_2.get(cell_type)
            cell_type_origin = reverse_mapping_1.get(tmp_re_1)
            cell_type = cell_type_origin.replace(' ','___').replace('/','---')
            # print('celltype')
            # print(cell_type)
            
            cell_name = str(row[the_length*the_length])  
            # print('cellname')
            # print(cell_name)
            
            # Create the directory to save the image (if it doesn't exist)  
            save_dir = root_dir + str(cell_type)  
            os.makedirs(save_dir, exist_ok=True)  
            
            # Save the image as a JPEG file  
            image.save(f'{save_dir}/{cell_name}.jpg')
            # break
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"time cost：{execution_time} s")

    def save_df_to_npys(self,root_dir = None,df = None,mapping_1 = None,mapping_2 = None,the_length = None):
        if the_length is None:
            the_length = 224
        
        start_time = time.time()
        if root_dir is None or df is None or mapping_1 is None or mapping_2 is None:
            raise PCmaster_anno_0_Error("root_dir == None or df == None or mapping_1 == None or mapping_2 == None.")
        # root_dir = './data_to_image_test/'
        reverse_mapping_1 = {v: k for k, v in mapping_1.items()}  
        reverse_mapping_2 = {v: k for k, v in mapping_2.items()} 
        # Read the DataFrame containing gene expression data  
        # df = pd.read_csv('your_dataframe.csv')  
        the_max = df.iloc[:, :the_length*the_length].values.max()
        print(the_max)
        # Create a MinMaxScaler object  
        # scaler = MinMaxScaler(feature_range=(0, 255))  
        print('ok')
        # Iterate over each cell  
        for index in range(df.shape[0]):
            row = df.iloc[index]
            # Extract the gene expression data  
            gene_expression = row[:the_length*the_length].values  
            # print(gene_expression)
            # the_max = gene_expression.max()
            # gene_expression = gene_expression*255.0/the_max
            
            # gene_expression = gene_expression.reshape(1,-1)
            # print(gene_expression)
            # break

            # Scale the gene_expression to have values between 0 and 255  
            # scaled_gene_expression = scaler.fit_transform(gene_expression)  
            # scaled_gene_expression = gene_expression  
            
            # Reshape the gene expression data into a 224x224 matrix  
            # matrix = scaled_gene_expression.reshape(the_length, the_length)  
            # scaled_matrix = matrix
            
            # Convert the scaled matrix to uint8 data type  
            # scaled_matrix = scaled_matrix.astype(np.uint8)  
            
            # Copy and expand the matrix into a 224x224x3 matrix  
            # tmp_matrix = np.repeat(scaled_matrix[:, :, np.newaxis], 3, axis=2)  
            
            # Create an image object  
            # image = Image.fromarray(tmp_matrix)  
            
            # Get the celltype and cellname of the cell  
            cell_type = row[the_length*the_length+1]
            tmp_re_1 = reverse_mapping_2.get(cell_type)
            cell_type_origin = reverse_mapping_1.get(tmp_re_1)
            cell_type = cell_type_origin.replace(' ','___').replace('/','---')
            # print('celltype')
            # print(cell_type)
            
            cell_name = str(row[the_length*the_length])  
            # print('cellname')
            # print(cell_name)
            
            # Create the directory to save the image (if it doesn't exist)  
            save_dir = root_dir + str(cell_type)  
            os.makedirs(save_dir, exist_ok=True)  
            
            # Save the image as a JPEG file  
            # image.save(f'{save_dir}/{cell_name}.jpg')
            np.save(f'{save_dir}/{cell_name}.npy', gene_expression)
            # break
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"time cost：{execution_time} s")







    # other functions
    
    def calculate_median_f1_score(self,int_list1=None, int_list2=None):
        # from sklearn.metrics import f1_score  
        # import numpy as np  
        f1_scores = f1_score(int_list1, int_list2, average=None)  
    
        # Calculate median F1 score  
        median_f1_score = np.median(f1_scores)  
          
        print('f1-scores')
        print(f1_scores)
        print('median_f1_score')
        print(median_f1_score)
        
        return f1_scores,median_f1_score
    
    def get_adata_gene_list_irgsp(self,the_adata = None):
        adata_gene_list_zh11 = []
        adata_gene_list_irgsp = []
        for i in the_adata.var_names:
            adata_gene_list_zh11.append(i.split('.')[0].strip())
        for i in adata_gene_list_zh11:
            if i in zh11_irgsp_dict.keys():
                adata_gene_list_irgsp.append(zh11_irgsp_dict[i])
            else:
                adata_gene_list_irgsp.append(i)
        # for i in adata_gene_list_irgsp:
        #     print(i)
        return adata_gene_list_irgsp
    
    def compare_two_adata_gene_names(self,adata1 = None,adata2 = None):
        set1 = set()
        set2 = set()
        for i in adata1.var_names:
            set1.add(str(i))
        for i in adata2.var_names:
            set2.add(str(i))
        count = 0
        for i in set1:
            if i in set2:
                count = count+1
        print(count)
    
    # GPU
    def evaluate_accuracy_gpu(self,net, data_iter, device=None): 
        if isinstance(net, nn.Module):
            net.eval()
            if not device:
                device = next(iter(net.parameters())).device
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(d2l.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]
    
    # GPU
    def train_with_GPU(self,net, train_iter, valid_iter, num_epochs, lr, device):
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)
        print('training on', device)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0.00001)
        loss = nn.CrossEntropyLoss()
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],ylim=[0,1],
                                legend=['train loss', 'train acc', 'valid acc'])
        timer, num_batches = d2l.Timer(), len(train_iter)
        for epoch in range(num_epochs):
            metric = d2l.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (train_l, train_acc, None))
            valid_acc = evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'valid acc {valid_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')
    
    def get_sequence_index_dict(self,n = None, input_dict = None):
        # Given sequence length  
        # Given input dictionary   
          
        # Getting the original index  
        origin_index = len(input_dict.keys())  
          
        # Generating all possible sequences  
        bases = ['A', 'T', 'C', 'G']  # Assuming the sequence can consist of A, T, C, G  
        all_sequences = [''.join(seq) for seq in itertools.product(bases, repeat=n)]  
          
        # Creating the new dictionary with sequences and indices  
        new_dict = {sequence: origin_index + index for index, sequence in enumerate(all_sequences)}  
          
        # Merging the new dictionary with the input dictionary  
        input_dict.update(new_dict)  
    
    def get_gene_sequence_folder(self,input_gene_fa_path = None,output_folder = None):
        if output_folder is None:
            output_folder = input_gene_fa_path.replace('.fa','')
        if not os.path.exists(output_folder):  
            os.makedirs(output_folder) 
        with open(input_gene_fa_path, 'r',encoding='utf-8') as the_fa_file:   
            the_dict = {}
            tmp_gene = ''
            tmp_sequence = ''
            for line in the_fa_file: 
                if line.startswith('>'):
                    if tmp_gene != '': # save old,reset
                        if os.path.exists(output_folder+'/'+tmp_gene+'.txt'): # add a new line
                            with open(output_folder+'/'+tmp_gene+'.txt', 'a',encoding='utf-8') as tmp_file:  
                                # Write the string to the file  
                                tmp_file.write('\n'+tmp_sequence)
                        else:
                            with open(output_folder+'/'+tmp_gene+'.txt', 'w',encoding='utf-8') as tmp_file:  
                                # Write the string to the file  
                                tmp_file.write(tmp_sequence) 
                        # the_dict[tmp_gene]=tmp_sequence
                        tmp_gene = line.split(' ')[1].strip().replace('\n','')
                        tmp_sequence = ''
                    else: # the_start
                        tmp_gene = line.split(' ')[1].strip().replace('\n','')
                        tmp_sequence = ''
                else: # append the gene sequence
                    tmp_sequence = tmp_sequence+str(line.strip().replace('\n',''))
                    # print(line)  
            # return the_dict
        
    def get_gene_sequence_vector(self,input_folder=None, output_folder=None, vector_len=None, seq_dict=None):  
        tmp_dict = {i: 0 for i in range(vector_len)}  # Step 1  
        seq_count_dict = {j:0.0 for j in seq_dict.keys()}
        gene_seq_num_list = []
        count_count = 0
        if not os.path.exists(output_folder):  
            os.makedirs(output_folder) 
        # import os  
        for file_name in os.listdir(input_folder):  
            if file_name.endswith(".txt"):  
                file_path = os.path.join(input_folder, file_name)  
                with open(file_path, 'r') as file:  
                    lines = file.readlines()  
                    line_count = len(lines)
                    for line in lines:
                        # print(line)
                        for i in range(len(line) - 5):  # Step 2.1  
                            sequence = line[i:i+5]  
                            if sequence in seq_dict:  
                                pos = seq_dict[sequence]  
                                tmp_dict[pos] = tmp_dict[pos] +1  
                                seq_count_dict[sequence] = seq_count_dict[sequence]+1.0/float(line_count)
                                # seq_count_dict[sequence] = tmp_dict[pos]
                        for i in range(len(line) - 6):  # Step 2.1  
                            sequence = line[i:i+6]  
                            if sequence in seq_dict:  
                                pos = seq_dict[sequence]  
                                tmp_dict[pos] = tmp_dict[pos] +1  
                                seq_count_dict[sequence] = seq_count_dict[sequence]+1.0/float(line_count)
                                # seq_count_dict[sequence] = tmp_dict[pos]
                        for i in range(len(line) - 7):  # Step 2.1  
                            sequence = line[i:i+7]  
                            if sequence in seq_dict:  
                                pos = seq_dict[sequence]  
                                tmp_dict[pos] = tmp_dict[pos] +1  
                                seq_count_dict[sequence] = seq_count_dict[sequence]+1.0/float(line_count)
                                # seq_count_dict[sequence] = tmp_dict[pos]
                line_count = len(lines)  # Step 2.2  
                gene_seq_num_list.append(line_count)
                # print(len(lines))
                # for seqseq in seq_count_dict.keys():
                #     print(seqseq)
                #     print(seq_count_dict[seqseq])
                tmp_dict_values = list(tmp_dict.values())  
                tmp_dict_values = [float(val)/float(line_count) for val in tmp_dict_values]  
                output_file_path = os.path.join(output_folder, file_name)  
                with open(output_file_path, 'w') as output_file:  
                    output_file.write('\n'.join(map(str, tmp_dict_values)))  # Step 2.3  
                count_count = count_count+1
            # break
            # if count_count >=50:
            #     break
        # print('len of gene_seq_num_list')
        # gene_seq_num_list_len = len(gene_seq_num_list)
        # print(gene_seq_num_list_len)
        # print('max min mean median q1 q3 of gene_seq_num_list')
        # max_value = max(gene_seq_num_list)  
        # min_value = min(gene_seq_num_list)  
        # mean_value = statistics.mean(gene_seq_num_list)  
        # median_value = statistics.median(gene_seq_num_list)  
        # q1 = statistics.quantiles(gene_seq_num_list, n=4)[0]  
        # q3 = statistics.quantiles(gene_seq_num_list, n=4)[-1]  
        # print(max_value)  
        # print(min_value)  
        # print(mean_value)  
        # print(median_value)  
        # print(q1)  
        # print(q3)  
        # for seqseq in seq_count_dict.keys():
        #     print(seqseq)
        #     print(seq_count_dict[seqseq]/gene_seq_num_list_len)
        return gene_seq_num_list,seq_count_dict
    
    def get_all_files_under_a_folder(self,input_folder = None,end_str = None):
        file_list = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(end_str)]  
        return file_list
    
    def get_all_biotypes_from_gtf(self,input_gtf = None):
        tmp_set = set()
        with open(input_gtf, 'r',encoding='utf-8') as the_gtf_file:
            for line in the_gtf_file:
                tmp_set.add(line.split('biotype ')[1].split(';')[0].replace('\"','').strip())
        print(input_gtf)
        for i in tmp_set:
            print(i)
        print('\n')
        
    def change_gtf_to_specific_biotype(self,input_gtf = None,output_gtf = None,the_biotype = None):
        tmp_set = set()
        with open(input_gtf, 'r',encoding='utf-8') as the_gtf_file:
            for line in the_gtf_file:
                if str(line.split('biotype ')[1].split(';')[0].replace('\"','').strip()) == the_biotype:
                    tmp_set.add(line.strip())
        print(input_gtf)
        print(len(tmp_set))
        
    def save_big_adata_to_small_adata(self,input_adata_path=None, input_adata_name=None, small_adata_size=None, output_folder=None):
        if not os.path.exists(output_folder):  
            os.makedirs(output_folder) 
        max_count = 0
        big_adata = sc.read(input_adata_path + input_adata_name)
        try:
            del big_adata.raw
        except Exception:
            max_count = 0
        # Define the start and end cell indices  
        tmp_list = []
        max_count = 0
        for i in range(0,len(big_adata.obs_names)):
            tmp_count = i//small_adata_size+1
            tmp_list.append(tmp_count)
            if tmp_count>max_count:
                max_count = tmp_count
        big_adata.obs['split_group_number'] = tmp_list
        # start_cell_index = 0  
        # end_cell_index = start_cell_index + small_adata_size
        # Loop through the big Anndata and save smaller parts 
        end_count = 1
        while end_count <= max_count - 1:  
            # Extract a smaller part of the big Anndata  
            tmp_adata = big_adata[big_adata.obs['split_group_number'] == end_count, :]  
            tmp_adata.obs.reset_index(inplace=True)  
            tmp_adata.var.reset_index(inplace=True) 
            # Save the smaller Anndata  
            tmp_adata.write(output_folder + input_adata_name.replace('.h5ad','') + \
                            f"_from_{small_adata_size*(end_count-1)}_to_{small_adata_size*(end_count)}.h5ad")  
            end_count = end_count+1
            # Update indices for the next iteration  
            # start_cell_index = end_cell_index  
            # end_cell_index = min(start_cell_index + small_adata_size, big_adata.shape[0])  
            
    def change_adata_to_cell_vector(self,input_adata=None,gene_seq_vector_folder=None, 
                                    output_folder=None, if_change_var_names=None,
                                    the_length=None,):
        max_count = 0
        gene_seq_vector_dict = get_gene_seq_vector_dict(gene_seq_vector_folder=gene_seq_vector_folder)
        the_adata = sc.read(input_adata)
        # try:
        #     del the_adata.raw
        # except Exception:
        #     max_count = 0
        if if_change_var_names is True:
            the_adata.var_names = the_adata.var['features']
        if the_length is None:
            the_length = 50176
        print(the_adata.var_names)
        # print(the_adata.var['features'])
        the_adata.obs['cellname_tmp'] = the_adata.obs_names
        the_adata.var['genename_tmp'] = the_adata.var_names
        for i in the_adata.obs_names:
            tmp_vector = np.zeros(the_length)
            tmp_adata = the_adata[the_adata.obs['cellname_tmp']==i,:]
            print(len(tmp_adata.var_names))
            sc.pp.filter_genes(tmp_adata, min_cells=1)
            print(len(tmp_adata.var_names))
            for j in tmp_adata.var_names:
                # print(j)
                if j in gene_seq_vector_dict.keys():
                    # print(j)
                    # with open(gene_seq_vector_dict[j], 'r',encoding='utf-8') as file:
                    #     value_list = []
                    #     for line in file:
                    #         value_list.append(float(line.strip()))
                    #     data_array = np.array(value_list)
                        # print(data_array)
                        # print(type(tmp_adata[:,tmp_adata.var['genename_tmp']==j].X[0,0])) 
                    data_array = np.load(gene_seq_vector_dict[j])
                    mask_log10 = (np.arange(the_length) >= 1) & (np.arange(the_length) <= 1024)  
                    mask_loge = (np.arange(the_length) >= 1025) & (np.arange(the_length) <= 5120)  
                      
                    # Apply the logarithm operations selectively  
                    data_array[mask_log10] = data_array[mask_log10]+1
                    data_array[mask_loge] = data_array[mask_loge]+1
                    data_array[mask_log10] = np.log10(data_array[mask_log10])  
                    data_array[mask_loge] = np.log(data_array[mask_loge])  
                      
                    # Print the resulting array to verify  
                    # print(array)  
                    tmp_value = tmp_adata[:,tmp_adata.var['genename_tmp']==j].X[0,0]
                    if tmp_value>0.0:
                        tmp_vector = tmp_vector+data_array*tmp_value
                # break
            array_sum = np.sum(tmp_vector)
            tmp_vector = tmp_vector * 100000 / array_sum
            tmp_vector = tmp_vector+1
            tmp_vector = np.log(tmp_vector)
            # print(tmp_vector)
            break
            
    def change_adata_to_cell_vector_2(self,input_adata=None,gene_seq_vector_folder=None, 
                                    output_folder=None, if_change_var_names=None,
                                    the_length=None,):
        max_count = 0
        gene_seq_vector_dict = get_gene_seq_vector_dict(gene_seq_vector_folder=gene_seq_vector_folder)
        the_adata = sc.read(input_adata)
        # try:
        #     del the_adata.raw
        # except Exception:
        #     max_count = 0
        if if_change_var_names is True:
            the_adata.var_names = the_adata.var['features']
        if the_length is None:
            the_length = 50176
        print(the_adata.var_names)
        # print(the_adata.var['features'])
        the_adata.obs['cellname_tmp'] = the_adata.obs_names
        the_adata.var['genename_tmp'] = the_adata.var_names
        for i in the_adata.obs_names:
            tmp_vector = np.zeros(the_length)
            tmp_adata = the_adata[the_adata.obs['cellname_tmp']==i,:]
            print(len(tmp_adata.var_names))
            # sc.pp.filter_genes(tmp_adata, min_cells=1)
            print(len(tmp_adata.var_names))
            for j in tmp_adata.var_names:
                # print(j)
                tmp_value = tmp_adata[:,tmp_adata.var['genename_tmp']==j].X[0,0]
                if j in gene_seq_vector_dict.keys() and tmp_value>0.0:
                    # print(j)
                    # with open(gene_seq_vector_dict[j], 'r',encoding='utf-8') as file:
                    #     value_list = []
                    #     for line in file:
                    #         value_list.append(float(line.strip()))
                    #     data_array = np.array(value_list)
                        # print(data_array)
                        # print(type(tmp_adata[:,tmp_adata.var['genename_tmp']==j].X[0,0])) 
                    data_array = np.load(gene_seq_vector_dict[j])
                    mask_log10 = (np.arange(the_length) >= 1) & (np.arange(the_length) <= 1024)  
                    mask_loge = (np.arange(the_length) >= 1025) & (np.arange(the_length) <= 5120)  
                      
                    # Apply the logarithm operations selectively  
                    data_array[mask_log10] = data_array[mask_log10]+1
                    data_array[mask_loge] = data_array[mask_loge]+1
                    data_array[mask_log10] = np.log10(data_array[mask_log10])  
                    data_array[mask_loge] = np.log(data_array[mask_loge])  
                      
                    # Print the resulting array to verify  
                    # print(array)  
                    # tmp_value = tmp_adata[:,tmp_adata.var['genename_tmp']==j].X[0,0]
                    # if tmp_value>0.0:
                    tmp_vector = tmp_vector+data_array*tmp_value
                # break
            array_sum = np.sum(tmp_vector)
            tmp_vector = tmp_vector * 100000 / array_sum
            tmp_vector = tmp_vector+1
            tmp_vector = np.log(tmp_vector)
            # print(tmp_vector)
            break
    
    def get_gene_seq_vector_dict(self,gene_seq_vector_folder=None):
        tmp_dict = {}
        files = os.listdir(gene_seq_vector_folder)
        for file in files:
            if file.endswith('.npy'):
                tmp_dict[file.replace('.npy','')] = gene_seq_vector_folder+file
        return tmp_dict
    
    def folder_merge(self,folders=None, output_folder=None):
        for folder in folders:  
            files = os.listdir(folder)  
            for file in files:  
                shutil.copy(os.path.join(folder, file), os.path.join(output_folder, file))  
                
    def change_txt_vector_to_npy_vector(self,input_folder=None,output_folder=None):
        files = os.listdir(input_folder)
        for the_file in files:
            with open(os.path.join(input_folder, the_file), 'r',encoding='utf-8') as file:
                value_list = []
                for line in file:
                    value_list.append(float(line.strip()))
                data_array = np.array(value_list)
                np.save(os.path.join(output_folder, the_file.replace('.txt','.npy')),data_array)
                
    def compare_load_content_and_speed(self,input_txt=None, input_npy=None):
        with open(input_txt, 'r',encoding='utf-8') as file:
            value_list = []
            for line in file:
                value_list.append(float(line.strip()))
            data_array1 = np.array(value_list)
            data_array2 = np.load(input_npy)
            # Check if the arrays have the same shape and elements  
            result = np.array_equal(data_array1, data_array2)  
            print(result)  # This will print True if the arrays have the same values, and False otherwise 
            
        # Process 1  
        start_time_p1 = time.time()  
        with open(input_txt, 'r',encoding='utf-8') as file:
            value_list = []
            for line in file:
                value_list.append(float(line.strip()))
            data_array1 = np.array(value_list)  
        end_time_p1 = time.time()  
        time_taken_p1 = end_time_p1 - start_time_p1  
        print(f"Time taken by process 1: {time_taken_p1} seconds")  
          
        # Process 2  
        start_time_p2 = time.time()  
        data_array2 = np.load(input_npy) 
        end_time_p2 = time.time()  
        time_taken_p2 = end_time_p2 - start_time_p2  
        print(f"Time taken by process 2: {time_taken_p2} seconds")  
        
    def change_npy_vector_length(self,input_folder=None,the_length=None, start_pos=None, end_pos=None, output_folder=None):
        if not os.path.exists(output_folder):  
            os.makedirs(output_folder)
        if the_length is None:
            the_length = 50176
        files = os.listdir(input_folder)
        for the_file in files:
            if the_file.endswith('.npy'):
                data_array = np.load(os.path.join(input_folder, the_file))
                the_mask = (np.arange(the_length) >= start_pos) & (np.arange(the_length) <= end_pos)
                np.save(os.path.join(output_folder, the_file),data_array[the_mask])
                
    def change_adata_to_cell_vector_3(self,input_adata=None,gene_seq_vector_folder=None, 
                                      output_folder=None, if_change_var_names=None,
                                      the_length=None,# all_gene_seq_possibilities
                                      gene_batch_size=None,gene_seq_dict=None,obs_celltype_label=None):
        max_count = 0
        gene_seq_vector_dict = get_gene_seq_vector_dict(gene_seq_vector_folder=gene_seq_vector_folder)
        the_adata = sc.read(input_adata)
        # try:
        #     del the_adata.raw
        # except Exception:
        #     max_count = 0
        if if_change_var_names is True:
            the_adata.var_names = the_adata.var['features']
        if the_length is None:
            the_length = 50176
        if gene_batch_size is None:
            gene_batch_size = 1000
        print(the_adata.var_names)
        # print(the_adata.var['features'])
        # the_adata.obs['cellname_tmp'] = the_adata.obs_names
        # the_adata.var['genename_tmp'] = the_adata.var_names
        sc.pp.filter_genes(the_adata, min_cells=1)
        the_number_of_cells = len(the_adata.obs_names)
        var_len = len(the_adata.var_names)
        start_len = 0
        result_matrix = np.zeros((the_number_of_cells, the_length))
        matched_count = 0
        miss_count = 0
        miss_list = []
        while start_len<var_len:
            matrix2_blocks = []
            tmp_list = []
            # tmp_vector = np.zeros(the_length)
            for i in range(start_len,min(start_len+gene_batch_size,var_len)):
                tmp_gene = the_adata.var_names[i]
                tmp_list.append(tmp_gene)
                if tmp_gene in gene_seq_vector_dict.keys(): 
                    matrix2_blocks.append(np.load(gene_seq_vector_dict[tmp_gene]))
                    matched_count = matched_count+1
                else:
                    matrix2_blocks.append(np.zeros(the_length))
                    miss_count = miss_count+1
                    miss_list.append(tmp_gene)
            matrix1 = the_adata[:,tmp_list].X.todense()
            matrix2 = np.vstack(matrix2_blocks) 
            print(matrix1.shape)
            print(matrix2.shape)
            result_matrix = result_matrix + the_adata[:,tmp_list].X.todense() @ matrix2
            print(result_matrix.shape)
            start_len = start_len+gene_batch_size
        print(matched_count)
        print(miss_count)
        # for i in miss_list:
        #     print(i)
        result_matrix = result_matrix[:, :21504]
        the_df = pd.DataFrame(result_matrix, index=the_adata.obs_names, columns=list(the_dict.keys()))
        return the_df,miss_list,the_adata.obs[obs_celltype_label]
    
    def change_adata_to_cell_vector_4(self,input_adata_dir=None,input_adata_name=None,gene_seq_vector_folder=None, 
                                      output_folder=None, if_change_var_names=None,if_change_obs_names=None,
                                      the_length=None,result_matrix_length=None,# all_gene_seq_possibilities,
                                      gene_batch_size=None,gene_seq_dict=None,obs_celltype_label=None):
        max_count = 0
        gene_seq_vector_dict = self.get_gene_seq_vector_dict(gene_seq_vector_folder=gene_seq_vector_folder)
        the_adata = sc.read(input_adata_dir+input_adata_name)
        # try:
        #     del the_adata.raw
        # except Exception:
        #     max_count = 0
        if result_matrix_length is None:
            result_matrix_length=21504
        if output_folder is None:
            output_folder=input_adata_dir
        if not os.path.exists(output_folder):  
            os.makedirs(output_folder)
        if if_change_var_names is True:
            the_adata.var_names = the_adata.var['features']
        if if_change_obs_names is True:
            the_adata.obs_names = list(the_adata.obs['index'])
        if the_length is None:
            the_length = 50176
        if gene_batch_size is None:
            gene_batch_size = 1000
        print(the_adata.var_names)
        # print(the_adata.var['features'])
        # the_adata.obs['cellname_tmp'] = the_adata.obs_names
        # the_adata.var['genename_tmp'] = the_adata.var_names
        sc.pp.filter_genes(the_adata, min_cells=1)
        the_number_of_cells = len(the_adata.obs_names)
        var_len = len(the_adata.var_names)
        start_len = 0
        result_matrix = np.zeros((the_number_of_cells, the_length))
        matched_count = 0
        miss_count = 0
        miss_list = []
        while start_len<var_len:
            matrix2_blocks = []
            tmp_list = []
            # tmp_vector = np.zeros(the_length)
            for i in range(start_len,min(start_len+gene_batch_size,var_len)):
                tmp_gene = the_adata.var_names[i]
                tmp_list.append(tmp_gene)
                if tmp_gene in gene_seq_vector_dict.keys(): 
                    matrix2_blocks.append(np.load(gene_seq_vector_dict[tmp_gene]))
                    matched_count = matched_count+1
                else:
                    matrix2_blocks.append(np.zeros(the_length))
                    miss_count = miss_count+1
                    miss_list.append(tmp_gene)
            matrix1 = the_adata[:,tmp_list].X.todense()
            matrix2 = np.vstack(matrix2_blocks) 
            print(matrix1.shape)
            print(matrix2.shape)
            result_matrix = result_matrix + the_adata[:,tmp_list].X.todense() @ matrix2
            print(result_matrix.shape)
            start_len = start_len+gene_batch_size
        print(matched_count)
        print(miss_count)
        # for i in miss_list:
        #     print(i)
        result_matrix = result_matrix[:, :result_matrix_length]
        the_df = pd.DataFrame(result_matrix, index=the_adata.obs_names, columns=list(gene_seq_dict.keys()))
        the_df.to_csv(output_folder+input_adata_name.replace('.h5ad','_changed.csv'))  
        del the_df
        gc.collect()
        new_anndata = sc.read_csv(output_folder+input_adata_name.replace('.h5ad','_changed.csv'))  
        true_gene_list = list(new_anndata.var_names)[0:]
        print(true_gene_list[0])
        new_anndata.var['seq_names']=new_anndata.var_names
        new_anndata=new_anndata[:,new_anndata.var['seq_names'].isin(true_gene_list)]
        new_anndata.obs['celltype']=the_adata.obs[obs_celltype_label]
        
        print(new_anndata.obs_names)
        print(new_anndata.var_names)
        print(new_anndata.X.max())
        print(new_anndata.obs['celltype'])
        new_anndata.write(output_folder+input_adata_name.replace('.h5ad','_changed.h5ad'))
        
        # return the_df,miss_list,the_adata.obs[obs_celltype_label]

    def downsample_celltypes(self, new_anndata=None, n=None, celltype_label='celltype',random_seed=42):
        # downsample each celltype, max is n
        np.random.seed(random_seed)
        celltype_counts = new_anndata.obs[celltype_label].value_counts()
        re_anndata = None
        for celltype, count in celltype_counts.items():
            if count > n:
                indices_to_keep = np.random.choice(
                    new_anndata.obs.index[new_anndata.obs[celltype_label] == celltype],
                    size=n,  
                    replace=False,)
                if re_anndata is None:
                    re_anndata = new_anndata[new_anndata.obs.index.isin(indices_to_keep)]
                else:
                    re_anndata = re_anndata.concatenate(new_anndata[new_anndata.obs.index.isin(indices_to_keep)], index_unique=None)
            else:
                if re_anndata is None:
                    re_anndata = new_anndata[new_anndata.obs.index.isin(indices_to_keep)]
                else:
                    re_anndata = re_anndata.concatenate(new_anndata[new_anndata.obs[celltype_label] == celltype], index_unique=None)
                    
        return re_anndata

    def split_anndata_by_celltype_to_train_valid_test(self, new_anndata=None, celltype_label='celltype', split_ratio=[0.8,0.1,0.1], random_seed=42):  
        # split adata to three part by given ratio
        # import scanpy as sc  
        # import numpy as np 
        np.random.seed(random_seed)
        train_anndata = None
        valid_anndata = None
        test_anndata = None
        
        for celltype in new_anndata.obs[celltype_label].unique(): 
            celltype_indices = new_anndata.obs[celltype_label] == celltype  
            celltype_data = new_anndata[celltype_indices]  
            
            total_cells = len(celltype_data)  
            x, y, z = split_ratio  
            x_count = int(total_cells * x / (x + y + z))  
            y_count = int(total_cells * y / (x + y + z))  
            z_count = total_cells - x_count - y_count  
            
            indices = np.random.permutation(total_cells)  
            train_indices = indices[:x_count]  
            valid_indices = indices[x_count:x_count + y_count]  
            test_indices = indices[x_count + y_count:]  
            
            train_data = celltype_data[train_indices]  
            valid_data = celltype_data[valid_indices]  
            test_data = celltype_data[test_indices]  
            
            if train_anndata is None:
                train_anndata = train_data
            else:
                train_anndata = train_anndata.concatenate(train_data, index_unique=None)  
            if valid_anndata is None:
                valid_anndata = valid_data
            else:
                valid_anndata = valid_anndata.concatenate(valid_data, index_unique=None)   
            if test_anndata is None:
                test_anndata = test_data
            else:
                test_anndata = test_anndata.concatenate(test_data, index_unique=None)   
            
        return train_anndata, valid_anndata, test_anndata  

    def save_to_pkl(obj=None, name=None):
        with open(name,'wb') as f:
            pickle.dump(obj,f)

    def load_pkl(name=None):
        with open(name,'rb') as f:
            obj = pickle.load(f)
            return obj
    