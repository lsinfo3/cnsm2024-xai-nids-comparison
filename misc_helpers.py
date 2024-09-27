# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:29:48 2024

@author: katha
"""
import pandas as pd
import torch.nn as nn
from scipy import stats
from scipy.cluster import hierarchy
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
import re

# simple NN -> SLP
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# DNN (still simple, but technically deep) -> MLP
class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    
# Multicollinearity Filter -> see noms2022 paper: https://github.com/lsinfo3/noms2022-sdn-performance-prediction
# not needed here though for now; alternative feature selection mechanism
class MulticollinearityFilter(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.selected_features = []
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr = stats.spearmanr(X_df).correlation
        corr_linkage = hierarchy.ward(corr)

        cluster_ids = hierarchy.fcluster(corr_linkage, self.n_clusters, criterion='maxclust')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
    
        self.selected_features = list(X_df.columns[[v[0] for v in cluster_id_to_feature_ids.values()]])
        
        return self 
    
    def transform(self, X, y=None):
        X_df = pd.DataFrame(X)
        X_df = X_df[self.selected_features]
        return X_df
    
# we need this for the CIDDS dataset, cause it has stuff like "1M" instead of 1000000 etc.
def convert_suffix_to_int(value):
    if isinstance(value, str):
        value = value.strip()
        if value.endswith('M'):
            return int(float(value[:-1]) * 1000000)
        elif value.endswith('K'):
            return int(float(value[:-1]) * 1000)
    return value


