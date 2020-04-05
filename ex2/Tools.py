import numpy as np
from scipy.stats import zscore

'''
The function change the first column from string letter for number with one hot vector, 
and change all the rest string number to float. 
the function add bais = 1
'''
def change_for_matrix(data):
    m = [0, 0, 1]
    F = [0, 1, 0]
    I = [1, 0, 0]
    n_feats = np.zeros((np.size(data, 0), np.size(data, 1) + 3))
    for i, f in enumerate(data):
        if f[0] == 'M':
            f[0] = 1
            n_feats[i] = np.concatenate((f, m), axis=None)
        if f[0] == 'F':
            f[0] = 1
            n_feats[i] = np.concatenate((f, F), axis=None)
        if f[0] == 'I':
            f[0] = 1
            n_feats[i] = np.concatenate((f, I), axis=None)
    return np.array(n_feats.astype(float))

'''
The function normalization the data by z- score algorithm
'''
def z_score_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for i,x in enumerate(std):
        if x==0.0:
            std[i]=1.0
    data=((data - mean) / std)
    return np.abs(data)

'''
The function normalization the data by min-max algorithm
'''
def min_max_normalization(data):
    data = np.transpose(data)
    for i, line in enumerate(data):
        if line.max() != line.min():
            data[i] = (line - line.min()) / (line.max() - line.min())
    return np.transpose(data)

'''
The function normalization all the data between 0 to 1
'''
def norm(data):
    return data/np.max(data,axis=0)