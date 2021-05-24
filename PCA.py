import numpy as np
import pandas as pd
from scipy import linalg
import scipy.stats as stats

def pca(x):
    '''Runs PCA analysis through a variety of steps
    1. drops NA values from data frame and transforms it into a numpy dataset 
    2. Standardizes data variables within dataset
    3. Creates covariance/correlation matrices 
    4. Extracts eigenvalues and eigenvectors 
    5. Removes imaginary numbers and computes diagonals
    Input: Dataset that has no qualitative data, only numeric data will work 
    Output: Data matrix with respective PC values that are NOT necessarily in order from greatest to least variance, the eigenvectors, and the normalized and transformed datasets that are used later in the code'''
    df = x.dropna()
    data = df.values
    data_norm = (data - np.mean(data, axis=0))/np.std(data, axis = 0, ddof=1) #standardizes data
    R = np.cov(data_norm, rowvar=False) #creates covariance matrix 
    val, vec = linalg.eig(R) #extracts eigenvalues and eigenvectors respectvely 
    value = np.diag(np.real(val)) #removes imaginary vaues and computes PC matrices via diagonals 
    return (value, vec, data_norm) #returns PC matrix NOT necessarily in order from greatest to least