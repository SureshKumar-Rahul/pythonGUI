# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:07:54 2019

@author: jozsef.suto
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import kurtosis
from random import randint
import copy
from statsmodels.tsa.ar_model import AR
from statsmodels.graphics.tsaplots import plot_acf
import sympy
import time
import scipy.signal as signal

classes = 10
features = 12
windowsize = 256
dt = 256

def get_params_from_file(path):
    """
    Given a file with path, return its content.

    Parameters
    ----------
    path : string
      file path

    Returns
    -------
    res : array list
      file content
    """
    h1 = []
    #h2 = []
    r = []

    with open(path) as f:
        dataContent = f.readlines()

    for row in dataContent:
        cols = row.split(' ')
        h1.append(float(cols[1]))
        #h2.append(float(cols[2]))
        r.append(float(cols[2]))
        
    return r, h1

def get_file_content(path):
    """
    Given a file with path, return its content.

    Parameters
    ----------
    path : string
      file path

    Returns
    -------
    res : array list
      file content
    """
    data = []

    with open(path) as f:
        dataContent = f.readlines()
        
    for row in dataContent:
        if row:
            cols = row.split(',')
            data.append(np.asarray([float(i) for i in cols]))
            #np.reshape([float(i) for i in cols], (-1,1)
    return np.asarray(data)

def read_files(indx):
    """
    Given a file with path, return its content.

    Parameters
    ----------
    path : string
      file path

    Returns
    -------
    res : array list
      file content
    """
    data = []
    
    if indx == 0:
        name = 'SutoJ_'
    elif indx == 1:
        name = 'SutoJb_'
    elif indx == 2:
        name = 'Erika_'
    elif indx == 3:
        name = 'Zsanett_'
        
    for i in range(10):
        d = get_file_content('d:/python/eeg/emotiv/data/eeg/' + 
                             name + str(i) + '.txt')
        data.append(d[640:-512])
    return np.asarray(data)

def transform_data(data, firfilter=None):
    p = 0.2
    banddata = []
    Fi = []
    Li = []
    
    if firfilter is not None:
        for i in range(len(data)):
            temp = np.transpose(data[i])
            for j in range(len(temp)):
                temp[j] = signal.convolve(temp[j], firfilter, 'same')
            banddata.append(np.transpose(temp))
    else:
        banddata = data
        
    for i in range(len(banddata)):
        Fi.append(windowing(banddata[i]))
        Li.append(i*np.ones(len(Fi[len(Fi)-1])))
      
    F = np.concatenate((Fi[0], Fi[1]), axis=0)
    L = np.concatenate((Li[0], Li[1]), axis=0)
    for i in range(2, len(Fi)):
        F = np.concatenate((F, Fi[i]), axis=0)
        L = np.concatenate((L, Li[i]), axis=0)
    F = normalize(F)
    mixrand(F, L)
    
    traindata = F[0:int(len(F) * (1.0 - p))]
    testdata = F[int(len(F) * (1.0 - p)):len(F)]
    trainlabels = L[0:int(len(L) * (1.0 - p))]
    testlabels = L[int(len(L) * (1.0 - p)):len(L)]
    return (traindata, trainlabels, testdata, testlabels)

def windowing(data):
    windows = int((len(data) - windowsize) / dt + 1)
    cols = np.size(data,1)
    F = np.zeros((windows, cols * features))    
    startindex = 0
    
    for i in range(0, windows):
        w = data[startindex:(startindex + windowsize)]
        F[i] = np.ravel([ext_features(w[:,i]) for i in range(cols)])
        startindex += dt
    #F = normalize(F)
    return F

def ext_features(w):
    f = np.zeros(features)
    fq = np.fft.fft(w)
    fq = np.array(np.absolute(fq)[1:int(len(w) / 2 + 1)])
    ar = ar_coefficients(w)
    f[0] = np.mean(w)       
    f[1] = np.std(w)
#   f[2] = np.median(w)
    f[2] = np.mean(np.absolute(w - np.mean(w)))             #MAD
#    f[3] = np.sqrt(np.mean(w**2))                           #RMS
    f[3] = np.percentile(w, 75) - np.percentile(w, 25)      #IQR
    f[4] = np.percentile(w, 75)                             #P75
    f[5] = kurtosis(w)
    #f[6] = np.mean(np.absolute(w))                          #SMA
    f[6] = np.max(w) - np.min(w)                            #RANGE
    #f[8] = np.size(np.nonzero(np.diff(w > 0)))              #Zero Crossing
    f[7] = np.sum(fq**2)
    f[8] = spectral_centroid(fq)
    f[9] = np.max(fq)
    f[10] = ar[0]
    f[11] = ar[1]
#    f[13] = spectral_entropy(fq)
    return f
    
def spectral_centroid(fq):
    c = [((i+1) * (fq[i])) for i in range(len(fq))]
    if np.sum(fq) > 0: 
        return np.sum(c) / np.sum(fq)   # return weighted mean
    else:
        return 0.0     

def ar_coefficients(data):
    model = AR(data)
    fit_model = model.fit()
    #print('Lag: %s' % fit_model.k_ar)
    #print('Coefficients: %s' % fit_model.params[0:fit_model.k_ar]) 
    return fit_model.params[0:fit_model.k_ar]    

def mixrand(data, labels):
    for i in range(len(data)):
        r = randint(0, len(data) - 1)
        datatmp = copy.deepcopy(data[i])
        labeltmp = copy.deepcopy(labels[i])
        data[i] = data[r]
        labels[i] = labels[r]
        data[r] = datatmp
        labels[r] = labeltmp

def normalize(data):
    """
    Given an array list, normalizes it with standard scale.

    Parameters
    ----------
    data: array list
    """
    ndata = np.zeros((np.size(data, 0), np.size(data, 1)))
    for i in range(np.size(data, 1)):
        m = np.mean(data[:,i])
        std = np.std(data[:,i])
        ndata[:,i] = ((data[:,i] - m) / std) if std > 0 else 0.0
    return ndata

def split_data(data): 
    p = 0.2
    traindata = []
    testdata = []
    for i in data:
        traindata.append(i[0:int(len(i) * (1.0 - p))])
        testdata.append(i[int(len(i) * (1.0 - p)):len(i)])
    return np.asarray(traindata), np.asarray(testdata)


def merge_data(data1, data2):
    """
    Given two array list, merge their content list by list.

    Parameters
    ----------
    data1 : array list
    data2 : array list
    """
    data = []
    if len(data1) == len(data2):
        for i in range(len(data1)):
            data.append(np.concatenate((data1[i], data2[i]), axis=0))
    return np.asarray(data)
    
def echelon(X):
    start_time = time.time()
    M = sympy.Matrix(X[0:6000])
    print('M: ', M.shape)
    enc = M.rref()
    print('%.3f'%(time.time() - start_time))
    return enc
    
def firwdsize(ftype, fl, fh, fs):
    """
    Determines filter size to the given window.

    Parameters
    ----------
    ftype: int
        window type
        0-rectangular
        1-hanning
        2-hamming
        3-blackman
    fl : lower frequency
    fh : upper frequency
    fs : sampling frequency
    """
    df = (fh - fl) / fs
    
    if ftype == 1:
        s = 3.1 / df
    elif ftype == 2:
        s = 3.3 / df
    elif ftype == 3:
        s = 5.5 / df
    else:
        s = 0.9 / df 
    
    s = math.ceil(s)
    if s % 2 == 0:
        return s + 1
    else:
        return s
    
def firwd(filtertype, win, fl, fh, fs):
    """
    Determines filter coefficients.

    Parameters
    ----------
    size: int
    filtertype: int
        0-lowpass
        1-highpass
        2-bandpass
        3-bandstop
    win: int
        window type
        0-rectangular
        1-hanning
        2-hamming
        3-blackman
    fl: float
        lower frequency
    fh: float
        upper frequency
    fs: float
        sampling frequency
    """
    size = firwdsize(filtertype, fl, fh, fs)
    M = int((size - 1) / 2);
    o1 = (2.0 * np.pi * fl) / fs;
    o2 = (2.0 * np.pi * fh) / fs;
    h = np.zeros(M + 1);
    w = np.ones(M + 1);
    hw = np.zeros(size)
    print('Filter size: ', size)
    
    if (size & 1) > 0:
        for n in range(M, 0, -1):
            if filtertype == 0:
                h[M-n] = np.sin(o1 * n) / (n * np.pi)
            elif filtertype == 1:
                h[M-n] = -np.sin(o1 * n) / (n * np.pi)
            elif filtertype == 2:
                h[M-n] = (np.sin(o2 * n) - np.sin(o1 * n)) / (n * np.pi)
            elif filtertype == 3:
                h[M-n] = (-np.sin(o2 * n) + np.sin(o1 * n)) / (n * np.pi)
            else:   
                h[M-n] = 0.0
        
        for n in range(M, -1, -1):
            if win == 1:
                w[M-n] = 0.5 + 0.5 * np.cos((n * np.pi) / M)
            elif win == 2:
                w[M-n] = 0.54 + 0.46 * np.cos((n * np.pi) / M)
            elif win == 3:
                w[M-n] = 0.42 + 0.5 * np.cos((n * np.pi) / M) + 0.08 * np.cos((2 * n * np.pi) / M)
            else:   
                w[M-n] = 1.0
        
        if filtertype == 0:
            h[M] = o1 / np.pi
        elif filtertype == 1:
            h[M] = (np.pi - o1) / np.pi
        elif filtertype == 2:
            h[M] = (o2 - o1)/ np.pi
        elif filtertype == 3:
            h[M] = (np.pi - o2 + o1) / np.pi
        else:   
            h[M] = 0.0
        
        hw[0:M+1] = np.multiply(h, w)
        hw[M+1:size] = hw[M-1::-1]
        return hw
    
def draw_freq_response(filters):
    fs = 128
    markers = ['b--', 'r:', 'g-.', 'k', 'c-']
    
    for i in range(len(filters)):    
        w, h = signal.freqz(filters[i])
        plt.plot(fs * (w / (2*np.pi)), 20*np.log10(np.abs(h)), markers[i])
    
    plt.ylabel('Amplitude response (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.grid()
    
    plt.text(-1, 2, r'$\delta_{f}$', fontsize=9)
    plt.text(5, 2, r'$\theta_{f}$', fontsize=9)
    plt.text(8, 2, r'$\alpha_{f}$', fontsize=9)
    plt.text(20, 2, r'$\beta_{f}$', fontsize=9)
    plt.text(65, 2, r'$\gamma_{f}$', fontsize=9)
    plt.show()
    
def draw_subplots(data):
    """
    Given a list, draws its content.

    Parameters
    ----------
    data : list
    """
    titlesize = 9
    f, axarr = plt.subplots(len(data), sharex=True)
    
    for i in range(len(data)):
        axarr[i].plot(range(len(data[i])), data[i], 'b-', range(len(data[i])), 
                      data[i], 'b', linewidth=1.0)
        axarr[i].set(ylabel='Amp.')
        axarr[i].set_title('Channel ' + str(i+1), fontsize=titlesize)
        axarr[i].grid(True)
    plt.subplots_adjust(hspace=0.5)
    plt.xlabel('Samples', fontsize=13)
    plt.show()
    
def drawsubplots(data0, data, data1, data2, data3, data4):
    """
    Given a list, draws its content.

    Parameters
    ----------
    data : list
    """
    titlesize = 12
    f, axarr = plt.subplots(6, sharex=True)
    axarr[0].plot(range(len(data0)), data0, 'k', range(len(data0)), data0, 'b', linewidth=1.0)
    axarr[0].set(ylabel='Amplitude')
    axarr[0].set_title('Origina signal', fontsize=titlesize)
    axarr[0].grid(True)
    axarr[1].plot(range(len(data)), data, 'k', range(len(data)), data, 'b', linewidth=1.0)
    axarr[1].set(ylabel='Amplitude')
    axarr[1].set_title(r'$\delta$ band signal', fontsize=titlesize)
    axarr[1].grid(True)
    axarr[2].plot(range(len(data1)), data1, 'k', range(len(data1)), data1, 'b', linewidth=1.0)
    axarr[2].set(ylabel='Amplitude')
    axarr[2].set_title(r'$\theta$ band signal', fontsize=titlesize)
    axarr[2].grid(True)
    axarr[3].plot(range(len(data2)), data2, 'k', range(len(data2)), data2, 'b', linewidth=1.0)
    axarr[3].set(ylabel='Amplitude')
    axarr[3].set_title(r'$\alpha$ band signal', fontsize=titlesize)
    axarr[3].grid(True)
    axarr[4].plot(range(len(data3)), data3, 'k', range(len(data3)), data3, 'b', linewidth=1.0)
    axarr[4].set(ylabel='Amplitude')
    axarr[4].set_title(r'$\beta$ band signal', fontsize=titlesize)
    axarr[4].grid(True)
    axarr[5].plot(range(len(data4)), data4, 'k', range(len(data4)), data4, 'b', linewidth=1.0)
    axarr[5].set(xlabel='Samples', ylabel='Amplitude')
    axarr[5].set_title(r'$\gamma$ band signal', fontsize=titlesize)
    axarr[5].grid(True)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
def draw_rec(data):
    """
    Given a list, draws its content.

    Parameters
    ----------
    data : list
    """
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    xy = sorted(zip(x,y))
    x = [i[0] for i in xy]
    y = [i[1] for i in xy]
    plt.plot(np.log10(x), y, 'b.', np.log10(x), y, 'b:', linewidth=1.3)
    plt.grid(True)
    plt.xlabel('Trials with random learning rate')
    plt.ylabel('Recognition accuracy (%)')
    plt.show()
    
def draw_res(full, fh, delta, dh, theta, th, alpha, ah, beta, bh, gamma, gh):
    """
    Given a list, draws its content.

    Parameters
    ----------
    data : list
    """
    xy = sorted(zip(fh, full))
    x = [i[0] for i in xy]
    y = [i[1] * 100 for i in xy]
    plt.plot(np.log10(x), y, 'b--',  linewidth=1.3)
    
    xy = sorted(zip(dh, delta))
    x = [i[0] for i in xy]
    y = [i[1] * 100 for i in xy]
    plt.plot(np.log10(x), y, 'g--', linewidth=1.3)
    
    xy = sorted(zip(th, theta))
    x = [i[0] for i in xy]
    y = [i[1] * 100 for i in xy]
    plt.plot(np.log10(x), y, 'y--', linewidth=1.3)
    
    xy = sorted(zip(ah, alpha))
    x = [i[0] for i in xy]
    y = [i[1] * 100 for i in xy]
    plt.plot(np.log10(x), y, 'c--', linewidth=1.3)
    
    xy = sorted(zip(bh, beta))
    x = [i[0] for i in xy]
    y = [i[1] * 100 for i in xy]
    plt.plot(np.log10(x), y, 'k--', linewidth=1.3)
    
    xy = sorted(zip(gh, gamma))
    x = [i[0] for i in xy]
    y = [i[1] * 100 for i in xy]
    plt.plot(np.log10(x), y, 'r--', linewidth=1.3)
    plt.legend(['full_specturm', 'delta', 'theta', 'alpha', 'beta', 'gamma'])
    plt.grid(True)
    plt.xlabel('Trials with random learning rate (log10(x))', fontsize=18)
    plt.ylabel('Recognition accuracy (%)', fontsize=18)
    plt.show()
    
def draw_autocorr(data):
    plot_acf(data, lags=30, zero=False)
    
    
def generate_data(sub_index, fir=None):
    np.random.seed(1990)
    data = read_files(sub_index)  
    #data2 = read_files(1)
    #data = merge_data(data1, data2)
    print('Data reading complete. Tensor size: ', data.shape)
    #train, test = split_data(data)
    #print('Data separation complete.')
    #print('Tr size: ', train.shape, ' Test size: ', test.shape)
    (traind, trainl, testd, testl) = transform_data(np.asarray(data), fir)
    print('Training data transformation complete.')
    print('Data size: ', traind.shape)
    #testd, testl = transform_data(test, fir)
    print('Test data transformation complete.')
    print('Data size: ', testd.shape)
    return traind, trainl.astype(int), testd, testl.astype(int)