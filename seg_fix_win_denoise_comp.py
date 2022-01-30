#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:25:14 2021

@author: amitkumarbhuyan
"""

import numpy as np
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import time

# Speech segmentation based on BIC
def calculate_BIC(mfcc_v, delta):
    m, n = mfcc_v.shape

    sigma0 = np.cov(mfcc_v).diagonal()
    eps = np.spacing(1)
    realmin = np.finfo(np.double).tiny
    det0 = max(np.prod(np.maximum(sigma0, eps)), realmin)

    flat_start = 5

    range_loop = range(flat_start, n, delta)
    x = np.zeros(len(range_loop))
    iter = 0
    for index in range_loop:
        part1 = mfcc_v[:, 0:index]
        part2 = mfcc_v[:, index:n]

        sigma1 = np.cov(part1).diagonal()
        sigma2 = np.cov(part2).diagonal()

        det1 = max(np.prod(np.maximum(sigma1, eps)), realmin)
        det2 = max(np.prod(np.maximum(sigma2, eps)), realmin)

        BIC = 0.5*(n*np.log(det0)-index*np.log(det1)-(n-index)*np.log(det2))-0.5*(m+0.5*m*(m+1))*np.log(n)
        #print("BIC: ",BIC)
        x[iter] = BIC
        iter = iter + 1

    maxBIC = x.max()
    maxIndex = x.argmax()
    if maxBIC > 0:
        return range_loop[maxIndex]-1, len(range_loop)
    else:
        return -1, len(range_loop)


def segment_speech(mfccs,win,delta):
    wStart = 0
    wEnd = win
    wGrow = win

    m, n = mfccs.shape

    store_cp = []
    BICl = 0
    index = 0
    while wEnd < n:
        featureSeg = mfccs[:, wStart:wEnd]
        detBIC,lBIC = calculate_BIC(featureSeg, delta)
        BICl = BICl + lBIC
        index = index + 1
        if detBIC > 0:
            temp = wStart + detBIC
            store_cp.append(temp)
            wStart = wStart + detBIC + win
            wEnd = wStart + wGrow
        else:
            wEnd = wEnd + wGrow

    return np.array(store_cp), BICl
    

def seg_with_silence(file, sr, srn, frame_size, frame_shift, win, delta, plot_seg=False):
    y, sr = librosa.load(file, sr=sr)
    nois = y[0:int(sr)]                          
    denoise_y = nr.reduce_noise(audio_clip=y, noise_clip=nois, verbose=True)
    
    #Noise Reduction for silence calculation
    z, srn = librosa.load(file, srn)
    noisy_part = z[0:int(srn)]                      
    reduced_noise = nr.reduce_noise(audio_clip=z, noise_clip=noisy_part, verbose=True)
    interval = librosa.effects.split(reduced_noise, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
    #Delete conversations of length less than 15% of a second
    intervaldiff = interval[:,1] - interval[:,0]
    decider = intervaldiff < int(round((srn*15)/100))
    fininterval = np.delete(interval,decider,0)

    m, n = fininterval.shape
    chg_pt = []
    sil_list = []
    nBIC = 0
    dur = 0
    
    #Start timer
    t_start = time.process_time()
    
    for i in range(m-1):
        #convert the centre silence points from sampling rate of 96000 to 16000
        sil = int(round((int(round(fininterval[i,1] + fininterval[i+1,0])/2) * sr) / srn))  
        start = sil - 14000
        if start < 0: start = 0
        end = sil + 14000
        if end > (len(y)-1): end = (len(y)-1)
        mfccs = librosa.feature.mfcc(denoise_y[start:end], sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
        seg_point,BICn = segment_speech(mfccs / mfccs.max(),win, delta)
        seg_point = seg_point * frame_shift
        nBIC = nBIC + BICn
        seg_point = seg_point + start
        seg = seg_point.tolist()
        chg_pt  = chg_pt + seg
        sil_list.append(sil/sr)
        
    dur = time.process_time() - t_start
    # Stop timer  
    
    if plot_seg:
        plt.figure('speech segmentation plot')
        plt.plot(np.arange(0, len(y)) / (float)(sr), y, "b-")
    
        for i in range(len(chg_pt)):
            plt.vlines(chg_pt[i] / (float)(sr), -1, 1, colors="r", linestyles="dashed")
        plt.xlabel("Time/s")
        plt.ylabel("Speech Amp")
        plt.grid(True)
        plt.show()
        
    return np.asarray(chg_pt) / float(sr) , sil_list, dur, nBIC
        
        
def seg_wo_silence(file, sr, frame_size, frame_shift, win, delta, plot_seg=False, save_seg=False, cluster_method=None):
    y, sr = librosa.load(file, sr=sr)
    #Remove noise from the signal 
    nois = y[0:int(sr)]  
    denoise_y = nr.reduce_noise(audio_clip=y, noise_clip=nois, verbose=True)
    nBIC = 0
    #Start timer
    t_start = time.process_time()
    
    mfccs = librosa.feature.mfcc(denoise_y, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
    m, n = mfccs.shape
    seg_point,nBIC = segment_speech(mfccs / mfccs.max(),win, delta)

    seg_point = seg_point * frame_shift
    
    dur = time.process_time() - t_start
    # Stop timer

    if plot_seg:
        plt.figure('speech segmentation plot')
        plt.plot(np.arange(0, len(y)) / (float)(sr), y, "b-")

        for i in range(len(seg_point)):
            plt.vlines(seg_point[i] / (float)(sr), -1, 1, colors="c", linestyles="dashed")
            plt.vlines(seg_point[i] / (float)(sr), -1, 1, colors="r", linestyles="dashed")
        plt.xlabel("Time/s")
        plt.ylabel("Speech Amp")
        plt.grid(True)
        plt.show()
        
    return np.asarray(seg_point) / float(sr), len(y)/16000, dur, nBIC