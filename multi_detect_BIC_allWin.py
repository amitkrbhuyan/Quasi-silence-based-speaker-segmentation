#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:23:36 2021

@author: amitkumarbhuyan
"""

from __future__ import print_function
import numpy as np
import seg_fix_win_denoise_comp as bic_seg

frame_size = 256
frame_shift = 128
sr = 16000
srn = 44100
without = []
withsil = []
without1 = np.zeros((4,3,3))
withsil1 = np.zeros((4,3,3))
bas_dur = []
pro_dur = []
bas_BIC = []
pro_BIC = []
base_dur = np.zeros((4,3))
prop_dur = np.zeros((4,3))
base_BIC = np.zeros((4,3))
prop_BIC = np.zeros((4,3))

for win in range(100,175,25):
    for delta in range(int(0.2*win),int(win),int(0.2*win)):
        chg_pt, silences, p_dur, p_BIC = bic_seg.seg_with_silence("/Users/amitkumarbhuyan/Documents/MSU/CODE/filed codes/py_speech_seg-master-new/downloaded conv/shop0101.mp3", sr, srn, frame_size, frame_shift, win, delta, plot_seg=True)
        seg_point, length, b_dur, b_BIC = bic_seg.multi_segmentation("/Users/amitkumarbhuyan/Documents/MSU/CODE/filed codes/py_speech_seg-master-new/downloaded conv/shop0101.mp3", sr, frame_size, frame_shift, win, delta, plot_seg=True, save_seg=False,
                                           cluster_method='bic')
    
        bas_dur.append(b_dur)
        pro_dur.append(p_dur)
        bas_BIC.append(b_BIC)
        pro_BIC.append(p_BIC)
        
        print('\n Length of the audio ',length)
        print('\n {} segmentation point for this audio file with silence (Unit: /s) {}'.format(len(chg_pt), chg_pt))
        print('\n {} segmentation point for this audio file without silence (Unit: /s) {}'.format(len(seg_point), seg_point))
        
        true_chg = [6.22875283446712,9.618866213151927,12.1556462585034,14.587936507936508,16.306213151927437]
        tp_s = 0  
        tp_s = len([i for i in true_chg if np.any(abs(i-chg_pt) < 0.5)])
        fn_s = len(true_chg)-tp_s
        fp_s = len(chg_pt)-tp_s
        tp_w = 0  
        tp_w = len([i for i in true_chg if np.any(abs(i-seg_point) < 0.5)])
        fn_w = len(true_chg)-tp_w
        fp_w = len(seg_point)-tp_w
        with_out = [len(seg_point),fp_w,fn_w]
        with_sil = [len(chg_pt),fp_s,fn_s]
        without = without + with_out
        withsil = withsil + with_sil
    
    without1[:,:,int((win-100)/25)] = np.asarray(without).reshape((-1,3))  
    withsil1[:,:,int((win-100)/25)] = np.asarray(withsil).reshape((-1,3))
    base_dur[:,int((win-100)/25)] = np.asarray(bas_dur)
    prop_dur[:,int((win-100)/25)] = np.asarray(pro_dur)
    base_BIC[:,int((win-100)/25)] = np.asarray(bas_BIC)
    prop_BIC[:,int((win-100)/25)] = np.asarray(pro_BIC)
    without = []
    withsil = []
    bas_dur = []
    pro_dur = []
    bas_BIC = []
    pro_BIC = []
    
without2 = np.concatenate((without1[:,:,0],without1[:,:,1],without1[:,:,2]),axis = 0)
withsil2 = np.concatenate((withsil1[:,:,0],withsil1[:,:,1],withsil1[:,:,2]),axis = 0)
base_dur2 = np.concatenate((base_dur[:,0],base_dur[:,1],base_dur[:,2]),axis = 0)
prop_dur2 = np.concatenate((prop_dur[:,0],prop_dur[:,1],prop_dur[:,2]),axis = 0)
base_BIC2 = np.concatenate((base_BIC[:,0],base_BIC[:,1],base_BIC[:,2]),axis = 0)
prop_BIC2 = np.concatenate((prop_BIC[:,0],prop_BIC[:,1],prop_BIC[:,2]),axis = 0)