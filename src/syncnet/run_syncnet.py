#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob

from syncnet.SyncNetInstance import *
import warnings
warnings.filterwarnings("ignore")
# ==================== PARSE ARGUMENT ====================
'''
parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='data/work', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();


'''

# ==================== LOAD MODEL AND FILE LIST ====================
def run_syncnet(opt):
    setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
    setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
    setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
    setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))

    s = SyncNetInstance();

    s.loadParameters(opt.initial_model);
    #print("Model %s loaded."%opt.initial_model);

    flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
    flist.sort()

    # ==================== GET OFFSETS ====================

    dists = []
    offset = -1
    conf = -1
    for idx, fname in enumerate(flist):
        offset, conf, dist = s.evaluate(opt,videofile=fname)
        dists.append(dist)
        
    # ==================== PRINT RESULTS TO FILE ====================
    return offset, conf, dists
