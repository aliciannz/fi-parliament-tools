#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob

from syncnet.SyncNetInstance import *
import warnings
warnings.filterwarnings("ignore")
# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='data/work', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ==================== LOAD MODEL AND FILE LIST ====================
def run_syncnet(opt):
    s = SyncNetInstance();

    s.loadParameters(opt.initial_model);
    print("Model %s loaded."%opt.initial_model);

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
    with open(os.path.join(opt.work_dir,opt.reference,'results.txt'), 'w') as fil:
        print(os.path.join(opt.work_dir,opt.reference,'results.txt'))
        results = f'Offset:{offset}, Confidence:{conf}'
        fil.write(results)

    with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
        pickle.dump(dists, fil)
    return offset, conf, dists
