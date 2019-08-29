#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import pickle

import numpy as np

def loadPickle(path):

    """
    Check and load pickle object.
    """
    assert osp.exists(path)
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

# For test
def test(fname):
    data = loadPickle(fname)
    #print(data['test_marks'])
    #print(data.keys())
    marks = np.asarray(data['test_marks'])
    qinds = marks == 0
    ginds = marks == 1
    print(marks[qinds].shape)
    print(np.sum(marks[ginds]))
    #im_names = data['trainval_im_names']
    #print(im_names[1])
    #file_path = im_names[1]
    #print(int(file_path.split('/')[-1].split('_')[1]))

import sys
if __name__ == '__main__':
    fname = sys.argv[1]
    test(fname)
