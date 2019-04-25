# -*-coding:utf-8-*-
import numpy as np
import os
import h5py


# Load points and labels 
def load_cls(filelist):
    points = []
    labels = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))
        if 'normal' in data:
            points.append(np.concatenate([data['data'][...], data['normal'][...]], axis= -1).astype(np.float32))
        else:
            points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int64))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0))


def group_shuffle(inputs):
    # Check the len of dataset
    for idx in range(len(inputs) - 1):
        assert(len(inputs[idx]) == len(inputs[idx+1]))
    
    shuffle_indices = np.arange(inputs[0].shape[0])
    np.random.shuffle(shuffle_indices)
    outputs = []
    for idx in range(len(inputs)):
        outputs.append(inputs[idx][shuffle_indices, ...])
    return outputs


def load_cls_train_val(filelist, filelist_val):
    data_train, label_trian = group_shuffle(load_cls(filelist))
    data_val, label_val = load_cls(filelist_val)
    return data_train, label_trian, data_val, label_val
