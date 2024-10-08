import numpy as np
import os
import sys
import torch

def send_to_cuda(model):
    for key in model.keys():
        model[key].cuda()

    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(path, model, optimizer=None):
    pass

def save_model(path, model, epoch, optimizer=None):
    state_dict = model.state_dict()

    data = {'epoch': epoch,
            'state_dict': state_dict}

    if not (optimizer is None):
        data['optimzer'] = optimizer.state_dict()

    torch.save(data, path)

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path