
import numpy as np
import torch
import logging
import math
import json
import torch.nn.functional as F
from copy import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].tolist() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}

def prepare_params(params, frame_mask, rel_trans = None, dtype = np.float32):
    n_params = {k: v[frame_mask].astype(dtype)  for k, v in params.items()}
    if rel_trans is not None:
        n_params['transl'] -= rel_trans
    return n_params

def torch2np(item, dtype=np.float32):
    out = {}
    for k, v in item.items():
        if v ==[] or v=={}:
            continue
        if isinstance(v, list):
            if isinstance(v[0],  str):
                out[k] = v
            else:
                if torch.is_tensor(v[0]):
                    v = [v[i].cpu() for i in range(len(v))]
                try:
                    out[k] = np.array(np.concatenate(v), dtype=dtype)
                except:
                    out[k] = np.array(np.array(v), dtype=dtype)
        elif isinstance(v, dict):
            out[k] = torch2np(v)
        else:
            if torch.is_tensor(v):
                v = v.cpu()
            out[k] = np.array(v, dtype=dtype) 
            
    return out

def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def append2dict(source, data):
    for k in data.keys():
        if k in source.keys():
            if isinstance(data[k], list):
                source[k] += data[k]
            else:
                source[k].append(data[k])

            
def append2list(source, data):
    # d = {}
    for k in data.keys():
        leng = len(data[k])
        break
    for id in range(leng):
        d = {}
        for k in data.keys():
            if isinstance(data[k], list):
                if isinstance(data[k][0], str):
                    d[k] = data[k]
                elif isinstance(data[k][0], np.ndarray):
                    d[k] = data[k][id]
                
            elif isinstance(data[k], str):
                    d[k] = data[k]
            elif isinstance(data[k], np.ndarray):
                    d[k] = data[k]
        source.append(d)
           
        # source[k] += data[k].astype(np.float32)
        
        # source[k].append(data[k].astype(np.float32))

def np2torch(item, dtype=torch.float32):
    out = {}
    for k, v in item.items():
        if v ==[] :
            continue
        if isinstance(v, str):
           out[k] = v 
        elif isinstance(v, list):
            # if isinstance(v[0], str):
            #    out[k] = v
            try:
                out[k] = torch.from_numpy(np.concatenate(v)).to(dtype)
            except:
                out[k] = v # torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            out[k] = np2torch(v)
        else:
            out[k] = torch.from_numpy(v).to(dtype)
    return out

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype).to(device)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

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

def lr_decay_step(optimizer, epo, lr, gamma):
    if epo % 3 == 0:
        lr = lr * gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def lr_decay_mine(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def save_csv_log(opt, head, value, is_create=False, file_name='train_log'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = opt.ckpt + '/{}.csv'.format(file_name)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def save_ckpt(state, epo, opt=None):
    file_path = os.path.join(opt.ckpt, 'ckpt_last.pth.tar')
    torch.save(state, file_path)
    # if epo ==24: # % 4 == 0 or epo>22 or epo<5:
    if epo % 5 == 0:
        file_path = os.path.join(opt.ckpt, 'ckpt_epo'+str(epo)+'.pth.tar')
        torch.save(state, file_path)


def save_options(opt):
    with open('option.json', 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=False, indent=4))


