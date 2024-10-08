import argparse
import itertools
import sys
import os
from ast import literal_eval

def get_args_update_dict(args):
    args_update_dict = {}
    for string in sys.argv:
        string = ''.join(string.split('-'))
        if string in args:
            args_update_dict.update({string: args.__dict__[string]})
    return args_update_dict


def argparseNloop():
    parser = argparse.ArgumentParser()
    
    '''Directories and data path'''
    parser.add_argument('--work-dir', default = os.path.join('src', 'Ninjutsu'), type=str,
                        help='The path to the downloaded data')
    parser.add_argument('--data-path', default = os.path.join('..', 'DATASETS', 'Ninjutsu_Data'), type=str,
                        help='The path to the folder that contains dataset before pre-processing')
    parser.add_argument('--model_path', default = 'smplx_model', type=str,
                        help='The path to the folder containing SMPLX model')
    parser.add_argument('--save_dir', default = os.path.join('save', 'Ninjutsu', 'diffusion'), type=str,
                        help='The path to the folder to save the processed data')
    parser.add_argument('--render_path', default = os.path.join('render', 'Ninjutsu'), type=str,
                        help='The path to the folder to save the rendered output')
    parser.add_argument('--data_dir', default = os.path.join('data', 'Ninjutsu'), type=str,
                        help='The path to the pre-processed data')
    
    
    '''Dataset Parameters'''
    parser.add_argument('-dataset', nargs='+', type=str, default='NinjutsuDataset',
                        help='name of the dataset')
    parser.add_argument('--frames', nargs='+', type=int, default=50,
                        help='Number of frames taken from each sequence in the dataset for training.')
    parser.add_argument('-seedLength', nargs='+', type=int, default=20,
                        help='initial length of inputs to seed the prediction; used when offset > 0')
    parser.add_argument('-exp', nargs='+', type=int, default=0,
                        help='experiment number')
    parser.add_argument('-scale', nargs='+', type=int, default=1000.0,
                        help='Data scale by this factor')
    parser.add_argument('-framerate', nargs='+', type=int, default=20,
                        help='frame rate after pre-processing.')
    parser.add_argument('-seed', nargs='+', type=int, default=4815,
                                                help='manual seed')
    parser.add_argument('-load', nargs='+', type=str, default=None,
                        help='Load weights from this file')
    parser.add_argument('-cuda', nargs='+', type=int, default=0,
                        help='choice of gpu device, -1 for cpu')
    parser.add_argument('-overfit', nargs='+', type=int, default=0,
                        help='disables early stopping and saves models even if the dev loss increases. useful for performing an overfitting check')
    
    '''Diffusion parameters'''
    parser.add_argument("--noise_schedule", default='linear', choices=['linear', 'cosine', 'sigmoid'], type=str,
                       help="Noise schedule type")
    parser.add_argument("--diffusion_steps", default=300, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    parser.add_argument("--sampler", default='uniform', type=str,
                       help="Create a Schedule Sampler")
    
    
    '''Diffusion transformer model parameters'''
    parser.add_argument('-model', nargs='+', type=str, default='DiffusionTransformer',
                        help='name of model')
    parser.add_argument('-input_feats', nargs='+', type=int, default=3,
                        help='number of input features ')
    parser.add_argument('-out_feats', nargs='+', type=int, default=3,
                        help='number of output features ')
    parser.add_argument('--jt_latent', nargs='+', type=int, default=32,
                        help='dimensionality of last dimension after GCN')
    parser.add_argument('--d_model', nargs='+', type=int, default=256,
                        help='dimensionality of model embeddings')
    parser.add_argument('--d_ff', nargs='+', type=int, default=512,
                        help='dimensionality of the inner layer in the feed-forward network')
    parser.add_argument('--num_layer', nargs='+', type=int, default=6,
                        help='number of layers in encoder-decoder of model')
    parser.add_argument('--num_head', nargs='+', type=int, default=4,
                        help='number of attention heads in the multi-head attention mechanism.')
    parser.add_argument("--activations", default='LeakyReLU', choices=['LeakyReLU', 'SiLU', 'GELU'], type=str,
                       help="Activation function")
    '''Diffusion transformer hand model parameters'''
    parser.add_argument('-hand_input_condn_feats', nargs='+', type=int, default=280,
                        help='number of input features ')
    parser.add_argument('-hand_out_feats', nargs='+', type=int, default=3,
                        help='number of output features ')
    parser.add_argument('--d_modelhand', nargs='+', type=int, default=256,
                        help='dimensionality of model embeddings')
    parser.add_argument('--d_ffhand', nargs='+', type=int, default=512,
                        help='dimensionality of the inner layer in the feed-forward network')
    parser.add_argument('--num_layer_hands', nargs='+', type=int, default=6,
                        help='number of layers in encoder-decoder of model')
    parser.add_argument('--num_head_hands', nargs='+', type=int, default=4,
                        help='number of attention heads in the multi-head attention mechanism.')

   
    '''Training parameters'''
    parser.add_argument('-batch_size', nargs='+', type=int, default=32,
                        help='minibatch size.')
    parser.add_argument('-num_epochs', nargs='+', type=int, default=5000,
                        help='number of epochs for training')
    parser.add_argument('--skip_train', nargs='+', type=int, default=1,
                        help='downsampling factor of the training dataset. For example, a value of s indicates floor(D/s) training samples are loaded, '
                        'where D is the total number of training samples (default: 1).')
    parser.add_argument('--skip_val', nargs='+', type=int, default=1,
                        help='downsampling factor of the validation dataset. For example, a value of s indicates floor(D/s) validation samples are loaded, '
                        'where D is the total number of validation samples (default: 1).')
    parser.add_argument('-early_stopping', nargs='+', type=int, default=0,
                        help='Use 1 for early stopping')
    parser.add_argument('--n_workers', default=0, type=int,
                        help='Number of PyTorch dataloader workers')
    parser.add_argument('-greedy_save', nargs='+', type=int, default=1,
                        help='save weights after each epoch if 1')
    parser.add_argument('-save_model', nargs='+', type=int, default=1,
                        help='flag to save model at every step')
    parser.add_argument('-stop_thresh', nargs='+', type=int, default=3,
                        help='number of consequetive validation loss increses before stopping')
    parser.add_argument('-eps', nargs='+', type=float, default=0,
                        help='if the decrease in validation is less than eps, it counts for one step in stop_thresh ')
    parser.add_argument('--curriculum', nargs='+', type=int, default=0,
                        help='if 1, learn generating time steps by starting with 2 timesteps upto time, increasing by a power of 2')
    parser.add_argument('--use-multigpu', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='If to use multiple GPUs for training')
    parser.add_argument('--load-on-ram', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='This will load all the data on the RAM memory for faster training.'
                             'If your RAM capacity is more than 40 Gb, consider using this.')
    
    '''Optimizer parameters'''
    parser.add_argument('--optimizer', default='optim.Adam', type=str,
                        help='Optimizer')
    parser.add_argument('-momentum', default=0.9, type=float,
                        help='Weight decay for SGD Optimizer')
    parser.add_argument('-lr', nargs='+', type=float, default=1e-5,
                        help='learning rate')
    
    '''Scheduler parameters'''
    parser.add_argument('--scheduler', default='torch.optim.lr_scheduler.StepLR', type=str,
                        help='Scheduler')
    parser.add_argument('--patience', default=3, type=float,
                        help='Step size for ReduceOnPlateau scheduler')
    parser.add_argument('--factor', default=0.99, type=float,
                        help='Decay rate for ReduceOnPlateau scheduler')
    parser.add_argument('--threshold', default=0.05, type=float,
                        help='THreshold for ReduceOnPlateau scheduler')
    
    parser.add_argument('--stepsize', default=5, type=float,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Decay rate for StepLR scheduler')
    parser.add_argument('--milestones', default=[50, 100], type=float,
                        help='List of epoch indices. Must be increasing for MultiStepLR scheduler')
    '''Loss parameters'''
    parser.add_argument('--lambda_loss', type=dict, default=None, 
                        help='weight of loss for VAE')                   

    
    args, unknown = parser.parse_known_args()
    return args
