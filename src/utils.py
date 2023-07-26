import os
import sys
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from importlib import import_module
from collections import defaultdict
from multiprocessing import Process

logger = logging.getLogger(__name__)


#########################
# Argparser Restriction #
#########################
class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


########
# Seed #
########
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f'[SEED] ...seed is set: {seed}!')


#########################
# Weight initialization #
#########################

#这里需要改

def init_weights(model, init_type, init_gain):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)


#####################
# Arguments checker #
#####################
def check_args(args):
    # check optimizer
    if args.optimizer not in torch.optim.__dict__.keys():
        err = f'`{args.optimizer}` is not a submodule of `torch.optim`... please check!'
        logger.exception(err)
        raise AssertionError(err)

    # check criterion
    if args.criterion not in torch.nn.__dict__.keys():
        err = f'`{args.criterion}` is not a submodule of `torch.nn`... please check!'
        logger.exception(err)
        raise AssertionError(err)

    # check algorithm
    if args.algorithm == 'fedsgd':
        args.E = 1
        args.B = 0

    # check lr step
    if args.lr_decay_step >= args.R:
        err = f'step size for learning rate decay (`{args.lr_decay_step}`) should be smaller than total round (`{args.R}`)... please check!'
        logger.exception(err)
        raise AssertionError(err)

    # check train only mode
    if args.test_fraction == 0:
        args._train_only = True
    else:
        args._train_only = False

    # check compatibility of evaluation metrics
    if hasattr(args, 'num_classes'):
        if args.num_classes > 2:
            if ('auprc' or 'youdenj') in args.eval_metrics:
                err = f'some metrics (`auprc`, `youdenj`) are not compatible with multi-class setting... please check!'
                logger.exception(err)
                raise AssertionError(err)
        else:
            if 'acc5' in args.eval_metrics:
                err = f'Top5 accruacy (`acc5`) is not compatible with binary-class setting... please check!'
                logger.exception(err)
                raise AssertionError(err)

        if ('mse' or 'mae' or 'mape' or 'rmse' or 'r2' or 'd2') in args.eval_metrics:
            err = f'selected dataset (`{args.dataset}`) is for a classification task... please check evaluation metrics!'
            logger.exception(err)
            raise AssertionError(err)
    else:
        if (
                'acc1' or 'acc5' or 'auroc' or 'auprc' or 'youdenj' or 'f1' or 'precision' or 'recall' or 'seqacc') in args.eval_metrics:
            err = f'selected dataset (`{args.dataset}`) is for a regression task... please check evaluation metrics!'
            logger.exception(err)
            raise AssertionError(err)

    # print welcome message
    logger.info('[CONFIG] List up configurations...')
    for arg in vars(args):
        logger.info(f'[CONFIG] - {str(arg).upper()}: {getattr(args, arg)}')
    else:
        print('')
    return args


##################
# Metric manager #
##################
class MetricManager:
    """Managing metrics to be used.
    """
    def __init__(self, eval_metrics):
        self.metric_funcs = {
            name: import_module(f'.metrics', package=__package__).__dict__[name.title()]()
            for name in eval_metrics
            }
        self.figures = defaultdict(int)
        self._results = dict()

    def track(self, loss, pred, true):
        # update running loss
        self.figures['loss'] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, curr_step=None):
        running_figures = {name: module.summarize() for name, module in self.metric_funcs.items()}
        running_figures['loss'] = self.figures['loss'] / total_len
        if curr_step is not None:
            self._results[curr_step] = {
                'loss': running_figures['loss'],
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        else:
            self._results = {
                'loss': running_figures['loss'],
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        self.figures = defaultdict(int)

    @property
    def results(self):
        return self._results
