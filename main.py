import os
import time
import torch
import argparse

from importlib import import_module
from torch.utils.tensorboard import SummaryWriter

from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model


def main(args, writer):
    set_seed(args.seed)

    server_dataset, client_datasets = load_dataset(args)

    # #check gpu
    # if 'cuda' in args.device:
    #     assert torch.cuda.is_available(), 'Please check if your GPU is available now!'
    #     args.device = 'cuda' if args.device_ids == [] else f'cuda:{args.device_ids[0]}'
    #

    model, args = load_model(args)

    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[f'{args.algorithm.title()}Server']
    server = server_class(args=args, writer=writer, server_dataset=server_dataset, client_datasets=client_datasets,
                          model=model)

    # federated learning
    for curr_round in range(1, args.R + 1):
        ## update round indicator
        server.round = curr_round

        ## update after sampling clients randomly
        selected_ids = server.update()

        ## evaluate on clients not sampled (for measuring generalization performance)
        if curr_round % args.eval_every == 0:
            server.evaluate(excluded_ids=selected_ids)
    else:
        ## wrap-up
        server.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    ####################
    # Default arguments #
    ####################
    parser.add_argument('__exp_name', help='experiment name', type=str, required=True)
    parser.add_argument('--seed', help='global random seed', type=int, default=5959)
    parser.add_argument('--device', help='device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`', type=str, default='cpu')
    parser.add_argument('--data_path', help='path to read data from', type=str, default='./data')

    #####################
    # Dataset arguments #
    #####################
    ## dataset



    #parse arguments
    args = parser.parse_args()

    #make path for saving losses & metrics & models
    curr_time = time.struct_time("%y%m%d", time.localtime())
    args.exp_name = f'{args.exp_name}_{args.seed}_{args.dataset.lower()}_{args.model_name.lower()}'
    args.result_path = os.path.join(args.result_path, f'{args.exp_name}_{curr_time}')
    if not os.patth.exists(args.result_path):
        os.makedirs(args.result_path)

    #make path for saving logs
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    #initialize logger
    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log', args)

    #check TensorBoard execution
    # if args.use_tb


    #define writer
    writer = SummaryWriter(
        log_dir=os.path.join(args.log_path, f'{args.exp_name}_{curr_time}'),
        filename_suffix=f'_{curr_time}'
    )
    
    #run main program
    main(args, writer)











