import time

import argparse
import logging
import numpy as np
import os
import odin.actions

import odin
from odin.compute import default_interface as co


def prepare_logging(args):
    log_dir = os.path.join('log', str(os.path.basename(args.model).split('.')[0]),
                           time.strftime('%Y-%m-%d_%H-%M-%S_') + str(time.time()).replace('.', ''))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_file, level=logging.DEBUG)
    logging.info(args)


def default_arguments_and_behavior():
    """
    Sets up a basic environment for chainer and arguments for most actions
    :return: args, model_wrapper
    """
    print("----SETUP----")

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="model to do action on")
    ap.add_argument("--new-model", "--new_model", required=False, type=bool,
                    default=False, help="load or start from scratch")

    ap.add_argument("-p", '--prefix', type=str, default='default',
                    help="An additional sub label for your model")
    ap.add_argument("-e", '--experiment', type=str, required=False, dest="experiment",
                    help="An additional experiment label for your model")

    ap.add_argument("-a", "--action", dest="action", required=True, nargs="?",
                    choices=odin.actions.action_map.keys(), help="action keyword")

    ap.add_argument("-f", "--file-prefix", required=False,
                    default="save", help="prefix for the files of computed values")
    ap.add_argument('--gpu', type=int, default=-1)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, required=False)
    ap.add_argument('--snapshot', type=int, default=10)
    ap.add_argument('-n', "--no-large-files", type=bool, default=False,
                    help="To prevent out of memory on test run")
    ap.add_argument("--ask", "--prompt", type=bool, dest="prompt", default=True,
                    help="Whether to ask before downloading.")

    # Model options
    ap.add_argument('--unit', '-u', type=int, default=1000,
                    help='Number of units')
    ap.add_argument("--use_bias", "--use-bias", required=False, type=bool,
                    default=False, help="Use a bias vector")

    ap.add_argument('--max', type=float, required=False,
                    help='Maximum; action specific.')
    ap.add_argument('--num', type=int, required=False,
                    help='Count; action specific.')

    # Plot and print
    ap.add_argument("--plot", dest="plot", action="store_true",
                    help="Plot the results.")
    ap.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                    help="Print detailed information. May slow down execution.")

    # optimization
    ap.add_argument('--opt', type=str, default='MomentumSGD',
                    choices=['MomentumSGD', 'Adam', 'AdaGrad'])
    ap.add_argument('--weight_decay', type=float, default=0.0001)
    ap.add_argument('--alpha', type=float, default=0.001)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--lr_decay_freq', type=int, default=5)
    ap.add_argument('--lr_decay_ratio', type=float, default=0.1)
    ap.add_argument('--validate_freq', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1701)
    ap.add_argument('--frequency', type=int, default=-1)

    ap.add_argument('--gradclip', '-c', type=float, default=5,
                    help='Gradient norm threshold to clip')

    ap.add_argument('--bproplen', '-l', type=int, default=35,
                    help='Number of words in each mini-batch '
                         '(= length of truncated BPTT)')

    ap.add_argument('--regularizing', type=str, default="sq", choices=["sq", "abs"],
                    help="Which regularizer to use in lambda optimizer")

    # RNN text generation
    ap.add_argument('--primetext', type=str, default='',
                    help='base text data, used for text generation')

    ap.add_argument('--available_cores', type=int, default=4,
                    help='The number of CPU cores that can be used for computation')

    args = ap.parse_args()

    prepare_logging(args)

    np.random.seed(args.seed)

    co.update_args(args)

    kwargs = vars(args)
    odin.update_config(kwargs)

    print(kwargs)
    print("----END-SETUP----")

    return kwargs
