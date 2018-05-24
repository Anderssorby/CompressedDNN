import argparse
import logging
import os
import time

import numpy as np

import odin
from odin import compute
from odin.models import load_model


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


def default_chainer():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False,
                    default="mnist_vgg2", help="model to compress")
    ap.add_argument("-f", "--file-prefix", required=False,
                    default="save", help="prefix for the files of computed values")
    ap.add_argument("--interface", required=False,
                    default="chainer", choices=["chainer", "tensorflow"])
    ap.add_argument('--gpu', type=int, default=-1)
    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--batchsize', type=int, default=128)
    ap.add_argument('--snapshot', type=int, default=10)
    ap.add_argument('--datadir', type=str, default='../data')
    ap.add_argument('-n', "--no-large-files", type=bool, default=False,
                    help="To prevent out of memory on test run")
    ap.add_argument("--ask", "--prompt", type=bool, dest="prompt", default=True,
                    help="Whether to ask before downloading.")

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

    args = ap.parse_args()

    prepare_logging(args)
    interface = compute.chainer.Chainer if args.interface is "chainer" else compute.TensorflowWrapper
    co = interface(args)

    np.random.seed(args.seed)

    kwargs = vars(args)
    model_wrapper = odin.model_wrapper = load_model(args.model, **kwargs)

    return co, args, model_wrapper
