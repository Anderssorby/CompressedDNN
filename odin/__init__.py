"""
Odin is a high level framework that enables deep-learning for the `brutal age` of machine learning.
It organizes the jungle of frameworks into something that's comprehensible and extendable.
Specifically it attempts to provide a common language for the different implementations and framework available.
"""

import os
import yaml

VERSION = 0.2

ODIN_CONFIG = os.getenv("ODIN_CONFIG", os.path.join(os.getcwd(), ".odin.yaml"))
results_dir = os.getenv("ODIN_RESULTS_DIR", os.path.join(os.getcwd(), "results"))
log_dir = os.path.join(os.getcwd(), "log")
data_dir = os.path.join(os.getcwd(), "data")

model_wrapper = None  # For convenience when testing

config = {}

ascii_logo = """

                               dddddddd                          
     OOOOOOOOO                 d::::::d  iiii                    
   OO:::::::::OO               d::::::d i::::i                   
 OO:::::::::::::OO             d::::::d  iiii                    
O:::::::OOO:::::::O            d:::::d                           
O::::::O   O::::::O    ddddddddd:::::d iiiiiii nnnn  nnnnnnnn    
O:::::O     O:::::O  dd::::::::::::::d i:::::i n:::nn::::::::nn  
O:::::O     O:::::O d::::::::::::::::d  i::::i n::::::::::::::nn 
O:::::O     O:::::Od:::::::ddddd:::::d  i::::i nn:::::::::::::::n
O:::::O     O:::::Od::::::d    d:::::d  i::::i   n:::::nnnn:::::n
O:::::O     O:::::Od:::::d     d:::::d  i::::i   n::::n    n::::n
O:::::O     O:::::Od:::::d     d:::::d  i::::i   n::::n    n::::n
O::::::O   O::::::Od:::::d     d:::::d  i::::i   n::::n    n::::n
O:::::::OOO:::::::Od::::::ddddd::::::ddi::::::i  n::::n    n::::n
 OO:::::::::::::OO  d:::::::::::::::::di::::::i  n::::n    n::::n
   OO:::::::::OO     d:::::::::ddd::::di::::::i  n::::n    n::::n
     OOOOOOOOO        ddddddddd   dddddiiiiiiii  nnnnnn    nnnnnn
                                                                 
"""


def check_or_create_dir(*d, file=None):
    if len(d) > 1:
        d = os.path.join(*d)
    else:
        d = d[0]
    if not os.path.isdir(d):
        os.makedirs(d)
    if file:
        d = os.path.join(d, file)
    return d


def default_save_path(name, category="", experiment=None):
    if not model_wrapper:
        model_name = "other"
        args = [results_dir, model_name]
    else:
        model_name = model_wrapper.model_name

        args = [results_dir, model_name]
        if model_wrapper.prefix:
            args.append(model_wrapper.prefix)

    if category:
        args.append(category)

    if experiment:
        args.append(experiment)

    check_or_create_dir(os.path.join(*args))

    args.append(name)

    return os.path.join(*args)


def update_config(new_args):
    config.update(new_args)
    return config


def load_yaml_config(file):
    if not os.path.isfile(file):
        raise ValueError(file + " is not a file.")

    stream = open(file, 'r')
    # For compatibility with older versions of pyyaml
    if hasattr(yaml, "FullLoader"):
        y_config = yaml.load(stream, Loader=yaml.FullLoader)
    else:
        y_config = yaml.load(stream)
    return update_config(y_config)


load_yaml_config(ODIN_CONFIG)

print(ascii_logo)

import odin.models
import odin.actions
import odin.compute
import odin.utils
import odin.callbacks
