"""
Odin is a high level framework that enables deep-learning for the `brutal age` of machine learning.
It organizes the jungle of frameworks into something that's comprehensible and extendable.
Specifically it attempts to provide a common language for the different implementations and framework available.
"""

import os

VERSION = 0.1

results_dir = os.getenv("ODIN_RESULTS_DIR", os.path.join(os.getcwd(), "results"))

model_wrapper = None  # For convenience when testing

config = {}


def check_or_create_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)
    return d


def default_save_path(name, category=""):
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

    check_or_create_dir(os.path.join(*args))

    args.append(name)

    return os.path.join(*args)


def update_config(new_args):
    config.update(new_args)


import odin.models
import odin.actions
import odin.compute
import odin.utils
