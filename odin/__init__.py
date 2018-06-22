import os
VERSION = 0.1

description = """
Odin is a high level framework that enables deep-learning for the brutal age of machine learning. 
It organizes the jungle of frameworks into something that's comprehensible and extendable.
"""
results_dir = os.path.join(os.getcwd(), "results")

model_save_dir = os.path.join(results_dir, "saved_models")

model_wrapper = None


def check_or_create_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)
    return d


def default_save_path(name, category=""):
    args = [results_dir, model_wrapper.model_name]
    if model_wrapper.prefix:
        args.append(model_wrapper.prefix)

    if category:
        args.append(category)

    check_or_create_dir(os.path.join(*args))

    args.append(name)

    return os.path.join(*args)
