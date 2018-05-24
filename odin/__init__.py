import os
VERSION = 0.1

description = """
Odin is a high level framework that enables deep-learning for the brutal age of machine learning. 
It organizes the jungle of frameworks into something that's comprehensible and extendable.
"""
results_dir = os.path.join(os.getcwd(), "results")

model_save_dir = os.path.join(results_dir, "saved_models")

model_wrapper = None


def default_save_path(name):
    return os.path.join(results_dir, model_wrapper.model_name, name)
