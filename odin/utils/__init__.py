import importlib


def dynamic_class_import(full_class_name):
    components = full_class_name.split('.')
    module = '.'.join(components[0:-1])
    class_name = components[-1]
    try:
        mod = importlib.import_module(module)
        class_obj = getattr(mod, class_name)
    except Exception as e:
        raise ValueError(e)
    return class_obj
