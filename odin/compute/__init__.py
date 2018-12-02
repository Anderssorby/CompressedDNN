import os


_backend_name = os.getenv("ODIN_BACKEND", "numpy").lower()


def _load_backend():
    if _backend_name == "tensorflow":
        interface = __import__('odin.compute.tf', fromlist=['TensorflowWrapper'])
        _class = getattr(interface, 'TensorflowWrapper')
    elif _backend_name == "numpy":
        interface = __import__('odin.compute.numpy', fromlist=['Numpy'])
        _class = getattr(interface, 'Numpy')
    else:
        raise EnvironmentError("Unknown backend %s" % _backend_name)

    print("Odin using backend %s" % _backend_name)

    return _class()


default_interface = _load_backend()
