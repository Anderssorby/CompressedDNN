import os


_backend_name = os.getenv("ODIN_BACKEND", "numpy")


def _load_backend():
    if _backend_name is "tensorflow":
        interface = __import__('odin.compute.tf', fromlist=['TensorflowWrapper'])
        _class = getattr(interface, 'TensorflowWrapper')
    else:
        interface = __import__('odin.compute.numpy', fromlist=['Numpy'])
        _class = getattr(interface, 'Numpy')

    return _class()


default_interface = _load_backend()
