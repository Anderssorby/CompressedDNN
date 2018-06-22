import os


_backend_name = os.getenv("ODIN_BACKEND", "chainer")


def _load_backend():
    if _backend_name is "tensorflow":
        interface = __import__('odin.compute.tf', fromlist=['TensorflowWrapper'])
        _class = getattr(interface, 'TensorflowWrapper')
    else:
        interface = __import__('odin.compute.chainer', fromlist=['Chainer'])
        _class = getattr(interface, 'Chainer')

    return _class()


default_interface = _load_backend()
