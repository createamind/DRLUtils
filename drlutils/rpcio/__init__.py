#coding: utf-8
import tensorflow as tf
import os

__all__ = ['dataio']

from ._load import _load_ops

__dataio_send, __dataio_recv = _load_ops()
def dataio_recv(ioname, pool, is_training, dtypes, shapes, names, **kwargs):
    return __dataio_recv(ioname, pool, is_training, dtypes, shapes, names,
                         name=ioname,
                         **kwargs)

def dataio_send(values, ioname, pool, names, **kwargs):
    return __dataio_send(values, ioname, pool,
                         names,
                         name=ioname,
                         **kwargs)
