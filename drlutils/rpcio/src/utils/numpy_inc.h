//
// Created by Robin Huang on 12/5/16.
//

#ifndef APPC_NUMPY_INC_H
#define APPC_NUMPY_INC_H

#define NPDATAARRAY_MAX_DIM (8)
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/arrayobject.h>

#include <atomic>
#include "src/utils/pyutils.h"
#if PYTHON_ABI_VERSION == 3
static void * init_numpy()
#else
static void init_numpy()
#endif
{
    static std::atomic_flag import_numpy;
    if(!import_numpy.test_and_set())
    {
        PyGILEnsure gilEnsure;
        import_array();
    }
#if PYTHON_ABI_VERSION == 3
    return NULL;
#endif
}

#endif //APPC_NUMPY_INC_H
