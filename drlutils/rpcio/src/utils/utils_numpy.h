//
// Created by Robin Huang on 4/1/17.
//

#ifndef APPC_UTILS_NUMPY_H
#define APPC_UTILS_NUMPY_H

#include "numpy_inc.h"
#include "slice/data.h"
PyObject * convertRLUtilNDArray2NPArray(const DataFlow::NDArrayPtr array);

#endif //APPC_UTILS_NUMPY_H
