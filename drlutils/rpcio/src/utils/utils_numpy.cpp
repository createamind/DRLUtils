//
// Created by Robin Huang on 4/1/17.
//
#include "stdafx.h"
#include "utils_numpy.h"
#include "slice/data.h"
#include "src/utils/pyutils.h"
//#include <boost/python/numpy.hpp>
//namespace p = boost::python;
//namespace np = boost::python::numpy;

PyObject * convertRLUtilNDArray2NPArray(const DataFlow::NDArrayPtr array) {
    PyGILEnsure gilEnsure;
    init_numpy();
    assert(array->buffer.size() > 0);
    assert(array->shape.size() > 0);
    int dtype = 0;
    uint32_t itemSize = 0;
    if(array->dtype == DataFlow::NDType::ndtUint8) { dtype = NPY_UINT8; itemSize = 1; }
    else if(array->dtype == DataFlow::NDType::ndtFloat32) { dtype = NPY_FLOAT; itemSize = sizeof(float); }
    else if(array->dtype == DataFlow::NDType::ndtInt32) { dtype = NPY_INT32; itemSize = sizeof(int32_t); }
    else {assert(0);}
    std::vector<npy_intp> dims;
    uint32_t array_size = 0;
    for(auto s: array->shape) {
        array_size = array_size == 0 ? (uint32_t)s : (array_size * s);
        dims.push_back((npy_intp)s);
    }
//    np::ndarray np::from_data()
//    assert((array_size * itemSize) == array->buffer.size());
    PyArrayObject * ret =  (PyArrayObject*)PyArray_SimpleNewFromData(dims.size(), dims.data(), dtype, array->buffer.data());
    return PyArray_Return(ret);
}