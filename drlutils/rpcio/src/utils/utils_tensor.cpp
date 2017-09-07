//
// Created by Robin Huang on 11/17/16.
//
#include "stdafx.h"
#include "utils_std.h"
#include <iostream>
#include "utils_tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "tf_tensor_utils.h"
#include "numpy_inc.h"

using namespace tensorflow;

void zero_tensor(tensorflow::Tensor & tensor)
{
//    tensor.flat<uint8_t>().setZero();
    assert(tensor.tensor_data().size() > 0);
    memset((uint8_t*)tensor.tensor_data().data(), 0, tensor.tensor_data().size());
}

const std::string describe_tensor(const tensorflow::Tensor & tensor)
{
    std::string ret = "unknow dtype";
    if(tensor.dtype() == DT_FLOAT)
    {
        long tensor_size = tensor.flat<float>().size();
        Eigen::Tensor<float, 0, Eigen::RowMajor> _tmax = tensor.flat<float>().maximum();
        Eigen::Tensor<float, 0, Eigen::RowMajor> _tmin = tensor.flat<float>().minimum();
        Eigen::Tensor<float, 0, Eigen::RowMajor> _tmean = tensor.flat<float>().mean();
        Eigen::Tensor<float, 0, Eigen::RowMajor> _tstd = ((tensor.flat<float>() - _tmean(0)).square().sum()/ static_cast<float>(tensor_size)).sqrt();
        float vmean = _tmean(0);
//        long size = tensor.flat<float>().size();
//        float * p = (float*)tensor.tensor_data().data();
//        float sum = 0.0;
//        for(long i = 0;i < size; i++)
//        {
//            sum += ((tensor.flat<float>()(i) - vmean) * (tensor.flat<float>()(i) - vmean));
//        }
//        sum /= size;
//        float vstd = std::sqrt(sum);
        ret = stdsprintf("mean/std/min/max=%.4f/%.4f/%.4f/%.4f", _tmean(0), _tstd(0), _tmin(0), _tmax(0));
    }
    return ret;
}

void summary_tensor(const tensorflow::Tensor & tensor, float & _mean, float & _std, float & _min, float & _max) {
    if(tensor.dtype() == DT_FLOAT) {
        long tensor_size = tensor.flat<float>().size();
        Eigen::Tensor<float, 0, Eigen::RowMajor> _tmax = tensor.flat<float>().maximum();
        Eigen::Tensor<float, 0, Eigen::RowMajor> _tmin = tensor.flat<float>().minimum();
        Eigen::Tensor<float, 0, Eigen::RowMajor> _tmean = tensor.flat<float>().mean();
        Eigen::Tensor<float, 0, Eigen::RowMajor> _tstd = ((tensor.flat<float>() - _tmean(0)).square().sum() /
                                                          static_cast<float>(tensor_size)).sqrt();
        _mean = _tmean(0);
        _std = _tstd(0);
        _min = _tmin(0);
        _max = _tmax(0);
    }
    else {assert(0);};
};

void copyObservationData(OBSERVATION_DATAS & dst, const OBSERVATION_DATAS & src)
{
    if(dst.size() == 0)
    {
        for(int i = 0; i < src.size(); i++)
        {
            std::shared_ptr<Tensor> data = src[i];
            Tensor * tensor = new Tensor(data->dtype(), data->shape());
            dst.push_back(std::shared_ptr<Tensor>(tensor));
        }
    }
    assert(dst.size() == src.size());
    for(int i = 0; i< src.size(); i++)
    {
        auto tsrc = *src[i];
        auto tdst = *dst[i];
        assert(tsrc.tensor_data().size() == tdst.tensor_data().size());
        memcpy((char*)tdst.tensor_data().data(), tsrc.tensor_data().data(), tsrc.tensor_data().size());
    }
}

void copyTensorMap(TENSOR_MAP & dst, const TENSOR_MAP & src) {
    if (dst.size() == 0) {
        for(auto it: src) {
            auto & data = it.second;
            Tensor * tensor = new Tensor(data->dtype(), data->shape());
            dst[it.first] = std::shared_ptr<Tensor>(tensor);
        }
    }
    assert(dst.size() == src.size());
    for(auto it: src) {
        auto & tsrc = *it.second;
        auto & tdst = *dst[it.first];
        assert(tsrc.tensor_data().size() == tdst.tensor_data().size());
        memcpy((char*)tdst.tensor_data().data(), tsrc.tensor_data().data(), tsrc.tensor_data().size());
    }
}
void copyTensor(Tensor & dst, const Tensor & src)
{
    if(dst.shape() != src.shape() || dst.dtype() != src.dtype())
        dst = _DeepCopy(src);
    else
    {
        DataTypeCanUseMemcpy(src.dtype());
        memcpy((char*)dst.tensor_data().data(), src.tensor_data().data(), src.tensor_data().size());
    }
}

void fillNan(tensorflow::Tensor & t) {
    assert(t.dtype() == DT_FLOAT);
    for(int i = 0; i < t.flat<float>().size(); i++) {
        t.flat<float>()(i) = floatNAN;
    }
}

bool hasNan(const tensorflow::Tensor & t, bool print_summary)
{
    tensorflow::DataType dataType = t.dtype();
    if(dataType == DT_FLOAT) {
        float *p = (float *) t.tensor_data().data();
        size_t len = t.tensor_data().size() / sizeof(float);
        bool ret = hasNan(p, len);
        if(ret && print_summary) {
            std::cerr << "tensor has nan: " << t.SummarizeValue(t.tensor_data().size()/sizeof(float)) << std::endl;
        }
        return ret;
    }
    else if(dataType == DT_INT8 || dataType == DT_UINT8 || dataType == DT_INT32 || dataType == DT_UINT16 || dataType == DT_INT16 || dataType == DT_INT64)
        return false;
    else {
        std::cerr << "hasNan(): unknow dtype: " << t.dtype() << std::endl;
        assert(0);
    }
    return false;
}

bool hasNan(const float * data, size_t len)
{
    for(size_t i = 0; i < len; i++)
    {
        if(std::isnan(data[i]))
            return true;
    }
    return false;
}

bool operator == (const TensorShape & shape0, std::vector<int> & shape1) {
    if(shape0.dims() != shape1.size()) return false;
    for(size_t idx = 0; idx < shape1.size(); idx++) {
        if(shape0.dim_size(idx) != shape1[idx])
            return false;
    }
    return true;
}

tensorflow::DataType convertRLUtilDType2TensorDType(DataFlow::NDType ndtype) {
    tensorflow::DataType dtype = DT_INVALID ;
    if(ndtype == DataFlow::NDType::ndtFloat32) dtype = DT_FLOAT;
    else if(ndtype == DataFlow::NDType::ndtUint8) dtype = DT_UINT8;
    else if(ndtype == DataFlow::NDType::ndtInt32) dtype = DT_INT32;
    else if(ndtype == DataFlow::NDType::ndtInt64) dtype = DT_INT64;
    else{assert(0);}
    return dtype;
}

DataFlow::NDType convertTensorDType2RLUtilDType(tensorflow::DataType dtype) {
    DataFlow::NDType ret;
    if(dtype == DT_FLOAT) ret = DataFlow::NDType::ndtFloat32;
    else if(dtype == DT_UINT8) ret = DataFlow::NDType::ndtUint8;
    else if(dtype == DT_INT32) ret = DataFlow::NDType::ndtInt32;
    else if(dtype == DT_INT64) ret = DataFlow::NDType::ndtInt64;
    else {assert(0);};
    return ret;
}

const tensorflow::TensorShape convertRLUtilShape2TensorShape(const DataFlow::Shape & shape) {
    TensorShape ret;
    for(auto s: shape) ret.AddDim(s);
    return ret;
}

void fillTensorWithRLUtilNDArray(tensorflow::Tensor & tensor, const DataFlow::NDArrayPtr & ndarray) {
    assert(ndarray->shape.size() >= 0);
    tensorflow::DataType dtype = convertRLUtilDType2TensorDType(ndarray->dtype) ;

    bool match = tensor.dtype() == dtype && tensor.shape() == ndarray->shape;

    if(!match) {
        TensorShape shape;
        for(auto s: ndarray->shape) shape.AddDim(s);
        tensor = Tensor(dtype, shape);
    }
    assert(tensor.tensor_data().size() == ndarray->buffer.size());
    memcpy((void*)tensor.tensor_data().data(), &ndarray->buffer[0], tensor.tensor_data().size());
    assert(!hasNan(tensor));
}

void copyTensorMapWithDataFlowTensorMap(TENSOR_MAP & tensor, const DataFlow::TensorMap & src) {
    if (tensor.size() != src.size()) {
        tensor.clear();
        for(auto it: src) {
            tensor[it.first] = std::shared_ptr<Tensor>(new Tensor(convertRLUtilDType2TensorDType(it.second->dtype),
                                          convertRLUtilShape2TensorShape(it.second->shape)));
        }
    }
    for(auto it: src) {
        fillTensorWithRLUtilNDArray(*tensor[it.first], it.second);
    }
}

void copyTensorListWithDataFlowTensorList(TENSOR_LIST & tensor, const DataFlow::TensorList & src) {
    if (tensor.size() != src.size()) {
        tensor.clear();
        for(size_t i = 0; i < src.size(); i++) {
            tensor.push_back(std::shared_ptr<Tensor>(new Tensor(convertRLUtilDType2TensorDType(src[i]->dtype),
                                                                  convertRLUtilShape2TensorShape(src[i]->shape))));
        }
    }
    for(size_t i = 0; i < src.size(); i++) {
        fillTensorWithRLUtilNDArray(*tensor[i], src[i]);
    }
}

void copyDataFlowTensorMapWithTensorMap(DataFlow::TensorMap & tensor, const TENSOR_MAP & src) {
    if(tensor.size() != src.size()) {
        tensor.clear();
        for(auto it: src) {
            tensor[it.first] = new DataFlow::NDArray;
        }
    }
    for(auto it: src) {
        fillRLUtilNDArrayWithTensor(*src.at(it.first), tensor[it.first]);
    }
}

void copyDataFlowTensorListWithTensorList(DataFlow::TensorList & tensor, const TENSOR_LIST & src) {
    if(tensor.size() != src.size()) {
        tensor.clear();
        for(size_t i = 0; i < src.size(); i++) {
            tensor.push_back(new DataFlow::NDArray);
        }
    }
    for(size_t i = 0; i < src.size(); i++) {
        fillRLUtilNDArrayWithTensor(*src[i], tensor[i]);
    }
}

void fillRLUtilNDArrayWithTensor(const tensorflow::Tensor &tensor, DataFlow::NDArrayPtr &ndarray) {
    if(!ndarray) ndarray = new DataFlow::NDArray;
    for(uint32_t idx = 0; idx < tensor.shape().dims(); idx++) ndarray->shape.push_back((int)tensor.shape().dim_size(idx));
//    if(ndarray->shape.size() == 0) ndarray->shape.push_back(0);
    assert(ndarray->shape.size() == tensor.shape().dims());
    ndarray->dtype = convertTensorDType2RLUtilDType(tensor.dtype());
//    assert(tensor.tensor_data().size() == 4);
    ndarray->buffer.resize(tensor.tensor_data().size());
    memcpy(ndarray->buffer.data(), tensor.tensor_data().data(), tensor.tensor_data().size());
    assert((int)ndarray->dtype >= 0 && (int)ndarray->dtype < (int)DataFlow::NDType::ndtUnknown);
}

bool checkRLUtilDTypeAndShapeMatch(const DataFlow::NDArrayPtr &ndarray, const tensorflow::Tensor &tensor) {
    return convertRLUtilDType2TensorDType(ndarray->dtype) == tensor.dtype() && (tensor.shape() == ndarray->shape);
}