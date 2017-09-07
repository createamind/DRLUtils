//
// Created by Robin Huang on 11/17/16.
//

#ifndef APPC_UTILS_TENSOR_H
#define APPC_UTILS_TENSOR_H

#include "stdafx.h"
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include "slice/data.h"
typedef std::vector<std::shared_ptr<tensorflow::Tensor>> OBSERVATION_DATAS;

typedef std::map<std::string, std::shared_ptr<tensorflow::Tensor>> TENSOR_MAP;
typedef std::vector<std::shared_ptr<tensorflow::Tensor>> TENSOR_LIST;

void copyObservationData(OBSERVATION_DATAS & dst, const OBSERVATION_DATAS & src);
void copyTensor(tensorflow::Tensor & dst, const tensorflow::Tensor & src);
void copyTensorMap(TENSOR_MAP & dst, const TENSOR_MAP & src);

void fillNan(tensorflow::Tensor & t);
bool hasNan(const tensorflow::Tensor & t, bool print_summary = true);
bool hasNan(const float * data, size_t len);

const std::string describe_tensor(const tensorflow::Tensor & tensor);
void summary_tensor(const tensorflow::Tensor & tensor, float & _mean, float & _std, float & _min, float & _max);

#define VALIDATE_VALUE_RANGE(v, vmin, vmax) assert(!std::isnan(v) && v >= vmin && v <= vmax);

void fillTensorWithRLUtilNDArray(tensorflow::Tensor & tensor, const DataFlow::NDArrayPtr & ndarray);
void fillRLUtilNDArrayWithTensor(const tensorflow::Tensor &tensor, DataFlow::NDArrayPtr &ndarray);
tensorflow::DataType convertRLUtilDType2TensorDType(DataFlow::NDType ndtype);
DataFlow::NDType convertTensorDType2RLUtilDType(tensorflow::DataType dtype);
const tensorflow::TensorShape convertRLUtilShape2TensorShape(const DataFlow::Shape & shape);
bool checkRLUtilDTypeAndShapeMatch(const DataFlow::NDArrayPtr &ndarray, const tensorflow::Tensor &tensor);
void copyTensorMapWithDataFlowTensorMap(TENSOR_MAP & tensor, const DataFlow::TensorMap & src);
void copyTensorListWithDataFlowTensorList(TENSOR_LIST & tensor, const DataFlow::TensorList & src);
void copyDataFlowTensorMapWithTensorMap(DataFlow::TensorMap & tensor, const TENSOR_MAP & src);
void copyDataFlowTensorListWithTensorList(DataFlow::TensorList & tensor, const TENSOR_LIST & src);
#endif //APPC_UTILS_TENSOR_H
