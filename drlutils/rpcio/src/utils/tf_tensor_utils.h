//
// Created by Robin Huang on 11/26/16.
//

#ifndef APPC_TF_TENSOR_UTILS_H
#define APPC_TF_TENSOR_UTILS_H
#include <Python.h>
#include <tensorflow/core/public/session.h>
typedef struct TF_Tensor TF_Tensor;

// Creates a numpy array in 'ret' and copies the content of tensor 't'
// into 'ret'.
tensorflow::Status _ConvertTensorToNdarray(const tensorflow::Tensor& t, PyObject** ret);

// Given an numpy ndarray object 'obj', creates a corresponding tf
// Tensor in '*ret'.
tensorflow::Status _ConvertNdarrayToTensor(PyObject* obj, tensorflow::Tensor* ret);

namespace tensorflow {
// Import numpy.  This wrapper function exists so that the
// PY_ARRAY_UNIQUE_SYMBOL can be safely defined in a .cc file to
// avoid weird linking issues.  Should be called only from our
// module initialization function.
    void ImportNumpy();

    namespace tensor {

// DeepCopy returns a tensor whose contents are a deep copy of the
// contents of 'other'.  This function is intended only for
// convenience, not speed.
//
// REQUIRES: 'other' must point to data stored in CPU memory.
// REQUIRES: 'other' must be a Tensor of a copy-able type if
//           'other' is not appropriately memory-aligned.
        Tensor DeepCopy(const Tensor& other);

// Concatenates 'tensors' into a single tensor, along their 0th dimension.
//
// REQUIRES: All members of 'tensors' must have the same data type parameter.
// REQUIRES: Each member of 'tensors' must have at least one dimension.
// REQUIRES: Each member of 'tensors' must point to data stored in CPU memory.
// REQUIRES: Each member of 'tensors' must be a Tensor of a copy-able type if it
//           is not appropriately memory-aligned.
        Tensor Concat(const gtl::ArraySlice<Tensor>& tensors);

// Splits 'tensor' into 'sizes.size()' individual tensors, along the 0th
// dimension. The ith output tensor has 0th-dimension size 'sizes[i]'.
//
// REQUIRES: 'tensor' must have at least one dimension.
// REQUIRES: 'tensor.dim_size(0)' must equal the sum of the elements of 'sizes'.
// REQUIRES: 'tensor' must point to data stored in CPU memory.
// REQUIRES: 'tensor' must be a Tensor of a copy-able type if it is not
//           appropriately memory-aligned.
//
// Split() and Concat() are inverse operations.
        std::vector<Tensor> Split(const Tensor& tensor,
                                  const gtl::ArraySlice<int64>& sizes);

    }  // namespace tensor

}

void zero_tensor(tensorflow::Tensor & tensor);
tensorflow::Tensor _DeepCopy(const tensorflow::Tensor& other);

tensorflow::Status _TF_Tensor_to_PyObject(TF_Tensor *tensor, PyObject **out_array);
void _TF_DeleteTensor(TF_Tensor*);

namespace tensorflow
{
    TF_Tensor* TF_Tensor_EncodeStrings(const Tensor& src);
}

tensorflow::Status TfDTypeToNpDType(const tensorflow::DataType& tf, int* np);

const std::string describe_tensor(tensorflow::Tensor & tensor);
#endif //APPC_TF_TENSOR_UTILS_H
