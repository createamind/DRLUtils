//
// Created by Robin Huang on 11/26/16.
//
#include "stdafx.h"
#include "src/utils/utils_tensor.h"
#include "tf_tensor_utils.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/core/status.h"
#include "src/utils/numpy_inc.h"

namespace tensorflow
{
    namespace core
    {
        extern int VarintLength(uint64_t v);
        extern char* EncodeVarint32(char* dst, uint32 v);
        extern char* EncodeVarint64(char* dst, uint64 v);

    }
}
using namespace tensorflow;

// --------------------------------------------------------------------------
// TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The enum values here are identical to corresponding values in types.proto.
typedef enum {
    TF_FLOAT = 1,
    TF_DOUBLE = 2,
    TF_INT32 = 3,  // Int32 tensors are always in 'host' memory.
    TF_UINT8 = 4,
    TF_INT16 = 5,
    TF_INT8 = 6,
    TF_STRING = 7,
    TF_COMPLEX64 = 8,  // Single-precision complex
    TF_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
    TF_INT64 = 9,
    TF_BOOL = 10,
    TF_QINT8 = 11,     // Quantized int8
    TF_QUINT8 = 12,    // Quantized uint8
    TF_QINT32 = 13,    // Quantized int32
    TF_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
    TF_QINT16 = 15,    // Quantized int16
    TF_QUINT16 = 16,   // Quantized uint16
    TF_UINT16 = 17,
    TF_COMPLEX128 = 18,  // Double-precision complex
    TF_HALF = 19,
} TF_DataType;

struct TF_Tensor {
    TF_DataType dtype;
    TensorShape shape;
    TensorBuffer* buffer;
};

void _TF_DeleteTensor(TF_Tensor* t) {
  t->buffer->Unref();
  delete t;
}

static TF_DataType TF_TensorType(const TF_Tensor* t) { return t->dtype; }
static int TF_NumDims(const TF_Tensor* t) { return t->shape.dims(); }
static int64_t TF_Dim(const TF_Tensor* t, int dim_index) {
  return static_cast<int64_t>(t->shape.dim_size(dim_index));
}
static size_t TF_TensorByteSize(const TF_Tensor* t) { return t->buffer->size(); }
static void* TF_TensorData(const TF_Tensor* t) { return t->buffer->data(); }

typedef void (*Py_DECREF_wrapper_type)(PyObject*);
typedef std::unique_ptr<PyObject, Py_DECREF_wrapper_type> Safe_PyObjectPtr;
typedef std::vector<Safe_PyObjectPtr> Safe_PyObjectVector;
static void Py_DECREF_wrapper(PyObject* o) { Py_DECREF(o); };
static Safe_PyObjectPtr make_safe(PyObject* o) {
    return Safe_PyObjectPtr(o, Py_DECREF_wrapper);
}

namespace tensorflow {
    namespace core {
        extern const char *GetVarint32Ptr(const char *p, const char *limit, uint32 *v);
        extern const char *GetVarint64Ptr(const char *p, const char *limit, uint64 *v);
    }
}

static Status TF_DataType_to_PyArray_TYPE(TF_DataType tf_datatype,
                                   int* out_pyarray_type) {
    switch (tf_datatype) {
        case TF_HALF:
            *out_pyarray_type = NPY_FLOAT16;
            break;
        case TF_FLOAT:
            *out_pyarray_type = NPY_FLOAT32;
            break;
        case TF_DOUBLE:
            *out_pyarray_type = NPY_FLOAT64;
            break;
        case TF_INT32:
            *out_pyarray_type = NPY_INT32;
            break;
        case TF_UINT8:
            *out_pyarray_type = NPY_UINT8;
            break;
        case TF_UINT16:
            *out_pyarray_type = NPY_UINT16;
            break;
        case TF_INT8:
            *out_pyarray_type = NPY_INT8;
            break;
        case TF_INT16:
            *out_pyarray_type = NPY_INT16;
            break;
        case TF_INT64:
            *out_pyarray_type = NPY_INT64;
            break;
        case TF_BOOL:
            *out_pyarray_type = NPY_BOOL;
            break;
        case TF_COMPLEX64:
            *out_pyarray_type = NPY_COMPLEX64;
            break;
        case TF_COMPLEX128:
            *out_pyarray_type = NPY_COMPLEX128;
            break;
        case TF_STRING:
            *out_pyarray_type = NPY_OBJECT;
            break;
            // TODO(keveman): These should be changed to NPY_VOID, and the type used for
            // the resulting numpy array should be the custom struct types that we
            // expect for quantized types.
        case TF_QINT8:
            *out_pyarray_type = NPY_INT8;
            break;
        case TF_QUINT8:
            *out_pyarray_type = NPY_UINT8;
            break;
        case TF_QINT16:
            *out_pyarray_type = NPY_INT16;
            break;
        case TF_QUINT16:
            *out_pyarray_type = NPY_UINT16;
            break;
        case TF_QINT32:
            *out_pyarray_type = NPY_INT32;
            break;
        case TF_BFLOAT16:
            *out_pyarray_type = NPY_UINT16;
            break;
        default:
            return errors::Internal("Unsupported fetch type");
    }
    return Status::OK();
}

// Determine the pointer and offset of the string at offset 'i' in the string
// tensor 'src', whose total length is 'num_elements'.
static Status TF_StringTensor_GetPtrAndLen(const TF_Tensor* src,
                                           tensorflow::int64 num_elements,
                                           tensorflow::int64 i,
                                           const char** ptr,
                                           tensorflow::uint64* len) {
    const char* input = reinterpret_cast<const char*>(TF_TensorData(src));
    const size_t src_size = TF_TensorByteSize(src);
    const char* data_start = input + sizeof(tensorflow::uint64) * num_elements;
    const char* limit = input + src_size;
    tensorflow::uint64 offset =
            reinterpret_cast<const tensorflow::uint64*>(input)[i];
    const char* p =
            tensorflow::core::GetVarint64Ptr(data_start + offset, limit, len);
    if (static_cast<int64>(offset) >= (limit - data_start) || !p ||
        static_cast<int64>(*len) > (limit - p)) {
        return errors::InvalidArgument("Malformed TF_STRING tensor; element ", i,
                                       " out of range");
    }
    *ptr = p;
    return Status::OK();
}

// Copy the string at offset 'i' in the (linearized) string tensor 'tensor' into
// 'pyarray' at offset pointed by the 'i_ptr' iterator.
static Status CopyStringToPyArrayElement(PyArrayObject* pyarray, void* i_ptr,
                                         TF_Tensor* tensor,
                                         tensorflow::int64 num_elements,
                                         tensorflow::int64 i) {
    const char* ptr = nullptr;
    tensorflow::uint64 len = 0;
    TF_RETURN_IF_ERROR(
            TF_StringTensor_GetPtrAndLen(tensor, num_elements, i, &ptr, &len));
    auto py_string = make_safe(PyBytes_FromStringAndSize(ptr, len));
    int success = PyArray_SETITEM(
            pyarray, static_cast<char*>(PyArray_ITER_DATA(i_ptr)), py_string.get());
    if (success != 0) {
        return errors::Internal("Error setting element ", i);
    }
    return Status::OK();
}

// Converts the given TF_Tensor to a Numpy array.
// If the returned status is OK, the caller becomes the owner of *out_array.
Status _TF_Tensor_to_PyObject(TF_Tensor* tensor, PyObject** out_array) {
    init_numpy();

    // A fetched operation will correspond to a null tensor, and a None
    // in Python.
    if (tensor == nullptr) {
        Py_INCREF(Py_None);
        *out_array = Py_None;
        return Status::OK();
    }

    const int ndims = TF_NumDims(tensor);
    gtl::InlinedVector<npy_intp, 4> dims(ndims);
    tensorflow::int64 nelems = 1;
    for (int i = 0; i < ndims; ++i) {
        dims[i] = TF_Dim(tensor, i);
        nelems *= dims[i];
    }

    // Convert TensorFlow dtype to numpy type descriptor.
    int type_num = -1;
    TF_RETURN_IF_ERROR(
            TF_DataType_to_PyArray_TYPE(TF_TensorType(tensor), &type_num));
    PyArray_Descr* descr = PyArray_DescrFromType(type_num);

    // Copy the TF_TensorData into a newly-created ndarray and return it.
    // TODO(mrry): Perhaps investigate zero-copy approaches. This would involve
    // creating an ndarray-like object that wraps the TF_Tensor buffer, and
    // maps its destructor to TF_DeleteTensor.
    Safe_PyObjectPtr safe_out_array =
            make_safe(PyArray_Empty(ndims, dims.data(), descr, 0));
    if (!safe_out_array) {
        return errors::Internal("Could not allocate ndarray");
    }
    PyArrayObject* py_array =
            reinterpret_cast<PyArrayObject*>(safe_out_array.get());
    if (PyArray_NBYTES(py_array) !=
        static_cast<int64>(TF_TensorByteSize(tensor))) {
        if (TF_TensorType(tensor) == TF_STRING) {
            // Copy element by element.
            auto iter = make_safe(PyArray_IterNew(safe_out_array.get()));
            for (tensorflow::int64 i = 0; i < nelems; ++i) {
                auto s =
                        CopyStringToPyArrayElement(py_array, iter.get(), tensor, nelems, i);
                if (!s.ok()) {
                    return s;
                }
                PyArray_ITER_NEXT(iter.get());
            }
        } else {
            return errors::Internal("ndarray was ", PyArray_NBYTES(py_array),
                                    " bytes but TF_Tensor was ",
                                    TF_TensorByteSize(tensor), " bytes");
        }
    } else {
        memcpy(PyArray_DATA(py_array), TF_TensorData(tensor),
               PyArray_NBYTES(py_array));
    }

    // PyArray_Return turns rank 0 arrays into numpy scalars
    *out_array = PyArray_Return(
            reinterpret_cast<PyArrayObject*>(safe_out_array.release()));
    return Status::OK();
}

typedef std::vector<std::pair<string, Tensor>> TF_INPUT;
Tensor _DeepCopy(const Tensor& other) {
    Tensor tmp = Tensor(other.dtype(), other.shape());
    if (DataTypeCanUseMemcpy(other.dtype())) {
        StringPiece other_data = other.tensor_data();

        // We use StringPiece as a convenient map over the tensor buffer,
        // but we cast the type to get to the underlying buffer to do the
        // copy.
        StringPiece tmp_data = tmp.tensor_data();
        memcpy(const_cast<char*>(tmp_data.data()), other_data.data(),
               other_data.size());
    } else {
        CHECK_EQ(DT_STRING, other.dtype());
        tmp.flat<string>() = other.flat<string>();
    }
    return tmp;
}
