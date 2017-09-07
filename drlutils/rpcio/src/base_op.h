//
// Created by Robin Huang on 7/21/17.
//

#ifndef RPCIO_BASE_OP_H
#define RPCIO_BASE_OP_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "slice/data.h"


class DataIOBaseOp;
class DataIOManager;
class IOStream;
class Pool;
class TensorIO;
class SubBatchInfo;
class SubBatchInfoPred;
class DataIOBaseOp : public tensorflow::OpKernel {
public:
    explicit DataIOBaseOp(tensorflow::OpKernelConstruction *context, DataFlow::FlowDir dir, bool isTrain = false);
    virtual ~DataIOBaseOp();

protected:
    std::shared_ptr<Pool> m_pool;
    std::string m_ioname;
    friend class Pool;
    std::shared_ptr<TensorIO> m_tensorIO;
    bool m_isTrain = false;
    virtual void Compute_Send(tensorflow::OpKernelContext *ctx);
    virtual void Compute_Recv(tensorflow::OpKernelContext *ctx);
};


#endif //RPCIO_BASE_OP_H
