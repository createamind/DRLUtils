#include "stdafx.h"
#include "../slice/data.h"
#include "base_op.h"
#include "dataio.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
using namespace std;
using namespace tensorflow;

MSP_MODULE_DECLARE("dio_recv");

Status ProbShape(shape_inference::InferenceContext* c) {
    auto & mgr = DataIOManager::getInstance();
    bool is_train;
    TF_RETURN_IF_ERROR(c->GetAttr("is_train", &is_train));
    return mgr.createTensorIO(c, DataFlow::FlowDir::fdRecv, is_train);
}


REGISTER_OP("DataIORecv")
        .Attr("ioname: string")
        .Attr("pool: string")
        .Attr("is_train: bool")
        .Attr("types: list(type) >= 1")
        .Attr("shapes: list(shape) >= 1")
        .Attr("names: list(string) >= 1")
        .Output("output: types")
        .SetShapeFn(ProbShape)
        .SetIsStateful()
        .Doc(R"doc(
)doc");

static bool checkOPIsTrain(OpKernelConstruction * context) {
    bool isTrain = false;
    context->GetAttr("is_train", &isTrain);
    return isTrain;
}
class DataIORecvOp : public DataIOBaseOp {
public:
    explicit DataIORecvOp(OpKernelConstruction *context) : DataIOBaseOp(context, DataFlow::FlowDir::fdRecv, checkOPIsTrain(context)) {
    }

    void Compute(OpKernelContext *ctx) override {
        Compute_Recv(ctx);
    }

};

REGISTER_KERNEL_BUILDER(Name("DataIORecv").Device(DEVICE_CPU), DataIORecvOp);
