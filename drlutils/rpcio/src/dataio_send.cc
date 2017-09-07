#include "stdafx.h"
#include <string>
#include <memory>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <Ice/Ice.h>
#include "base_op.h"
#include "dataio.h"
using namespace std;
using namespace tensorflow;

Status ProbSendShape(shape_inference::InferenceContext* c) {
    for (int i = 0; i < c->num_inputs(); ++i) {
//        shape_inference::ShapeHandle shapeHandle;
//        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &shapeHandle));
//        c->set_output(i, shapeHandle);
    }
    auto & mgr = DataIOManager::getInstance();
    return mgr.createTensorIO(c, DataFlow::FlowDir::fdSend, false);
}

REGISTER_OP("DataIOSend")
        .Input("inputs: Tinputs")
        .Attr("ioname: string")
        .Attr("pool: string")
        .Attr("names: list(string) >= 0")
        .Attr("Tinputs: list(type) >= 0")
        .SetShapeFn(ProbSendShape)
        .SetIsStateful()
        .Doc(R"doc(
Send a serialized list of Tensors
    )doc");

class DataIOSendOp : public DataIOBaseOp {
public:
    explicit DataIOSendOp(OpKernelConstruction *context) : DataIOBaseOp(context, DataFlow::FlowDir::fdSend) {
    }

    void Compute(OpKernelContext *ctx) override {
        Compute_Send(ctx);
    }

private:
};


REGISTER_KERNEL_BUILDER(Name("DataIOSend").Device(DEVICE_CPU), DataIOSendOp);