//
// Created by Robin Huang on 7/21/17.
//
#include "stdafx.h"
#include "base_op.h"
#include "dataio.h"
MSP_MODULE_DECLARE("baseop");

using namespace tensorflow;

DataIOBaseOp::DataIOBaseOp(OpKernelConstruction *context, DataFlow::FlowDir dir, bool isTrain) : OpKernel(context), m_isTrain(isTrain) {
    auto & ioname = m_ioname;
    OP_REQUIRES_OK(context, context->GetAttr("ioname", &ioname));
    std::string poolname;
    OP_REQUIRES_OK(context, context->GetAttr("pool", &poolname));
    auto & mgr = DataIOManager::getInstance();
    m_pool = mgr.findPool(poolname);
    assert(m_pool);
    if (!m_pool) {context->CtxFailure(errors::InvalidArgument("IOStream ", poolname, " with dir ", mgr.getFlowDirName(dir), " not exist"));}

    m_tensorIO = isTrain ? m_pool->m_trainTensorIOs.at(ioname) : m_pool->m_predictTensorIOs.at(ioname);
    MSP_INFO("create op %s: dir=%s, isTrain=%d", ioname.c_str(), dir == DataFlow::FlowDir::fdRecv ? "recv" : "send", (int)m_isTrain);
}

DataIOBaseOp::~DataIOBaseOp() {
    MSP_INFO("op %s closed", m_ioname.c_str());
}

void DataIOBaseOp::Compute_Recv(tensorflow::OpKernelContext *ctx) {
    OP_REQUIRES(ctx, DataIOManager::hasInstance(), errors::Cancelled("already closed"));

    auto &mgr = DataIOManager::getInstance();
    OP_REQUIRES(ctx, !mgr.isFlagEnd(), errors::Cancelled("already closed"));

    int start, stop;
    TF_CHECK_OK(this->OutputRange("output", &start, &stop));

    OpOutputList outputs;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &outputs));

    auto &initParams = m_pool->getInitParams();
    const bool isTrain = m_isTrain;
    auto tensorIO = m_tensorIO;
    assert(tensorIO->isTrain() == isTrain);

//    auto &subInfos = m_tensorIO->m_subBatchInfos;
    auto &subInfos = m_pool->m_subBatchInfos;
    assert(subInfos.size() > 0);

    std::vector<const _TensorListDatas*> dps;
    int batchSize = isTrain ? (int) initParams.trainBatchSize : -1;

    int countBatch = 0;
    uint32_t minBatchSize = isTrain ? initParams.trainBatchSize : initParams.predictMinBatchSize;
    uint32_t maxBatchSize = isTrain ? initParams.trainBatchSize : (initParams.predictMaxBatchSize > 0
                                                                   ? initParams.predictMaxBatchSize : initParams.size);
    assert(maxBatchSize >= minBatchSize);
    assert(maxBatchSize <= initParams.size);
    std::vector<std::shared_ptr<SubBatchInfo>> subInfosOwner;
    for (auto & subInfo: subInfos) {
        if (subInfo->matchTensorIO(tensorIO)) {
            subInfosOwner.push_back(subInfo);
        }
    }

    auto & queueRecv = isTrain ? m_tensorIO->m_queueRecvTrain : m_tensorIO->m_queueRecvPredict;
    do {
        const auto dp = queueRecv->pop();
        if (!mgr.hasInstance() || mgr.isFlagEnd()) ctx->CtxFailure(errors::Cancelled("already closed"));
        dps.push_back(dp);
        auto status = subInfos[dp->subBatchInfoIdx]->getStatus();
        (isTrain ? status->packetTrain : status->packetPredict)->packetRecvProcessedCount++;
        auto & t = dp->datas.at("agentIdent");
        if (t->dims() > 0) countBatch += t->dim_size(0);
        else countBatch += 1;
        if (maxBatchSize > 0 and countBatch >= maxBatchSize)
            break;
        if (!isTrain and queueRecv->size() == 0)
            break;
    } while(1);

    OP_REQUIRES(ctx, !mgr.isFlagEnd(), errors::Cancelled("already closed"));
    OP_REQUIRES(ctx, countBatch >= minBatchSize && countBatch <= maxBatchSize,
                errors::InvalidArgument("[", m_pool->m_name, "]: countBatch ", countBatch, " invalid, min/max=",
                                        minBatchSize, "/", maxBatchSize));

    batchSize = countBatch;
    MSP_DEBUG("[%s]: %s: no full batch mode got batch %d, isTrain=%d, min/max=%d/%d, queueSize=%d",
              m_pool->m_name.c_str(), m_tensorIO->name().c_str(),
              batchSize, isTrain, minBatchSize, maxBatchSize, queueRecv->size());


    auto &dtypes = m_tensorIO->m_recvDtypes;
    auto &shapes = m_tensorIO->m_recvShapes;
    auto &names = m_tensorIO->m_recvVarNames;
    Tensor *tensorSubBatchInfo = nullptr;
    {
        TensorShape shape;
        shape.AddDim(dps.size());
        shape.AddDim(2);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(start, shape, &tensorSubBatchInfo));
        int idx = 0;
        _SubInfoIdxs * pbuf = nullptr;
        if (!isTrain) {
            assert(m_tensorIO->m_queueRecvPredictSubIdxs->sizeEmpty() > 0);
            pbuf = m_tensorIO->m_queueRecvPredictSubIdxs->getEmpty();
            pbuf->subInfoIdxs.clear();
        }
        for (auto & it: dps) {
            assert(it->subBatchInfoIdx >= 0 && it->subBatchInfoIdx < m_pool->m_subBatchInfos.size());
            tensorSubBatchInfo->matrix<int>()(idx, 0) = it->subBatchInfoIdx;
            tensorSubBatchInfo->matrix<int>()(idx, 1) = it->datas.at("agentIdent")->dim_size(0);
            if(pbuf)
                pbuf->subInfoIdxs.push_back(it->subBatchInfoIdx);
            idx++;
        }
        assert(idx == dps.size());
        if(pbuf) {
            assert(pbuf->subInfoIdxs.size() == idx);
            assert(pbuf->subInfoIdxs.size() == tensorSubBatchInfo->dim_size(0));
            assert(pbuf->subInfoIdxs.size() == dps.size());
            pbuf->push2queue();
        }
    }

    for (int i = start+1; i < stop; ++i) {
        Tensor *output = nullptr;
        int tensorIdx = i - start;
        const auto &shape_def = shapes[tensorIdx];
        std::vector<int64> shape_dims;
        for (size_t t = 0; t < shape_def.dims(); t++) shape_dims.push_back(shape_def.dim_size(t));
        shape_dims[0] = batchSize;
        size_t itemBytes = 0;
        bool hasNotMatchTimestep = false;
        if (shape_dims.size() > 1 && shape_dims[1] < 0) {
            for (auto dp: dps) {
                assert(dp->datas.find(names[tensorIdx]) != dp->datas.end());
                const auto &data = *dp->datas.at(names[tensorIdx]);
                auto dim_timestep = data.shape().dim_size(1);
                if (shape_dims[1] != dim_timestep) hasNotMatchTimestep = true;
                if (shape_dims[1] < dim_timestep) {
                    shape_dims[1] = dim_timestep;
                    itemBytes = data.tensor_data().size() / data.shape().dim_size(0);
                }
            }
        } else {
            for (auto dp: dps) {
                assert(dp->datas.find(names[tensorIdx]) != dp->datas.end());
                const auto &data = *dp->datas.at(names[tensorIdx]);
                itemBytes = data.tensor_data().size() / data.shape().dim_size(0);
                break;
            }
        }
        TensorShape shape(shape_dims);
        for (int j = 0; j < shape.dims(); j++) { assert(shape.dim_size(j) > 0); };

        Tensor *bddata = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(i, shape, &bddata));
        assert(bddata->tensor_data().size() == itemBytes * batchSize);
        if (hasNotMatchTimestep) {
            if (bddata->dtype() == DT_FLOAT) bddata->flat<float>().setZero();
            else if (bddata->dtype() == DT_DOUBLE) bddata->flat<double>().setZero();
            else if (bddata->dtype() == DT_INT32 || bddata->dtype() == DT_UINT8 || bddata->dtype() == DT_INT8 ||
                     bddata->dtype() == DT_INT64)
                memset((char *) bddata->tensor_data().data(), 0, bddata->tensor_data().size());
        }
        uint32_t batch_data_bytes = 0;
        for (auto dp: dps) {
            auto & subInfo = subInfos[dp->subBatchInfoIdx];
            std::shared_ptr<const Tensor> src = dp->datas.at(names[tensorIdx]);
            OP_REQUIRES(ctx, src->dim_size(0) == subInfo->size(),
                        errors::InvalidArgument("sub processor ", subInfo->identStart(), " data ",
                                                src->shape().DebugString(), " not match batch_size ", subInfo->size()));
            int64_t subBatchSize = src->shape().dim_size(0);
            const char *psrc = src->tensor_data().data();
            const char *pdst = bddata->tensor_data().data() + batch_data_bytes;
            memcpy((char *) pdst, psrc, src->tensor_data().size());
            batch_data_bytes += itemBytes * subBatchSize;
        }
        assert(batch_data_bytes == bddata->tensor_data().size());
    }
    for (auto dp: dps) {
        dp->recycle();
//        subInfo->m_status->packetCount++;
    }
//    if (senderStream) {
//        senderStream->m_semRecvDataAvaiable.release();
//    }
}

void DataIOBaseOp::Compute_Send(tensorflow::OpKernelContext *ctx) {
    OP_REQUIRES(ctx, DataIOManager::hasInstance(), errors::Cancelled("already closed"));

    auto & mgr = DataIOManager::getInstance();
    OP_REQUIRES(ctx, !mgr.isFlagEnd(), errors::Cancelled("already closed"))
    int start, stop;
    TF_CHECK_OK(this->InputRange("inputs", &start, &stop));
//    OP_REQUIRES(ctx, stop-start == m_stream->m_dtypes.size(), errors::InvalidArgument("stop ", stop, " - start ", start, " not == dtypes size ", m_stream->m_dtypes.size()));
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    auto & subInfos = m_pool->m_subBatchInfos;
    assert(subInfos.size() > 0);
    const auto & names = m_tensorIO->m_sendVarNames;
//    if (m_stream->m_minBatchSize > 0) {
//        // no full batch mode is sync mode
//        m_stream->m_semRecvDataAvaiable.acquire();
//        if (!DataIOManager::hasInstance() || mgr.isFlagEnd()) ctx->CtxFailure(errors::Cancelled("already closed"));
//    }
    assert(m_tensorIO->m_queueRecvPredictSubIdxs->size() > 0);

    assert(stop > (start + 2));
    const auto & tensorSubBatchInfo = inputs[start];
    const auto & tensorIdents = inputs[start+1];
    OP_REQUIRES(ctx, tensorIdents.dims() == 1, errors::InvalidArgument("send op: tensor 1 should be agentIdents"));
    OP_REQUIRES(ctx, tensorSubBatchInfo.dims() == 2 && tensorSubBatchInfo.dim_size(0) <= m_pool->m_subBatchInfos.size(),
                errors::InvalidArgument("send op: tensor 0 dim0 ", tensorSubBatchInfo.dim_size(0), " not match predict SubBatch ", m_pool->m_subBatchInfos.size()));
#ifndef NDEBUG
    auto descSubBatch = tensorSubBatchInfo.DebugString();
    auto summarySubBatch = tensorSubBatchInfo.SummarizeValue(16);
    auto descIdents = tensorIdents.DebugString();
    auto summaryIdents = tensorIdents.SummarizeValue(16);
#endif
    std::unordered_map<int, _TensorListDatas*> dps;
    auto pbuf = m_tensorIO->m_queueRecvPredictSubIdxs->pop();
    assert(pbuf->subInfoIdxs.size() == tensorSubBatchInfo.dim_size(0));
    for(uint32_t i = 0; i < tensorSubBatchInfo.dim_size(0); i++) {
        auto subIdx = tensorSubBatchInfo.matrix<int32_t>()(i, 0);
        assert(subIdx == pbuf->subInfoIdxs[i]);
        auto subInfo = subInfos[subIdx];
        auto dp = subInfo->m_queueSendPredict->getEmpty();
        dp->datas.clear();
        assert((stop-start) == names.size());
        for (int k = 1; k < (stop-start); k++)
            dp->datas[names[k]] = std::make_shared<Tensor>();
        assert(dp->datas.size() == (names.size() - 1));
        dps[subIdx] = dp;
        auto status = subInfo->getStatus();
        status->packetPredict->packetSendInQueueCount++;
    }
    pbuf->recycle();

    // ignore first subBatchInfo

    for (int i = start+1; i < stop; ++i) {
        int tensorIdx = i - start;
        const auto & tensor = inputs[i];
//        MSP_INFO("send tensor %d: %s", tensorIdx, tensor.SummarizeValue(16).c_str());
        OP_REQUIRES(ctx, tensor.shape().dims() >= 0, errors::InvalidArgument("input ", tensorIdx, "shape rank ", tensor.shape().dims(), " must > 0"));

#ifndef NDEBUG
        auto descTensor = tensor.DebugString();
#endif
        assert(tensor.dim_size(0) == tensorIdents.dim_size(0));
        int64_t sliceStart = 0;
        for(uint32_t sidx = 0; sidx < tensorSubBatchInfo.dim_size(0); sidx++) {
            int subIdx = tensorSubBatchInfo.matrix<int32_t>()(sidx, 0);
            auto subInfo = subInfos[subIdx];
            auto dp = dps[subIdx];
            int subSliceCount = tensorSubBatchInfo.matrix<int32_t>()(sidx, 1);
            copyTensor(*dp->datas.at(names[tensorIdx]), tensor.Slice(sliceStart, sliceStart+subSliceCount));
            sliceStart += subSliceCount;
        }
        assert(sliceStart == tensor.dim_size(0));
//        uint32_t batchStart = 0;
//        uint32_t identStart = (uint32_t)tensorIdents.flat<int32_t>()(batchStart);
//        for(auto & subInfo: subInfos) {
//            auto dp = subInfo->m_lastSendData;
//            if (dp == nullptr) continue;
//            if (dp->datas.size() != (stop-start)) {
//                dp->datas.clear();
//                for(int k = 0; k < (stop-start); k++)
//                          dp->datas.push_back(std::shared_ptr<Tensor>(new Tensor()));
//            }
//            if (m_stream->m_minBatchSize > 0) { assert(subInfo->m_sendDataDim0ForNotFullBatchMode > 0); }
//            int64_t sliceC = m_stream->m_minBatchSize > 0 ? subInfo->m_sendDataDim0ForNotFullBatchMode : subInfo->m_batchSize;
//            assert(batchStart < tensor.dim_size(0));
//            copyTensor(*dp->datas[tensorIdx], tensor.Slice(batchStart, batchStart+sliceC));
//            batchStart += sliceC;
//        }
//        assert(batchStart == tensor.dim_size(0));
    }
    for(auto it: dps) {
        it.second->push2queue();
    }
}