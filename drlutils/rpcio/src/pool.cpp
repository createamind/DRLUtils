//
// Created by Robin Huang on 8/27/17.
//

#include "stdafx.h"
#include "dataio.h"

MSP_MODULE_DECLARE("iostream");

SubBatchInfo::SubBatchInfo(Pool & pool, const std::string & name, uint32_t identStart, uint32_t size, uint32_t idx)
: m_pool(pool), m_name(name), m_identStart(identStart), m_size(size), m_idx(idx) {

    uint32_t qsize = 4;
//    m_queueRecvTrain = std::make_shared<RecycleQueue<_TensorListDatas>>(qsize);
//    m_queueRecvPredict = std::make_shared<RecycleQueue<_TensorListDatas>>(qsize);
    m_queueSendPredict = std::make_shared<RecycleQueue<_TensorListDatas>>(qsize);
    m_status = new DataFlow::BatchDataProcessorStatus;
    m_status->batchSize = m_size;
    m_status->batchIdxStart = m_identStart;
    m_status->packetTrain = new DataFlow::PacketStatistic;
    m_status->packetPredict = new DataFlow::PacketStatistic;
}

bool SubBatchInfo::matchTensorIO(std::shared_ptr<TensorIO> tensorIO) {
    return tensorIO == (tensorIO->isTrain() ? m_tensorIOTrain : m_tensorIOPredict);
}

TensorIO::TensorIO(Pool &pool, const std::string &name, int idx, bool isTrain) : m_pool(pool), m_name(name), m_idx(idx), m_isTrain(isTrain) {
    m_tensorInfos = new DataFlow::TensorInfos;
    uint32_t qsize = isTrain ? pool.getInitParams().trainBatchSize : pool.getInitParams().predictMinBatchSize;
    qsize = MAX(64, qsize * 2);
    m_queueRecvTrain = std::make_shared<RecycleQueue<_TensorListDatas>>(qsize);
    m_queueRecvPredict = std::make_shared<RecycleQueue<_TensorListDatas>>(qsize);
    m_queueRecvPredictSubIdxs = std::make_shared<RecycleQueue<_SubInfoIdxs>>(qsize);
}

Pool::Pool(DataIOManager &manager, const PoolInitParams & params)
        : m_manager(manager), m_name(params.name), m_size(params.size), m_initParams(params)
{
    assert(params.size > 0);
    assert(params.subBatchSize == 1);
    assert(params.predictMinBatchSize > 0 && params.predictMinBatchSize < params.size && params.predictMaxBatchSize < params.size && params.trainBatchSize < params.size);

    uint32_t subIdx = 0;
    for (uint32_t i = 0; i < params.size;) {
        uint32_t size = MIN(params.size - i, params.subBatchSize);
        auto subInfo = std::make_shared<SubBatchInfo>(*this, params.name, i, size, subIdx);
        m_subBatchInfos.push_back(subInfo);
        i += size;
        subIdx++;
    }
    MSP_INFO("create Pool: %s, size=%d, subBatchSize=%d, batch.train/predict_min/predict_max=%d/%d/%d",
             params.name.c_str(), params.size, params.subBatchSize, params.trainBatchSize, params.predictMinBatchSize, params.predictMaxBatchSize);
}

#include "base_op.h"
void Pool::start() {
    for(auto it: m_trainTensorIOs) {
        for (auto subInfo: m_subBatchInfos) {
            if(subInfo->idx() % m_trainTensorIOs.size() == it.second->idx()) {
                assert(subInfo->m_tensorIOTrain == nullptr);
                subInfo->m_tensorIOTrain = it.second;
            }
        }
    }
    assert(m_predictTensorIOs.size() > 0);
    for(auto it: m_predictTensorIOs) {
        for (auto subInfo: m_subBatchInfos) {
            if (subInfo->idx() % m_predictTensorIOs.size() == it.second->idx()) {
                assert(subInfo->m_tensorIOPredict == nullptr);
                subInfo->m_tensorIOPredict = it.second;
            }
        }
    }
    for(auto subInfo: m_subBatchInfos) {
        subInfo->m_status->tensorInfos = new DataFlow::TensorInfos;
        auto tis = subInfo->m_status->tensorInfos;
        if (subInfo->m_tensorIOTrain) {
            tis->train = subInfo->m_tensorIOTrain->getTensorInfos()->train;
            assert(tis->train.size() > 0);
        }
        if (subInfo->m_tensorIOPredict) {
            tis->predict = subInfo->m_tensorIOPredict->getTensorInfos()->predict;
            assert(tis->predict.size() > 0);
        }

    }
}

void Pool::putData(const ::DataFlow::IODataPutPtr & dataPtr) {
    if(!DataIOManager::hasInstance() || m_manager.isFlagEnd())
        throw DataFlow::ExceptionClosed();

    const bool isTrain = dataPtr->isTrain;
    auto & subInfos = m_subBatchInfos;
    int subIdx = dataPtr->processorIdx;
    if (subIdx < 0 || subIdx >= subInfos.size()) {
        throw std::invalid_argument(stdsprintf("[%s]: invalid subIdx %d", m_name.c_str(), subIdx));
    }
    auto subInfo = subInfos[subIdx];
    auto tensorIO = isTrain ? subInfo->m_tensorIOTrain : subInfo->m_tensorIOPredict;
    auto & dtypes = tensorIO->m_recvDtypes;
    auto & shapes = tensorIO->m_recvShapes;
    auto & names = tensorIO->m_recvVarNames;
    if (dataPtr->datas.size() != (dtypes.size() - 1)) {
        std::stringstream strm;
        strm << "data=[";
        for(auto it: dataPtr->datas) {
            strm << it.first << ",";
        }
        strm << "],local=[";
        for(auto it: names) strm << it << ",";

        throw std::invalid_argument(stdsprintf("[%s]: recv[%d], datas.size=%d not match local dtypes.size %d, %s",
                                               m_name.c_str(), subIdx,
                                               dataPtr->datas.size(), dtypes.size(), strm.str().c_str()));
    }
    auto funcDumpShape = [](const std::vector<int> & shape)->std::string {
        std::stringstream strm;

        strm << "[";
        for(size_t i = 0; i < shape.size(); i++) {
            strm << shape[i];
            if (i > 0) strm << ",";
        }
        strm << "]";
        return strm.str();
    };

    for(size_t didx = 0; didx < dataPtr->datas.size(); didx++) {
        const auto & name = names[didx+1];
        auto & d = dataPtr->datas[name];
        auto & shape = shapes[didx+1];
        if (d->shape.size() != shape.dims()) {
            throw std::logic_error(stdsprintf("[%s]: recv [%d]: data %s shape %s not match local shape %s",
                                              m_name.c_str(), subIdx, name.c_str(),
                                              funcDumpShape(d->shape).c_str(), shape.DebugString().c_str()));
        }
        for (size_t idx = 1; idx < d->shape.size(); idx++) {
            if (shape.dim_size(idx) > 0 && shape.dim_size(idx) != d->shape[idx]) {
                throw std::invalid_argument(stdsprintf("[%s]: recv[%d]: data %s shape %s not match local shape %s",
                                                       m_name.c_str(), subIdx, name.c_str(),
                                                       funcDumpShape(d->shape).c_str(), shape.DebugString().c_str()));
            }
        }
    }
//    auto queue = isTrain ? subInfo->m_queueRecvTrain : subInfo->m_queueRecvPredict;
//    auto dp = queue->getEmpty();
//    if(!DataIOManager::hasInstance() || m_manager.isFlagEnd())
//        throw DataFlow::ExceptionClosed();

    auto & queueRecv = isTrain ? tensorIO->m_queueRecvTrain : tensorIO->m_queueRecvPredict;
    auto dp = queueRecv->getEmpty();
    if(!DataIOManager::hasInstance() || m_manager.isFlagEnd())
        throw DataFlow::ExceptionClosed();

    copyTensorMapWithDataFlowTensorMap(dp->datas, dataPtr->datas);
    dp->subBatchInfoIdx = subIdx;
    (isTrain ? subInfo->m_status->packetTrain : subInfo->m_status->packetPredict)->packetRecvCount++;
    dp->push2queue();
    // no full batch mode
    auto & sem = isTrain ? m_semRecvTrainDataAvaiable : m_semRecvPredictDataAvaiable;
    if (sem.available() < 1)
        sem.release();

    MSP_LOG("[%s]: putData, isTrain=%d", dataPtr->name.c_str(), isTrain);
}

::DataFlow::IODataGetPtr Pool::getData(::Ice::Int subIdx) {
    // getData only avaiable in predict mode
    if(!DataIOManager::hasInstance() || m_manager.isFlagEnd())
        throw DataFlow::ExceptionClosed();

    auto & subInfos = m_subBatchInfos;
    if (subIdx < 0 || subIdx >= subInfos.size()) {
        throw std::invalid_argument(stdsprintf("[%s]: invalid subIdx %d", m_name.c_str(), subIdx));
    }

    DataFlow::IODataGetPtr ret = new DataFlow::IODataGet;
    auto & subInfo = subInfos[subIdx];
    const auto dp = subInfo->m_queueSendPredict->pop();
    if(!DataIOManager::hasInstance() || m_manager.isFlagEnd())
        throw DataFlow::ExceptionClosed();

    copyDataFlowTensorMapWithTensorMap(ret->datas, dp->datas);
    dp->recycle();
    subInfo->m_status->packetPredict->packetSendCount++;
    MSP_LOG("[%s]: getData", m_name.c_str());
    return ret;
}

void Pool::close() {
    for(auto & it: m_subBatchInfos) {
        it->close();
    }
}

void Pool::printStatus(std::ostream &strm) const {
    strm << "Pool: " << m_name << ENDL;

    if (m_predictTensorIOs.size() > 0) {
        strm << " Predict: " << ENDL;
        for (auto &it: m_predictTensorIOs) {
            auto & tensorIO = it.second;
            strm << "   " << tensorIO->name() << ": queue.recv=" << tensorIO->m_queueRecvPredict->size()
                 << ENDL;
        }
    }
    if (m_trainTensorIOs.size() > 0) {
        strm << " Train: " << ENDL;
        for(auto it: m_trainTensorIOs) {
            auto & tensorIO = it.second;
            strm << "   " << tensorIO->name() << ": queue.recv=" << tensorIO->m_queueRecvTrain->size() << ENDL;
        }
    }
    strm << " SubBatchInfos: " << ENDL;
    for(auto subInfo: m_subBatchInfos) {
        auto status = subInfo->getStatus();
        auto trainPS = status->packetTrain;
        auto predictPS = status->packetPredict;
        strm << "  [" << subInfo->idx() << "]" << ": queue.send=" << subInfo->m_queueSendPredict->size() << ENDL
             << "     predict: packet.recv/processed/in.send/sent=" << predictPS->packetRecvCount << "/" << predictPS->packetRecvProcessedCount << "/" << predictPS->packetSendInQueueCount << "/" << predictPS->packetSendCount << ENDL
             << "     train  : packet.recv/processed/in.send/sent=" << trainPS->packetRecvCount << "/" << trainPS->packetRecvProcessedCount << "/" << trainPS->packetSendInQueueCount << "/" << trainPS->packetSendCount << ENDL;
    }
}
