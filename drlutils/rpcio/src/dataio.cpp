//
// Created by Robin Huang on 7/21/17.
//
#include "stdafx.h"
#include "dataio.h"
#include "base_op.h"
#include "utils/utils_std.h"

MSP_MODULE_DECLARE("manager");
using namespace tensorflow;

DataIOManager* DataIOManager::m_instance = nullptr;
DataIOManager::DataIOManager() {
    assert(DataIOManager::m_instance == nullptr);
    DataIOManager::m_instance = this;

    m_status = new DataFlow::DSStatus;
}

DataIOManager::~DataIOManager() {
    DataIOManager::m_instance = nullptr;

}

DataIOManager& DataIOManager::getInstance() {
    assert(DataIOManager::m_instance);
    return *DataIOManager::m_instance;
}

bool DataIOManager::hasInstance() {
    return m_instance != nullptr;
}

int DataIOManager::createPool(const PoolInitParams &params) {
    assert(!m_finialized);
    if (params.name.empty()) {
        MSP_ERROR("empty pool name");
        return -1;
    }
    if (m_pools.find(params.name) != m_pools.end()) {
        MSP_ERROR("pool %s already created", params.name.c_str());
        return -1;
    }
    {
        QWriteLocker locker(&m_rwlPools);
        m_pools[params.name] = std::make_shared<Pool>(*this, params);
    }
    return 0;
}

const Status DataIOManager::createTensorIO(shape_inference::InferenceContext *context, DataFlow::FlowDir dir, bool isTrain) {
    assert(!m_finialized);

    assert(dir >= DataFlow::FlowDir::fdRecv && dir <= DataFlow::FlowDir::fdSend);
    std::string ioname;
    TF_RETURN_IF_ERROR(context->GetAttr("ioname", &ioname));

    std::string poolname;
    TF_RETURN_IF_ERROR(context->GetAttr("pool", &poolname));
    QReadLocker locker(&m_rwlPools);
    if (m_pools.find(poolname) == m_pools.end())
        return errors::InvalidArgument("pool ", poolname, " not found, please call createPool first");

    std::shared_ptr<Pool> pool = m_pools[poolname];
    auto & tensorIOs = isTrain ? pool->m_trainTensorIOs : pool->m_predictTensorIOs;
    if (tensorIOs.find(ioname) != tensorIOs.end()) {
        auto tensorIO = tensorIOs.at(ioname);
        if (isTrain || (tensorIO->m_recvInited && tensorIO->m_sendInited))
            return errors::InvalidArgument("tensorio ", ioname, " already exist");
    }
    else {
        auto threadIdx = tensorIOs.size();
        tensorIOs[ioname] = std::make_shared<TensorIO>(*pool, ioname, threadIdx, isTrain);
        MSP_INFO("create pool tensorIO %s, isTrain=%d, idx=%d, size.train/pred=%d/%d",
                 ioname.c_str(), isTrain, threadIdx, pool->m_trainTensorIOs.size(), pool->m_predictTensorIOs.size());
    }

    auto tensorIO = tensorIOs.at(ioname);
    if (dir == DataFlow::FlowDir::fdRecv) {
        tensorIO->m_recvInited = true;
        if (isTrain && pool->getInitParams().trainBatchSize == 0)
            return errors::InvalidArgument("pool trainBatchSize == 0 when is_train=true");

//        auto & names = (isTrain ? pool->m_trainRecvVarNames : pool->m_predictRecvVarNames);
        auto &names = tensorIO->m_recvVarNames;
        assert(names.size() == 0);
        TF_RETURN_IF_ERROR(context->GetAttr("names", &names));
        auto &dtypes = tensorIO->m_recvDtypes;
        assert(dtypes.size() == 0);
        TF_RETURN_IF_ERROR(context->GetAttr("types", &dtypes));
        std::vector<tensorflow::PartialTensorShape> shapes;
        TF_RETURN_IF_ERROR(context->GetAttr("shapes", &shapes));

        for (const auto &dtype : dtypes) {
            if (!(dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_INT64))
                return errors::InvalidArgument("dtype: ", dtype, " now only support dtype int8/uint8/float32/int32/int64");
        }
        if (dtypes.size() != shapes.size())
            return errors::InvalidArgument("shapes rank ", shapes.size(),
                                           " must equal to dtypes rank ", dtypes.size());
        if (dtypes.size() != names.size())
            return errors::InvalidArgument("shapes rank ", shapes.size(),
                                           " must equal to names rank ", names.size());
        CHECK(shapes.size() == context->num_outputs());

        assert(dtypes.size() > 2);

        // first is subBatchInfo, no need send to client
        auto &tis = isTrain ? tensorIO->m_tensorInfos->train : tensorIO->m_tensorInfos->predict;
        std::stringstream strm;
        for (int i = 0; i < context->num_outputs(); i++) {
            auto &shape = shapes[i];
            auto &name = names[i];
            auto &dtype = dtypes[i];
            shape_inference::ShapeHandle shapeHandle, subShapeHandle;
            TF_RETURN_IF_ERROR(context->MakeShapeFromPartialTensorShape(shape, &subShapeHandle));
            int batchDim = isTrain ? (int) pool->m_initParams.trainBatchSize : -1;
            TF_RETURN_IF_ERROR(context->Concatenate(context->Vector(batchDim), subShapeHandle, &shapeHandle));
            context->set_output(i, shapeHandle);

            tensorflow::PartialTensorShape shapeFull;
            tensorflow::PartialTensorShape::MakePartialShape<int>(&batchDim, 1, &shapeFull);
            shapeFull = shapeFull.Concatenate(shape);
//            printf("%s: shape = %s\n", name.c_str(), shapeFull.DebugString().c_str());

            // first item is subBatchInfo, ignore it to output
            if (i > 0) {
                DataFlow::TensorInfoPtr p = new DataFlow::TensorInfo;
                p->dtype = convertTensorDType2RLUtilDType(dtypes[i]);
                //            printf("%s dtype = %d, shape = %s\n", names[i].c_str(), p->dtype, shapes[i].DebugString().c_str());
                p->name = names[i];
                for (int dim = 0; dim < shapeFull.dims(); dim++) p->shape.push_back(shapeFull.dim_size(dim));
                tis.push_back(p);
            }
            auto &_shapes = tensorIO->m_recvShapes;
            _shapes.push_back(shapeFull);
            strm << name << ":shape=" << shapeFull.DebugString() << ",";
        }
//        for (const auto &subShape : shapes) {
//            std::vector<int64> vshape;
//            vshape.push_back(streamio->m_poolSize);
//            for(size_t tidx = 0; tidx < subShape.dims(); tidx++) vshape.push_back(subShape.dim_size(tidx));
//            PartialTensorShape shape(vshape);
////            MSP_INFO("push shape %s", shape.DebugString().c_str());
//            streamio->m_shapes.push_back(shape);
//            if (shape.dims() < 0) return errors::InvalidArgument("shape ", shape.DebugString(), " must have known rank.");
//            for (size_t didx = 2; didx < shape.dims(); didx++) {
//                if (shape.dim_size(didx) <= 0)
//                    return errors::InvalidArgument("shape: ", shape.DebugString(), " has None dim except dim (batch, timestep)");
//            }
//        }
        MSP_INFO("[TensorIO]: create recv %s, tensors: %s", ioname.c_str(), strm.str().c_str());
    }
    else if(dir == DataFlow::FlowDir::fdSend) {
        tensorIO->m_sendInited = true;
        TF_RETURN_IF_ERROR(context->GetAttr("names", &tensorIO->m_sendVarNames));
        MSP_INFO("[TensorIO]: create send %s, tensors: %s", ioname.c_str(), print_vector_string(tensorIO->m_sendVarNames).c_str());
    }
    return Status::OK();
}

std::shared_ptr<Pool> DataIOManager::findPool(const std::string &name) {
    QReadLocker locker(&m_rwlPools);
    auto it = m_pools.find(name);
    if (it == m_pools.end()) return nullptr;
    return it->second;
}

//void DataIOManager::removeOp(DataIOBaseuOp *op) {
//    assert(op);
//    MSP_INFO("remove op %s %s", op->getDirName().c_str(), op->getName().c_str());
//    QMutexLocker locker(&m_mutexOps);
//    assert(m_ops.find(op->getName()) != m_ops.end());
//    auto & pair = *m_ops.at(op->getName());
//    if (op->getDir() == DataFlow::FlowDir::fdSend) {
//        pair.m_sender = nullptr;
//    }
//    else if(op->getDir() == DataFlow::FlowDir::fdRecv) {
//        pair.m_receiver = nullptr;
//    }
//    else {assert(0);};
//    if (pair.m_receiver == nullptr && pair.m_sender == nullptr) {
//        m_ops.erase(op->getName());
//    }
//}

state_t DataIOManager::initialize(const InitParams & params) {
    _init_cli();
    return 0;
}

void DataIOManager::start() {
    m_finialized = true;
    for(auto it: m_pools) {
        it.second->start();
    }
}

void DataIOManager::init(const ::DataFlow::InitServerParamsPtr &, const ::Ice::Current &) {

}
::DataFlow::DSStatusPtr DataIOManager::getStatus(const ::Ice::Current &) {
    return m_status;
}

void DataIOManager::putData(const ::DataFlow::IODataPutPtr & data, const ::Ice::Current &) {
    if(!hasInstance() || isFlagEnd())
        throw DataFlow::ExceptionClosed();

    if (m_pools.find(data->name) == m_pools.end()) {
        std::stringstream strm;
        for(auto it: m_pools) {
            strm << it.first << ",";
        }
        throw std::invalid_argument(stdsprintf("pool %s not found, avaiable pools = %s", data->name, strm.str().c_str()));
    }
    std::shared_ptr<Pool> pool = m_pools.at(data->name);
    return pool->putData(data);
}

::DataFlow::IODataGetPtr DataIOManager::getData(const ::std::string & name, ::Ice::Int subProcessorIdx, const ::Ice::Current &) {
    if(!hasInstance() || isFlagEnd())
        throw DataFlow::ExceptionClosed();
    std::shared_ptr<Pool> pool = m_pools.at(name);
    return pool->getData(subProcessorIdx);
}


void DataIOManager::close() {
    m_flagEnd = true;
//    QMutexLocker locker(&m_mutexOps);
//    for(auto it: m_ops) {
//        if (it.second->m_receiver) it.second->m_receiver->close();
//        if (it.second->m_sender) it.second->m_sender->close();
//    }
//    m_ops.clear();
//    QWriteLocker locker(&m_rwlPools);
    for (auto it: m_pools) {
        it.second->close();
    }
}

void DataIOManager::onEvent(const EventParams &params) {
    if (params.name == "epoch_num") {
        m_status->epoch = params.paramInt;
        MSP_INFO("event: epoch_num = %d", params.name.c_str(), m_status->epoch);
    }
}

const std::string &DataIOManager::getFlowDirName(DataFlow::FlowDir dir) {
    static std::string names[] = {"recv", "send"};
    return names[(int)dir];
}

::DataFlow::BatchDataProcessorStatusPtr DataIOManager::getBatchDataProcessorStatus(const ::std::string & name, ::Ice::Int processorIdx,
                                                                                   const ::Ice::Current &) {
    auto pool = findPool(name);
    if (pool == nullptr) {
        throw std::invalid_argument(stdsprintf("can not find pool %s", name.c_str()));
    }
    if (processorIdx < 0 || processorIdx >= pool->m_subBatchInfos.size()) {
        throw std::invalid_argument(stdsprintf("[%s]: invalid processorIdx %d, max=%d", name.c_str(), processorIdx, pool->m_subBatchInfos.size()));
    }
    auto & subInfo = pool->m_subBatchInfos[processorIdx];
    auto ret = subInfo->getStatus();
    assert(ret->tensorInfos);
    assert(ret->tensorInfos->predict.size() > 0);
    if (ret->tensorInfos) {
        for (auto ti: ret->tensorInfos->train) {
            assert(ti->dtype >= 0 && ti->dtype < DataFlow::NDType::ndtUnknown);
        }
        for (auto ti: ret->tensorInfos->predict) {
            assert(ti->dtype >= 0 && ti->dtype < DataFlow::NDType::ndtUnknown);
        }
    }
    return ret;
//    return stream->m_subProcessorInfos[processorIdx]->m_status;
    return nullptr;
}
