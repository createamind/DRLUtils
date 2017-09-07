//
// Created by Robin Huang on 7/21/17.
//

#ifndef RPCIO_RPCIO_H
#define RPCIO_RPCIO_H
#include <Ice/Ice.h>
#include "msp_common.h"
#include "msp_cli.h"
#include "cdataio.h"
#include "utils/utils_tensor.h"
#include "utils/recycle_queue.h"
class DataIOBaseOp;
class IOStream;
class IOPair {
    std::string m_name;
public:
    IOPair(const std::string & name): m_name(name) {
    }
    std::shared_ptr<IOStream> m_receiver;
    std::shared_ptr<IOStream> m_sender;
};

class _TensorListDatas : public RecycleQueueDataBase {
public:
    TENSOR_MAP datas;
    int32_t subBatchInfoIdx = -1;
};

class _SubInfoIdxs : public RecycleQueueDataBase {
public:
    std::vector<uint32_t> subInfoIdxs;
};

class Pool;
class TensorIO;
class SubBatchInfo {
protected:
    friend class Pool;
    Pool & m_pool;
    std::string m_name;
    uint32_t m_identStart;
    uint32_t m_size;
    uint32_t m_idx;
    DataFlow::BatchDataProcessorStatusPtr m_status;
    std::shared_ptr<TensorIO> m_tensorIOTrain;
    std::shared_ptr<TensorIO> m_tensorIOPredict;
public:
//    std::shared_ptr<RecycleQueue<_TensorListDatas>> m_queueRecvTrain;
//    std::shared_ptr<RecycleQueue<_TensorListDatas>> m_queueRecvPredict;
    std::shared_ptr<RecycleQueue<_TensorListDatas>> m_queueSendPredict;
//    const _TensorListDatas * m_lastRecvData = nullptr;
    SubBatchInfo(Pool & pool, const std::string & name, uint32_t identStart, uint32_t size, uint32_t idx);
    virtual void close() {
//        if(m_queueRecvTrain) m_queueRecvTrain->close();
//        if(m_queueRecvPredict) m_queueRecvPredict->close();
        if(m_queueSendPredict) m_queueSendPredict->close();
    };
    uint32_t size() const { return m_size; };
    uint32_t identStart() const { return m_identStart; };
    uint32_t idx() const { return m_idx; };
    DataFlow::BatchDataProcessorStatusPtr getStatus() { return m_status;};
//    std::shared_ptr<RecycleQueue<_TensorListDatas>> & getQueueRecv(bool isTrain) { return isTrain ? m_queueRecvTrain : m_queueRecvPredict;}
    std::shared_ptr<RecycleQueue<_TensorListDatas>> & getQueueSend(bool isTrain) { return m_queueSendPredict; };

    bool matchTensorIO(std::shared_ptr<TensorIO> tensorIO);
};

class DataIOBaseOp;
class TensorIO {
protected:
    Pool & m_pool;
    int m_idx;
    std::string m_name;
    bool m_isTrain;
    friend class DataIOManager;
    friend class Pool;
    friend class DataIOBaseOp;
    bool m_sendInited = false;
    bool m_recvInited = false;

//    std::vector<std::shared_ptr<SubBatchInfo>> m_subBatchInfos;

    std::vector<std::string> m_recvVarNames;
    std::vector<std::string> m_sendVarNames;

    tensorflow::DataTypeVector m_recvDtypes;
    std::vector<tensorflow::PartialTensorShape> m_recvShapes;
    tensorflow::DataTypeVector m_sendDtypes;
    DataFlow::TensorInfosPtr m_tensorInfos;

    std::shared_ptr<RecycleQueue<_TensorListDatas>> m_queueRecvTrain;
    std::shared_ptr<RecycleQueue<_SubInfoIdxs>> m_queueRecvPredictSubIdxs;
    std::shared_ptr<RecycleQueue<_TensorListDatas>> m_queueRecvPredict;
public:
    TensorIO(Pool & pool, const std::string & name, int idx, bool isTrain);
    int idx() const { return m_idx; };
    const std::string & name() const { return m_name; };
    DataFlow::TensorInfosPtr getTensorInfos() const { return m_tensorInfos; };
    bool isTrain() const { return m_isTrain; };
};

class Pool {
protected:
    friend class DataIOManager;
    friend class DataIOBaseOp;
    DataIOManager & m_manager;
    std::string m_name;
    uint32_t m_size;
    PoolInitParams m_initParams;
    std::unordered_map<std::string, std::shared_ptr<TensorIO>> m_trainTensorIOs;
    std::unordered_map<std::string, std::shared_ptr<TensorIO>> m_predictTensorIOs;

    std::vector<std::shared_ptr<SubBatchInfo>> m_subBatchInfos;
//    std::vector<std::shared_ptr<SubBatchInfoPred>> m_predictBatchInfos;

//    tensorflow::DataTypeVector m_trainDtypes;
    QSemaphore m_semRecvTrainDataAvaiable;
    QSemaphore m_semRecvPredictDataAvaiable;

public:
    Pool(DataIOManager & manager, const PoolInitParams & params);
    virtual void putData(const ::DataFlow::IODataPutPtr&);
    virtual ::DataFlow::IODataGetPtr getData(::Ice::Int);
    void start();
    void close();
    const PoolInitParams & getInitParams() const { return m_initParams; };

    void printStatus(std::ostream & strm) const;
};

class DataIOManager : public DataFlow::DataServer {
public:
    DataIOManager();
    virtual ~DataIOManager();
    static DataIOManager & getInstance();
    virtual state_t initialize(const InitParams & params);
    void start();
    void close();
    const tensorflow::Status createTensorIO(tensorflow::shape_inference::InferenceContext *ctx, DataFlow::FlowDir dir, bool isTrain);
    std::shared_ptr<Pool> findPool(const std::string & name);
//    void removeStream(const std::string name, DataFlow::FlowDir dir);
    void onEvent(const EventParams & params);
    const std::string & getFlowDirName(DataFlow::FlowDir dir);
    bool isFlagEnd() const { return m_flagEnd; };
    static bool hasInstance();
    int createPool(const PoolInitParams &params);

protected:
    friend class CDataIO;
    state_t _init_cli();
    msp_cli::CCommandLineInterface * m_cli = nullptr;
    DECLARE_CLI_IN_CLASS(DataIOManager);
    DELCARE_CLI_COMMAND_IN_CLASS(DataIOManager, status);
    Ice::CommunicatorPtr m_iceCommunicatorPtr;
    Ice::ObjectAdapterPtr m_iceAdapter;
    bool m_flagEnd = false;
    static DataIOManager * m_instance;

//    QMutex m_mutexOps;
//    std::map<std::string, std::shared_ptr<IOPair>> m_ops;

    DataFlow::DSStatusPtr m_status;

    std::map<std::string, std::shared_ptr<Pool>> m_pools;
    QReadWriteLock m_rwlPools;
    bool m_finialized = false;

    virtual ::DataFlow::DSStatusPtr getStatus(const ::Ice::Current& = ::Ice::Current());
    virtual void init(const ::DataFlow::InitServerParamsPtr&, const ::Ice::Current& = ::Ice::emptyCurrent);
    virtual void putData(const ::DataFlow::IODataPutPtr&, const ::Ice::Current& = ::Ice::emptyCurrent);
    virtual ::DataFlow::IODataGetPtr getData(const ::std::string&, ::Ice::Int, const ::Ice::Current& = ::Ice::emptyCurrent);
    virtual ::DataFlow::BatchDataProcessorStatusPtr getBatchDataProcessorStatus(const ::std::string&, ::Ice::Int, const ::Ice::Current& = ::Ice::emptyCurrent);
};


#endif //RPCIO_RPCIO_H
