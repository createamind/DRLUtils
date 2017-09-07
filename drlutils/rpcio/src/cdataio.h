//
// Created by Robin Huang on 7/21/17.
//

#ifndef RPCIO_CRPCIO_H
#define RPCIO_CRPCIO_H
class DataIOManager;
typedef struct InitParams {
    std::string host;
    int port = 6000;
}InitParams;

typedef struct EventParams {
    std::string name;
    std::string paramString;
    int paramInt = 0;
}EventParams;

typedef struct PoolInitParams {
    std::string name;
    int size;
    int subBatchSize = 1;
    int trainBatchSize = 0;
    int predictMinBatchSize = 1;
    int predictMaxBatchSize = 32;
}PoolInitParams;


class CDataIO {
public:
    CDataIO();
    int initialize(const InitParams & params);
    void onEvent(const EventParams & params);
    void start();
    void close();
    int createPool(const PoolInitParams & params);
private:
    state_t _init_ice(const InitParams & params);
    DataIOManager * m_manager;
};



#endif //RPCIO_CRPCIO_H
