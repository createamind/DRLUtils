//
// Created by Robin Huang on 7/21/17.
//

#include "stdafx.h"
#include "dataio.h"
MSP_MODULE_DECLARE("manager");

state_t CDataIO::_init_ice(const InitParams & params) {
    MSP_INFO("init Ice on %s:%d", params.host.c_str(), params.port);
    QString adpname;
    adpname.sprintf("CDataIO");

    Ice::PropertiesPtr props = Ice::createProperties();
    props->setProperty("Ice.MessageSizeMax", "0");
    props->setProperty("Ice.ThreadPool.Server.SizeMax", "256");
    props->setProperty("Ice.ThreadPool.Server.Size", "8");
    props->setProperty("Ice.ThreadPool.Client.Size", "8");
    props->setProperty("Ice.ThreadPool.Client.SizeMax", "256");
//    props->setProperty("Ice.Trace.ThreadPool", "5");
//    props->setProperty("Ice.Trace.Network", "5");
//    props->setProperty("Ice.Trace.Protocol", "1");

    int thread_pool_max_size = getEnvUInt("RPCIO_THREAD_POOL_MAX", 256);

    props->setProperty(stdsprintf("%s.ThreadPool.SizeMax", adpname.toStdString().c_str()) , stdsprintf("%d", thread_pool_max_size));
    props->setProperty(stdsprintf("%s.ThreadPool.Size", adpname.toStdString().c_str()) , stdsprintf("%d", thread_pool_max_size/4));
    Ice::InitializationData data;
    data.properties = props;
    Ice::CommunicatorPtr ic = Ice::initialize(data);
    if(!ic)
    {
        MSP_ERROR("can not init ice communicator ptr");
        return -1;
    }
    MSP_INFO("create IC");
//    ic->stringToProxy(server);
    QString str;
    assert(params.port > 0 && params.port < 65535);
    uint32_t port = params.port > 0 ? ((uint32_t)params.port) : getEnvUInt("ICE_RPCIO_LISTEN_PORT_BASE", 50000);
    // TODO: 此处尽量不要使用0.0.0.0，因为Ice会选择所有interface，导致新的连接会连接到其他地址去，然后长时间等待而超时
    // 一定注意，被坑过两次了
    std::string host = params.host.length() > 0 ? params.host : getEnvString("ICE_RPCIO_LISTEN_ADDRESS", "0.0.0.0");
    uint32_t timeout = getEnvUInt("ICE_RPCIO_TIMEOUT", 3000);
    str.sprintf("tcp -h %s -p %d -t %u", host.c_str(), port, timeout);
    Ice::ObjectAdapterPtr adapter;
    for(int retry = 0; retry < 10; retry++)
    {
        try
        {
            MSP_INFO("try to create endpoint %s", str.toStdString().c_str());
            adapter = ic->createObjectAdapterWithEndpoints(adpname.toStdString(), str.toStdString());
            if(adapter)
            {
                break;
            }
        }
        catch(Ice::Exception & e)
        {
            MSP_ERROR("createObjectAdapterWithEndpoints: %s failed, exception=%s, param=%s", adpname.toStdString().c_str(), e.what(), str.toStdString().c_str());
        }
        catch(...)
        {
            MSP_ERROR("createObjectAdapterWithEndpoints: %s failed, unknow exception, param=%s", adpname.toStdString().c_str(), str.toStdString().c_str());
        }
        sleep(1);
    }
    if(!adapter)
    {
        MSP_ERROR("can not init ice adapter");
        assert(0);
        return -1;
    }
    adapter->add(m_manager, Ice::stringToIdentity(adpname.toStdString()));
//    adapter->activate();
    MSP_INFO("Ice running on %s:%d", params.host.c_str(), params.port);
    m_manager->m_iceCommunicatorPtr = ic;
    m_manager->m_iceAdapter = adapter;
    return 0;
}

void CDataIO::onEvent(const EventParams & params) {
    if (m_manager) {
        m_manager->onEvent(params);
    }
}
