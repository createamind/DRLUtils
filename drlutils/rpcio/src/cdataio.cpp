//
// Created by Robin Huang on 7/21/17.
//



#include "stdafx.h"
#include "cdataio.h"
#include "dataio.h"
#include "src/utils/utils_stack_trace.h"
#include "src/utils/numpy_inc.h"
#include "src/utils/tf_tensor_utils.h"
MSP_MODULE_DECLARE("dataio");

CDataIO::CDataIO() {
    msp_log_init();
    if (getEnvBool("ENABLE_STACK_TRACE", false))
        init_stack_trace();
    init_numpy();
    tensorflow::ImportNumpy();
}

int CDataIO::initialize(const InitParams &params) {
    MSP_INFO("CDataIO::initializing...");
    assert(m_manager == nullptr);
    m_manager = new DataIOManager();
    _init_ice(params);
    m_manager->initialize(params);
    MSP_INFO("CDataIO::initialized");
    return 0;
}

void CDataIO::start() {
    MSP_INFO("CDataIO::start");
    if (m_manager) {
        m_manager->start();
        if (m_manager->m_iceAdapter) {
            m_manager->m_iceAdapter->activate();
        }
    }
}

int CDataIO::createPool(const PoolInitParams &params) {
    if (m_manager) {
        return m_manager->createPool(params);
    }
    return -1;
}

void CDataIO::close() {
    MSP_INFO("CDataIO::close");
    if(m_manager) {
        m_manager->close();
        auto iceAdapter = m_manager->m_iceAdapter;
        auto ic = m_manager->m_iceCommunicatorPtr;
        if (iceAdapter) {
            iceAdapter->deactivate();
        }
        if (ic) {
            ic->destroy();
        }
        else {
            delete m_manager;
        }
        m_manager = nullptr;
    }
    MSP_INFO("CDataIO::closed");
}
