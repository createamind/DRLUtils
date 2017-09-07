//
// Created by Robin Huang on 7/21/17.
//
#include "stdafx.h"
#include "dataio.h"
using namespace msp_cli;

DEFINE_CLI_COMMAND_IN_CLASS(DataIOManager, status) {
    for(auto it: m_pools) {
        it.second->printStatus(ctx);
    }
}

void DataIOManager::_callback_cli_add_command(msp_cli::CContext &ctx)
{
    CCommand * c = nullptr;

    CLI_ADD_COMMAND_IN_CLASS("status", status, NULL, NULL, "print manager status");
}


state_t DataIOManager::_init_cli()
{
#define BANNER "##################################DataIO##################################"
    auto port = (uint16_t )getEnvInt("CLI_TELNET_PORT", 7000);
    if (port > 0) {
        CLI_START_IN_CLASS("DataIO", BANNER, (uint16_t) port);
        m_cli = cli;
    }
    return 0;
}