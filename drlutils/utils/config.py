
import os
if 'ICE_RPCIO_LISTEN_PORT_BASE' not in os.environ:
    os.environ['ICE_RPCIO_LISTEN_PORT_BASE'] = "50000"
ICE_RPCIO_LISTEN_PORT_BASE = int(os.environ['ICE_RPCIO_LISTEN_PORT_BASE'])
