# -*- coding: utf-8 -*-
# **********************************************************************
#
# Copyright (c) 2003-2016 ZeroC, Inc. All rights reserved.
#
# This copy of Ice is licensed to you under the terms described in the
# ICE_LICENSE file included in this distribution.
#
# **********************************************************************
#
# Ice version 3.6.3
#
# <auto-generated>
#
# Generated from file `agent.ice'
#
# Warning: do not edit this file.
#
# </auto-generated>
#

from sys import version_info as _version_info_
import Ice, IcePy

# Start of module DataPool
_M_DataPool = Ice.openModule('DataPool')
__name__ = 'DataPool'

if '_t_StringSeq' not in _M_DataPool.__dict__:
    _M_DataPool._t_StringSeq = IcePy.defineSequence('::DataPool::StringSeq', (), IcePy._t_string)

if '_t_StringMap' not in _M_DataPool.__dict__:
    _M_DataPool._t_StringMap = IcePy.defineDictionary('::DataPool::StringMap', (), IcePy._t_string, IcePy._t_string)

_M_DataPool.rpcPortDataServer = 21000

if 'InitParamManager' not in _M_DataPool.__dict__:
    _M_DataPool.InitParamManager = Ice.createTempClass()
    class InitParamManager(Ice.Object):
        def __init__(self):
            pass

        def ice_ids(self, current=None):
            return ('::DataPool::InitParamManager', '::Ice::Object')

        def ice_id(self, current=None):
            return '::DataPool::InitParamManager'

        def ice_staticId():
            return '::DataPool::InitParamManager'
        ice_staticId = staticmethod(ice_staticId)

        def __str__(self):
            return IcePy.stringify(self, _M_DataPool._t_InitParamManager)

        __repr__ = __str__

    _M_DataPool.InitParamManagerPrx = Ice.createTempClass()
    class InitParamManagerPrx(Ice.ObjectPrx):

        def checkedCast(proxy, facetOrCtx=None, _ctx=None):
            return _M_DataPool.InitParamManagerPrx.ice_checkedCast(proxy, '::DataPool::InitParamManager', facetOrCtx, _ctx)
        checkedCast = staticmethod(checkedCast)

        def uncheckedCast(proxy, facet=None):
            return _M_DataPool.InitParamManagerPrx.ice_uncheckedCast(proxy, facet)
        uncheckedCast = staticmethod(uncheckedCast)

        def ice_staticId():
            return '::DataPool::InitParamManager'
        ice_staticId = staticmethod(ice_staticId)

    _M_DataPool._t_InitParamManagerPrx = IcePy.defineProxy('::DataPool::InitParamManager', InitParamManagerPrx)

    _M_DataPool._t_InitParamManager = IcePy.defineClass('::DataPool::InitParamManager', InitParamManager, -1, (), False, False, None, (), ())
    InitParamManager._ice_type = _M_DataPool._t_InitParamManager

    _M_DataPool.InitParamManager = InitParamManager
    del InitParamManager

    _M_DataPool.InitParamManagerPrx = InitParamManagerPrx
    del InitParamManagerPrx

if 'PoolInfo' not in _M_DataPool.__dict__:
    _M_DataPool.PoolInfo = Ice.createTempClass()
    class PoolInfo(object):
        def __init__(self, host='', port=0):
            self.host = host
            self.port = port

        def __hash__(self):
            _h = 0
            _h = 5 * _h + Ice.getHash(self.host)
            _h = 5 * _h + Ice.getHash(self.port)
            return _h % 0x7fffffff

        def __compare(self, other):
            if other is None:
                return 1
            elif not isinstance(other, _M_DataPool.PoolInfo):
                return NotImplemented
            else:
                if self.host is None or other.host is None:
                    if self.host != other.host:
                        return (-1 if self.host is None else 1)
                else:
                    if self.host < other.host:
                        return -1
                    elif self.host > other.host:
                        return 1
                if self.port is None or other.port is None:
                    if self.port != other.port:
                        return (-1 if self.port is None else 1)
                else:
                    if self.port < other.port:
                        return -1
                    elif self.port > other.port:
                        return 1
                return 0

        def __lt__(self, other):
            r = self.__compare(other)
            if r is NotImplemented:
                return r
            else:
                return r < 0

        def __le__(self, other):
            r = self.__compare(other)
            if r is NotImplemented:
                return r
            else:
                return r <= 0

        def __gt__(self, other):
            r = self.__compare(other)
            if r is NotImplemented:
                return r
            else:
                return r > 0

        def __ge__(self, other):
            r = self.__compare(other)
            if r is NotImplemented:
                return r
            else:
                return r >= 0

        def __eq__(self, other):
            r = self.__compare(other)
            if r is NotImplemented:
                return r
            else:
                return r == 0

        def __ne__(self, other):
            r = self.__compare(other)
            if r is NotImplemented:
                return r
            else:
                return r != 0

        def __str__(self):
            return IcePy.stringify(self, _M_DataPool._t_PoolInfo)

        __repr__ = __str__

    _M_DataPool._t_PoolInfo = IcePy.defineStruct('::DataPool::PoolInfo', PoolInfo, (), (
        ('host', (), IcePy._t_string),
        ('port', (), IcePy._t_int)
    ))

    _M_DataPool.PoolInfo = PoolInfo
    del PoolInfo

if 'InitParamPool' not in _M_DataPool.__dict__:
    _M_DataPool.InitParamPool = Ice.createTempClass()
    class InitParamPool(Ice.Object):
        def __init__(self, name='', batchSize=0, subBatchSize=0, isTrain=False, dataioHost='', dataioPort=0, trainTargets=None, kwargs=None, isContinue=False):
            self.name = name
            self.batchSize = batchSize
            self.subBatchSize = subBatchSize
            self.isTrain = isTrain
            self.dataioHost = dataioHost
            self.dataioPort = dataioPort
            self.trainTargets = trainTargets
            self.kwargs = kwargs
            self.isContinue = isContinue

        def ice_ids(self, current=None):
            return ('::DataPool::InitParamPool', '::Ice::Object')

        def ice_id(self, current=None):
            return '::DataPool::InitParamPool'

        def ice_staticId():
            return '::DataPool::InitParamPool'
        ice_staticId = staticmethod(ice_staticId)

        def __str__(self):
            return IcePy.stringify(self, _M_DataPool._t_InitParamPool)

        __repr__ = __str__

    _M_DataPool.InitParamPoolPrx = Ice.createTempClass()
    class InitParamPoolPrx(Ice.ObjectPrx):

        def checkedCast(proxy, facetOrCtx=None, _ctx=None):
            return _M_DataPool.InitParamPoolPrx.ice_checkedCast(proxy, '::DataPool::InitParamPool', facetOrCtx, _ctx)
        checkedCast = staticmethod(checkedCast)

        def uncheckedCast(proxy, facet=None):
            return _M_DataPool.InitParamPoolPrx.ice_uncheckedCast(proxy, facet)
        uncheckedCast = staticmethod(uncheckedCast)

        def ice_staticId():
            return '::DataPool::InitParamPool'
        ice_staticId = staticmethod(ice_staticId)

    _M_DataPool._t_InitParamPoolPrx = IcePy.defineProxy('::DataPool::InitParamPool', InitParamPoolPrx)

    _M_DataPool._t_InitParamPool = IcePy.defineClass('::DataPool::InitParamPool', InitParamPool, -1, (), False, False, None, (), (
        ('name', (), IcePy._t_string, False, 0),
        ('batchSize', (), IcePy._t_int, False, 0),
        ('subBatchSize', (), IcePy._t_int, False, 0),
        ('isTrain', (), IcePy._t_bool, False, 0),
        ('dataioHost', (), IcePy._t_string, False, 0),
        ('dataioPort', (), IcePy._t_int, False, 0),
        ('trainTargets', (), _M_DataPool._t_StringSeq, False, 0),
        ('kwargs', (), _M_DataPool._t_StringMap, False, 0),
        ('isContinue', (), IcePy._t_bool, False, 0)
    ))
    InitParamPool._ice_type = _M_DataPool._t_InitParamPool

    _M_DataPool.InitParamPool = InitParamPool
    del InitParamPool

    _M_DataPool.InitParamPoolPrx = InitParamPoolPrx
    del InitParamPoolPrx

if 'Manager' not in _M_DataPool.__dict__:
    _M_DataPool.Manager = Ice.createTempClass()
    class Manager(Ice.Object):
        def __init__(self):
            if Ice.getType(self) == _M_DataPool.Manager:
                raise RuntimeError('DataPool.Manager is an abstract class')

        def ice_ids(self, current=None):
            return ('::DataPool::Manager', '::Ice::Object')

        def ice_id(self, current=None):
            return '::DataPool::Manager'

        def ice_staticId():
            return '::DataPool::Manager'
        ice_staticId = staticmethod(ice_staticId)

        def createPool(self, param, current=None):
            pass

        def closePool(self, name, current=None):
            pass

        def __str__(self):
            return IcePy.stringify(self, _M_DataPool._t_Manager)

        __repr__ = __str__

    _M_DataPool.ManagerPrx = Ice.createTempClass()
    class ManagerPrx(Ice.ObjectPrx):

        def createPool(self, param, _ctx=None):
            return _M_DataPool.Manager._op_createPool.invoke(self, ((param, ), _ctx))

        def begin_createPool(self, param, _response=None, _ex=None, _sent=None, _ctx=None):
            return _M_DataPool.Manager._op_createPool.begin(self, ((param, ), _response, _ex, _sent, _ctx))

        def end_createPool(self, _r):
            return _M_DataPool.Manager._op_createPool.end(self, _r)

        def closePool(self, name, _ctx=None):
            return _M_DataPool.Manager._op_closePool.invoke(self, ((name, ), _ctx))

        def begin_closePool(self, name, _response=None, _ex=None, _sent=None, _ctx=None):
            return _M_DataPool.Manager._op_closePool.begin(self, ((name, ), _response, _ex, _sent, _ctx))

        def end_closePool(self, _r):
            return _M_DataPool.Manager._op_closePool.end(self, _r)

        def checkedCast(proxy, facetOrCtx=None, _ctx=None):
            return _M_DataPool.ManagerPrx.ice_checkedCast(proxy, '::DataPool::Manager', facetOrCtx, _ctx)
        checkedCast = staticmethod(checkedCast)

        def uncheckedCast(proxy, facet=None):
            return _M_DataPool.ManagerPrx.ice_uncheckedCast(proxy, facet)
        uncheckedCast = staticmethod(uncheckedCast)

        def ice_staticId():
            return '::DataPool::Manager'
        ice_staticId = staticmethod(ice_staticId)

    _M_DataPool._t_ManagerPrx = IcePy.defineProxy('::DataPool::Manager', ManagerPrx)

    _M_DataPool._t_Manager = IcePy.defineClass('::DataPool::Manager', Manager, -1, (), True, False, None, (), ())
    Manager._ice_type = _M_DataPool._t_Manager

    Manager._op_createPool = IcePy.Operation('createPool', Ice.OperationMode.Normal, Ice.OperationMode.Normal, False, None, (), (((), _M_DataPool._t_InitParamPool, False, 0),), (), ((), _M_DataPool._t_PoolInfo, False, 0), ())
    Manager._op_closePool = IcePy.Operation('closePool', Ice.OperationMode.Normal, Ice.OperationMode.Normal, False, None, (), (((), IcePy._t_string, False, 0),), (), None, ())

    _M_DataPool.Manager = Manager
    del Manager

    _M_DataPool.ManagerPrx = ManagerPrx
    del ManagerPrx

if 'Pool' not in _M_DataPool.__dict__:
    _M_DataPool.Pool = Ice.createTempClass()
    class Pool(Ice.Object):
        def __init__(self):
            if Ice.getType(self) == _M_DataPool.Pool:
                raise RuntimeError('DataPool.Pool is an abstract class')

        def ice_ids(self, current=None):
            return ('::DataPool::Pool', '::Ice::Object')

        def ice_id(self, current=None):
            return '::DataPool::Pool'

        def ice_staticId():
            return '::DataPool::Pool'
        ice_staticId = staticmethod(ice_staticId)

        def shutdown(self, current=None):
            pass

        def getPid(self, current=None):
            pass

        def __str__(self):
            return IcePy.stringify(self, _M_DataPool._t_Pool)

        __repr__ = __str__

    _M_DataPool.PoolPrx = Ice.createTempClass()
    class PoolPrx(Ice.ObjectPrx):

        def shutdown(self, _ctx=None):
            return _M_DataPool.Pool._op_shutdown.invoke(self, ((), _ctx))

        def begin_shutdown(self, _response=None, _ex=None, _sent=None, _ctx=None):
            return _M_DataPool.Pool._op_shutdown.begin(self, ((), _response, _ex, _sent, _ctx))

        def end_shutdown(self, _r):
            return _M_DataPool.Pool._op_shutdown.end(self, _r)

        def getPid(self, _ctx=None):
            return _M_DataPool.Pool._op_getPid.invoke(self, ((), _ctx))

        def begin_getPid(self, _response=None, _ex=None, _sent=None, _ctx=None):
            return _M_DataPool.Pool._op_getPid.begin(self, ((), _response, _ex, _sent, _ctx))

        def end_getPid(self, _r):
            return _M_DataPool.Pool._op_getPid.end(self, _r)

        def checkedCast(proxy, facetOrCtx=None, _ctx=None):
            return _M_DataPool.PoolPrx.ice_checkedCast(proxy, '::DataPool::Pool', facetOrCtx, _ctx)
        checkedCast = staticmethod(checkedCast)

        def uncheckedCast(proxy, facet=None):
            return _M_DataPool.PoolPrx.ice_uncheckedCast(proxy, facet)
        uncheckedCast = staticmethod(uncheckedCast)

        def ice_staticId():
            return '::DataPool::Pool'
        ice_staticId = staticmethod(ice_staticId)

    _M_DataPool._t_PoolPrx = IcePy.defineProxy('::DataPool::Pool', PoolPrx)

    _M_DataPool._t_Pool = IcePy.defineClass('::DataPool::Pool', Pool, -1, (), True, False, None, (), ())
    Pool._ice_type = _M_DataPool._t_Pool

    Pool._op_shutdown = IcePy.Operation('shutdown', Ice.OperationMode.Normal, Ice.OperationMode.Normal, False, None, (), (), (), None, ())
    Pool._op_getPid = IcePy.Operation('getPid', Ice.OperationMode.Normal, Ice.OperationMode.Normal, False, None, (), (), (), ((), IcePy._t_int, False, 0), ())

    _M_DataPool.Pool = Pool
    del Pool

    _M_DataPool.PoolPrx = PoolPrx
    del PoolPrx

if 'SubBatchPool' not in _M_DataPool.__dict__:
    _M_DataPool.SubBatchPool = Ice.createTempClass()
    class SubBatchPool(Ice.Object):
        def __init__(self):
            pass

        def ice_ids(self, current=None):
            return ('::DataPool::SubBatchPool', '::Ice::Object')

        def ice_id(self, current=None):
            return '::DataPool::SubBatchPool'

        def ice_staticId():
            return '::DataPool::SubBatchPool'
        ice_staticId = staticmethod(ice_staticId)

        def __str__(self):
            return IcePy.stringify(self, _M_DataPool._t_SubBatchPool)

        __repr__ = __str__

    _M_DataPool.SubBatchPoolPrx = Ice.createTempClass()
    class SubBatchPoolPrx(Ice.ObjectPrx):

        def checkedCast(proxy, facetOrCtx=None, _ctx=None):
            return _M_DataPool.SubBatchPoolPrx.ice_checkedCast(proxy, '::DataPool::SubBatchPool', facetOrCtx, _ctx)
        checkedCast = staticmethod(checkedCast)

        def uncheckedCast(proxy, facet=None):
            return _M_DataPool.SubBatchPoolPrx.ice_uncheckedCast(proxy, facet)
        uncheckedCast = staticmethod(uncheckedCast)

        def ice_staticId():
            return '::DataPool::SubBatchPool'
        ice_staticId = staticmethod(ice_staticId)

    _M_DataPool._t_SubBatchPoolPrx = IcePy.defineProxy('::DataPool::SubBatchPool', SubBatchPoolPrx)

    _M_DataPool._t_SubBatchPool = IcePy.defineClass('::DataPool::SubBatchPool', SubBatchPool, -1, (), False, False, None, (), ())
    SubBatchPool._ice_type = _M_DataPool._t_SubBatchPool

    _M_DataPool.SubBatchPool = SubBatchPool
    del SubBatchPool

    _M_DataPool.SubBatchPoolPrx = SubBatchPoolPrx
    del SubBatchPoolPrx

# End of module DataPool