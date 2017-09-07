#coding: utf-8

import numpy as np
from ..utils import logger
from .base import AgentBase, StepResult
import Ice
import six

class AgentSingleLoop(AgentBase):
    def _init(self):
        super(AgentSingleLoop, self)._init()
        self._dataioClient = None
        dio_addr = self._kwargs.get('dio_addr', None)
        self._dio = None
        if dio_addr:
            from ..rpcio.pyclient import DataIOClient
            dio_name = self._kwargs.get('dio_name')
            self._dio = DataIOClient(dio_name, self._agentIdent, dio_addr[0], dio_addr[1])

    def run(self):
        from ..dataflow.load_slice import DataFlow
        from .base import StepResult
        dio = self._dio
        if dio:
            dio.waitForConnected()
            logger.info("[{}]: dio connected, start to run".format(self._agentIdent))
        gamma = self._kwargs.get("gamma", 0.99)
        local_t_max = self._kwargs.get("local_t_max", 5)
        enable_gae = self._kwargs.get("enable_gae", False)

        logger.info("[{:04}]: agent start run with gamma={}, local_t_max={}, enable_gae={}, isTrain={}"
                    .format(self._agentIdent, gamma, local_t_max, enable_gae, self._isTrain))
        memory = []     # type: list[StepResult]
        state = self.reset()

        pred_seqlen = np.array([1], dtype=np.int32)
        pred_resetRNN = np.zeros([1], dtype=np.int32)
        _agentIdent = np.array([self._agentIdent], dtype=np.int32)
        is_over = True
        predictCount = 0
        trainCount = 0
        while True:
            try:
                if len(state.shape) < 3:
                    state = state[np.newaxis, np.newaxis]
                # logger.info("[{:04}]: before putData, state = {}, predictCount={}".format(self._agentIdent, state.shape, predictCount))
                if is_over: pred_resetRNN[0] = 1
                dio.putData(isTrain=False, agentIdent=_agentIdent, state=state, sequenceLength=pred_seqlen, resetRNN=pred_resetRNN)
                # logger.info("[{:04d}]: after putData")
                pred_resetRNN[0] = 0
                pred = dio.getData()
                # logger.info("[{}]: got data: {}".format(self._agentIdent, pred))
                predictCount += 1
                policy = pred['policy'][0]
                value = pred['value'][0]
                ident = pred['agentIdent'][0]
                assert(ident == self._agentIdent)
                result = self.step({k:v[0] for k, v in pred.items()})
                state = result.state
                # logger.info("state = {}, actoin={}, reward={}, is_over={}".format(state.shape, action.shape, reward.shape, is_over))
                # logger.info("act={}, reward = {}, value= {:04f}, policy = {}".format(result.action, result.reward, value, policy))
                is_over = result.isOver
                if self._isTrain:
                    memory.append(result)
                    len_mem = len(memory)
                    if len_mem > local_t_max or is_over:
                        if (not is_over) or (is_over and len_mem > local_t_max):
                            last = memory[-1]
                            mem = memory[:-1]
                            init_r = last.value
                        else:
                            init_r = 0.
                            mem = memory

                        states = np.concatenate([m.state[np.newaxis, np.newaxis] for m in mem], axis=1).astype(np.float32)
                        actions = np.concatenate([m.action[np.newaxis, np.newaxis] for m in mem], axis=1)
                        seqLength = np.array([states.shape[1]], dtype=np.int32)
                        isOver = np.array([is_over], dtype=np.int32)

                        if enable_gae:
                            def discount(x, gamma):
                                from scipy.signal import lfilter
                                return lfilter(
                                    [1], [1, -gamma], x[::-1], axis=0)[::-1]

                            rewards_plus = np.asarray([m.reward for m in mem] + [float(init_r)])
                            discounted_rewards = discount(rewards_plus, gamma)[:-1]
                            values_plus = np.asarray([m.value for m in mem] + [float(init_r)])
                            _rewards = np.asarray([m.reward for m in mem]).astype(np.float32)
                            advantages = (_rewards + gamma * values_plus[1:] - values_plus[:-1]).astype(np.float32)
                            advantages = discount(advantages, gamma)
                            rewards = discounted_rewards
                        else:
                            R = float(init_r)
                            Rs = []
                            for idx, k in enumerate(mem[::-1]):
                                R = k.reward + gamma * R
                                Rs.append(R)
                            Rs.reverse()
                            # logger.info("state={}, init_r = {:.3f}, Rs={}, reward={}, action={}"
                            #             .format(states.dtype, init_r,
                            #                     ['{:.3f}'.format(r) for r in Rs],
                            #                     ['{:.3f}'.format(float(k.reward)) for k in mem],
                            #                     actions.flatten().tolist(),
                            #                     ))
                            rewards = np.asarray(Rs)
                            advantages = rewards - np.asarray([m.value for m in mem])
                        kwargs = {}
                        for k, v in mem[0]._kwargs.items():
                            if isinstance(v, np.ndarray):
                                kwargs[k] = np.concatenate([m._kwargs[k][np.newaxis, np.newaxis] for m in mem], axis=1).astype(np.float32)
                            elif isinstance(v, int) or isinstance(v, np.int32):
                                kwargs[k] = np.concatenate([np.array([[m._kwargs[k]]]) for m in mem],axis = 1).astype(np.int32)
                            elif isinstance(v, float) or isinstance(v, np.float32) or isinstance(v, np.float64):
                                kwargs[k] = np.concatenate([np.array([[m._kwargs[k]]]) for m in mem], axis=1).astype(np.float32)
                            else:
                                raise Exception("unknow type {} of {}".format(type(v), k))
                        # logger.info("try putTrain, trainCount={}".format(trainCount))
                        dio.putData(isTrain=True,
                                    agentIdent=_agentIdent,
                                    state = states,
                                    action = actions,
                                    reward = rewards[np.newaxis],
                                    advantage = advantages[np.newaxis],
                                    sequenceLength = seqLength,
                                    resetRNN = isOver,
                                    **kwargs
                                    )
                        # logger.info("after putTrain, trainCount={}".format(trainCount))
                        trainCount += 1
                        if not is_over:
                            memory = [last]
                        else:
                            memory = []
                if is_over:
                    self.finishEpisode()
                    state = self.reset()
            except DataFlow.ExceptionClosed as e:
                break
            except Ice.ConnectionRefusedException as e:
                break
            except Ice.ConnectionLostException as e:
                break
            except Ice.Exception as e:
                raise e
