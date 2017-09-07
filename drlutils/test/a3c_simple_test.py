#coding: utf-8


import numpy as np
# import six
import tensorflow as tf
from drlutils.utils import logger

FEATURE_DIM = 5
ACTION_DIM = 2
GAMMA = 0.9
LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 1000
EVAL_EPISODE = 0
POOL_SIZE = 128
TRAIN_BATCH_SIZE=32
PREDICT_MIN_BATCH_SIZE = 1    # batch for efficient forward
PREDICT_MAX_BATCH_SIZE = 16    # batch for efficient forward
PREDICTOR_THREAD_PER_GPU = 4
INIT_LEARNING_RATE_A = 1e-4
INIT_LEARNING_RATE_C = 1e-3

evaluators = []

import docopt, os
args = docopt.docopt(
'''
Usage:
    main.py train [--gpu GPU] [options]
    main.py dataserver [options]
    main.py infer  [--gpu GPU] [--load MODULEWEIGHTS] [options]
    main.py test  [options]

Options:
    -h --help                   Show the help
    --version                   Show the version
    --gpu GPU                   comma separated list of GPU(s)
    --load MODELWEIGHTS         load weights from file
    --simulators SIM            simulator count             [default: 16]
    --debug_mode                set debug mode
    --a3c_instance_idx IDX      set a3c_instance_idx            [default: 0]
    --continue                  continue mode, load saved weights
    --tfdbg
    --log LEVEL                 log level                       [default: info]
    --target TARGET             test target
    --fake_agent                use fake agent to debug          
''', version='0.1')

if args['train']:
    from drlutils.dataflow.tensor_io import TensorIO_AgentPools
    data_io = TensorIO_AgentPools(train_targets=['Pendulum'])

# from tensorpack.tfutils import summary
from drlutils.model.base import ModelBase
class Model(ModelBase):
    def _get_inputs(self, select = None):
        from drlutils.model.base import ModelBase, InputDesc
        inputs = [
                        InputDesc(tf.float32, (None, FEATURE_DIM), 'state'),
                        InputDesc(tf.float32, (None, ACTION_DIM), 'action'),
                        InputDesc(tf.float32, (None,), 'reward'),
                        InputDesc(tf.float32, (None,), 'advantage'),
                        InputDesc(tf.int32, (), 'sequenceLength'),
                        InputDesc(tf.int32, (), 'resetRNN'),
                        # InputDesc(tf.float32, (None,), 'action_prob'),
                        ]
        if select is None:
            return inputs

        assert(type(select) in [list, tuple])
        return [i for i in inputs if i.name in select]

    def _build_nn(self, tensor_io):
        from drlutils.dataflow.tensor_io import TensorIO
        assert(isinstance(tensor_io, TensorIO))
        from drlutils.model.base import get_current_nn_context
        from tensorpack.tfutils.common import get_global_step_var
        global_step = get_global_step_var()
        nnc = get_current_nn_context()
        is_training = nnc.is_training
        i_state = tensor_io.getInputTensor('state')
        i_agentIdent = tensor_io.getInputTensor('agentIdent')
        i_sequenceLength = tensor_io.getInputTensor('sequenceLength')
        i_resetRNN = tensor_io.getInputTensor('resetRNN')

        prefix = 'T:' if is_training else 'P:'
        # i_state = tf.Print(i_state, [i_agentIdent, tf.shape(i_agentIdent)], prefix + 'agentIdent = ', summarize=POOL_SIZE * 10)
        # i_state = tf.Print(i_state, [i_state, tf.shape(i_state)], prefix + 'State = ', summarize=POOL_SIZE*10)
        # i_state = tf.Print(i_state, [i_sequenceLength, tf.shape(i_sequenceLength)], prefix + 'SeqLen = ', summarize=POOL_SIZE*10)
        # i_state = tf.Print(i_state, [i_resetRNN, tf.shape(i_resetRNN)], prefix + 'resetRNN = ', summarize=POOL_SIZE*10)
        rnn_hidde_size = 64
        cell = tf.nn.rnn_cell.LSTMCell(rnn_hidde_size)
        rnn_output = self._buildRNN(i_state, cell, batchSize=POOL_SIZE, i_sequenceLength=i_sequenceLength, i_resetRNN=i_resetRNN, i_agentIdent=i_agentIdent)
        rnn_output = tf.reshape(rnn_output, [-1, rnn_hidde_size])
        with tf.control_dependencies([rnn_output]):
            policy = tf.identity(i_state[:, -1, 2], name='policy')
            value = tf.identity(i_state[:, -1, 3] + 1, name='value')

        lp = tf.layers.dense(rnn_output, 128, name='fc-p')
        lv = tf.layers.dense(rnn_output, 128, name='fc-v')

        if not is_training:
            # policy = tf.Print(policy, [policy], prefix + 'Policy = ', summarize=BATCH_SIZE)
            # policy = tf.Print(policy, [value], prefix + 'Value = ', summarize=BATCH_SIZE)
            tensor_io.setOutputTensors(policy, value, tf.identity(i_state, name='state'))
            return
        i_action = tensor_io.getInputTensor('action')
        # i_action = tf.reshape(i_action, (-1, ACTION_DIM))
        i_futurereward = tensor_io.getInputTensor('reward')


        lv = tf.Print(lv, [i_state, tf.shape(i_state)], prefix + 'State = ', summarize=POOL_SIZE*10)
        # lv = tf.Print(lv, [i_action, tf.shape(i_action)], prefix + 'Action = ', summarize=POOL_SIZE*10)
        # lv = tf.Print(lv, [i_agentIdent, tf.shape(i_agentIdent)], prefix + 'agentIdent = ', summarize=POOL_SIZE*10)
        lv = tf.Print(lv, [i_sequenceLength, tf.shape(i_sequenceLength)], prefix + 'SeqLen = ', summarize=POOL_SIZE*10)
        lv = tf.Print(lv, [i_resetRNN, tf.shape(i_resetRNN)], prefix + 'resetRNN = ', summarize=POOL_SIZE*10)
        lv = tf.Print(lv, [i_futurereward, tf.shape(i_futurereward)], prefix + 'reward = ', summarize=POOL_SIZE*10)

        with tf.control_dependencies([tf.assert_equal(i_action, i_state[:, :, -2:], message="no equal")]):
            loss_policy = tf.reduce_mean(tf.square(lp - 0.5))
            loss_value = tf.reduce_mean(tf.square(lv - 0.1))
            return loss_policy, loss_value


    def _build_graph(self):
        from drlutils.model.base import NNContext
        gpu_towers = [0]
        for towerIdx, tower in enumerate(gpu_towers):
            with NNContext("TestTrain", device='/device:GPU:%d' % tower, summary_collection=towerIdx==0, is_training=True):
                data_io.createPool('Test-%d'%towerIdx, POOL_SIZE, sub_batch_size = 1, is_training = True,
                                   train_batch_size=TRAIN_BATCH_SIZE,
                                   predict_min_batch_size=PREDICT_MIN_BATCH_SIZE,
                                   predict_max_batch_size=PREDICT_MAX_BATCH_SIZE,
                                   torcsIdxOffset = data_io.getAgentCountTotal())
                tensor_io = data_io.getTensorIO("Test-%d/train"%towerIdx, self._get_inputs(), queue_size=50,
                                                min_batch_size=TRAIN_BATCH_SIZE,
                                                max_batch_size=TRAIN_BATCH_SIZE,
                                                is_training=True)
                loss_policy, loss_value = self._build_nn(tensor_io)
                self._addLoss('policy%d'%towerIdx, loss_policy, opt=self._get_optimizer('actor'),
                              trainOpGroup='Test', tensor_io=tensor_io)
                self._addLoss('value%d'%towerIdx, loss_value, opt=self._get_optimizer('critic'),
                              trainOpGroup='Test', tensor_io=tensor_io)
                if not hasattr(self, '_weights_test_train'):
                    self._weights_test_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            with NNContext("TestPred", device='/device:GPU:%d'%tower, summary_collection=False, is_training=False):
                for pidx in range(PREDICTOR_THREAD_PER_GPU):
                    with tf.variable_scope('pred', reuse=(towerIdx>0 or pidx>0)) as vs:
                        tensor_io = data_io.getTensorIO("Test-%d/pred%d"%(towerIdx, pidx),
                                                        self._get_inputs(['state', 'sequenceLength', 'resetRNN']),
                                                        is_training=False,
                                                        min_batch_size = PREDICT_MIN_BATCH_SIZE,
                                                        max_batch_size=PREDICT_MAX_BATCH_SIZE,
                                                        )
                        self._build_nn(tensor_io)
                        self._addThreadOp('Test-%d/pred'%towerIdx, tensor_io.getOutputOp())
                        if not hasattr(self, '_weights_test_pred'):
                            self._weights_test_pred = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
                            assert(len(self._weights_test_pred) == len(self._weights_test_train))
                            self._sync_op_pred = tf.group(*[d.assign(s) for d, s in zip(self._weights_test_pred, self._weights_test_train)])
                        # self._sync_op_pred = tf.group(*[d.assign(s + tf.random_normal(tf.shape(s), stddev=2e-4)) for d, s in zip(self._weights_ad_pred, self._weights_ad_train)])

        for eidx, evaluator in enumerate(evaluators):
            with NNContext(evaluator.name, device='/device:GPU:%d' % gpu_towers[eidx%len(gpu_towers)], add_variable_scope=True, summary_collection=True, is_evaluating=True):
                data_io.createPool(evaluator.name.replace('/', '_'),
                                   evaluator._batch_size,
                                   sub_batch_size=1,
                                   is_training=False,
                                   torcsIdxOffset = data_io.getAgentCountTotal())
                tensor_io = evaluator.getTensorIO(self._get_inputs(['state', 'sequenceLength', 'resetRNN']))
                self._build_nn(tensor_io)
                evaluator.set_weights(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=evaluator.name))

    def _calc_learning_rate(self, name, epoch, lr):
        def _calc():
            lr_init = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
            lrs = [(0, lr_init * 0.25),
                   (1, lr_init * 0.5),
                   (2, lr_init * 1.0),
                   (3, lr_init * 0.5),
                   (4, lr_init * 0.25),
                   (5, lr_init * 0.128),
                   # (100, lr_init/16),
                   ]
            for idx in range(len(lrs) - 1):
                if epoch >= lrs[idx][0] and epoch < lrs[idx+1][0]:
                    return lrs[idx][1]
            return lrs[-1][1]
        ret = _calc()
        return ret

    def _get_optimizer(self, name):
        from tensorpack.tfutils import optimizer
        from tensorpack.tfutils.gradproc import SummaryGradient, GlobalNormClip, MapGradient
        init_lr = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
        import tensorpack.tfutils.symbolic_functions as symbf
        lr = symbf.get_scalar_var('learning_rate/' + name, init_lr, summary=True)
        opt = tf.train.RMSPropOptimizer(lr)
        logger.info("create opt {}".format(name))
        # if name == 'critic':
        #     gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.05), regex='^critic/.*')]
        # elif name == 'actor':
        #     gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1), regex='^actor/.*')]
        # else: assert(0)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.05))]
        gradprocs.append(SummaryGradient())
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt

def get_config():
    M = Model()

    dataflow = data_io
    from tensorpack.callbacks.base import Callback
    class CBSyncWeight(Callback):
        def _before_run(self, ctx):
            if self.local_step % 10 == 0:
                return [M._sync_op_pred]
    import functools
    from tensorpack.train.config import TrainConfig
    from tensorpack.callbacks.saver import ModelSaver
    from tensorpack.callbacks.graph import RunOp
    from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter, HyperParamSetterWithFunc
    from tensorpack.tfutils import sesscreate
    from tensorpack.tfutils.common import get_default_sess_config
    import tensorpack.tfutils.symbolic_functions as symbf

    return TrainConfig(
        model=M,
        data=dataflow,
        callbacks=[
            ModelSaver(),
            HyperParamSetterWithFunc(
                'learning_rate/actor',
                functools.partial(M._calc_learning_rate, 'actor')),
            HyperParamSetterWithFunc(
                'learning_rate/critic',
                functools.partial(M._calc_learning_rate, 'critic')),
            CBSyncWeight(),
            data_io,
        ] + evaluators,
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    from drlutils.dataflow.server import DataFlowServer
    from drlutils.dataflow.pool import AgentPoolNoBatchMode
    from drlutils.agent.single_loop import AgentSingleLoop

    class Agent(AgentSingleLoop):
        def _init(self):
            super(Agent, self)._init()
            self._stepCount = 0

        def _current_state(self):
            self._cur_ob = self._agentIdent + self._episodeSteps * np.ones(FEATURE_DIM, dtype=np.float32)
            self._cur_ob[0] = self._agentIdent
            self._cur_ob[1] = self._maxStep - self._episodeSteps
            self._cur_ob[2] = self._stepCount
            self._curStep = self._episodeSteps
            self._stepCount += 1
            return self._cur_ob

        def _reset(self):
            self._maxStep = self._rng.randint(100, 105)
            self._current_state()
            return self._cur_ob

        def _step(self, pred):
            act, value, i_state = pred['policy'], pred['value'], pred['state']
            assert(np.allclose(i_state, self._cur_ob))
            assert(np.allclose(act, self._stepCount-1)), '[{:04d}]: act={}, episodeSteps={}[{}], pred={}'\
                .format(self._agentIdent, act, self._curStep, self._episodeSteps, pred)
            assert(np.allclose(value, self._agentIdent + self._curStep+1))
            ob = self._current_state()
            act = self._cur_ob[-2:]
            reward = 1.
            isOver = self._episodeSteps >= self._maxStep
            from drlutils.agent.base import StepResult
            return StepResult(ob, act, reward, isOver, value)

    class AgentPool(AgentPoolNoBatchMode):
        def _init(self):
            self._cls_agent = Agent
            super(AgentPool, self)._init()

    if args['train']:
        from drlutils.utils import logger
        logger.info("Begin train task")
        clsPool = AgentPool
        from drlutils.evaluator.evaluator import EvaluatorBase
        class Evaluator(EvaluatorBase):
            def _init(self):
                super(Evaluator, self)._init()

        if args['--gpu']:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(sorted(list(set(args['--gpu'].split(',')))))

        dirname = '/tmp/a3c_test/trainlog'
        from tensorpack.utils import logger
        logger.set_logger_dir(dirname, action='k' if args['--continue'] else 'b')
        logger.info("Backup source to {}/source/".format(logger.LOG_DIR))
        source_dir = os.path.dirname(__file__)
        os.system('rm -f {}/sim-*; mkdir -p {}/source; rsync -a --exclude="core*" --exclude="cmake*" --exclude="build" {} {}/source/'
                  .format(source_dir, logger.LOG_DIR, source_dir, logger.LOG_DIR))

        # if not args['--fake_agent']:
        #     logger.info("Create simulators, please wait...")
            # clsPool.startNInstance(BATCH_SIZE)
        evaluators = [
            Evaluator(data_io, 'evaluate/valid', batch_size=4, is_training=False, cls_pool=clsPool, sync_step = 10),
        ]
        from tensorpack.utils.gpu import get_nr_gpu
        from tensorpack.train.feedfree import QueueInputTrainer
        from tensorpack.tfutils.sessinit import get_model_loader
        nr_gpu = get_nr_gpu()
        # trainer = QueueInputTrainer
        assert(nr_gpu > 0)
        if nr_gpu > 1:
            predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
        from drlutils.train.multigpu import MultiGPUTrainer
        trainer = MultiGPUTrainer
        config = get_config()
        if os.path.exists(logger.LOG_DIR + '/checkpoint'):
            from tensorpack.tfutils.sessinit import SaverRestore
            config.session_init = SaverRestore(logger.LOG_DIR + '/checkpoint')
        elif args['--load']:
            config.session_init = get_model_loader(args['--load'])
        config.tower = train_tower
        config.predict_tower = predict_tower
        trainer(config).train()
        import sys
        sys.exit(0)
    elif args['dataserver']:
        import os
        from drlutils.dataflow.server import DataFlowServer
        clsPool = AgentPool
        try:
            ds = DataFlowServer(clsPool, local_t_max=LOCAL_TIME_MAX, gamma=GAMMA)
            ds.run()
        except KeyboardInterrupt:
            pass
        import sys
        sys.exit(0)


