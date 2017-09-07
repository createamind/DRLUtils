#coding: utf-8


import numpy as np
# import six
import tensorflow as tf
from drlutils.utils import logger

FEATURE_DIM = 3
ACTION_DIM = 1
GAMMA = 0.9
LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 1000
EVAL_EPISODE = 0
POOL_SIZE = 64
TRAIN_BATCH_SIZE = 16
PREDICT_MAX_BATCH_SIZE = 8    # batch for efficient forward
PREDICTOR_THREAD_PER_GPU = 2
INIT_LEARNING_RATE_A = 5e-5
INIT_LEARNING_RATE_C = 5e-4

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
        inputs = [InputDesc(tf.int32, (), 'agentIdent'),
                        InputDesc(tf.float32, (None, FEATURE_DIM), 'state'),
                        InputDesc(tf.float32, (None, ACTION_DIM), 'action'),
                        InputDesc(tf.float32, (None,), 'futurereward'),
                        InputDesc(tf.float32, (None,), 'advantage'),
                        InputDesc(tf.int32, (), 'sequenceLength'),
                        InputDesc(tf.int32, (), 'resetRNN'),
                        # InputDesc(tf.float32, (None,), 'action_prob'),
                        ]
        if select is None:
            return inputs

        assert(type(select) in [list, tuple])
        return [i for i in inputs if i.name in select]

    def _build_ad_nn(self, tensor_io):
        return self._build_ad_rnn(tensor_io)
        # return self._build_ad_ff(tensor_io)

    def _build_ad_rnn(self, tensor_io):
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
        l = i_state
        # l = tf.Print(l, [i_state, tf.shape(i_state)], 'State = ')
        # l = tf.Print(l, [i_agentIdent, tf.shape(i_agentIdent)], 'agentIdent = ')
        # l = tf.Print(l, [i_sequenceLength, tf.shape(i_sequenceLength)], 'SeqLen = ')
        # l = tf.Print(l, [i_resetRNN, tf.shape(i_resetRNN)], 'resetRNN = ')
        w_init = None #tf.random_normal_initializer(0., .1)
        with tf.variable_scope('critic', reuse=nnc.reuse) as vs:
            cell_size = 32
            def _get_cell():
                cell = tf.nn.rnn_cell.BasicRNNCell(cell_size)
                # if is_training:
                #     cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.9, input_keep_prob=0.9)
                return cell

            cell = _get_cell()
            rnn_outputs = self._buildRNN(l, cell, tensor_io.batchSize,
                                         i_agentIdent=i_agentIdent,
                                         i_sequenceLength=i_sequenceLength,
                                         i_resetRNN=i_resetRNN,
                                         )
            rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size])
            l = rnn_outputs
            l = tf.layers.dense(l, 50, activation=tf.nn.relu6, kernel_initializer=w_init, name='fc-0')
            value = tf.layers.dense(l, 1, kernel_initializer=w_init, name='fc-value')
            value = tf.squeeze(value, [1], name="value")
            if not hasattr(self, '_weights_critic'):
                self._weights_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('actor', reuse=nnc.reuse) as vs:
            l = tf.stop_gradient(rnn_outputs)
            l = tf.layers.dense(l, 80, activation=tf.nn.relu6, kernel_initializer=w_init, name='la-0')
            mu = tf.layers.dense(l, 1, activation=tf.nn.tanh, kernel_initializer=w_init, name='fc-mu')
            sigma = tf.layers.dense(l, 1, activation=tf.nn.softplus, kernel_initializer=w_init, name='fc-sigma')

            if not nnc.is_evaluating:
                sigma_beta = tf.get_default_graph().get_tensor_by_name('actor/sigma_beta:0')
            else:
                sigma_beta = tf.constant(0.)
            sigma = (sigma + sigma_beta + 1e-4)

            from tensorflow.contrib.distributions import Normal
            dists = Normal(mu, sigma)
            policy = tf.squeeze(dists.sample([1]), [0])
            policy = tf.clip_by_value(policy, -2., 2.)
            if is_training:
                self._addMovingSummary(tf.reduce_mean(mu, name='mu/mean'),
                                       tf.reduce_mean(sigma, name='sigma/mean'),
                                       )
            # actions = tf.Print(actions, [mus, sigmas, tf.concat([sigma_steering_, sigma_accel_], -1), actions],
            #                    'mu/sigma/sigma.orig/act=', summarize=4)
            if not hasattr(self, '_weights_actor'):
                self._weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        if not is_training:
            tensor_io.setOutputTensors(policy, value, mu, sigma)
            return

        i_actions = tensor_io.getInputTensor("action")
        # i_actions = tf.Print(i_actions, [i_actions], 'actions = ')
        i_actions = tf.reshape(i_actions, [-1] + i_actions.get_shape().as_list()[2:])
        i_futurereward = tensor_io.getInputTensor("futurereward")
        i_futurereward = tf.reshape(i_futurereward, [-1] + i_futurereward.get_shape().as_list()[2:])
        i_advantage = tensor_io.getInputTensor("advantage")
        i_advantage = tf.reshape(i_advantage, [-1] + i_advantage.get_shape().as_list()[2:])
        i_advantage =  value - i_futurereward

        log_probs = dists.log_prob(i_actions)
        _log_probs = tf.reduce_sum(log_probs, axis=-1)
        loss_policy = tf.reduce_mean(_log_probs*i_advantage)

        entropy = tf.reduce_sum(-dists.entropy(), 1)
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        loss_entropy = tf.reduce_mean(entropy_beta * entropy, name='xentropy_loss')
        loss_policy = tf.add(loss_policy, loss_entropy, name='loss/policy')

        loss_value = tf.reduce_mean(tf.square(value - i_futurereward))

        # from tensorflow.contrib.layers.python.layers.regularizers import apply_regularization, l2_regularizer
        # loss_l2_regularizer = apply_regularization(l2_regularizer(1e-4), self._weights_critic)
        # loss_l2_regularizer = tf.identity(loss_l2_regularizer, 'loss/l2reg')
        # loss_value += loss_l2_regularizer
        loss_value = tf.identity(loss_value, name='loss/value')

        self._addParamSummary([('.*', ['rms', 'absmax'])])
        pred_reward = tf.reduce_mean(value, name='predict_reward')
        import tensorpack.tfutils.symbolic_functions as symbf
        advantage = symbf.rms(i_advantage, name='rms_advantage')
        self._addMovingSummary(loss_policy, loss_value,
                                   loss_entropy,
                                   pred_reward, advantage,
                                   # loss_l2_regularizer,
                                   tf.reduce_mean(policy, name='actor/mean'),
                                   )
        return loss_policy, loss_value

    def _build_ad_ff(self, tensor_io):
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
        l = i_state
        # l = tf.Print(l, [i_state, tf.shape(i_state)], 'State = ')
        # l = tf.Print(l, [i_agentIdent, tf.shape(i_agentIdent)], 'agentIdent = ')
        # l = tf.Print(l, [i_sequenceLength, tf.shape(i_sequenceLength)], 'SeqLen = ')
        # l = tf.Print(l, [i_resetRNN, tf.shape(i_resetRNN)], 'resetRNN = ')
        w_init = None #tf.random_normal_initializer(0., .1)
        with tf.variable_scope('critic', reuse=nnc.reuse) as vs:
            i_state = tf.reshape(i_state, [-1, FEATURE_DIM])
            l = i_state
            l = tf.layers.dense(l, 100, activation=tf.nn.relu6, kernel_initializer=w_init, name='fc-critic')
            value = tf.layers.dense(l, 1, kernel_initializer=w_init, name='fc-value')
            value = tf.squeeze(value, [1], name="value")
            if not hasattr(self, '_weights_critic'):
                self._weights_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('actor', reuse=nnc.reuse) as vs:
            l = i_state
            l = tf.layers.dense(l, 200, activation=tf.nn.relu6, kernel_initializer=w_init, name='la-0')
            mu = tf.layers.dense(l, 1, activation=tf.nn.tanh, kernel_initializer=w_init, name='fc-mu')
            sigma = tf.layers.dense(l, 1, activation=tf.nn.softplus, kernel_initializer=w_init, name='fc-sigma')

            if not nnc.is_evaluating:
                sigma_beta = tf.get_default_graph().get_tensor_by_name('actor/sigma_beta:0')
                # sigma_beta_steering_exp = tf.train.exponential_decay(0.3, global_step, 1000, 0.5, name='sigma/beta/steering/exp')
                # sigma_beta_accel_exp = tf.train.exponential_decay(0.5, global_step, 5000, 0.5, name='sigma/beta/accel/exp')
            else:
                sigma_beta = 0.
            sigma = (sigma + sigma_beta + 1e-4)

            from tensorflow.contrib.distributions import Normal
            dists = Normal(mu, sigma)
            policy = tf.squeeze(dists.sample([1]), [0])
            # 裁剪到两倍方差之内
            policy = tf.clip_by_value(policy, -2., 2.)
            if is_training:
                self._addMovingSummary(tf.reduce_mean(mu, name='mu/mean'),
                                       tf.reduce_mean(sigma, name='sigma/mean'),
                                       )
            # actions = tf.Print(actions, [mus, sigmas, tf.concat([sigma_steering_, sigma_accel_], -1), actions],
            #                    'mu/sigma/sigma.orig/act=', summarize=4)
            if not hasattr(self, '_weights_actor'):
                self._weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        if not is_training:
            tensor_io.setOutputTensors(policy, value, mu, sigma)
            return

        i_actions = tensor_io.getInputTensor("action")
        # i_actions = tf.Print(i_actions, [i_actions], 'actions = ')
        i_actions = tf.reshape(i_actions, [-1] + i_actions.get_shape().as_list()[2:])
        i_futurereward = tensor_io.getInputTensor("futurereward")
        i_futurereward = tf.reshape(i_futurereward, [-1] + i_futurereward.get_shape().as_list()[2:])
        i_advantage = tensor_io.getInputTensor("advantage")
        i_advantage = tf.reshape(i_advantage, [-1] + i_advantage.get_shape().as_list()[2:])
        i_advantage =  tf.stop_gradient(value) - i_futurereward

        log_probs = dists.log_prob(i_actions)
        _log_probs = tf.reduce_mean(log_probs, axis=1)
        loss_policy = tf.reduce_mean(_log_probs*i_advantage)

        entropy = tf.reduce_mean(-dists.entropy(), 1)
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        loss_entropy = tf.reduce_mean(entropy_beta * entropy, name='xentropy_loss')
        loss_policy = tf.add(loss_policy, loss_entropy, name='loss/policy')

        loss_value = tf.reduce_mean(tf.square(value - i_futurereward))

        # from tensorflow.contrib.layers.python.layers.regularizers import apply_regularization, l2_regularizer
        # loss_l2_regularizer = apply_regularization(l2_regularizer(1e-4), self._weights_critic)
        # loss_l2_regularizer = tf.identity(loss_l2_regularizer, 'loss/l2reg')
        # loss_value += loss_l2_regularizer
        loss_value = tf.identity(loss_value, name='loss/value')

        self._addParamSummary([('.*', ['rms', 'absmax'])])
        pred_reward = tf.reduce_mean(value, name='predict_reward')
        import tensorpack.tfutils.symbolic_functions as symbf
        advantage = symbf.rms(i_advantage, name='rms_advantage')
        self._addMovingSummary(loss_policy, loss_value,
                                   loss_entropy,
                                   pred_reward, advantage,
                                   # loss_l2_regularizer,
                                   tf.reduce_mean(policy, name='actor/mean'),
                                   )
        return loss_policy, loss_value

    def _build_graph(self):
        from drlutils.model.base import NNContext
        gpu_towers = [0]
        for towerIdx, tower in enumerate(gpu_towers):
            with NNContext("PendulumTrain", device='/device:GPU:%d' % tower, summary_collection=towerIdx==0, is_training=True):
                data_io.createPool('P-%d'%towerIdx, POOL_SIZE, sub_batch_size = 1, is_training = True, torcsIdxOffset = data_io.getAgentCountTotal())
                tensor_io = data_io.getTensorIO("P-%d/train"%towerIdx, self._get_inputs(), queue_size=50, is_training=True,
                                                min_batch_size=TRAIN_BATCH_SIZE,
                                                max_batch_size = TRAIN_BATCH_SIZE,
                                                )
                loss_policy, loss_value = self._build_ad_nn(tensor_io)
                self._addLoss('policy%d'%towerIdx, loss_policy, opt=self._get_optimizer('actor'),
                              trainOpGroup='P', tensor_io=tensor_io, var_list=self._weights_actor)
                self._addLoss('value%d'%towerIdx, loss_value, opt=self._get_optimizer('critic'),
                              trainOpGroup='P', tensor_io=tensor_io, var_list=self._weights_critic)
                if not hasattr(self, '_weights_ad_train'):
                    self._weights_ad_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


            with NNContext("PendulumPred", device='/device:GPU:%d'%tower, summary_collection=False, is_training=False, reuse=True):
                tensor_io = data_io.getTensorIO("P-%d/pred" % (towerIdx),
                                                self._get_inputs(['agentIdent', 'state', 'sequenceLength', 'resetRNN']),
                                                is_training=False,
                                                min_batch_size = 1,
                                                max_batch_size=PREDICT_MAX_BATCH_SIZE,
                                                )
                # for predictIdx in range(PREDICTOR_THREAD_PER_GPU):
                self._build_ad_nn(tensor_io)
                self._addThreadOp('P-%d/pred'%(towerIdx), tensor_io.getOutputOp())

        for eidx, evaluator in enumerate(evaluators):
            with NNContext(evaluator.name, device='/device:GPU:%d' % gpu_towers[eidx%len(gpu_towers)], add_variable_scope=True, summary_collection=True, is_evaluating=True):
                data_io.createPool(evaluator.name.replace('/', '_'), evaluator._batch_size, sub_batch_size=1, is_training=False, torcsIdxOffset = data_io.getAgentCountTotal())
                tensor_io = evaluator.getTensorIO(self._get_inputs(['agentIdent', 'state', 'sequenceLength', 'resetRNN']))
                self._build_ad_nn(tensor_io)
                evaluator.set_weights(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=evaluator.name))

    def _calc_learning_rate(self, name, epoch, lr):
        def _calc():
            lr_init = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
            lrs = [(0, lr_init),
                   (5, lr_init / 2.),
                   (10, lr_init / 4.),
                   (20, lr_init / 16),
                   # (100, lr_init/16),
                   ]
            for idx in range(len(lrs) - 1):
                if epoch >= lrs[idx][0] and epoch < lrs[idx+1][0]:
                    return lrs[idx][1]
            return lrs[-1][1]
        # return INIT_LEARNING_RATE_A
        # ret = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
        ret = _calc()
        return ret

    def _get_optimizer(self, name):
        from tensorpack.tfutils import optimizer
        from tensorpack.tfutils.gradproc import SummaryGradient, GlobalNormClip, MapGradient
        init_lr = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
        import tensorpack.tfutils.symbolic_functions as symbf
        lr = symbf.get_scalar_var('learning_rate/' + name, init_lr, summary=True)
        opt = tf.train.AdamOptimizer(lr)
        # opt = tf.train.RMSPropOptimizer(lr)
        logger.info("create opt {}".format(name))
        # if name == 'critic':
        #     gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.05), regex='^critic/.*')]
        # elif name == 'actor':
        #     gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1), regex='^actor/.*')]
        # else: assert(0)
        gradprocs = [
            # MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.05))
        ]
        gradprocs.append(SummaryGradient())
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt

def get_config():
    M = Model()

    dataflow = data_io
    from tensorpack.callbacks.base import Callback
    import functools
    from tensorpack.train.config import TrainConfig
    from tensorpack.callbacks.saver import ModelSaver
    from tensorpack.callbacks.graph import RunOp
    from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter, HyperParamSetterWithFunc
    from tensorpack.tfutils import sesscreate
    from tensorpack.tfutils.common import get_default_sess_config
    import tensorpack.tfutils.symbolic_functions as symbf

    sigma_beta = symbf.get_scalar_var('actor/sigma_beta', 5., summary=True, trainable=False)

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

            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
            # HumanHyperParamSetter('entropy_beta'),
            ScheduledHyperParamSetter('actor/sigma_beta', [(1, 0.)]),
            # CBSyncWeight(),
            data_io,
            # PeriodicTrigger(Evaluator(
            #     EVAL_EPISODE, ['state'], ['policy'], get_player),
            #     every_k_epochs=3),
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

    class AgentPendulum(AgentSingleLoop):
        def _init(self):
            super(AgentPendulum, self)._init()
            import gym
            self._env = gym.make('Pendulum-v0')

        def _reset(self):
            return self._env.reset()

        def _step(self, pred):
            act, value, mu, sigma = pred
            ob, reward, isover, info = self._env.step(act)
            if self._isTrain and self._agentIdent == 0:
                self._env.render()
            reward /= 10.
            return ob, act, reward, isover


    class AgentPool(AgentPoolNoBatchMode):
        def _init(self):
            self._cls_agent = AgentPendulum
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

        dirname = '/tmp/pendulum/trainlog'
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


