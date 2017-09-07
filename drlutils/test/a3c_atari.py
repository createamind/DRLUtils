#coding: utf-8


import numpy as np
# import six
import tensorflow as tf
from drlutils.utils import logger

ENABLE_GAE = True
ENABLE_RNN = True
ENV_NAME = 'Breakout-v0'

FEATURE_DIM = 84*84*3 if ENABLE_RNN else 84*84*12
NUM_ACTIONS = 4
GAMMA = 0.99
LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 0
POOL_SIZE = 256
TRAIN_BATCH_SIZE = 25
PREDICT_MIN_BATCH_SIZE = 1    # batch for efficient forward
PREDICT_MAX_BATCH_SIZE = 16    # batch for efficient forward
PREDICTOR_THREAD_PER_GPU = 3
INIT_LEARNING_RATE_A = 1e-3 * 1.
INIT_LEARNING_RATE_C = 1e-3 * 1.

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
    --log LEVEL                 log level                   ,    [default: info]
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
            InputDesc(tf.uint8, (None, FEATURE_DIM), 'state'),
            InputDesc(tf.int32, (None, 1), 'action'),
            InputDesc(tf.float32, (None,), 'reward'),
            InputDesc(tf.float32, (None,), 'action_prob'),
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
        if ENABLE_RNN:
            return self._build_ad_rnn(tensor_io)
        else:
            return self._build_ad_ff(tensor_io)

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
        with tf.variable_scope('critic', reuse=nnc.reuse) as vs:
            image = tf.cast(tf.reshape(i_state, [-1, 84, 84, 3]), tf.float32)
            if nnc.is_training:
                tf.summary.image('input', image[::5, :, :, -3:])
                # image = tf.Print(image, [i_agentIdent, tf.reduce_sum(tf.cast(i_state[0], tf.float32), axis=1)], 'image.sum=', summarize=8)

            image /= 255.0
            from tensorpack.tfutils.argscope import argscope
            from tensorpack.models.conv2d import Conv2D
            from tensorpack.models.pool import MaxPooling
            from tensorpack.models.fc import FullyConnected
            from tensorpack.models.nonlin import PReLU
            with argscope(Conv2D, nl=tf.nn.relu):
                l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
                l = MaxPooling('pool0', l, 2)
                l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
                l = MaxPooling('pool1', l, 2)
                l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
                l = MaxPooling('pool2', l, 2)
                l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

            shape = tf.shape(i_state)
            batch_size = shape[0]
            step_size = shape[1]
            l = tf.reshape(l, [batch_size, step_size, np.prod(l.get_shape().as_list()[1:])])
            cell_size = 256
            def _get_cell():
                cell = tf.nn.rnn_cell.LSTMCell(cell_size)
                # if is_training:
                #     cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.9, input_keep_prob=0.9)
                return cell

            cell = _get_cell()
            rnn_outputs = self._buildRNN(l, cell, tensor_io.batchSizeMax,
                                         i_agentIdent=i_agentIdent,
                                         i_sequenceLength=i_sequenceLength,
                                         i_resetRNN=i_resetRNN,
                                         )
            rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size])
            value = tf.layers.dense(rnn_outputs, 1, name='fc-value')
            value = tf.squeeze(value, [1], name="value")
            if not hasattr(self, '_weights_critic'):
                self._weights_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('actor', reuse=nnc.reuse) as vs:
            # l = tf.stop_gradient(rnn_outputs)
            l = rnn_outputs
            policy = tf.layers.dense(l, NUM_ACTIONS, name='la-0')
            policy = tf.nn.softmax(policy, name='policy')
            if not hasattr(self, '_weights_actor'):
                self._weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        if not is_training:
            tensor_io.setOutputTensors(policy, value,
                                       # tf.identity(i_state, 'image'),
                                       )
            return

        i_actions = tensor_io.getInputTensor("action")
        i_actions = tf.reshape(i_actions, [-1])  # + i_actions.get_shape().as_list()[2:])
        i_futurereward = tensor_io.getInputTensor("reward")
        i_futurereward = tf.reshape(i_futurereward, [-1] + i_futurereward.get_shape().as_list()[2:])
        i_advantage = tensor_io.getInputTensor("advantage")
        i_advantage = tf.reshape(i_advantage, [-1] + i_advantage.get_shape().as_list()[2:])
        i_action_prob = tensor_io.getInputTensor("action_prob")
        i_action_prob = tf.reshape(i_action_prob, [-1])
        # i_actions = tf.Print(i_actions, [i_actions, i_action_prob], 'action/prob = ', summarize=8)
        # i_futurereward = tf.Print(i_futurereward, [i_futurereward, value], 'reward/value = ', summarize=8)

        log_probs = tf.log(policy + 1e-6)
        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(i_actions, NUM_ACTIONS, 1., 0.), 1)
        if ENABLE_GAE:
            advantage = tf.identity(-i_advantage, name='advantage')
        else:
            advantage = tf.subtract(tf.stop_gradient(value), i_futurereward, name='advantage')
        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(i_actions, NUM_ACTIONS), 1)  # (B,)
        importance = 1.0 #tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (i_action_prob + 1e-8), 0, 10))
        policy_loss = tf.reduce_mean(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_mean(
            policy * log_probs, name='xentropy_loss')
        value_loss = 0.5 * tf.reduce_mean(tf.square(value - i_futurereward))
        value_loss = tf.identity(value_loss, name='value_loss')

        import tensorpack.tfutils.symbolic_functions as symbf
        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        with tf.variable_scope('beta', reuse=nnc.reuse):
            entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                           initializer=tf.constant_initializer(0.01), trainable=False)
        loss = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss],
                                 name='loss/policy')

        self._addMovingSummary(policy_loss, value_loss,
                               xentropy_loss,
                               pred_reward, advantage,
                               tf.reduce_mean(policy, name='actor/mean'),
                               tf.reduce_mean(importance, name='importance'),
                               tf.add(policy_loss, value_loss, name='cost'),
                               )
        return loss

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
        with tf.variable_scope('critic', reuse=nnc.reuse) as vs:
            image = tf.cast(tf.reshape(i_state, [-1, 84, 84, 3*4]), tf.float32)
            # tf.summary.image('input', image[:3, :, :, -3:])
            image /= 255.0
            from tensorpack.tfutils.argscope import argscope
            from tensorpack.models.conv2d import Conv2D
            from tensorpack.models.pool import MaxPooling
            from tensorpack.models.fc import FullyConnected
            from tensorpack.models.nonlin import PReLU

            with argscope(Conv2D, nl=tf.nn.relu):
                l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
                l = MaxPooling('pool0', l, 2)
                l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
                l = MaxPooling('pool1', l, 2)
                l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
                l = MaxPooling('pool2', l, 2)
                l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

            l = FullyConnected('fc0', l, 512, nl=tf.identity)
            l = PReLU('prelu', l)
            value = FullyConnected('fc-v', l, 1, nl=tf.identity)
            value = tf.squeeze(value, [1], name="value")
            if not hasattr(self, '_weights_critic'):
                self._weights_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('actor', reuse=nnc.reuse) as vs:
            policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
            if not hasattr(self, '_weights_actor'):
                self._weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            policy = tf.nn.softmax(policy, name='policy')
        if not is_training:
            tensor_io.setOutputTensors(policy, value)
            return
        # self._addParamSummary([('.*', ['rms', 'absmax'])])

        i_actions = tensor_io.getInputTensor("action")
        i_actions = tf.reshape(i_actions, [-1]) # + i_actions.get_shape().as_list()[2:])
        i_futurereward = tensor_io.getInputTensor("reward")
        i_futurereward = tf.reshape(i_futurereward, [-1] + i_futurereward.get_shape().as_list()[2:])
        i_advantage = tensor_io.getInputTensor("advantage")
        i_advantage = tf.reshape(i_advantage, [-1] + i_advantage.get_shape().as_list()[2:])
        i_action_prob = tensor_io.getInputTensor("action_prob")
        i_action_prob = tf.reshape(i_action_prob, [-1])
        # i_actions = tf.Print(i_actions, [i_actions, i_action_prob], 'action/prob = ', summarize=8)
        # i_futurereward = tf.Print(i_futurereward, [i_futurereward, value], 'reward/value = ', summarize=8)

        log_probs = tf.log(policy + 1e-6)
        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(i_actions, NUM_ACTIONS), 1)
        if ENABLE_GAE:
            advantage = tf.identity(-i_advantage, name='advantage')
        else:
            advantage = tf.subtract(tf.stop_gradient(value), i_futurereward, name='advantage')
        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(i_actions, NUM_ACTIONS), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (i_action_prob + 1e-8), 0, 10))
        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(
            policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - i_futurereward, name='value_loss')

        import tensorpack.tfutils.symbolic_functions as symbf
        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        with tf.variable_scope('beta', reuse=nnc.reuse):
            entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                           initializer=tf.constant_initializer(0.01), trainable=False)
        policy_loss = tf.truediv(policy_loss + xentropy_loss * entropy_beta, tf.cast(tf.shape(i_futurereward)[0], tf.float32), name='loss/policy')
        value_loss = tf.truediv(value_loss, tf.cast(tf.shape(i_futurereward)[0], tf.float32), name='loss/value')

        self._addMovingSummary(policy_loss, value_loss,
                               xentropy_loss,
                               pred_reward, advantage,
                               tf.reduce_mean(policy, name='actor/mean'),
                               tf.reduce_mean(importance, name='importance'),
                               tf.add(policy_loss, value_loss, name='cost'),
                               )
        return policy_loss, value_loss

    def _build_graph(self):
        from drlutils.model.base import NNContext
        train_towers = [0]
        pred_towers = [1]
        data_io.createPool('main', POOL_SIZE, sub_batch_size=1, is_training=True,
                           train_batch_size=TRAIN_BATCH_SIZE,
                           predict_min_batch_size=PREDICT_MIN_BATCH_SIZE,
                           predict_max_batch_size=PREDICT_MAX_BATCH_SIZE,
                           )
        for towerIdx, tower in enumerate(train_towers):
            with NNContext("MainTrain", device='/device:GPU:%d' % tower, summary_collection=towerIdx==0, is_training=True, reuse=towerIdx > 0):
                tensor_io = data_io.getTensorIO("main/train%d"%towerIdx, self._get_inputs(), queue_size=50, is_training=True)
                loss = self._build_ad_nn(tensor_io)
                if not hasattr(self, '_weights_main_train'):
                    self._weights_main_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if type(loss) in [tuple, list]:
                    loss_policy, loss_value = loss
                    self._addLoss('loss%d'%towerIdx, loss_policy, opt=self._get_optimizer('actor'),
                                  trainOpGroup='P', tensor_io=tensor_io, var_list=self._weights_actor)
                    self._addLoss('value%d'%towerIdx, loss_value, opt=self._get_optimizer('critic'),
                                  trainOpGroup='P', tensor_io=tensor_io, var_list=self._weights_critic)
                else:
                    self._addLoss('loss%d'%towerIdx, loss, opt=self._get_optimizer('actor'),
                              trainOpGroup='main', tensor_io=tensor_io, var_list=self._weights_main_train)


        for towerIdx, tower in enumerate(pred_towers):
            with NNContext("MainPred", device='/device:GPU:%d'%tower, summary_collection=False, is_training=False, reuse=True):
                for predictIdx in range(PREDICTOR_THREAD_PER_GPU):
                    tensor_io = data_io.getTensorIO("main/pred-%d" % (predictIdx),
                                                    self._get_inputs(['agentIdent', 'state', 'sequenceLength', 'resetRNN']),
                                                    is_training=False,
                                                    )
                    self._build_ad_nn(tensor_io)
                    self._addThreadOp('main-%d/pred-%d'%(towerIdx, predictIdx), tensor_io.getOutputOp())

        for eidx, evaluator in enumerate(evaluators):
            with NNContext(evaluator.name, device='/device:GPU:%d' % pred_towers[eidx%len(pred_towers)], add_variable_scope=True, summary_collection=True, is_evaluating=True):
                data_io.createPool(evaluator.name.replace('/', '_'), evaluator._batch_size, sub_batch_size=1, is_training=False     )
                tensor_io = evaluator.getTensorIO(self._get_inputs(['agentIdent', 'state', 'sequenceLength', 'resetRNN']))
                self._build_ad_nn(tensor_io)
                evaluator.set_weights(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=evaluator.name))

    def _calc_learning_rate(self, name, epoch, lr):
        def _calc():
            lr_init = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
            lrs = [(0, lr_init),
                   (20, lr_init / 2.),
                   (100, lr_init / 4.),
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
        lr = tf.get_default_graph().get_tensor_by_name('learning_rate/'+name + ':0')
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        # opt = tf.train.RMSPropOptimizer(lr)
        logger.info("create opt {}".format(name))
        # if name == 'critic':
        #     gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.05), regex='^critic/.*')]
        # elif name == 'actor':
        #     gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1), regex='^actor/.*')]
        # else: assert(0)
        gradprocs = [
            # MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.05))
            MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1))
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
    lr = symbf.get_scalar_var('learning_rate/critic', INIT_LEARNING_RATE_C, summary=True)
    lr = symbf.get_scalar_var('learning_rate/actor', INIT_LEARNING_RATE_A, summary=True)
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
            ScheduledHyperParamSetter('beta/entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
            # HumanHyperParamSetter('entropy_beta'),
            # ScheduledHyperParamSetter('actor/sigma_beta', [(1, 0.)]),
            # CBSyncWeight(),
            data_io,
            # PeriodicTrigger(Evaluator(
            #     EVAL_EPISODE, ['state'], ['policy'], get_player),
            #     every_k_epochs=3),
        ] + evaluators,
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(1.)),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    from drlutils.dataflow.server import DataFlowServer
    from drlutils.dataflow.pool import AgentPoolNoBatchMode
    from drlutils.agent.single_loop import AgentSingleLoop, StepResult

    class Agent(AgentSingleLoop):
        def _init(self):
            super(Agent, self)._init()
            import gym
            self._env = gym.make(ENV_NAME)
            self._env.env.frameskip = 4
            assert(self._env.action_space.n == NUM_ACTIONS), self._env.action_space
            self._key_fire = self._env.env.get_action_meanings().index('FIRE')
            self._maxEpisodeSteps = -1
            if self._isTrain and self._episodeCount <= 2:
                self._maxEpisodeSteps =self._rng.randint(50, 200)

        def _reset(self):
            ob = self._env.reset()
            if ENABLE_RNN:
                self._ob = self._resize(ob)
            else:
                self._ob = np.zeros((84, 84, 3 if ENABLE_RNN else 12), dtype=np.uint8)
                self._ob[:, :, 9:] = self._resize(ob)
            self._lives = -1
            self._new_live_start = 0
            self._last_ob = ob
            self._same_ob_count = 0
            return self._ob.flatten()

        def _resize(self, s):
            from scipy.misc import imresize
            s = imresize(s, (84, 84))
            return s

        def _step(self, pred):
            policy, value = pred['policy'], pred['value']
            assert(len(policy.shape) == 1 and policy.shape[0] == NUM_ACTIONS)
            if self._isTrain:
                assert(policy.shape[0] == NUM_ACTIONS), policy.shape
                act = self._rng.choice(NUM_ACTIONS, p=policy)
                # act = np.random.choice(NUM_ACTIONS, p=policy)
            else:
                act = np.argmax(policy)

            env = self._env

            ob, reward, isover, _ = env.step(act)
            if np.allclose(ob, self._last_ob):
                self._same_ob_count += 1
            else:
                self._same_ob_count = 0
            if self._same_ob_count >= 10:
                act = self._key_fire
                ob, reward, isover, _ = env.step(act)
                logger.info("press fire key, same_ob = {}".format(self._same_ob_count))
            self._last_ob = ob

            if self._agentIdent < 1:
                env.render()
            ob = self._resize(ob)

            if not ENABLE_RNN:
                ob = np.concatenate([self._ob[:, :, 3:], ob], axis=-1)

            ob = ob.flatten()
            return StepResult(ob, np.array([act]), reward, isover, value, action_prob=policy[act])

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
        from tensorpack.tfutils.sessinit import get_model_loader
        nr_gpu = get_nr_gpu()
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
            ds = DataFlowServer(clsPool, local_t_max=LOCAL_TIME_MAX, gamma=GAMMA, enable_gae=ENABLE_GAE)
            ds.run()
        except KeyboardInterrupt:
            pass
        import sys
        sys.exit(0)


