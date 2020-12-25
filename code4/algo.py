import numpy as np
import random

from abc import abstractmethod
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


def convert_obs(obs):
    new_obs = str(obs)
    return new_obs


class MyQAgent(QAgent):
    def __init__(self, lr=0.1, discount=0.8, action_num=4):
        super(MyQAgent, self).__init__()
        self.lr = lr
        self.discount = discount
        self.action_num = action_num

        self.q_table = {}

    def create_actions(self, state):
        state = convert_obs(state)
        if state not in self.q_table.keys():
            actions = np.zeros(self.action_num)
            self.q_table[state] = actions

    def select_action(self, ob):
        ob = convert_obs(ob)
        if ob not in self.q_table.keys():
            self.create_actions(ob)
            return np.random.randint(self.action_num)
        return np.argmax(self.q_table[ob])

    def update_table(self, s, a, sp, r, d, clip=False):
        s = convert_obs(s)
        sp = convert_obs(sp)
        self.create_actions(s)
        self.create_actions(sp)
        self.q_table[s][a] += self.lr * (r + (1 - d) * self.discount * np.max(self.q_table[sp]) - self.q_table[s][a])
        if clip:
            self.q_table[s][a] = np.clip(self.q_table[s][a], -100, 100)


class Model:
    def __init__(self, width, height, policy):
        self.width = width
        self.height = height
        self.policy = policy
        pass

    @abstractmethod
    def store_transition(self, s, a, r, s_):
        pass

    @abstractmethod
    def sample_state(self):
        pass

    @abstractmethod
    def sample_action(self, s):
        pass

    @abstractmethod
    def predict(self, s, a):
        pass


def convert_sa(s, a):
    return str(s) + ':' + str(a)


class DynaModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.transitions = {}

    def store_transition(self, s, a, r, s_):
        if str(s) not in self.transitions.keys():
            self.transitions[str(s)] = {}
        self.transitions[str(s)][a] = (r, s_)

    def sample_state(self):
        result = random.sample(self.transitions.keys(), 1)[0].strip('[').strip(']').split()
        result = np.array(list(map(int, result)))
        return result, None

    def sample_action(self, s):
        result = random.sample(self.transitions[str(s)].keys(), 1)[0]
        return result

    def predict(self, s, a):
        result = self.transitions[str(s)][a][1]
        return result

    def train_transition(self):
        pass


class NetworkModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        self.x_next_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x_next')
        self.a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='a')
        self.r_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        h1 = tf.layers.dense(tf.concat([self.x_ph, self.a_ph], axis=-1), units=256, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, units=256, activation=tf.nn.relu)
        self.next_x = tf.layers.dense(h2, units=3, activation=tf.nn.tanh) * 1.3 + self.x_ph
        self.x_mse = tf.reduce_mean(tf.square(self.next_x - self.x_next_ph))
        self.opt_x = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(self.x_mse)
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.variables_initializer(tf.global_variables()))
        self.buffer = []
        self.sensitive_index = []

    def norm_s(self, s):
        return s

    def de_norm_s(self, s):
        s = np.clip(np.round(s), 0, self.width - 1).astype(np.int32)
        s[2] = np.clip(s[2], 0, 1).astype(np.int32)
        return s

    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)

        x_mse = self.sess.run([self.x_mse, self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        s, a, r, s_ = self.buffer[idx]
        return self.de_norm_s(s), idx

    def sample_action(self, s):
        return self.policy.select_action(s)

    def predict(self, s, a):
        s_ = self.sess.run(self.next_x, feed_dict={self.x_ph: [s], self.a_ph: [[a]]})
        return self.de_norm_s(s_[0])


class AdvNetworkModel(NetworkModel):
    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)

        if len(self.sensitive_index) > 0:
            for _ in range(batch_size):
                idx = np.random.randint(0, len(self.sensitive_index))
                idx = self.sensitive_index[idx]
                s, a, r, s_ = self.buffer[idx]
                s_list.append(s)
                a_list.append([a])
                r_list.append(r)
                s_next_list.append(s_)

        x_mse = self.sess.run([self.x_mse, self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse
