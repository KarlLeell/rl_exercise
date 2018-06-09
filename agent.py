import mxnet as mx
from mxnet import nd, autograd, gluon
import gym
import numpy as np
import random


class Agent(object):
    def __init__(self, env):
        self._env = env
        self.experience = []
        self.batch_size = 128
        self.max_exp = 2000
        self.alpha = 0.001
        self.gamma = 0.9
        self.epsilon = 1.0
        self.decay = 0.99
        self.min_epsilon = 0.05
        self.ob_size = 4
        self.hidden_size = 16
        self.out_size = 2
        self.scale = 0.1
        self.model_ctx = mx.cpu()
        self.net = gluon.nn.Sequential()
        with self.net.name_scope():
            self.net.add(gluon.nn.Dense(self.hidden_size, activation="relu"))
            self.net.add(gluon.nn.Dense(self.hidden_size, activation="relu"))
            self.net.add(gluon.nn.Dense(self.out_size))
        self.net.collect_params().initialize(mx.init.Xavier(), ctx=self.model_ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': self.alpha})
        self.square_loss = gluon.loss.L2Loss()



    def remember(self, ob, next_ob, reward, action, done):
        self.experience.append((ob, next_ob, reward, action, done))
        if len(self.experience) > self.max_exp:
            self.experience.pop(0)



    def replay(self):
        if len(self.experience) < self.batch_size:
            return
        # build replay batch
        sample = random.sample(self.experience, self.batch_size)
        ob = nd.array([s[0] for s in sample])
        next_ob = nd.array([s[1] for s in sample])
        reward = nd.array([s[2] for s in sample])
        action = nd.array([s[3] for s in sample])
        next_q = nd.max(self.net(next_ob), axis=1)
        for i in range(self.batch_size):
            if sample[i][4]:
                next_q[i] = 0
        with autograd.record():
            current_q = self.net(ob).pick(action)
            target = next_q * self.gamma + reward
            loss = self.square_loss(current_q, target)
        loss.backward()
        self.trainer.step(self.batch_size)



    def learn(self, max_episodes=1000):
        for e in range(max_episodes):
            ob = self._env.reset()
            done = False
            timestep = 0
            while not done:
                action = self.query(ob)
                next_ob, reward, done, info = self._env.step(action)
                # remember current trasition for replay
                self.remember(ob, next_ob, reward, action, done)
                ob = next_ob
                timestep += 1
                if done and (e+1)%25==0:
                    self.replay()
                    print("Episode {} finishes after {} steps".format(e+1, timestep))



    def query(self, observation):
        # with probability epsilon, randomly explore
        if self.epsilon >= self.min_epsilon:
            self.epsilon *= self.decay
        observation = nd.array(observation)
        if random.random() < self.epsilon:
            return self._env.action_space.sample()
        else:
            return int(nd.argmax(self.net(nd.reshape(observation, [-1, self.ob_size])), axis=1).asscalar())


