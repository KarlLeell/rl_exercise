import mxnet as mx
from mxnet import nd, autograd, gluon
import gym
import numpy as np
import random


class Agent(object):
    def __init__(self, env):
        self._env = env
        self.experience = []
        self.batch_size = 64
        self.max_exp = 500
        self.alpha = 0.001
        self.gamma = 0.9
        self.epsilon = 1.0
        self.decay = 0.99
        self.ob_size = 4
        self.hidden_size = 32
        self.out_size = 2
        self.scale = 0.1
        self.model_ctx = mx.cpu()
        self.net = gluon.nn.Sequential()
        with self.net.name_scope():
            self.net.add(gluon.nn.Dense(self.hidden_size, activation="relu"))
            self.net.add(gluon.nn.Dense(self.hidden_size, activation="relu"))
            self.net.add(gluon.nn.Dense(self.out_size))
        self.net.collect_params().initialize(mx.init.Normal(), ctx=self.model_ctx)
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
        x = []
        target = []
        sample = random.sample(self.experience, self.batch_size)
        for ob, next_ob, reward, action, done in sample:
            y = self.net(nd.array(np.reshape(ob, [-1, self.ob_size])))
            y[0][action] = reward
            if not done:
                y[0][action] += np.max(self.net(nd.array(np.reshape(next_ob, [-1, self.ob_size]))).asnumpy()) * self.gamma
            x.append(ob)
            target.append(y[0].asnumpy().tolist())
        x = nd.array(x)
        target = nd.array(target)
        # update model
        with autograd.record():
            f_y = self.net(x)
            loss = self.square_loss(target, f_y)
            #print("loss {}".format(nd.mean(loss).asscalar()))
        loss.backward()
        self.trainer.step(self.batch_size)



    def learn(self, max_episodes=1000):
        for e in range(max_episodes):
            ob = self._env.reset()
            done = False
            timestep = 0
            while not done:
                action = self.query(ob)
                print(action)
                next_ob, reward, done, info = self._env.step(action)
                # remember current trasition for replay
                self.remember(ob, next_ob, reward, action, done)
                ob = next_ob
                timestep += 1
                if done and (e+1)%25==0:
                    print("Episode {} finishes after {} steps".format(e+1, timestep))
            self.replay()



    def query(self, observation):
        # with probability epsilon, randomly explore
        self.epsilon *= self.decay
        if random.random() < self.epsilon:
            return self._env.action_space.sample()
        else:
            return np.argmax((self.net(nd.reshape(nd.array(observation), [-1, self.ob_size]))).asnumpy())


