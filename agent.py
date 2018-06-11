import mxnet as mx
from mxnet import nd, autograd, gluon
import gym
import random


class Agent(object):
    def __init__(self, env):
        self._env = env
        self.success_train = 20
        self.experience = []
        self.batch_size = 64
        self.max_exp = 25000
        self.alpha = 0.001
        self.gamma = 0.9
        self.epsilon = 0.5
        self.decay = 0.9
        self.min_epsilon = 0.05
        self.ob_size = 4
        self.hidden_size = 16
        self.update_freq = 25
        self.set_target_freq = 100
        self.out_size = 2
        self.model_ctx = mx.cpu()
        self.net = self.build_network()
        self.target_net = self.build_network()
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': self.alpha})
        self.square_loss = gluon.loss.L2Loss()


    def build_network(self):
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(self.hidden_size, activation="tanh"))
            net.add(gluon.nn.Dense(self.hidden_size, activation="tanh"))
            net.add(gluon.nn.Dense(self.out_size))
        net.collect_params().initialize(mx.init.Xavier(), ctx=self.model_ctx)
        return net



    def remember(self, ob, next_ob, reward, action, done):
        self.experience.append((ob, next_ob, reward, action, done))
        if len(self.experience) > self.max_exp:
            self.experience.pop(0)



    def react(self, observation):
        # with probability epsilon, randomly explore
        if random.random() < self.epsilon:
            return self._env.action_space.sample()
        else:
            return self.query(observation)



    def replay(self):
        if len(self.experience) < self.batch_size:
            return
        if self.epsilon >= self.min_epsilon:
            self.epsilon *= self.decay
        # build replay batch
        sample = random.sample(self.experience, self.batch_size)
        ob = nd.array([s[0] for s in sample])
        next_ob = nd.array([s[1] for s in sample])
        reward = nd.array([s[2] for s in sample])
        action = nd.array([s[3] for s in sample])
        next_q = nd.max(self.target_net(next_ob), axis=1)
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
        
        success_count = 0
        
        for e in range(max_episodes):
        
            ob = self._env.reset()
            done = False
            timestep = 0
            
            while not done:
                action = self.react(ob)
                next_ob, reward, done, info = self._env.step(action)
                # remember current trasition for replay
                self.remember(ob, next_ob, reward, action, done)
                ob = next_ob
                timestep += 1
            
                if done:
                    if (e+1)%self.update_freq==0:
                        print("Episode {} finishes after {} steps, success count {}".format(e+1, timestep, success_count))
                        self.replay()
                    if (e+1)%self.set_target_freq==0:
                        self.target_net = self.net


            # check for training success
            if timestep == 200:
                success_count += 1
            else:
                success_count = 0
            if success_count >= self.success_train:
                return

 


    def query(self, observation):
        observation = nd.array(observation)
        return int(nd.argmax(self.net(nd.reshape(observation, [-1, self.ob_size])), axis=1).asscalar())


