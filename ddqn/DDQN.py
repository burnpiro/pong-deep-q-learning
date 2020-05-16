import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class TransitionsStore:
    def __init__(self, state_size, store_size):
        self.size = store_size
        self.state_size = state_size

        self.states = np.empty((store_size, state_size))
        self.actions = np.empty((store_size,), dtype=np.int8)
        self.rewards = np.empty((store_size,))
        self.next_states = np.empty((store_size, state_size))
        self.dones = np.empty((store_size,))

        self.current_element = 0
        self.full = False

    def store_transition(self, state, action, reward, next_state, done):
        # assert state.shape == (self.state_size,)
        # assert next_state.shape == (self.state_size,)
        # assert type(action) == int, "type is %s" % type(action)
        # assert type(reward) == float, "type is %s" % type(reward)
        # assert type(done) == bool, "type is %s" % type(done)

        self.states[self.current_element] = state
        self.actions[self.current_element] = action
        self.rewards[self.current_element] = reward
        self.next_states[self.current_element] = next_state
        self.dones[self.current_element] = done

        self.current_element = (self.current_element+1) % self.size
        if self.current_element == 0:
            self.full = True

    def sample_minibatch(self, batch_size):
        current_size = self.size if self.full else self.current_element
        batch_size = min(batch_size, current_size)
        indexes = np.random.choice(current_size, batch_size, replace=False)

        return self.states[indexes], self.actions[indexes], self.rewards[indexes], self.next_states[indexes], self.dones[indexes]


class DQN:
    def __init__(self, state_size, action_space, initial_epsilon=1.0):
        self.gamma = 0.99
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        # self.epsilon_decay_steps = int(1e6)//50
        self.epsilon_min = 0.1
        self.epsilon_decay = (initial_epsilon-self.epsilon_min)/100000

        self.target_update_frequency = 100

        self.action_space = action_space
        self.model = build_model(state_size, action_space.n)
        self.target_model = build_model(state_size, action_space.n)
        self.target_model.set_weights(self.model.get_weights())
        self.store = TransitionsStore(state_size, int(1e6))

        self._train_steps = 0

    def store_transition(self, state, action, reward, next_state, done):
        self.store.store_transition(state, action, reward, next_state, done)

    def act(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = self.model.predict(np.expand_dims(state, 0))[0]
            return int(np.argmax(q_values))

    def value(self, state):
        return self.model.predict(np.expand_dims(state, 0))[0]

    def train(self, batch_size=32):
        if self.store.current_element < batch_size:
            return

        self._train_steps += 1

        state, action, reward, new_state, done = \
            self.store.sample_minibatch(batch_size)

        next_values = self.target_model.predict(new_state)
        values = self.model.predict(new_state)
        target = self.model.predict(state)

        max_actions = np.argmax(values, axis=1)

        mb_index = np.arange(batch_size, dtype=np.int32)

        target[mb_index, action] = reward + \
            self.gamma*next_values[mb_index,
                                   max_actions.astype(int)]*(1.-done)

        self.model.fit(state, target, verbose=0)

        self.epsilon = self.epsilon-self.epsilon_decay if self.epsilon > \
            self.epsilon_min else self.epsilon_min

        if self._train_steps % self.target_update_frequency == 0:
            self.update_target()

    @tf.function
    def update_target(self):
        for weights, target_weights in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            target_weights.assign(weights)


def build_model(state_size, actions_size):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(state_size,)))
    # model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(actions_size, activation='linear'))
    optimizer = Adam(lr=0.0005)
    model.compile(optimizer, loss='mse')
    return model
