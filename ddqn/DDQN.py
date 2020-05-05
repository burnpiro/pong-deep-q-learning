import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import random


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
        assert state.shape == (self.state_size,)
        assert next_state.shape == (self.state_size,)
        assert type(action) == int, "type is %s" % type(action)
        assert type(reward) == float, "type is %s" % type(reward)
        assert type(done) == bool, "type is %s" % type(done)

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
        self.epsilon = initial_epsilon
        self.epsilon_decay = 0.99999
        self.tau = 0.01

        self.action_space = action_space
        self.model = build_model(state_size, action_space.n)
        self.target_model = build_model(state_size, action_space.n)
        self.target_model.set_weights(self.model.get_weights())
        self.store = TransitionsStore(state_size, 65536)

    def store_transition(self, state, action, reward, next_state, done):
        self.store.store_transition(state, action, reward, next_state, done)

    def act(self, state):
        self.epsilon *= self.epsilon_decay

        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = self.model.predict(np.expand_dims(state, 0))[0]
            return int(np.argmax(q_values))

    def train(self, batch_size=32):
        states, actions, rewards, next_states, dones = self.store.sample_minibatch(
            batch_size)
        batch_size = states.shape[0]
        target_values = self.model.predict(states)
        next_actions = np.argmax(self.model.predict(next_states), 1)
        next_values = self.target_model.predict(next_states)

        for i in range(batch_size):
            target_values[i][actions[i]] = rewards[i] + \
                (1.-dones[i])*self.gamma*next_values[i][next_actions[i]]

        return self.model.train_on_batch(states, target_values)

    @tf.function
    def update_target(self):
        for weights, target_weights in zip(self.model.trainable_weights, self.target_model.trainable_weights):
            target_weights.assign(
                target_weights*(1-self.tau) + self.tau*weights)


def build_model(state_size, actions_size):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(state_size,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(actions_size, activation='linear'))
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-8)
    model.compile(optimizer, loss='mse')
    return model
