"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A Q-learning agent for a stochastic task environment
"""

import random
import math
import sys


class RL_Agent(object):

    def __init__(self, states, valid_actions, parameters):
        self.alpha = parameters["alpha"]
        self.epsilon = parameters["epsilon"]
        self.gamma = parameters["gamma"]
        self.Q0 = parameters["Q0"]

        self.states = states
        self.Qvalues = {}
        for state in states:
            for action in valid_actions(state):
                self.Qvalues[(state, action)] = parameters["Q0"]


    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        self.alpha = alpha


    def choose_action(self, state, valid_actions):
        """ Choose an action using epsilon-greedy selection.
 
        Args:
            state (tuple): Current robot state.
            valid_actions (list): A list of possible actions.
        Returns:
            action (string): Action chosen from valid_actions.
        """
        if not valid_actions:
            return None
 
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
 
        # computes best actions and selects one randomly in the event of ties.
        best_value = -float('inf')
        for action in valid_actions:
            best_value = max(best_value, self.Qvalues[(state, action)])

        best_actions = []
        for action in valid_actions:
            if self.Qvalues[(state, action)] == best_value:
                best_actions.append(action)

        return random.choice(best_actions)
 
 
    def update(self, state, action, reward, successor, valid_actions):
        """ Update self.Qvalues for (state, action) given reward and successor.
 
        Args:
            state (tuple): Current robot state.
            action (string): Action taken at state.
            reward (float): Reward given for transition.
            successor (tuple): Successor state.
            valid_actions (list): A list of possible actions at successor state.
        """
        if successor is None or not valid_actions:
            # terminal successor
            best_next_q = 0
        else:
            best_next_q = -float('inf')
            for a in valid_actions:
                best_next_q = max(best_next_q, self.Qvalues[(successor, a)])
 
        sample = reward + self.gamma * best_next_q
        old_q = self.Qvalues[(state, action)]
        self.Qvalues[(state, action)] = (1 - self.alpha) * old_q + self.alpha * sample
