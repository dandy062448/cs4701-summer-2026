"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A dynamic programming agent for a stochastic task environment
"""

import random
import math
import sys


class DP_Agent(object):

    def __init__(self, states, parameters):
        self.gamma = parameters["gamma"]
        self.V0 = parameters["V0"]

        self.states = states
        self.values = {}
        self.policy = {}

        for state in states:
            self.values[state] = parameters["V0"]
            self.policy[state] = None


    def setEpsilon(self, epsilon):
        pass

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        pass


    def choose_action(self, state, valid_actions):
        return self.policy[state]

    def update(self, state, action, reward, successor, valid_actions):
        pass


    def compute_value(self, state, action, transition):
        """
        Helper function that scores the action.
        """
        successor_state, reward = transition(state, action)
        if successor_state is None:
            return reward + self.gamma * 0
        return reward + self.gamma * self.values[successor_state]
    

    def policy_evaluation(self, transition):
        """ Computes all values for current policy by iteration and stores them in self.values.
            Implemented with in place value iteration.
        Args:
            transition (Callable): Function that returns successor state and reward given a state and action.
        """

        VALUE_CONVERGENCE_THRESHOLD = 1e-6

        convergent = False
        while not convergent:
            # Initial assumption
            convergent = True

            for state in self.policy:
                action = self.policy[state]
                new_value = self.compute_value(state, action, transition)

                if new_value - self.values[state] > VALUE_CONVERGENCE_THRESHOLD:
                    convergent = False

                self.values[state] = new_value
    

    def policy_extraction(self, valid_actions, transition):
        """ Computes all optimal actions using value iteration and stores them in self.policy.
        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """

        for state in self.policy:
            best_action = max(
                valid_actions(state), 
                key=lambda action: self.compute_value(state, action, transition)
            )
            self.policy[state] = best_action


    def policy_iteration(self, valid_actions, transition):
        """ Runs policy iteration to learn an optimal policy. Calls policy_evaluation() and policy_extraction().
        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        # randomize initial policy.
        for state in self.policy:
            self.policy[state] = random.choice(valid_actions(state))

        while True:
            old_policy = self.policy.copy()
            self.policy_evaluation(transition)
            self.policy_extraction(valid_actions, transition)
            if self.policy == old_policy:
                break