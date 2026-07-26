#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
MCTS AI player for Othello.
"""

import random
import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility

def switch_player(player):
    """
    Alternate between dark (1) and light (2) players.
    """
    return player % 2 + 1


class Node:
    def __init__(self, state, player, parent, children, v=0, N=0):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = children
        self.value = v
        self.N = N

    def get_child(self, state):
        for c in self.children:
            if (state == c.state).all():
                return c
        return None


def select(root, alpha):
    """ Starting from given node, find a terminal node or node with unexpanded children.
    If all children of a node are in tree, move to the one with the highest UCT value.

    Args:
        root (Node): MCTS tree root node
        alpha (float): Weight of exploration term in UCT

    Returns:
        node (Node): Node at bottom of MCTS tree
    """
    node = root

    while True:
        moves = get_possible_moves(node.state, node.player)
        if not moves:
            return node

        for move in moves:
            successor = play_move(node.state, node.player, move[0], move[1])
            if node.get_child(successor) is None:
                return node

        best_child = None
        best_uct = -float("inf")
        for child in node.children:
            exploit_factor = child.value / child.N
            explore_factor = alpha * np.sqrt(np.log(node.N) / child.N)
            uct = exploit_factor + explore_factor
            if uct > best_uct:
                best_uct = uct
                best_child = child

        node = best_child


def expand(node):
    """ Add a child node of state into the tree if it's not terminal.

    Args:
        node (Node): Node to expand

    Returns:
        leaf (Node): Newly created node (or given Node if already leaf)
    """
    moves = get_possible_moves(node.state, node.player)
    if not moves:
        return node

    for move in moves:
        successor = play_move(node.state, node.player, move[0], move[1])
        if node.get_child(successor) is None:
            child = Node(successor, switch_player(node.player), node, [])
            node.children.append(child)
            return child

    return node


def simulate(node):
    """ Run one game rollout using from state to a terminal state.
    Use random playout policy.

    Args:
        node (Node): Leaf node from which to start rollout.

    Returns:
        utility (int): Utility of final state
    """
    state = node.state
    player = node.player

    while True:
        moves = get_possible_moves(state, player)
        if not moves:
            next_player = switch_player(player)
            next_moves = get_possible_moves(state, next_player)
            if not next_moves:
                break
            player = next_player
            continue

        move = random.choice(moves)
        state = play_move(state, player, move[0], move[1])
        player = switch_player(player)

    return compute_utility(state)


def backprop(node, utility):
    """ Backpropagate result from state up to the root.
    Every node has N, number of plays, incremented
    If node's parent is dark (1), then node's value increases
    Otherwise, node's value decreases.

    Args:
        node (Node): Leaf node from which rollout started.
        utility (int): Utility of simulated rollout.
    """
    current = node
    while current is not None:
        current.N += 1
        if current.player == 1:
            current.value -= utility
        else:
            current.value += utility
        current = current.parent


def mcts(state, player, rollouts=100, alpha=5):
    # MCTS main loop: Execute four steps rollouts number of times
    # Then return successor with highest number of rollouts
    root = Node(state, player, None, [], 0, 1)
    for i in range(rollouts):
        leaf = select(root, alpha)
        new = expand(leaf)
        utility = simulate(new)
        backprop(new, utility)

    move = None
    plays = 0
    for m in get_possible_moves(state, player):
        s = play_move(state, player, m[0], m[1])
        if root.get_child(s).N > plays:
            plays = root.get_child(s).N
            move = m

    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("MCTS AI")        # First line is the name of this AI
    color = int(input())    # 1 for dark (first), 2 for light (second)

    while True:
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":
            print()
        else:
            board = np.array(eval(input()))
            movei, movej = mcts(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()