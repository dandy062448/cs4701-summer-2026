#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Alpha-beta minimax AI player for Othello.
"""

import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility

def switch_player(player):
    """
    Alternate between dark (1) and light (2) players.
    """
    return player % 2 + 1

def max_value(state, player, alpha, beta):
    """
    Args:
        state: Board state
        player: Dark (1) or light (2)
        alpha, beta values

    Returns:
        value (int): Minimax value of state
        move (tuple): Best move to make
    """
    if not get_possible_moves(state, player):
        return compute_utility(state), None

    value = -float("inf")
    for possible_move in get_possible_moves(state, player):
        next_state = play_move(state, player, possible_move[0], possible_move[1])
        next_value, _ = min_value(next_state, switch_player(player), alpha, beta)
        if next_value > value:
            value = next_value
            move = possible_move
            alpha = max(alpha, value)
        if value >= beta:
            return value, move
    return value, move



def min_value(state, player, alpha, beta):
    """
    Args:
        state: Board state
        player: Dark (1) or light (2)
        alpha, beta values

    Returns:
        value (int): Minimax value of state
        move (tuple): Best move to make
    """
    if not get_possible_moves(state, player):
        return compute_utility(state), None

    value = float("inf")
    for possible_move in get_possible_moves(state, player):
        next_state = play_move(state, player, possible_move[0], possible_move[1])
        next_value, _ = max_value(next_state, switch_player(player), alpha, beta)
        if next_value < value:
            value = next_value
            move = possible_move
            beta = min(beta, value)
        if value <= alpha:
            return value, move
    return value, move


def minimax(state, player):
    """
    Minimax main loop
    Call max_value if player is 1 (dark), min_value if player is 2 (light)
    Then return the resultant move
    """
    if player == 1:
        _, move = max_value(state, player, -float('inf'), float('inf'))
    else:
        _, move = min_value(state, player, -float('inf'), float('inf'))
    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Minimax AI")     # First line is the name of this AI
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
            movei, movej = minimax(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()