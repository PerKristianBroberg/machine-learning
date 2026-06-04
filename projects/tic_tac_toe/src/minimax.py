"""Depth-limited minimax with alpha-beta pruning for tic-tac-toe.

This is a direct implementation of the classic pseudocode:

    function alphabeta(node, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or node is terminal:
            return heuristic value of node
        if maximizingPlayer:
            value := -inf
            for each child of node:
                value := max(value, alphabeta(child, depth-1, alpha, beta, FALSE))
                if value >= beta: break          # beta cutoff
                alpha := max(alpha, value)
            return value
        else:
            value := +inf
            for each child of node:
                value := min(value, alphabeta(child, depth-1, alpha, beta, TRUE))
                if value <= alpha: break          # alpha cutoff
                beta := min(beta, value)
            return value

Here a "node" is (board, player_to_move). The maximizing player is the AI; the
minimizing player is its opponent. The heuristic rewards faster wins and slower
losses by folding the remaining depth into the score.
"""

import math

from .game import available_moves, is_terminal, other, winner

INF = math.inf


def heuristic(board: list, ai_player: str, depth: int) -> int:
    """Score a terminal/leaf node from the AI's point of view.

    +(10 + depth) for an AI win, -(10 + depth) for an AI loss, 0 for a draw or
    a non-terminal leaf (depth limit reached). Adding ``depth`` makes the engine
    prefer winning sooner and losing later, since deeper nodes have smaller depth.
    """
    win = winner(board)
    if win == ai_player:
        return 10 + depth
    if win == other(ai_player):
        return -(10 + depth)
    return 0


def alphabeta(board, player, depth, alpha, beta, maximizing, ai_player):
    """Return the minimax value of ``board`` with alpha-beta pruning.

    Args:
        board: current 9-cell board.
        player: whose turn it is to move on this board.
        depth: remaining search depth (0 stops the search).
        alpha: best value the maximizer can already guarantee.
        beta: best value the minimizer can already guarantee.
        maximizing: True if ``player`` is the AI (maximizing) player.
        ai_player: the mark the AI plays as ("X" or "O").
    """
    if depth == 0 or is_terminal(board):
        return heuristic(board, ai_player, depth)

    if maximizing:
        value = -INF
        for move in available_moves(board):
            board[move] = player
            value = max(
                value,
                alphabeta(board, other(player), depth - 1, alpha, beta, False, ai_player),
            )
            board[move] = ""  # undo
            if value >= beta:
                break  # beta cutoff
            alpha = max(alpha, value)
        return value
    else:
        value = INF
        for move in available_moves(board):
            board[move] = player
            value = min(
                value,
                alphabeta(board, other(player), depth - 1, alpha, beta, True, ai_player),
            )
            board[move] = ""  # undo
            if value <= alpha:
                break  # alpha cutoff
            beta = min(beta, value)
        return value
