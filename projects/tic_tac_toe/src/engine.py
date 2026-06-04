"""High-level tic-tac-toe engine: pick the AI's best move and report game state."""

from .game import available_moves, is_full, other, winner
from .minimax import INF, alphabeta

VALID_CELLS = {"", "X", "O"}


def validate_board(board: list) -> None:
    """Raise ValueError if the board is not a legal 9-cell tic-tac-toe state."""
    if not isinstance(board, list) or len(board) != 9:
        raise ValueError("Board must be a list of exactly 9 cells.")
    if any(cell not in VALID_CELLS for cell in board):
        raise ValueError('Each cell must be "", "X", or "O".')

    x_count = board.count("X")
    o_count = board.count("O")
    # X always moves first, so X has either as many or one more mark than O.
    if o_count not in (x_count, x_count - 1):
        raise ValueError("Illegal board: move counts for X and O are inconsistent.")


def game_status(board: list) -> dict:
    """Return whether the game is over and who won."""
    win = winner(board)
    if win:
        return {"over": True, "winner": win, "draw": False}
    if is_full(board):
        return {"over": True, "winner": None, "draw": True}
    return {"over": False, "winner": None, "draw": False}


def best_move(board: list, ai_player: str = "O", depth: int = 9) -> dict:
    """Return the AI's optimal move for ``board`` using alpha-beta minimax.

    Args:
        board: the current 9-cell board.
        ai_player: the mark the AI plays as ("X" or "O"). Defaults to "O".
        depth: maximum search depth; 9 covers the whole game tree.

    Returns:
        A dict with the chosen ``move`` index (or None if the game is already
        over), the move's ``score``, and the resulting ``status``.
    """
    validate_board(board)
    if ai_player not in ("X", "O"):
        raise ValueError('ai_player must be "X" or "O".')

    status = game_status(board)
    if status["over"]:
        return {"move": None, "score": None, "status": status}

    best_idx = None
    best_score = -INF
    alpha, beta = -INF, INF

    # Root is the maximizing layer: try each move, then let the opponent reply.
    for move in available_moves(board):
        board[move] = ai_player
        score = alphabeta(
            board, other(ai_player), depth - 1, alpha, beta, False, ai_player
        )
        board[move] = ""  # undo

        if score > best_score:
            best_score = score
            best_idx = move
        alpha = max(alpha, best_score)

    # Report the status of the board *after* the AI plays its chosen move.
    board[best_idx] = ai_player
    resulting_status = game_status(board)
    board[best_idx] = ""

    return {"move": best_idx, "score": best_score, "status": resulting_status}
