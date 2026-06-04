"""Core tic-tac-toe board logic.

A board is a list of 9 strings: "X", "O", or "" (empty), indexed left-to-right,
top-to-bottom:

    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

EMPTY = ""
PLAYERS = ("X", "O")

# All winning index triplets: rows, columns, diagonals.
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


def other(player: str) -> str:
    """Return the opponent of the given player."""
    return "O" if player == "X" else "X"


def winner(board: list) -> str:
    """Return "X" or "O" if a player has three in a row, else ""."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return EMPTY


def available_moves(board: list) -> list:
    """Return the indices of all empty cells."""
    return [i for i, cell in enumerate(board) if cell == EMPTY]


def is_full(board: list) -> bool:
    """Return True if there are no empty cells left."""
    return EMPTY not in board


def is_terminal(board: list) -> bool:
    """A node is terminal when someone has won or the board is full (draw)."""
    return winner(board) != EMPTY or is_full(board)
