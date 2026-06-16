import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

from projects.tic_tac_toe.src.engine import best_move

logger = logging.getLogger(__name__)

router = APIRouter()


class BoardInput(BaseModel):
    # Exactly 9 cells; the engine validates the contents, but bounding the
    # length here rejects oversized lists before they reach it.
    board: List[str] = Field(
        ..., min_length=9, max_length=9, description='9 cells, each "", "X", or "O".'
    )
    ai_player: str = Field("O", description='Mark the AI plays as: "X" or "O".')
    depth: int = Field(9, ge=1, le=9, description="Max minimax search depth.")


@router.post("/predict/tic-tac-toe")
def tic_tac_toe_move(input_data: BoardInput):
    try:
        return best_move(input_data.board, input_data.ai_player, input_data.depth)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Tic-tac-toe move computation failed")
        raise HTTPException(status_code=500, detail="Move computation failed")
