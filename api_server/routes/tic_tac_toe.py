from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from projects.tic_tac_toe.src.engine import best_move

router = APIRouter()


class BoardInput(BaseModel):
    board: List[str] = Field(..., description='9 cells, each "", "X", or "O".')
    ai_player: str = Field("O", description='Mark the AI plays as: "X" or "O".')
    depth: int = Field(9, ge=1, le=9, description="Max minimax search depth.")


@router.post("/predict/tic-tac-toe")
def tic_tac_toe_move(input_data: BoardInput):
    try:
        return best_move(input_data.board, input_data.ai_player, input_data.depth)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Move computation failed: {e}")
