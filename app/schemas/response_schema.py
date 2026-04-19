from pydantic import BaseModel
from typing import Any, Dict


class StandardResponse(BaseModel):
    success: bool
    data: Dict[str, Any]