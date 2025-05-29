# Placeholder for shared data schemas (e.g., Pydantic models)
from typing import List, Optional
from pydantic import BaseModel

class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None 