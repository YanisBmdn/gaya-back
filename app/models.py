from dataclasses import dataclass
from pydantic import BaseModel
from typing import Dict
from enum import Enum
import pandas as pd

@dataclass
class VisualizationNeed(BaseModel):
    need_visualization: int

@dataclass
class PersonaSelection(BaseModel):
    persona_id: int

@dataclass
class APISelection(BaseModel):
    url: str

@dataclass
class VisualizationType(BaseModel):
    json_definition: str

@dataclass
class ProcessedData:
    """
    A generic container for preprocessed dataframes.
    
    Attributes:
        main_data (pd.DataFrame): The base dataframe with non-nested columns
        nested_dataframes (Dict[str, pd.DataFrame]): Dictionary of expanded nested dataframes
    """
    main_data: pd.DataFrame
    nested_dataframes: Dict[str, pd.DataFrame]


class OutputType(Enum):
    VISUALIZATION = "visualization"
    TEXT = "text"