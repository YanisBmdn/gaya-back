from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd

@dataclass
class VisualizationNeed(BaseModel):
    need_visualization: int = Field(description="Whether the user needs a visualization or not")
    topic_of_interest: str = Field(description="The topic of interest for the visualization")

@dataclass
class PersonaSelection(BaseModel):
    persona_id: int = Field(description="The persona ID that fits the user")

@dataclass
class ProcessedData:
    """
    A generic container for processed dataframes.
    
    Attributes:
        main_data (pd.DataFrame): The base dataframe with non-nested columns
        nested_dataframes (dict[str, pd.DataFrame]): Dictionary of expanded nested dataframes
    """
    main_data: pd.DataFrame
    nested_dataframes: dict[str, pd.DataFrame]

    def __str__(self):
        def generate_description(dataframes_dict: dict[str, pd.DataFrame]) -> str:
            """
            Generate a formatted description of multiple dataframes
            Args:
                dataframes_dict: Dictionary of dataframe name and corresponding dataframe
            Returns:
                Formatted string describing each dataframe
            """
            if not dataframes_dict:
                return ""
                
            descriptions = []
            for name, df in dataframes_dict.items():
                descriptions.append(
                    f"Dataset: {name}\n"
                    f"Preview:\n{df.head()}\n"
                    f"Shape: {df.shape}"
                )
                
            return "\n\n".join(descriptions)
        
        main_data_description = (
            f"Main Data:\n"
            f"Preview:\n{self.main_data.head()}\n"
            f"Shape: {self.main_data.shape}"
        )
        
        nested_dataframes_description = generate_description(self.nested_dataframes)
        
        return f"{main_data_description}\n\nNested Dataframes:\n{nested_dataframes_description}"
    
    def describe_patterns(self) -> str:
        """
        Generate a simple description of patterns in any time series data.
        Works with any dataframe that has a datetime index.
        """
        summary = ""
        
        for df_name, df in self.nested_dataframes.items():
            summary += f"Dataset: {df_name}\n"
            summary += f"Time range: {df.index.min()} to {df.index.max()}\n\n"
            
            # Analyze each numerical column
            for column in df.select_dtypes(include=['float64', 'int64']).columns:
                summary += f"Variable: {column}\n"
                
                # Basic statistics
                avg = df[column].mean()
                min_val = df[column].min()
                max_val = df[column].max()
                
                # Overall trend
                start_val = df[column].iloc[0]
                end_val = df[column].iloc[-1]
                total_change = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
                
                summary += f"Range: {min_val:.2f} to {max_val:.2f}\n"
                summary += f"Average: {avg:.2f}\n"
                summary += f"Overall change: {total_change:.1f}%\n\n"
            
            summary += "---\n\n"
        
        return summary


## Visualization classes

class VisualizationType(BaseModel):
    visualization: str = Field(description="Clear, descriptive name of the visualization")
    chart_type: str = Field(description="Visualization chart type (e.g. bar chart, scatter plot)")
    focus: str = Field(description="What aspect of climate change this visualization reveals")
    visual_elements: str = Field(description="Important visualization elemnts (e.g. axes, labels)")

    def __str__(self):
        return f"""
        Visualization: {self.visualization}
        Chart Type: {self.chart_type}
        Focus: {self.focus}
        Visual Elements: {self.visual_elements}
        """


class APIEndpoint(BaseModel):
    url: str = Field(description="API endpoint URL with inline parameters")

class APIEndpointResponse(BaseModel):
    endpoints: List[APIEndpoint] = Field(description="List of API endpoints to query")


class DataProcessingType(BaseModel):
    needed_data: str = Field(description="List of the data needed for visualization, including time range and location")
    data_processing_steps: str = Field(description="Step by step process to prepare data for visualization")


class NormalizedOpenMeteoData(BaseModel):
    metadata: Optional[pd.DataFrame] = Field(description="Dataframe containing data unrelated to time resolution")
    hourly_data: Optional[pd.DataFrame] = Field(description="Dataframe with hourly data")
    daily_data: Optional[pd.DataFrame] = Field(description="Dataframe with daily data")

    def __str__(self):
        return f"""
        Metadata: {self.metadata.head()} shape: {self.metadata.shape}
        Hourly Data: {self.hourly_data.head()} shape: {self.hourly_data.shape}
        Daily Data: {self.daily_data.head()} shape: {self.daily_data.shape}
        """

    class Config:
        arbitrary_types_allowed = True

    def generate_data_description(self) -> str:
        """
        Generate a statistical description of temporal data.
        Returns overall statistics for each numeric column in hourly and daily data.
        
        Returns:
            String containing statistical description
        """
        description = []
        
        if self.hourly_data is not None:
            numeric_cols = self.hourly_data.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col != 'time']
            
            if len(numeric_cols) > 0:
                stats = self.hourly_data[numeric_cols].describe()
                description.append("Hourly Data:")
                description.append(f"Time range: {self.hourly_data['time'].min()} to {self.hourly_data['time'].max()}")
                for col in numeric_cols:
                    description.append(f"{col}: mean={stats[col]['mean']:.2f}, min={stats[col]['min']:.2f}, max={stats[col]['max']:.2f}")
        
        if self.daily_data is not None:
            numeric_cols = self.daily_data.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col != 'time']
            
            if len(numeric_cols) > 0:
                stats = self.daily_data[numeric_cols].describe()
                description.append("\nDaily Data:")
                description.append(f"Time range: {self.daily_data['time'].min()} to {self.daily_data['time'].max()}")
                for col in numeric_cols:
                    description.append(f"{col}: mean={stats[col]['mean']:.2f}, min={stats[col]['min']:.2f}, max={stats[col]['max']:.2f}")
        
        return "\n".join(description)



#--- API SCHEMA ---#

class ChatVisualizationRequest(BaseModel):
    chat_id: str = Field(description="The chat ID of the user")
    complexity_level: int = Field(description="The complexity level of the user")
    user_description: str = Field(description="The description of the user")
    location: str = Field(description="The location of interest")
    message: str = Field(description="The message from the user")
    scenario: str = Field(description="The scenario proposed to the user")
    topic: str = Field(description="The topic of interest for the visualization")
    options: List[str] = Field(description="The options available for the selected scenario")

class ChatDescriptionRequest(BaseModel):
    chat_id: str = Field(description="The chat ID of the user")
    image: str = Field(description="The base64 image of the visualization to describe")

class ChatVisualizationResponse(BaseModel):
    visualization: str = Field(description="The visualization generated for the user")

class ChatExplanationResponse(BaseModel):
    explanation: str = Field(description="The explanation generated for the user")

class ScenarioRequest(BaseModel):
    chat_id: str = Field(description="The chat ID of the user")
    age_group: str = Field(description="The age of the user")
    location: str = Field(description="The location of interest")
    user_description: str = Field(description="The description of the user")


class ScenarioResponse(BaseModel):
    scenario: str = Field(description="The generated scenario")
    budget: int = Field(description="The budget for the selected scenario")
    options: List[str] = Field(description="The options available for the selected scenario")

class PersonaRequest(BaseModel):
    description: str = Field(description="The description of the user")
    age_group: str = Field(description="The age group of the user")

class ChatRequest(BaseModel):
    message: str = Field(description="The message from the user")