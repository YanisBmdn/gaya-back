import json
import logging

from functools import wraps
from typing import Callable, Any, TypeVar

import plotly.graph_objects as go
import plotly

T = TypeVar('T')

def handle_exceptions(
    default_return: Any = None,
    reraise: bool = True,
    log_exception: bool = True,
    specific_exceptions: tuple = (Exception,)
) -> Callable:
    """
    A decorator to handle exceptions in a consistent way across functions.
    
    Args:
        default_return: Value to return if an exception occurs
        reraise: Whether to re-raise the caught exception
        log_exception: Whether to log the exception
        specific_exceptions: Tuple of exception types to catch
    
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except specific_exceptions as e:
                if log_exception:
                    logging.error(
                        f"Error in {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator

    
def figure_to_json(fig: go.Figure) -> str:
    """
    Turn a Plotly figure to a JSON serializable dictionary.

    Args:
        fig: The figure to convert

    Returns:
        dict: The JSON serializable dictionary

    """
    try:
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return fig_json

    except Exception as e:
        logging.error(f"Error converting figure to JSON: {e}")
        return ""
    
def enhance_plotly_figure(fig: go.Figure) -> go.Figure:
    """
    Function to improve accessibility and styling of a plotly figure.

    Args:
        fig: The Plotly figure to enhance

    Returns:
        go.Figure: The enhanced Plotly figure

    """
    # 1. Improve Layout and Background
    fig.update_layout(
        template="simple_white",  # Clean, professional template
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=12),  # Readable base font size
    )

    # 2. Enhance Grid and Axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E5E5',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='#808080'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E5E5',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='#808080'
    )

    """
    # 3. Improve Interactivity
    fig.update_layout(
        hovermode='closest',  # Works for most plot types
        modebar_add=['zoom', 'pan', 'select', 'lasso2d', 'reset'],
    )

    # 4. Ensure Legend is Well Positioned (if applicable)
    fig.update_layout(
        showlegend=True,  # Will automatically hide if no legend items
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'  # Semi-transparent background
        )
    )
    """

    # 5. Enhanced Accessibility - Generic improvements
    fig.update_layout(
        title=dict(
            font=dict(size=16),  # Larger title font
            x=0.5,  # Centered title
            xanchor='center'
        )
    )

    return fig


def dict_to_json_file(conversation_id: int, data: dict) -> None:
    """
    Convert a dictionary to a JSON string.

    Args:
        data: The dictionary to convert

    Returns:
        str: The JSON string

    """
    with open(f"conversation_{conversation_id}.json", "a") as f:
        json.dump(data, f)
        logging.info(f"Data saved to conversation_{conversation_id}.json")