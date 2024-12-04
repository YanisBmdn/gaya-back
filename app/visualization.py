import logging
import plotly.io as pio

def generate_plotly_chart(json_str: str) -> pio.Figure:
    try: 
        fig = pio.from_json(json_str, skip_invalid=False)
        return fig
    
    except ValueError as e:
        logging.info(f"Error in JSON validation: {e}")
        fig = pio.Figure()
        return fig
