import logging
import base64

import plotly.graph_objects as go

def figure_to_base64(fig: go.Figure) -> str:
    try:
        img_bytes = fig.to_image(format="png", engine="kaleido", width=800)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

    except Exception as e:
        logging.error(f"Error converting figure to base64: {e}")
        return ""