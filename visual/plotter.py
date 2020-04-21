import plotly.graph_objects as go
import numpy as np


def plot_lines(lines: 'dict', save_path, file_type="pdf", plot_number=20, xaxis_title=None, yaxis_title=None):
    fig = go.Figure()
    for i, key in enumerate(lines.keys()):
        plot_x = np.arange(0, len(lines[key]), max(1, len(lines[key]) // plot_number))
        fig.add_trace(go.Scatter(x=plot_x, y=np.array(lines[key])[plot_x],
                                 mode='lines+markers',
                                 marker={"symbol": i + 1, "size": 15},
                                 name=key))
    fig.update_layout(
        plot_bgcolor="white",
        showlegend=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0},
        legend={"font": {"size": 25}, "x": 0.68, "y": 0.01, "bordercolor": "black", "borderwidth": 1},
        xaxis={"title": {"text": xaxis_title, "font": {"size": 30}, "standoff": 10}, "tickfont": {"size": 25}},
        yaxis={"title": {"text": yaxis_title, "font": {"size": 30}, "standoff": 15}, "tickfont": {"size": 25}},
        width=550,
        height=500,
    )
    fig.write_image(save_path + "." + file_type)



