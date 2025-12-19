
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis import get_rgb_string
from analysis import simplex_to_polygon_coords
import numpy as np


def plotly_simplex_vs_raw(states, dir="./figs"):

    states_flat = states.reshape(-1, states.shape[-1])

    color_codes = get_rgb_string(states_flat)
    #true_states_flat_shifted = np.array(test_states[:,1:].reshape(-1, 3))   

    # Create subplots with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Raw State Components', 'Simplex Projection'),
        horizontal_spacing=0.1,
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}]]
    )   

    # Transform the states data for simplex projection
    poly_coords, vertices = simplex_to_polygon_coords(states_flat)   


    # Subplot 1: Raw state components
    fig.add_trace(go.Scatter3d(  
        x=states_flat[:,0], 
        y=states_flat[:,1],
        z=states_flat[:,2],
        mode='markers',
        marker=dict(
            size=3,
            color=color_codes,
            opacity=1,
        ),
        showlegend=False
    ), row=1, col=1)    

    # Subplot 2: Simplex projection
    fig.add_trace(go.Scatter(
        x=poly_coords[:, 0],
        y=poly_coords[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color=color_codes,
            opacity=1,
        ),
        showlegend=False  # Don't duplicate in legend
    ), row=1, col=2)    

    # Add boundaries for simplex projection subplot
    vertices.append(vertices[0])
    fig.add_trace(go.Scatter(
        x=[item[0] for item in vertices],
        y=[item[1] for item in vertices],
        mode='lines',
        showlegend=False,
        opacity=1,
        line=dict(color='black', width=2)
    ), row=1, col=2)    

    # Update layout
    fig.update_layout(
        title='Visualization of True Hidden States: Raw vs 2D Simplex Projection',
        height=600,
        width=1400,
        showlegend=True
    )   

    # Update subplot axes
        # Update 3D subplot scene
    fig.update_scenes(
        xaxis_title='Belief State Component 1',
        yaxis_title='Belief State Component 2', 
        zaxis_title='Belief State Component 3',
        row=1, col=1
    )

    fig.update_xaxes(title_text='Simplex Projection Dimension 1', row=1, col=2)
    fig.update_yaxes(title_text='Simplex Projection Dimension 2', row=1, col=2)    

    # Keep aspect ratio for simplex plot
    fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(constrain="domain", row=1, col=2)  

    fig.write_html(f"{dir}/simplex_vs_raw.html")
    return fig



def add_simplex_projection(fig, states, row=1, col=1):
    """
    Add simplex projection plot to an existing figure
    
    Args:
        fig: Plotly figure object to add the plot to
        states: Array of states to project
        row: Row position in subplot grid
        col: Column position in subplot grid
        title_text: Title for this subplot
    """
    
    states_flat = states.reshape(-1, states.shape[-1])

    color_codes = get_rgb_string(states_flat)
    
    # Transform the states data for simplex projection
    poly_coords, vertices = simplex_to_polygon_coords(states_flat)
    
    # Add simplex projection points
    fig.add_trace(go.Scatter(
        x=poly_coords[:, 0],
        y=poly_coords[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color=color_codes,
            opacity=1,
        ),
        showlegend=False,
        name=f'States {col}'
    ), row=row, col=col)
    
    # Add boundaries for simplex projection
    vertices_copy = vertices.copy()
    vertices_copy.append(vertices_copy[0])
    fig.add_trace(go.Scatter(
        x=[item[0] for item in vertices_copy],
        y=[item[1] for item in vertices_copy],
        mode='lines',
        showlegend=False,
        opacity=1,
        line=dict(color='black', width=2),
        name=f'Boundary {col}'
    ), row=row, col=col)
    
    # Update axes for this subplot
    fig.update_xaxes(title_text='Simplex Projection Dimension 1', row=row, col=col)
    fig.update_yaxes(title_text='Simplex Projection Dimension 2', row=row, col=col)
    
    # Keep aspect ratio
    fig.update_xaxes(scaleanchor=f"y{col}", scaleratio=1, row=row, col=col)
    fig.update_yaxes(constrain="domain", row=row, col=col)
    
    return fig


