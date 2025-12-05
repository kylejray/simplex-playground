
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

    # fig.write_html(f"{dir}/simplex_vs_raw.html")
    return fig

def plotly_pca_explained_variance_only(pca_results, step_id, dir="./figs"):
    """
    Plot PCA components vs explained variance
    
    Args:
        pca_results: PCA object with explained_variance_ratio_ attribute
        dir: Directory to save the plot
    """
    
    explained_variance = pca_results.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find where cumulative variance exceeds 99%
    cutoff_idx = np.where(cumulative_variance >= 0.99)[0]
    if len(cutoff_idx) > 0:
        max_components = max(cutoff_idx[0] + 1, 5)  # Always show at least 5 components
    else:
        max_components = max(len(explained_variance), 5)  # Always show at least 5 components
    
    # Make sure we don't exceed available components
    max_components = min(max_components, len(explained_variance))
    
    # Trim arrays to only include meaningful components
    component_numbers = np.arange(1, max_components + 1)
    explained_variance = explained_variance[:max_components]
    cumulative_variance = cumulative_variance[:max_components]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Individual explained variance
    fig.add_trace(
        go.Bar(
            x=component_numbers,
            y=explained_variance,
            name='Individual Explained Variance',
            marker_color='lightblue',
            opacity=0.7
        ),
        secondary_y=False,
    )
    
    # Cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=component_numbers,
            y=cumulative_variance,
            mode='lines+markers',
            name='Cumulative Explained Variance',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True,
    )
    
    # Add 99% threshold line
    fig.add_hline(
        y=0.99, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="99% threshold",
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f'PCA Explained Variance Analysis (Min 5 components, up to 99%)',
        height=500,
        width=800,
        showlegend=True
    )
    
    # Update x-axis
    fig.update_xaxes(
        title_text='Principal Component Number',
        dtick=1
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text='Individual Explained Variance Ratio',
        secondary_y=False
    )
    fig.update_yaxes(
        title_text='Cumulative Explained Variance Ratio',
        secondary_y=True,
        range=[0, 1]
    )

    fig.write_html(f"{dir}/pca_explained_variance_{step_id}.html")
    return

def add_pca_explained_variance(fig, pca_results, row=1, col=1, title_text="PCA Explained Variance", target_cumulative=0.99, min_components=5):
    """
    Add PCA explained-variance bars and cumulative line to an existing subplot.

    Args:
        fig: Plotly figure (from make_subplots).
        pca_results: Fitted PCA object with explained_variance_ratio_.
        row, col: Subplot coordinates.
        title_text: Optional title text for this subplot.
        target_cumulative: Cumulative variance target to highlight (e.g., 0.99).
        min_components: Always show at least this many components.
    Returns:
        fig (for chaining)
    """
    explained_variance = np.asarray(pca_results.explained_variance_ratio_)
    cumulative_variance = np.cumsum(explained_variance)

    cutoff_idx = np.where(cumulative_variance >= target_cumulative)[0]
    if len(cutoff_idx) > 0:
        max_components = max(cutoff_idx[0] + 1, min_components)
    else:
        max_components = max(len(explained_variance), min_components)
    max_components = min(max_components, len(explained_variance))

    component_numbers = np.arange(1, max_components + 1)
    ev = explained_variance[:max_components]
    cv = cumulative_variance[:max_components]

    # Bars: individual explained variance
    fig.add_trace(
        go.Bar(
            x=component_numbers,
            y=ev,
            name='Individual Explained Variance',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=row, col=col
    )

    # Line: cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=component_numbers,
            y=cv,
            mode='lines+markers',
            name='Cumulative Explained Variance',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ),
        row=row, col=col
    )

    # Threshold line
    fig.add_hline(
        y=target_cumulative,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{int(target_cumulative*100)}% threshold",
        row=row, col=col
    )

    # Axes for this subplot
    fig.update_xaxes(
        title_text='Principal Component Number',
        dtick=1,
        row=row, col=col
    )
    fig.update_yaxes(
        title_text='Explained Variance Ratio',
        range=[0, 1],
        row=row, col=col
    )

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

def add_3d_pca(fig, pca_activations, row=1, col=1):


    for i, (name, data) in enumerate(pca_activations.items()):

    
        # Subsample for clearer visualization if needed
        subsample = data if len(data) < 1000 else data[:1000]
    
        fig.add_trace(go.Scatter3d(
            x=subsample[:, 0],  # PC1
            y=subsample[:, 1],  # PC2
            z=subsample[:, 2],  # PC3
            mode='markers',
            name=f'Position {i}',
            marker=dict(
            size=4,
            opacity=0.9,
            line=dict(width=0.5, color='DarkSlateGrey')
            ),
            showlegend=False),
            row=row, col=col
        )
 

    # Update subplot axes
        # Update 3D subplot scene
    fig.update_scenes(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2', 
        zaxis_title='PCA Component 3',
        row=row, col=col
    )


    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)

    # Keep aspect ratio
    fig.update_xaxes(scaleanchor=f"y{col}", scaleratio=1, row=row, col=col)
    fig.update_yaxes(constrain="domain", row=row, col=col)

    return fig

def comparative_2d_projection(true_states, predicted_states, step, dir="./figs"):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "scatter"}]])

    fig = add_simplex_projection(fig, true_states, row=1, col=1)
    fig = add_simplex_projection(fig, predicted_states, row=1, col=2)

    fig.write_html(f"{dir}/comparative_simplex_projection_{step}.html")
    return 

def plotly_pca_explained_variance(pca_results, pca_activations, step_id, dir="./figs"):
    """
    Standalone figure that uses add_pca_explained_variance and saves to HTML.
    """
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "scatter3d"}]])
    fig = add_pca_explained_variance(fig, pca_results, row=1, col=1)
    fig = add_3d_pca(fig, pca_activations, row=1, col=2)

    fig.update_layout(
        title='PCA Plot',
        height=600,
        width=1400,
    )  
    
    fig.write_html(f"{dir}/pca_plot_{step_id}.html")
    return fig


