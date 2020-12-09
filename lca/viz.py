import numpy as np
import plotly.graph_objects as go


# constants, need to be pulled programmatically
layer_breaks = np.cumsum([78400,100,5000,50, 500,10])
layer_shapes = [(784, 100), (100,), (100, 50), (50,), (50, 10), (10,)]

layer_index_shape = dict(zip(
    layer_breaks,
    layer_shapes
))


def plot_lcas(lcas, t=0, viz_layer=0):
    '''
    Parameters:
    -----------
    lcas : np.ndarray
    t : int
    viz_layer : int
    '''
    unformatted_lcas_t = lcas[t]
    formatted_lcas_t = format_lca(unformatted_lcas_t)
    fig = get_plot_layer_lcas(formatted_lcas_t, viz_layer=viz_layer)
    return fig


def format_lca(unformatted_lcas):
    '''
    '''
    formatted_lcas_t = []
    layer_index_start = 0
    for layer_index in layer_breaks:
        layer_index_end = layer_index
        layer_lcas = unformatted_lcas[layer_index_start:layer_index_end]
        layer_lcas = layer_lcas.reshape(layer_index_shape[layer_index])
        formatted_lcas_t.append(layer_lcas)
        layer_index_start = layer_index_end
    return formatted_lcas_t

def get_plot_layer_lcas(formatted_lcas, viz_layer=0):
    lcas_t_viz_layer = formatted_lcas[viz_layer]
    fig = go.Figure(data=go.Heatmap(
                        z=lcas_t_viz_layer,
                        zmid=0,
                        colorscale=['green','white','red'])
                )
    return fig