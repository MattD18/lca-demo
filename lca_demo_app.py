import pickle

import streamlit as st
import numpy as np

from lca.viz import plot_lcas

st.header("LCA Demo")



## Load LCAS
with open('data/experiment_0/lcas.pickle', 'rb') as f:
    lcas = pickle.load(f)

# plot LCAs
num_steps = lcas.shape[0]
t = st.slider("Training step", min_value=0, max_value=num_steps, value=0, step=None, format=None, key=None)


col1, col2 = st.beta_columns(2)

lca_plot_1 = plot_lcas(lcas, t=t, viz_layer=0)
with st.beta_expander("Layer 1:"):
    st.plotly_chart(lca_plot_1)


lca_plot_2 = plot_lcas(lcas, t=t, viz_layer=2)
with st.beta_expander("Layer 2:"):
    st.plotly_chart(lca_plot_2)


lca_plot_3 = plot_lcas(lcas, t=t, viz_layer=4)
with st.beta_expander("Layer 3:"):
    st.plotly_chart(lca_plot_3)





