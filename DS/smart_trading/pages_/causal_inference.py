import streamlit as st
import yfinance as yf
from datetime import date
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
import lightgbm as lgb
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Standard market indices
indices = {
    'S&P 500': '^GSPC',
    'Dow Jones Industrial Average': '^DJI',
    'NASDAQ Composite': '^IXIC',
    'Russell 2000': '^RUT'  # Reflects smaller companies
}
# Sector ETFs
sector_etfs = {
    'Financial Sector': 'XLF',
    'Technology Sector': 'XLK',
    'Energy Sector': 'XLE',
    'Industrial Sector': 'XLI',
    'Health Care Sector': 'XLV',
    'Consumer Discretionary Sector': 'XLY',
    'Consumer Staples Sector': 'XLP',
    'Utilities Sector': 'XLU',
    'Real Estate Sector': 'XLRE',
    'Communication Services Sector': 'XLC'
}
# Descriptions for each ETF
etf_descriptions = {
    'XLF': 'Financial Sector ETF',
    'XLK': 'Technology Sector ETF',
    'XLE': 'Energy Sector ETF',
    'XLI': 'Industrial Sector ETF',
    'XLV': 'Health Care Sector ETF',
    'XLY': 'Consumer Discretionary Sector ETF',
    'XLP': 'Consumer Staples Sector ETF',
    'XLU': 'Utilities Sector ETF',
    'XLRE': 'Real Estate Sector ETF',
    'XLC': 'Communication Services Sector ETF'
}
# Volatility Index
vix = {'VIX': '^VIX'}
# Commodity ETFs as proxies
commodity_etfs = {
    'Gold': 'GLD',
    'Silver': 'SLV',
    'Oil': 'USO',
    'Natural Gas': 'UNG',
    'Gasoline': 'UGA',
    'Brent Oil': 'BNO',
    'Copper': 'COPX',
    'Palladium': 'PALL',
    'Platinum': 'PLTM',
    'Corn': 'CORN',
    'Soybeans': 'SOYB',
    'Wheat': 'WEAT',
    'Agribusiness': 'MOO',
    'Livestock': 'COW',
    'Physical Gold (Switzerland)': 'SGOL',
    'Physical Silver': 'SIVR',
    'Rare Earth/Strategic Metals': 'REMX',
    'Timber & Forestry': 'WOOD',
    'Cotton': 'BAL'
}

def fetch_data(tickers, start_date, end_date):
    """
    Fetch adjusted close prices for provided tickers between specific dates.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data


# def causal_discovery(data):

#     # Use another scoring method
#     bic_score = BicScore(data)
#     k2_score = K2Score(data)

#     # Initialize Hill Climb Search with more detailed output
#     hc = HillClimbSearch(data)

#     # Estimate the model with increased verbosity and a different scoring function
#     best_model = hc.estimate(scoring_method=k2_score, max_indegree=2, max_iter=100000, epsilon=1e-4, show_progress=False)

#     # Streamlit method to display edges found in the model
#     if best_model.edges():
#         st.write("Edges in the best model found:", best_model.edges())
#     else:
#         st.write("No significant edges found.")

# def causal_discovery(data):
#     from pgmpy.estimators import HillClimbSearch, BicScore, K2Score

#     # Use another scoring method
#     bic_score = BicScore(data)
#     k2_score = K2Score(data)

#     # Initialize Hill Climb Search with more detailed output
#     hc = HillClimbSearch(data)

#     # Estimate the model with increased verbosity and a different scoring function
#     best_model = hc.estimate(scoring_method=k2_score, max_indegree=2, max_iter=100000, epsilon=1e-4, show_progress=False)

#     # Create and show network graph
#     if best_model.edges():
#         G = nx.DiGraph()
#         G.add_edges_from(best_model.edges())
#         pos = nx.spring_layout(G, seed=42)  # positions for all nodes
#         plt.figure(figsize=(10, 8))
#         nx.draw(G, pos, node_color='skyblue', node_size=2000, edge_color='black', linewidths=1, font_size=10, with_labels=True)
#         st.pyplot(plt)
#     else:
#         st.write("No significant edges found.")
# def causal_discovery(data):
#     # Initialize and estimate the model
#     hc = HillClimbSearch(data)
#     best_model = hc.estimate(scoring_method=K2Score(data), max_indegree=2, max_iter=100000, epsilon=1e-4, show_progress=False)

#     # Check if there are any edges
#     if best_model.edges():
#         # Create a directed graph
#         G = nx.DiGraph()
#         G.add_edges_from(best_model.edges())
#         pos = nx.spring_layout(G, seed=42)  # Layout for consistent positioning

#         # Create edge traces for Plotly
#         edge_x = []
#         edge_y = []
#         for edge in G.edges():
#             x0, y0 = pos[edge[0]]
#             x1, y1 = pos[edge[1]]
#             edge_x.extend([x0, x1, None])
#             edge_y.extend([y0, y1, None])

#         edge_trace = go.Scatter(
#             x=edge_x, y=edge_y,
#             line=dict(width=2, color='grey'),
#             hoverinfo='none',
#             mode='lines')

#         # Create node traces
#         node_x = []
#         node_y = []
#         node_text = []
#         for node in G.nodes():
#             x, y = pos[node]
#             node_x.append(x)
#             node_y.append(y)
#             node_text.append(node)

#         node_trace = go.Scatter(
#             x=node_x, y=node_y,
#             mode='markers+text',
#             hoverinfo='text',
#             text=node_text,
#             marker=dict(
#                 showscale=False,
#                 colorscale='Viridis',
#                 size=10,
#                 color='blue',
#                 line_width=2))

#         # Create figure
#         fig = go.Figure(data=[edge_trace, node_trace],
#                         layout=go.Layout(
#                             showlegend=False,
#                             hovermode='closest',
#                             margin=dict(b=0, l=0, r=0, t=0),
#                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                             clickmode='event+select'))

#         # Display the figure
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.write("No significant edges found.")

# def causal_discovery(data):
#     # Initialize and estimate the model
#     hc = HillClimbSearch(data)
#     best_model = hc.estimate(scoring_method=K2Score(data), max_indegree=2, max_iter=100000, epsilon=1e-4, show_progress=False)

#     # Check if there are any edges
#     if best_model.edges():
#         G = nx.DiGraph()
#         G.add_edges_from(best_model.edges())
#         pos = nx.spring_layout(G, seed=42)  # Layout for consistent positioning

#         edge_trace = go.Scatter(
#             x=[],
#             y=[],
#             line=dict(width=0.5, color='#888'),
#             hoverinfo='none',
#             mode='lines')

#         for edge in G.edges():
#             x0, y0 = pos[edge[0]]
#             x1, y1 = pos[edge[1]]
#             edge_trace['x'] += tuple([x0, x1, None])
#             edge_trace['y'] += tuple([y0, y1, None])

#         node_trace = go.Scatter(
#             x=[],
#             y=[],
#             text=[],
#             mode='markers',
#             hoverinfo='text',
#             marker=dict(
#                 showscale=True,
#                 colorscale='YlGnBu',
#                 size=10,
#                 color=[],
#                 colorbar=dict(
#                     thickness=15,
#                     title='Node Connections',
#                     xanchor='left',
#                     titleside='right'
#                 ),
#                 line_width=2))

#         for node in G.nodes():
#             x, y = pos[node]
#             node_trace['x'] += tuple([x])
#             node_trace['y'] += tuple([y])
#             node_trace['text'] += tuple([f'{node}'])

#         fig = go.Figure(data=[edge_trace, node_trace],
#                         layout=go.Layout(
#                             showlegend=False,
#                             hovermode='closest',
#                             margin=dict(b=0, l=0, r=0, t=0),
#                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                             clickmode='event+select'))

#         fig.update_layout(
#             updatemenus=[
#                 dict(
#                     type="buttons",
#                     buttons=[
#                         dict(label="Reset",
#                              method="relayout",
#                              args=["line.color", 'grey'])
#                     ]
#                 )
#             ]
#         )

#         # Display the figure
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.write("No significant edges found.")

# def causal_discovery(data):
#     hc = HillClimbSearch(data)
#     best_model = hc.estimate(scoring_method=K2Score(data), max_indegree=2, max_iter=100000, epsilon=1e-4, show_progress=False)

#     if best_model.edges():
#         G = nx.DiGraph()
#         G.add_edges_from(best_model.edges())
#         pos = nx.spring_layout(G, seed=42)  # Consistent positioning

#         edge_trace = go.Scatter(
#             x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
#         for edge in G.edges():
#             x0, y0 = pos[edge[0]]
#             x1, y1 = pos[edge[1]]
#             edge_trace['x'] += (x0, x1, None)
#             edge_trace['y'] += (y0, y1, None)

#         node_trace = go.Scatter(
#             x=[], y=[], text=[], mode='markers+text', hoverinfo='text', marker=dict(
#                 showscale=True, colorscale='YlOrRd', size=[], color=[], colorbar=dict(
#                     thickness=15, title='Node Connections', xanchor='left', titleside='right'),
#                 line_width=2))

#         for node in G.nodes():
#             x, y = pos[node]
#             node_trace['x'] += (x,)
#             node_trace['y'] += (y,)
#             node_trace['text'] += (node,)
#             node_trace['marker']['size'] += (10 * len(G.edges(node)),)  # Node size by connectivity
#             node_trace['marker']['color'] += (len(G.edges(node)),)  # Node color by connectivity

#         fig = go.Figure(data=[edge_trace, node_trace],
#                         layout=go.Layout(
#                             title='<br>Network graph of causal relationships',
#                             titlefont_size=16, showlegend=False, hovermode='closest',
#                             margin=dict(b=20, l=5, r=5, t=40),
#                             annotations=[dict(text="PGMPY and Plotly visualization", showarrow=False,
#                                               xref="paper", yref="paper", x=0.005, y=-0.002)],
#                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.write("No significant edges found.")
# def causal_discovery(data, focal_node):
#     hc = HillClimbSearch(data)
#     best_model = hc.estimate(scoring_method=K2Score(data), max_indegree=2, max_iter=100000, epsilon=1e-4, show_progress=False)

#     if best_model.edges():
#         G = nx.DiGraph()
#         G.add_edges_from(best_model.edges())

#         if focal_node in G:
#             # Extract the subgraph for nodes directly connected to the focal node
#             connected_nodes = list(G.neighbors(focal_node)) + [focal_node]
#             H = G.subgraph(connected_nodes)  # Create the subgraph of interest
#         else:
#             st.write(f"No connections found for {focal_node}. Showing full graph.")
#             H = G  # Use the full graph if no connections

#         pos = nx.spring_layout(H, seed=42)  # Layout for consistent positioning

#         edge_trace = go.Scatter(
#             x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
#         for edge in H.edges():
#             x0, y0 = pos[edge[0]]
#             x1, y1 = pos[edge[1]]
#             edge_trace['x'] += (x0, x1, None)
#             edge_trace['y'] += (y0, y1, None)

#         node_trace = go.Scatter(
#             x=[], y=[], text=[], mode='markers+text', hoverinfo='text', marker=dict(
#                 showscale=True, colorscale='YlOrRd', size=[], color=[], colorbar=dict(
#                     thickness=15, title='Node Connections', xanchor='left', titleside='right'),
#                 line_width=2))

#         for node in H.nodes():
#             x, y = pos[node]
#             node_trace['x'] += (x,)
#             node_trace['y'] += (y,)
#             node_trace['text'] += (node,)
#             node_trace['marker']['size'] += (10 * len(H.edges(node)),)
#             node_trace['marker']['color'] += (len(H.edges(node)),)

#         fig = go.Figure(data=[edge_trace, node_trace],
#                         layout=go.Layout(
#                             title=f'<br>Network graph centered on {focal_node}',
#                             titlefont_size=16, showlegend=False, hovermode='closest',
#                             margin=dict(b=20, l=5, r=5, t=40),
#                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.write("No significant edges found.")

def causal_discovery(data, focal_node):
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=K2Score(data), max_indegree=2, max_iter=100000, epsilon=1e-4, show_progress=False)

    if best_model.edges():
        G = nx.DiGraph()
        G.add_edges_from(best_model.edges())

        if focal_node in G:
            # Get directly connected nodes
            direct_neighbors = set(G.neighbors(focal_node))
            extended_neighbors = set(direct_neighbors)  # Initialize with direct neighbors

            # Add neighbors of each directly connected node
            for neighbor in direct_neighbors:
                extended_neighbors.update(G.neighbors(neighbor))

            # Ensure the focal node is included
            extended_neighbors.add(focal_node)

            # Create the subgraph for these nodes
            H = G.subgraph(extended_neighbors)
        else:
            st.write(f"No connections found for {focal_node}. Showing full graph.")
            H = G  # Use the full graph if no connections for focal node

        pos = nx.spring_layout(H, seed=42)  # Position nodes using spring layout

        # Edge traces for Plotly visualization
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        for edge in H.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)

        # Node traces
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text', hoverinfo='text', marker=dict(
                showscale=True, colorscale='YlOrRd', size=[], color=[], colorbar=dict(
                    thickness=15, title='Node Connections', xanchor='left', titleside='right'),
                line_width=2))

        for node in H.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node,)
            # Adjust node size based on degree within subgraph
            node_trace['marker']['size'] += (10 * len(H.edges(node)),)
            node_trace['marker']['color'] += (len(H.edges(node)),)

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'<br>Network graph centered on {focal_node}',
                            titlefont_size=16, showlegend=False, hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No significant edges found.")

def causal_inference(data, treatments, covariates, outcome):
    # First Stage: Get treatment residuals
    treatment_residuals = {}
    for treatment in treatments:
        model_treatment = lgb.LGBMRegressor(random_state=123, cv=5, force_col_wise=True, verbose=-1)
        predicted_treatment = cross_val_predict(model_treatment, data[covariates], data[treatment], cv=5)
        residuals = data[treatment] - predicted_treatment
        treatment_residuals[treatment] = residuals

    # First Stage for Outcome: Predict outcome from covariates to get outcome residuals
    # model_outcome = LassoCV(cv=5)
    model_outcome = lgb.LGBMRegressor(random_state=123, cv=5, force_col_wise=True, verbose=-1)
    predicted_outcome = cross_val_predict(model_outcome, data[covariates], data[outcome], cv=5)
    outcome_residuals = data[outcome] - predicted_outcome

    # Second Stage: Regress outcome residuals on treatment residuals
    residuals_df = pd.DataFrame(treatment_residuals)  # DataFrame of treatment residuals
    model_final = LassoCV(cv=5)
    model_final.fit(residuals_df, outcome_residuals)  # Regress outcome residuals on treatment residuals

    # Coefficients from this model represent the causal effects
    effects = model_final.coef_
    return effects

    # print("Estimated causal effects of treatments on outcome:")
    # for i, treatment in enumerate(treatments):
    #     print(f"{treatment}: {effects[i]}")


def show_causal_inference():
    st.title("Causal Inference for Trading")

    user_ticker = st.sidebar.text_input('Enter Stock Ticker', value='AAPL').upper()
    start_date = st.sidebar.date_input('Start Date', value=date(2021, 1, 1))
    end_date = st.sidebar.date_input('End Date', value=date.today())

    # Combine all tickers into a single dictionary for simplicity
    all_tickers = {**indices, **sector_etfs, **vix, **commodity_etfs}

    if st.sidebar.button('Fetch Data'):
        with st.spinner('Fetching data...'):

            # Fetch the data
            all_data = fetch_data(list(all_tickers.values()), start_date, end_date)
            st.write("All Tickers Data:")
            st.dataframe(all_data.tail())
            returns = all_data.pct_change().dropna()
            # Fetch data for the user-selected ticker
            additional_data = yf.download(user_ticker, start=start_date, end=end_date)['Adj Close']
            returns[user_ticker] = additional_data.pct_change().dropna()
            data = returns
            st.write(f"{user_ticker} Data:")
            st.line_chart(additional_data)

            # Define outcome and treatments
            
            outcome = user_ticker
            treatments_tickers = {**sector_etfs}
            treatments = list(treatments_tickers.values())  # Assuming all sector ETFs are potential treatments
            covariates_tickers = {**indices, **vix, **commodity_etfs}
            covariates = list(covariates_tickers.values())     # Add any other covariates if necessary

            if not data.empty:
                # Perform causal discovery
                st.subheader("Causal Discovery")
                causal_discovery(data, user_ticker)
                
                # Perform causal inference
                st.subheader("Causal Inference Results")
                effects = causal_inference(data, treatments, covariates, outcome)

                # Create DataFrame with descriptions
                effects_df = pd.DataFrame({
                    'Treatment': treatments,
                    'Effect': effects,
                    'Description': [etf_descriptions[t] for t in treatments]
                })

                # Display causal effects in a table
                st.write("Estimated causal effects of treatments on the outcome:")
                st.table(effects_df)

                # Optionally, you could add visualizations of these relationships/effects
                st.bar_chart(effects_df.drop(['Description'], axis=1).set_index('Treatment'))

                # Optionally, you could add visualizations of these relationships/effects
                st.write("Graphical representation of causal effects not implemented.")
            else:
                st.write("No data available for this ticker.")
