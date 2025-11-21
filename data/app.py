# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np

st.set_page_config(layout="wide", page_title="Global Development Dashboard")

@st.cache_data
def load_data(path="Global_Development_CLEAN.csv"):
    df = pd.read_csv(path)
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)
    return df

# Load dataset (file must be in same folder as app.py)
df = load_data("Global_Development_CLEAN.csv")

st.title("Global Development Dashboard — 2000–2020")
st.markdown("Interactive dashboard prepared for CEN445 — Data Visualization. Use the sidebar to filter and explore.")

# Sidebar controls
st.sidebar.header("Controls & Filters")
year = st.sidebar.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))
primary_indicator = st.sidebar.selectbox("Primary indicator (for many charts)",
                                         options=[c for c in df.select_dtypes(include=[np.number]).columns if c not in ['year']],
                                         index=0)
regions = st.sidebar.multiselect("Regions", options=sorted(df['region'].dropna().unique()), default=sorted(df['region'].dropna().unique()))
income_groups = st.sidebar.multiselect("Income groups", options=sorted(df['income_group'].dropna().unique()), default=sorted(df['income_group'].dropna().unique()))
top_n = st.sidebar.slider("Top N countries (rankings, networks)", 5, 50, 20)

mask = (df['year']==year) & (df['region'].isin(regions)) & (df['income_group'].isin(income_groups))
df_year = df[mask].copy()
st.sidebar.write("Rows after filtering:", df_year.shape[0])

# KPIs
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Selected Year", year)
with c2:
    if primary_indicator in df_year.columns and not df_year[primary_indicator].isna().all():
        st.metric(primary_indicator, f"{df_year[primary_indicator].median():.2f}")
    else:
        st.metric(primary_indicator, "n/a")
with c3:
    st.metric("Countries (filtered)", df_year['country_name'].nunique())

# 1) Time series
st.header("1) Time Series — GDP per Capita examples")
countries_default = df['country_name'].value_counts().index[:4].tolist()
sel_countries = st.multiselect("Select countries for time series", options=sorted(df['country_name'].unique()), default=countries_default)
if sel_countries:
    ts = df[df['country_name'].isin(sel_countries)].sort_values(['country_name','year'])
    fig = px.line(ts, x='year', y='gdp_per_capita', color='country_name', markers=True, title="GDP per Capita (2000–2020)")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# 2) Choropleth map
st.header("2) World Map — Primary Indicator")
if 'country_code' in df.columns:
    map_df = df[df['year']==year]
    if primary_indicator in map_df.columns:
        fig = px.choropleth(map_df, locations='country_code', color=primary_indicator, hover_name='country_name', title=f"{primary_indicator} — {year}", color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Primary indicator not available for map.")
else:
    st.info("No country_code column — choropleth unavailable.")

# 3) Treemap
st.header("3) Treemap — Regional / Income composition")
if 'gdp_usd' in df_year.columns:
    tr = df_year.dropna(subset=['gdp_usd'])
    fig = px.treemap(tr, path=['region','income_group','country_name'], values='gdp_usd', color=primary_indicator, title=f"Treemap of GDP (by region & income) — {year}", hover_data=[primary_indicator])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("gdp_usd not present for treemap.")

# 4) Scatter GDP per capita vs CO2 per capita (bubble)
st.header("4) Scatter — GDP per Capita vs CO2 per Capita")
if all(c in df.columns for c in ['gdp_per_capita','co2_emissions_per_capita_tons','population']):
    scdf = df[df['year']==year].dropna(subset=['gdp_per_capita','co2_emissions_per_capita_tons','population'])
    fig = px.scatter(scdf, x='gdp_per_capita', y='co2_emissions_per_capita_tons', size='population', hover_name='country_name', log_x=True, title=f"GDP per Capita vs CO2 per Capita — {year}", labels={'gdp_per_capita':'GDP per capita','co2_emissions_per_capita_tons':'CO2 t per capita'})
    x = np.log1p(scdf['gdp_per_capita']); y = scdf['co2_emissions_per_capita_tons']
    if len(x)>1:
        m,b = np.polyfit(x,y,1)
        xp = np.linspace(scdf['gdp_per_capita'].min(), scdf['gdp_per_capita'].max(), 200)
        xp_log = np.log1p(xp)
        yp = m*xp_log + b
        fig.add_trace(go.Scatter(x=xp, y=yp, mode='lines', name='Trend (log fit)'))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Required columns for scatter not available.")

# 5) Correlation heatmap
st.header("5) Correlation Heatmap — Selected indicators")
default_inds = ['gdp_per_capita','life_expectancy','internet_usage_pct','human_development_index']
available_inds = [c for c in default_inds if c in df.columns]
chosen_inds = st.multiselect("Choose indicators for correlation", options=available_inds, default=available_inds)
if len(chosen_inds) >= 2:
    corr = df[chosen_inds].corr()
    heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis', zmin=-1, zmax=1))
    annotations = []
    for i,row in enumerate(corr.values):
        for j,val in enumerate(row):
            annotations.append(dict(x=corr.columns[j], y=corr.index[i], text=f"{val:.2f}", showarrow=False, font=dict(size=10)))
    heat.update_layout(title="Correlation matrix", annotations=annotations, height=600)
    st.plotly_chart(heat, use_container_width=True)

# 6) Parallel coordinates
st.header("6) Parallel Coordinates — Multi-indicator comparison")
pc_inds = st.multiselect("Indicators (parallel coords)", options=[c for c in df.select_dtypes(include=[np.number]).columns if c!='year'], default=available_inds[:3], max_selections=6)
if len(pc_inds) >= 2:
    pc_df = df[df['year']==year].dropna(subset=pc_inds+['country_name'])
    pc_norm = pc_df.copy()
    for c in pc_inds:
        rng = pc_norm[c].max() - pc_norm[c].min()
        pc_norm[c] = (pc_norm[c] - pc_norm[c].min()) / (rng if rng!=0 else 1)
    fig = px.parallel_coordinates(pc_norm, color=pc_norm[pc_inds[0]], labels={c:c for c in pc_inds}, title="Parallel Coordinates")
    st.plotly_chart(fig, use_container_width=True)

# 7) Bubble chart renewable vs CO2
st.header("7) Bubble Chart — Renewable % vs CO2 per Capita")
if all(c in df.columns for c in ['renewable_energy_pct','co2_emissions_per_capita_tons','gdp_per_capita']):
    bdf = df[df['year']==year].dropna(subset=['renewable_energy_pct','co2_emissions_per_capita_tons','gdp_per_capita'])
    fig = px.scatter(bdf, x='renewable_energy_pct', y='co2_emissions_per_capita_tons', size='gdp_per_capita', hover_name='country_name', title=f"Renewable % vs CO2 per Capita — {year}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Required columns for bubble chart not available.")

# 8) Boxplot life expectancy by income group
st.header("8) Boxplot — Life Expectancy by Income Group")
if 'life_expectancy' in df.columns and 'income_group' in df.columns:
    bp = df[df['year']==year].dropna(subset=['life_expectancy','income_group'])
    fig = px.box(bp, x='income_group', y='life_expectancy', points="all", title=f"Life expectancy by income group — {year}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Required columns for boxplot not available.")

# 9) Network graph — country similarity by GDP per capita trajectories
st.header("9) Network — Country similarity (GDP per capita time series)")
if 'gdp_per_capita' in df.columns:
    net_n = min(top_n, 20)
    topc = df[df['year']==year].sort_values('gdp_usd', ascending=False)['country_name'].unique()[:net_n]
    ts = df.pivot_table(index='year', columns='country_name', values='gdp_per_capita')
    corr = ts.corr().fillna(0)
    sub = corr.loc[[c for c in topc if c in corr.index], [c for c in topc if c in corr.columns]]
    G = nx.Graph()
    thresh = st.sidebar.slider("Network correlation threshold", 0.6, 0.95, 0.75)
    for n in sub.index:
        G.add_node(n)
    for i in sub.index:
        for j in sub.columns:
            if i!=j and sub.loc[i,j] >= thresh:
                G.add_edge(i,j, weight=sub.loc[i,j])
    pos = nx.spring_layout(G, seed=42)
    edge_x=[]; edge_y=[]
    for e in G.edges():
        x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    node_x=[pos[n][0] for n in G.nodes()]; node_y=[pos[n][1] for n in G.nodes()]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1,color='#888'), hoverinfo='none')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition="bottom center", marker=dict(size=10), hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=f"Country similarity network (threshold={thresh})"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("gdp_per_capita not present — network skipped.")

st.markdown("---")
st.write("Project deliverables: 9 interactive visualizations, README, and one-page report. Use the sidebar to explore and export visuals.")