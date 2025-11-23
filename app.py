import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Global Development Indicators Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        # Load data
        df = pd.read_csv('Global_Development_Indicators_2000_2020.csv')
        
        df = df.dropna(subset=['region'])
        
      
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
      
        for col in numerical_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna('Unknown')


        df_clean_backup = df.copy()
        
        def remove_outliers_iqr(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

        key_columns = ['gdp_per_capita', 'life_expectancy', 'co2_emissions_per_capita_tons']
        
        for col in key_columns:
            if col in df.columns:
                temp_df = remove_outliers_iqr(df, col)
                
                if len(temp_df) >= len(df_clean_backup) * 0.8:
                    df = temp_df

    
        if 'year' in df.columns:
            df['year'] = df['year'].astype(int)
            
        
        if 'gdp_usd' in df.columns and 'population' in df.columns:
            df['gdp_per_capita_calc'] = df['gdp_usd'] / df['population']
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


df = load_and_preprocess_data()

if df is not None:

    st.sidebar.header("Dashboard Filters")
    

    year_min = int(df['year'].min())
    year_max = int(df['year'].max())
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1
    )
    
    
    regions = ['All'] + sorted(df['region'].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    income_groups = ['All'] + sorted(df['income_group'].dropna().unique().tolist())
    selected_income = st.sidebar.selectbox("Select Income Group", income_groups)

    filtered_df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    if selected_income != 'All':
        filtered_df = filtered_df[filtered_df['income_group'] == selected_income]
    
    
    st.markdown('<h1 class="main-header"> Global Development Indicators Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")
    with col2:
        st.metric("Unique Countries", filtered_df['country_name'].nunique())
    with col3:
        avg_gdp = filtered_df['gdp_per_capita'].mean() if 'gdp_per_capita' in filtered_df.columns else 0
        st.metric("Avg GDP per Capita", f"${avg_gdp:,.0f}")
    with col4:
        avg_life = filtered_df['life_expectancy'].mean() if 'life_expectancy' in filtered_df.columns else 0
        st.metric("Avg Life Expectancy", f"{avg_life:.1f} years")
    
    st.markdown("---")
    
    # Visualization 1: Treemap - GDP by Region and Country
    st.header("Visualization 1: GDP Distribution Treemap")
    st.markdown("**Treemap** - Interactive hierarchical view of GDP distribution")
    if 'gdp_usd' in filtered_df.columns and 'region' in filtered_df.columns:
        
        treemap_data = filtered_df.groupby(['region', 'country_name'])['gdp_usd'].sum().reset_index()
        treemap_data = treemap_data[treemap_data['gdp_usd'] > 0]
        
        fig_treemap = px.treemap(
            treemap_data,
            path=['region', 'country_name'],
            values='gdp_usd',
            color='gdp_usd',
            color_continuous_scale='Viridis',
            title='GDP Distribution by Region and Country (Hover for details, click to zoom)',
            hover_data=['gdp_usd']
        )
        fig_treemap.update_traces(
            hovertemplate='<b>%{label}</b><br>GDP: $%{value:,.0f}<extra></extra>',
            textinfo='label+value'
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization 2: Parallel Coordinates 
    st.header("Visualization 2: Parallel Coordinates Plot")
    st.markdown("**Parallel Coordinates** - Multi-dimensional analysis of development indicators")
    
    if len(filtered_df) > 0:
    
        parallel_cols = ['gdp_per_capita', 'life_expectancy', 'co2_emissions_per_capita_tons', 
                        'internet_usage_pct', 'renewable_energy_pct', 'child_mortality']
        available_cols = [col for col in parallel_cols if col in filtered_df.columns]
        
        if len(available_cols) >= 3:
        
            sample_df = filtered_df[available_cols + ['region']].dropna()
            if len(sample_df) > 500:
                sample_df = sample_df.sample(500)
            
            fig_parallel = px.parallel_coordinates(
                sample_df,
                dimensions=available_cols[:6],
                color='gdp_per_capita' if 'gdp_per_capita' in available_cols else available_cols[0],
                color_continuous_scale=px.colors.sequential.Viridis,
                title='Multi-dimensional Development Indicators (Drag axes to filter, hover for values)',
                labels={col: col.replace('_', ' ').title() for col in available_cols}
            )
            st.plotly_chart(fig_parallel, use_container_width=True)
    
    st.markdown("---")

    # Visualization 3: Top Countries Ranking (Bar Chart - Basic)
    st.header("Visualization 3: Top Countries Ranking")
    st.markdown("**Bar Chart** - Ranking countries by key metrics")
    
    
    col1, col2 = st.columns(2)
    with col1:
        
        rank_metric = st.selectbox(
            "Select Metric for Ranking",
            ['gdp_usd', 'population', 'co2_emissions_kt', 'life_expectancy'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    with col2:
        
        top_n = st.slider("Number of Countries to Show", 5, 50, 15)

    if rank_metric in filtered_df.columns:
    
        ranking_data = filtered_df.groupby('country_name')[rank_metric].mean().reset_index()
        ranking_data = ranking_data.sort_values(by=rank_metric, ascending=False).head(top_n)
        
        fig_bar = px.bar(
            ranking_data,
            x=rank_metric,
            y='country_name',
            orientation='h', 
            color=rank_metric,
            title=f"Top {top_n} Countries by {rank_metric.replace('_', ' ').title()}",
            labels={
                rank_metric: rank_metric.replace('_', ' ').title(),
                'country_name': 'Country'
            },
            color_continuous_scale='Viridis',
            text_auto='.2s' 
        )
        
        
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization 4: Sankey Diagram - Flow of Economic Indicators
    st.header("Visualization 4: Economic Flow Sankey Diagram")
    st.markdown("**Sankey Diagram** - Flow visualization of economic relationships")
    
    if 'region' in filtered_df.columns and 'income_group' in filtered_df.columns:
        
        flow_data = filtered_df.groupby(['region', 'income_group']).size().reset_index(name='count')
        flow_data = flow_data[flow_data['count'] > 0]
        

        regions_list = flow_data['region'].unique().tolist()
        income_list = flow_data['income_group'].unique().tolist()
        
    
        all_labels = regions_list + income_list
        label_dict = {label: i for i, label in enumerate(all_labels)}
        
        
        source = [label_dict[row['region']] for _, row in flow_data.iterrows()]
        target = [label_dict[row['income_group']] for _, row in flow_data.iterrows()]
        value = flow_data['count'].tolist()
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color="lightblue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>'
            )
        )])
        
        fig_sankey.update_layout(
            title="Economic Flow: Region to Income Group Distribution (Hover on flows for details)",
            font_size=12,
            height=500
        )
        st.plotly_chart(fig_sankey, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization 5: Interactive Heatmap
    st.header("Visualization 5: Correlation Heatmap")
    st.markdown("**Heatmap** - Correlation matrix of key development indicators")
    
    heatmap_cols = ['gdp_per_capita', 'life_expectancy', 'co2_emissions_per_capita_tons',
                    'internet_usage_pct', 'renewable_energy_pct', 'child_mortality',
                    'inflation_rate', 'unemployment_rate']
    available_heatmap_cols = [col for col in heatmap_cols if col in filtered_df.columns]
    
    if len(available_heatmap_cols) >= 3:
        corr_data = filtered_df[available_heatmap_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_data,
            labels=dict(x="Indicator", y="Indicator", color="Correlation"),
            x=corr_data.columns,
            y=corr_data.columns,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="Correlation Matrix of Development Indicators (Hover for correlation values, click to zoom)"
        )
        fig_heatmap.update_xaxes(side="bottom")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
   
    
    # Visualization 6: Interactive Time Series (Normalized)
    st.header("Visualization 6: Time Series Analysis")
    st.markdown("**Interactive Line Chart** - Temporal trends of key indicators")
    

    col1, col2, col3 = st.columns(3)
    with col1:
        show_gdp = st.checkbox("GDP per Capita", value=True)
        show_life = st.checkbox("Life Expectancy", value=True)
    with col2:
        show_co2 = st.checkbox("CO2 Emissions", value=False)
        show_internet = st.checkbox("Internet Usage", value=False)
    with col3:
        show_renewable = st.checkbox("Renewable Energy", value=False)
        show_mortality = st.checkbox("Child Mortality", value=False)
    
    st.markdown("---")
    use_normalization = st.checkbox("Normalize Data (Scale to 0-100%) - Best for comparing trends", value=False)
    

    metrics_to_plot = []
    if show_gdp and 'gdp_per_capita' in filtered_df.columns:
        metrics_to_plot.append('gdp_per_capita')
    if show_life and 'life_expectancy' in filtered_df.columns:
        metrics_to_plot.append('life_expectancy')
    if show_co2 and 'co2_emissions_per_capita_tons' in filtered_df.columns:
        metrics_to_plot.append('co2_emissions_per_capita_tons')
    if show_internet and 'internet_usage_pct' in filtered_df.columns:
        metrics_to_plot.append('internet_usage_pct')
    if show_renewable and 'renewable_energy_pct' in filtered_df.columns:
        metrics_to_plot.append('renewable_energy_pct')
    if show_mortality and 'child_mortality' in filtered_df.columns:
        metrics_to_plot.append('child_mortality')
    
    if len(metrics_to_plot) > 0 and 'year' in filtered_df.columns:
        
        time_series_data = filtered_df.groupby('year')[metrics_to_plot].mean().reset_index()
        
        fig_line = go.Figure()
        
        for metric in metrics_to_plot:
        
            y_values = time_series_data[metric]
            original_values = y_values.copy() 
            
            
            if use_normalization:
                min_val = y_values.min()
                max_val = y_values.max()
                
                if max_val - min_val != 0:
                    
                    y_values = (y_values - min_val) / (max_val - min_val) * 100
            
            fig_line.add_trace(go.Scatter(
                x=time_series_data['year'],
                y=y_values,
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                
                customdata=original_values,
                
                hovertemplate='Year: %{x}<br>' + 
                              ('Normalized Score: %{y:.1f}<br>' if use_normalization else '') +
                              'Actual Value: %{customdata:.2f}<extra></extra>',
                line=dict(width=3)
            ))
        

        y_title = "Normalized Score (0-100)" if use_normalization else "Average Value"
        
        fig_line.update_layout(
            title="Time Series Trends (Toggle 'Normalize' to compare shapes)",
            xaxis_title="Year",
            yaxis_title=y_title,
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization 7: Box Plot 
    st.header("Visualization 7: Distribution Analysis - Box Plot")
    st.markdown("**Box Plot** - Statistical distribution of indicators by region")
    
    if 'region' in filtered_df.columns:
        box_metric = st.selectbox(
            "Select Metric for Box Plot",
            ['gdp_per_capita', 'life_expectancy', 'co2_emissions_per_capita_tons', 
             'internet_usage_pct', 'renewable_energy_pct'],
            key='box_metric'
        )
        
        if box_metric in filtered_df.columns:
            box_data = filtered_df[[box_metric, 'region']].dropna()
            
            fig_box = px.box(
                box_data,
                x='region',
                y=box_metric,
                color='region',
                title=f"Distribution of {box_metric.replace('_', ' ').title()} by Region (Hover for quartile values, click legend to filter)",
                labels={box_metric: box_metric.replace('_', ' ').title()}
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization 8: Geographic Map - If we have country data
    st.header("Visualization 8: Geographic Distribution Map")
    st.markdown("**Geographic Map** - Spatial distribution of development indicators")
    
    
    if 'country_code' in filtered_df.columns and 'gdp_per_capita' in filtered_df.columns:
        map_data = filtered_df.groupby('country_code').agg({
            'gdp_per_capita': 'mean',
            'life_expectancy': 'mean',
            'country_name': 'first'
        }).reset_index()
        
        
        try:
            fig_map = px.choropleth(
                map_data,
                locations='country_code',
                color='gdp_per_capita',
                hover_name='country_name',
                hover_data=['life_expectancy'],
                color_continuous_scale='Viridis',
                title="Geographic Distribution of GDP per Capita (Hover for country details, zoom to explore)",
                labels={'gdp_per_capita': 'GDP per Capita (USD)'}
            )
            fig_map.update_geos(projection_type="natural earth")
            st.plotly_chart(fig_map, use_container_width=True)
        except:
            
            st.info("Choropleth map not available. Using alternative visualization.")
            fig_scatter = px.scatter(
                map_data,
                x='gdp_per_capita',
                y='life_expectancy',
                size='gdp_per_capita',
                color='gdp_per_capita',
                hover_name='country_name',
                color_continuous_scale='Viridis',
                title="GDP per Capita vs Life Expectancy (Hover for details, drag to select)",
                labels={'gdp_per_capita': 'GDP per Capita', 'life_expectancy': 'Life Expectancy'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization 9: Sunburst Chart
    st.header("Visualization 9: Hierarchical Sunburst Chart")
    st.markdown("**Sunburst Chart** - Hierarchical view of development indicators")
    
    if 'region' in filtered_df.columns and 'income_group' in filtered_df.columns:
        sunburst_data = filtered_df.groupby(['region', 'income_group']).agg({
            'gdp_per_capita': 'mean',
            'life_expectancy': 'mean'
        }).reset_index()
        sunburst_data = sunburst_data.dropna()
        
        if len(sunburst_data) > 0:
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=['region', 'income_group'],
                values='gdp_per_capita',
                color='life_expectancy',
                color_continuous_scale='RdYlGn',
                title="Hierarchical View: Region → Income Group (Click to drill down, hover for values)",
                hover_data=['life_expectancy']
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
    
    st.markdown("---")
    
    
    with st.expander("View Processed Data Summary"):
        st.subheader("Data Overview")
        st.write(f"**Total Records:** {len(filtered_df):,}")
        st.write(f"**Date Range:** {filtered_df['year'].min()} - {filtered_df['year'].max()}")
        st.write(f"**Number of Countries:** {filtered_df['country_name'].nunique()}")
        st.write(f"**Number of Regions:** {filtered_df['region'].nunique()}")
        
        st.subheader("Sample Data")
        st.dataframe(filtered_df.head(100), use_container_width=True)
        
        st.subheader("Data Statistics")
        st.dataframe(filtered_df.describe(), use_container_width=True)
    
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Global Development Indicators Dashboard | Data: 2000-2020</p>
        <p>Interactive visualizations with hover effects, zooming, filtering, and panning capabilities</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Failed to load data. Please ensure the CSV file is available.")
