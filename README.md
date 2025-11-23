[README.md](https://github.com/user-attachments/files/23696571/README.md)
# Global Development Indicators Dashboard

## Project Description

This project presents an interactive data visualization dashboard analyzing global development indicators from 2000 to 2020. The dashboard provides comprehensive insights into economic, social, environmental, and technological development metrics across different regions and countries worldwide.

## Dataset Details

- **Dataset Name:** Global Development Indicators 2000-2020
- **Source:** Provided CSV file (`Global_Development_Indicators_2000_2020.csv`)
- **Rows:** 5,558+ records
- **Columns:** 40+ columns including:
  - **Temporal:** year, years_since_2000
  - **Geographic:** country_code, country_name, region, income_group
  - **Economic:** gdp_usd, gdp_per_capita, inflation_rate, unemployment_rate, fdi_pct_gdp
  - **Environmental:** co2_emissions_kt, energy_use_per_capita, renewable_energy_pct, forest_area_pct
  - **Social:** life_expectancy, child_mortality, school_enrollment_secondary
  - **Health:** health_expenditure_pct_gdp, hospital_beds_per_1000, physicians_per_1000
  - **Technology:** internet_usage_pct, mobile_subscriptions_per_100
  - **Composite Indices:** human_development_index, climate_vulnerability_index, digital_readiness_score, etc.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone or download this repository**
   ```bash
   cd denemeproje
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset file is in the project directory**
   - The file `Global_Development_Indicators_2000_2020.csv` should be in the same directory as `app.py`

4. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   - The application will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

## Dashboard Features

### Data Preprocessing
- Automatic handling of missing values (median imputation for numerical, 'Unknown' for categorical)
- Outlier detection and removal using IQR method
- Data type conversion and validation
- Derived metric calculations

### Visualizations (9 Total)

#### Advanced Visualizations (6):
1. **Treemap** - Hierarchical GDP distribution by region and country
   - Interactive: Hover for details, click to zoom, color highlighting

2. **Parallel Coordinates** - Multi-dimensional analysis of development indicators
   - Interactive: Drag axes to filter, hover for values, color coding

3. **Sankey Diagram** - Flow visualization of economic relationships (Region → Income Group)
   - Interactive: Hover on flows for details, click to explore

4. **Correlation Heatmap** - Correlation matrix of key indicators
   - Interactive: Hover for correlation values, click to zoom, color scale

5. **3D Scatter Plot** - Three-dimensional analysis (GDP, Life Expectancy, CO2)
   - Interactive: Drag to rotate, zoom, pan, hover for details

6. **Geographic Map (Choropleth)** - Spatial distribution of development indicators
   - Interactive: Hover for country details, zoom to explore regions

#### Standard Visualizations (3):
7. **Time Series Line Chart** - Temporal trends with multiple metrics
   - Interactive: Hover for values, double-click legend to isolate, drag to zoom, checkbox filtering

8. **Box Plot** - Statistical distribution by region
   - Interactive: Hover for quartile values, click legend to filter, dropdown selection

9. **Sunburst Chart** - Hierarchical view (Region → Income Group)
   - Interactive: Click to drill down, hover for values, color coding

### Interactive Components

1. **Year Range Slider** - Filter data by time period
2. **Region Dropdown** - Select specific regions
3. **Income Group Dropdown** - Filter by income classification
4. **Metric Checkboxes** - Toggle visibility of time series metrics
5. **Metric Dropdown** - Select metric for box plot analysis

### Interactivity Features

Each visualization includes:
- **Mouse hover effects** - Detailed tooltips with values
- **Color highlighting** - Visual emphasis on selected elements
- **Zooming** - Click and drag or use mouse wheel
- **Panning** - Click and drag to move view
- **Filtering** - Interactive legends and controls
- **Tooltips** - Rich hover information
- **Selection** - Click to highlight and filter

## Key Insights

The dashboard enables analysis of:
- Economic development patterns across regions and time
- Relationships between economic, social, and environmental indicators
- Regional disparities in development metrics
- Temporal trends in global development
- Correlation patterns between different indicators
- Hierarchical structures in development data

## Technical Stack

- **Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly (Express and Graph Objects)
- **Language:** Python 3.8+

## Project Structure

```
denemeproje/
├── app.py                                    # Main Streamlit application
├── Global_Development_Indicators_2000_2020.csv  # Dataset file
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

## Team Member Contributions

This project was developed as a comprehensive dashboard solution. Key contributions include:

- **Data Preprocessing:** Implementation of cleaning, missing value handling, and outlier detection
- **Visualization Design:** Creation of 9 distinct visualizations with advanced types
- **Interactivity Implementation:** Integration of hover effects, filtering, zooming, and panning
- **Dashboard Layout:** Organized, clean, and user-friendly interface design
- **Documentation:** Comprehensive README and code documentation

## Usage Tips

1. **Start with filters:** Use the sidebar filters to narrow down your analysis
2. **Explore interactively:** Hover over visualizations to see detailed information
3. **Use zoom and pan:** Click and drag to explore specific areas of charts
4. **Toggle metrics:** Use checkboxes to show/hide different time series
5. **Drill down:** Click on hierarchical visualizations (Treemap, Sunburst) to explore deeper
6. **Compare regions:** Use the region filter to compare different areas

## Future Enhancements

Potential improvements:
- Export functionality for visualizations
- Additional advanced visualization types
- Real-time data updates
- Custom metric calculations

## License

This project is for educational purposes.

## Contact

For questions or issues, please refer to https://github.com/dogakarzann/global-development-analysis/tree/main/data

---

**Note:** This dashboard requires the dataset file to be present in the project directory. Ensure `Global_Development_Indicators_2000_2020.csv` is available before running the application.

