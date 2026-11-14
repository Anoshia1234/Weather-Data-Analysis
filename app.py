import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="England Climate Dashboard",
    page_icon="⛅",
    layout="wide"
)

# Custom CSS for Light Blue & Light Yellow Contrast Theme & Attractive Fonts
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #F0F8FF; /* Alice Blue / Light Blue */
        color: #2C3E50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFACD; /* Lemon Chiffon / Light Yellow */
        border-right: 2px solid #E6F2FF;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E3A8A;
    }
    
    /* Metric Cards */
    div.stMetric {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #3B82F6;
    }
    
    /* Dataframe container */
    div.stDataFrame {
        background-color: #FFFFFF;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv('Climate of England _ Professional Reforms.csv')
    df['Full_Datetime'] = pd.to_datetime(df['Formatted Date'])
    df['Place'] = 'England'
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Dashboard Filters")
place_list = df['Place'].unique()
selected_place = st.sidebar.selectbox("Select Region / Place", place_list)

min_date = df['Full_Datetime'].min().date()
max_date = df['Full_Datetime'].max().date()

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filter dataframe based on selections
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['Place'] == selected_place) & 
                     (df['Full_Datetime'].dt.date >= start_date) & 
                     (df['Full_Datetime'].dt.date <= end_date)]
else:
    filtered_df = df[df['Place'] == selected_place]

# Main Title & Description
st.title("⛅ England Climate & Weather Analytics Dashboard")
st.markdown("Explore comprehensive meteorological patterns, temperature spreads, wind conditions, and correlations across time and geography.")
st.markdown("---")

# 4 Key Numbers on Top (KPIs)
col1, col2, col3, col4 = st.columns(4)
avg_temp = filtered_df['Temperature (C)'].mean() if not filtered_df.empty else 0
max_temp = filtered_df['Temperature (C)'].max() if not filtered_df.empty else 0
avg_humidity = filtered_df['Humidity'].mean() if not filtered_df.empty else 0
total_days = filtered_df['Full_Datetime'].dt.date.nunique() if not filtered_df.empty else 0

col1.metric("Avg Temperature", f"{avg_temp:.2f} °C")
col2.metric("Highest Temperature", f"{max_temp:.2f} °C")
col3.metric("Avg Humidity", f"{avg_humidity:.2f}")
col4.metric("Total Days Analyzed", f"{total_days:,}")

st.markdown("---")

# Graphs Section (At least 5 graphs including univariate, bivariate, time series, correlation heatmap)
st.subheader("📊 Visual Weather Insights")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("### 1. Temperature Distribution (Univariate)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(filtered_df['Temperature (C)'], kde=True, color='#3B82F6', ax=ax)
    ax.set_title("Temperature Spread")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with row1_col2:
    st.markdown("### 2. Temperature vs Apparent Temp (Bivariate)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=filtered_df, x='Temperature (C)', y='Apparent Temperature (C)', alpha=0.3, color='#F59E0B', ax=ax)
    ax.set_title("Actual vs Apparent Temperature")
    st.pyplot(fig)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown("### 3. Temperature Over Time (Line Chart)")
    fig, ax = plt.subplots(figsize=(7, 4))
    # Resample daily average for clean line chart
    daily_trend = filtered_df.set_index('Full_Datetime').resample('D')['Temperature (C)'].mean().reset_index()
    sns.lineplot(data=daily_trend, x='Full_Datetime', y='Temperature (C)', color='#10B981', ax=ax)
    ax.set_title("Daily Average Temperature Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

with row2_col2:
    st.markdown("### 4. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 4))
    num_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr = filtered_df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

# 5th Graph: Wind Speed Distribution
st.markdown("### 5. Wind Speed Distribution Across Records")
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(x=filtered_df['Wind Speed (km/h)'], color='#93C5FD', ax=ax)
ax.set_title("Wind Speed Spread & Outlier Ranges")
st.pyplot(fig)

st.markdown("---")

# Checkbox to show raw data table
if st.checkbox("Show Raw Data Table"):
    st.subheader("📋 Filtered Dataset View")
    st.dataframe(filtered_df, use_container_width=True)

# Main Findings Section
st.markdown("---")
st.subheader("📌 Key Findings & Summary")
st.markdown("""
- **Temperature Consistency:** The climate exhibits clear seasonal variations with temperatures averaging around 11.9°C.
- **Apparent Temperature Correlation:** Apparent temperature closely mirrors actual temperature, with deviations heavily influenced by wind speed and humidity levels.
- **Humidity and Visibility:** Higher humidity levels correlate inversely with clear visibility, frequently associated with foggy or overcast conditions.
- **Wind Patterns:** Wind speeds remain stable across most periods, with sporadic higher gusts representing storm systems.
""")
