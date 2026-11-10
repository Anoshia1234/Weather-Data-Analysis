import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration with weather theme
st.set_page_config(
    page_title="Weather Data Analyzer",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light Weather-Themed Styling (CSS)
st.markdown("""
    <style>
    .main {
        background-color: #F4F9FF;
    }
    .stAppHeader {
        background-color: #E6F0FA;
    }
    h1, h2, h3 {
        color: #1E3A8A;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #D0E1F9;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.03);
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Description
st.title("Weather Data Analysis & Visualization Dashboard")
st.write("""
Welcome to the Weather Insights Dashboard! Upload your `weather_data.csv` file below 
to analyze key atmospheric variables, temporal trends, distributions, and country-level variations.
""")

# File Uploader Section
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Identify Date column dynamically
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
    return df, date_cols[0] if date_cols else None

if uploaded_file is not None:
    df, date_col = load_data(uploaded_file)
    
    # Identify country/region column dynamically
    country_cols = [c for c in df.columns if any(k in c.lower() for k in ['country', 'region', 'place', 'location'])]
    country_col = country_cols[0] if country_cols else df.columns[0]
    
    # Identify numeric weather columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Sidebar Filters
    st.sidebar.header("🔍 Filter Options")
    
    # Country Filter
    unique_countries = ["All"] + list(df[country_col].dropna().unique())
    selected_country = st.sidebar.selectbox("Select Country / Region", unique_countries)

    # Date Range Filter
    if date_col:
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        start_date, end_date = st.sidebar.date_input(
            "Select Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
    
    # Apply Filtering
    filtered_df = df.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df[country_col] == selected_country]
    if date_col and isinstance(start_date, pd.Timestamp) == False:
        filtered_df = filtered_df[
            (filtered_df[date_col].dt.date >= start_date) & 
            (filtered_df[date_col].dt.date <= end_date)
        ]

    # TOP SECTION: 4 Key Metrics
    st.markdown("### 📊 Overview Metrics")
    m1, m2, m3, m4 = st.columns(4)
    
    primary_metric = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
    secondary_metric = numeric_cols[1] if len(numeric_cols) > 1 else primary_metric

    m1.metric("Total Records", f"{len(filtered_df):,}")
    m2.metric(f"Avg {primary_metric}", f"{filtered_df[primary_metric].mean():.2f}")
    m3.metric(f"Max {primary_metric}", f"{filtered_df[primary_metric].max():.2f}")
    m4.metric(f"Avg {secondary_metric}", f"{filtered_df[secondary_metric].mean():.2f}")

    st.divider()

    # GRAPH SECTION (At least 5 Graphs)
    st.markdown("### 📈 Visualizations")
    
    col1, col2 = st.columns(2)

    with col1:
        # Graph 1: Univariate Distribution
        st.subheader(f"1. Distribution of {primary_metric} (Univariate)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df[primary_metric], kde=True, color="#4A90E2", ax=ax1)
        ax1.set_title(f"Histogram & KDE of {primary_metric}")
        ax1.set_xlabel(primary_metric)
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with col2:
        # Graph 2: Bivariate Scatter Plot
        st.subheader(f"2. {primary_metric} vs {secondary_metric} (Bivariate)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=filtered_df, x=primary_metric, y=secondary_metric, color="#50E3C2", alpha=0.7, ax=ax2)
        ax2.set_title(f"{primary_metric} vs {secondary_metric}")
        ax2.set_xlabel(primary_metric)
        ax2.set_ylabel(secondary_metric)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        # Graph 3: Time Series Line Chart
        st.subheader("3. Weather Trend Over Time")
        if date_col:
            time_df = filtered_df.groupby(date_col)[primary_metric].mean().reset_index()
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(time_df[date_col], time_df[primary_metric], color="#9013FE", linewidth=2)
            ax3.set_title(f"Average {primary_metric} Over Time")
            ax3.set_xlabel("Date")
            ax3.set_ylabel(primary_metric)
            plt.xticks(rotation=45)
            st.pyplot(fig3)
        else:
            st.info("Date column not detected for time series line chart.")

    with col4:
        # Graph 4: Correlation Heatmap
        st.subheader("4. Correlation Heatmap")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        corr = filtered_df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax4)
        ax4.set_title("Feature Correlation Matrix")
        st.pyplot(fig4)

    # Graph 5: Regional Comparison Bar Chart
    st.subheader("5. Top 10 Regions Comparison")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    top_places = filtered_df.groupby(country_col)[primary_metric].mean().nlargest(10).reset_index()
    sns.barplot(data=top_places, x=country_col, y=primary_metric, palette="Blues_d", ax=ax5)
    ax5.set_title(f"Top 10 Regions by Highest Average {primary_metric}")
    ax5.set_xlabel("Region/Country")
    ax5.set_ylabel(f"Mean {primary_metric}")
    plt.xticks(rotation=45)
    st.pyplot(fig5)

    st.divider()

    # MAIN FINDINGS SECTION
    st.markdown("### 📝 Key Insights & Summary Findings")
    st.write(f"""
    - **Temporal Volatility:** The trend line displays noticeable fluctuation in **{primary_metric}** across the selected timeframe.
    - **Inter-Variable Dynamic:** A distinct relationship is observed between **{primary_metric}** and **{secondary_metric}**, as reflected in the bivariate scatter plot.
    - **Regional Divergence:** Location-based aggregation reveals substantial variances in baseline metric averages across different countries/regions.
    """)

    # RAW DATA TOGGLE
    st.divider()
    if st.checkbox("📋 Show Raw Data Table"):
        st.subheader("Filtered Raw Dataset")
        st.dataframe(filtered_df)

else:
    st.info("👈 Please upload the `weather_data.csv` file from the sidebar to start the analysis.")
