import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib

# Load sample data from CSV
@st.cache_data
def load_data():
    df = pd.read_csv("model_monitoring_sample_data.csv", parse_dates=["Uploaded_On"])
    df["Anomaly_Flag"] = df["Anomaly_Flag"].astype(str).str.upper().eq("TRUE")
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filter Reports")

partners = st.sidebar.multiselect("Select Partner(s)", options=df["Partner_Name"].unique())
models = st.sidebar.multiselect("Select Model(s)", options=df["Model_Name"].unique())
tiers = st.sidebar.multiselect("Select Tier(s)", options=df["Tier"].unique())
kpis = st.sidebar.multiselect("Select KPI(s)", options=df["KPI"].unique())

# Apply filters only if user selects something; otherwise, show all
filtered_df = df.copy()

if partners:
    filtered_df = filtered_df[filtered_df["Partner_Name"].isin(partners)]
if models:
    filtered_df = filtered_df[filtered_df["Model_Name"].isin(models)]
if tiers:
    filtered_df = filtered_df[filtered_df["Tier"].isin(tiers)]
if kpis:
    filtered_df = filtered_df[filtered_df["KPI"].isin(kpis)]

# Small CSS Tweaks
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-size: 16px;
        }
        .main-title {
            font-size: 20px;
            font-weight: 600;
            margin-top: -30px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Force override Streamlit's content width restriction */
        .st-emotion-cache-1w723zb {
            max-width: 90% !important;
            width: 90% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding : 1rem 1rem 10rem;
        }
    </style>
""", unsafe_allow_html=True)


# 🔰 Title
st.markdown('<div class="main-title">Model Report Monitoring Dashboard</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 KPI Trends", "🚨 Anomaly Details", "📄 Raw Data", "📑 Model Reports"])

with tab1:
    # Summary KPI Cards
    total_partners = filtered_df["Partner_Name"].nunique()
    total_models = filtered_df["Model_ID"].drop_duplicates().shape[0]
    unique_metrics = filtered_df["KPI"].nunique()
    total_alerts = filtered_df["Anomaly_Flag"].sum()
    most_common_metric = filtered_df["KPI"].mode()[0] if not filtered_df["KPI"].mode().empty else "N/A"
    models_by_tier = filtered_df.groupby("Tier")["Model_ID"].nunique().to_dict()
    report_frequencies = (
        filtered_df
        .drop_duplicates(subset=["Partner_Name", "Model_ID", "Year", "Period"])
        .groupby("Frequency")
        .size()
        .to_dict()
    )
    latest_upload = filtered_df["Uploaded_On"].max()
    total_reports = filtered_df.drop_duplicates(subset=["Partner_Name", "Model_ID", "Year", "Period"]).shape[0]

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label="Total Partners", value=total_partners)
    col2.metric(label="Total Models", value=total_models)
    col3.metric(label="Unique KPIs", value=unique_metrics)
    col4.metric(label="Anomaly Flags", value=int(total_alerts))
    col5.metric(label="Total Reports", value=total_reports)

    # Inline Breakdown
    col5, col6 = st.columns(2)
    with col5:
        st.write("**Most Common Metric:**", most_common_metric)
        tier_summary = ", ".join([f"{tier}: {count}" for tier, count in models_by_tier.items()])
        st.write("**Models by Tier:**", tier_summary)

    with col6:
        st.write("**Latest Upload:**", latest_upload.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(latest_upload) else "N/A")
        freq_summary = ", ".join([f"{freq}: {count}" for freq, count in report_frequencies.items()])
        st.write("**Report Frequencies:**", freq_summary)

    # Executive Summary Section
    with st.expander("📋 Executive Summary", expanded=False):

        #tier_summary = ", ".join([f"{tier}: {count}" for tier, count in models_by_tier.items()])
        #freq_summary = ", ".join([f"{freq}: {count}" for freq, count in report_frequencies.items()])

        st.markdown(f"""
        We analyzed a total of **{total_reports} model monitoring reports** across various partners. These reports covered **{unique_metrics} unique KPIs**, including the most frequently monitored metric, **{most_common_metric}**.

        A total of **{total_models} models** have been processed so far, distributed across:
        - {tier_summary}

        Across all reports, we identified **{int(total_alerts)} anomaly flags**, indicating potential risks or performance drift in select models.

        Report frequency distribution shows:
        - {freq_summary}

        The **latest report upload** occurred on **{latest_upload.strftime("%Y-%m-%d")}**, indicating active and ongoing model monitoring.

        These insights suggest a consistent reporting cadence with emerging anomalies that may require deeper investigation, especially for high-risk (Tier 1) models.
        """)    

with tab2:
    # Place line chart, filters, insights about trends
    # KPI Trends Chart
    #st.markdown('<h5 style="margin-bottom:10px;">📈 KPI Trends Over Time</h5>', unsafe_allow_html=True)
    kpi_plot_df = filtered_df.copy()
    kpi_plot_df["Period_Label"] = kpi_plot_df["Year"].astype(str) + " " + kpi_plot_df["Period"]

    fig = px.line(
        kpi_plot_df.sort_values("Uploaded_On"),
        x="Period_Label",
        y="Value",
        color="KPI",
        markers=True,
        line_group="Model_Name",
        hover_data=["Partner_Name", "Tier", "Summary"]
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 KPI Trend Summary (Click to Expand)"):
        num_kpis = filtered_df["KPI"].nunique()
        num_models = filtered_df[["Partner_Name", "Model_ID"]].drop_duplicates().shape[0]
        periods = filtered_df["Period"].nunique()
        latest_upload = filtered_df["Uploaded_On"].max()
        anomalies = filtered_df["Anomaly_Flag"].astype(str).str.upper().eq("TRUE").sum()

        top_kpi = (
            filtered_df["KPI"].value_counts().idxmax()
            if not filtered_df["KPI"].empty
            else "N/A"
        )

        tier_volatility = (
            filtered_df.groupby("Tier")["Value"].std().sort_values(ascending=False).index[0]
            if not filtered_df.empty
            else "N/A"
        )

        st.markdown(f"""
        - The current view includes **{num_kpis} KPIs** across **{num_models} models**.
        - The trend spans **{periods} periods**, with the latest report uploaded on **{latest_upload.date()}**.
        - **{top_kpi}** is the most frequently reported metric.
        - A total of **{anomalies} anomalies** were flagged in the filtered data.
        - Models tagged as **{tier_volatility}** exhibit the highest KPI variability.
        """)


with tab3:
    # Show anomaly records, trends in anomaly counts etc.
    # Anomaly Records Table
    anomaly_df = filtered_df[filtered_df["Anomaly_Flag"] == True]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Anomalies", len(anomaly_df))

    with col2:
        st.metric("Affected Models", anomaly_df['Model_ID'].nunique())

    with col3:
        st.metric("Affected KPIs", anomaly_df['KPI'].nunique())

    # anomaly_trend = (
    #     anomaly_df.groupby("Uploaded_On")
    #     .size()
    #     .reset_index(name="Anomaly Count")
    # )
    # fig = px.bar(anomaly_trend, x="Uploaded_On", y="Anomaly Count", title="🕒 Anomalies Over Time")
    # st.plotly_chart(fig, use_container_width=True)

    heatmap_df = (
        anomaly_df.groupby(["Model_Name", "KPI"]).size().unstack(fill_value=0)
    )
    st.dataframe(heatmap_df)#.style.background_gradient(cmap="Reds"))

    with st.expander("📌 Summary / Recommendations"):
        st.markdown(f"""
        - **{anomaly_df['Model_ID'].nunique()} models** had at least one anomaly.
        - **{anomaly_df['KPI'].nunique()} KPIs** were involved in flagged reports.
        - The most recent anomaly was recorded on **{anomaly_df['Uploaded_On'].max().date()}**.
        - Consider deep-diving into models with recurring anomalies.
        """)
 
    st.markdown('<h5 style="margin-bottom:10px;">Anomaly Records</h5>', unsafe_allow_html=True)

    if not anomaly_df.empty:
        st.dataframe(anomaly_df.sort_values("Uploaded_On", ascending=False), use_container_width=True)
    else:
        st.info("No anomalies found in selected filters.")
    
with tab4:
    # Raw Data Table (Expandable)
    #with st.expander("🧾 View Full Data Table"):
    st.dataframe(filtered_df.sort_values("Uploaded_On", ascending=False), use_container_width=True)

with tab5:
    unique_models = filtered_df["Model_Name"].dropna().unique().tolist()
    tabs = st.tabs(unique_models)

    for tab, model in zip(tabs, unique_models):
        with tab:
            model_df = filtered_df[filtered_df["Model_Name"] == model]
            summary = model_df["Report_Summary"].iloc[0]
            st.markdown(f"**Report Summary:** {summary}")
            # Display charts, metrics, etc. here

