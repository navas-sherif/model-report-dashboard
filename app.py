import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ---------- Configuration ----------
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("large_dataset.csv", parse_dates=["Uploaded_On"])
    df["Uploaded_On"] = pd.to_datetime(df["Uploaded_On"], errors='coerce')
    df["Anomaly_Flag"] = df["Anomaly_Flag"].astype(str).str.upper().eq("TRUE")
    return df

df = load_data()

# ---------- Utility Functions ----------
quarter_start = {"Q1": "-01-01", "Q2": "-04-01", "Q3": "-07-01", "Q4": "-10-01"}

def get_report_date(row):
    if row["Period"].startswith("Q"):
        return pd.to_datetime(f"{row['Year']}{quarter_start[row['Period'][:2]]}")
    elif row["Period"].startswith("Y"):
        return pd.to_datetime(f"{row['Year']}-01-01")
    else:
        return pd.NaT

# ---------- Preprocessing ----------
df["Report_Date"] = df.apply(get_report_date, axis=1)
min_report_date = df["Report_Date"].min().date()
max_report_date = df["Report_Date"].max().date()
default_start = max(min_report_date, max_report_date - timedelta(days=730))
default_end = max_report_date

# ---------- Sidebar Filters ----------
st.sidebar.header("🔎 Filter Reports")
report_range = st.sidebar.date_input(
    "Report Date Range", (default_start, default_end), min_value=min_report_date, max_value=max_report_date
)
if isinstance(report_range, tuple) and len(report_range) == 2:
    start_date, end_date = report_range
    df = df[(df["Report_Date"] >= pd.to_datetime(start_date)) & (df["Report_Date"] <= pd.to_datetime(end_date))]

filter_base_df = df.copy()

# Tier, Partner, Model, KPI Filters
tiers = st.sidebar.multiselect("Select Tier(s)", sorted(filter_base_df["Tier"].dropna().unique()), default=None)
if tiers:
    filter_base_df = filter_base_df[filter_base_df["Tier"].isin(tiers)]

partners = st.sidebar.multiselect("Select Partner(s)", sorted(filter_base_df["Partner_Name"].dropna().unique()), default=None)
if partners:
    filter_base_df = filter_base_df[filter_base_df["Partner_Name"].isin(partners)]

models = st.sidebar.multiselect("Select Model(s)", sorted(filter_base_df["Model_Name"].dropna().unique()), default=None)
if models:
    filter_base_df = filter_base_df[filter_base_df["Model_Name"].isin(models)]

anomaly_filter = st.sidebar.selectbox("Filter by Flagged Items", ["All", "Only Flagged Items", "Only Normal"])
if anomaly_filter == "Only Flagged Items":
    filter_base_df = filter_base_df[filter_base_df["Anomaly_Flag"] == True]
elif anomaly_filter == "Only Normal":
    filter_base_df = filter_base_df[filter_base_df["Anomaly_Flag"] == False]

filtered_df = filter_base_df.copy()

# Initialize session state variables if not set
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "selected_kpis" not in st.session_state:
    st.session_state.selected_kpis = []

# ---------- Main UI ----------
st.markdown("""
    <style>
        html, body, [class*="css"] { font-size: 16px; }
        .main-title { font-size: 24px; font-weight: bold; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Model Report Monitoring Dashboard</div>', unsafe_allow_html=True)


# ---------- Main Tabs ----------
tab1, tab2, tab3, tab4= st.tabs([
    "Executive Summary", "Model KPI Trends", "Flagged Items", "Raw Data"
])

# ---------- Overview Tab ----------
with tab1:
    total_partners = filtered_df["Partner_Name"].nunique()
    total_models = filtered_df["Model_ID"].nunique()
    unique_metrics = filtered_df["KPI"].nunique()
    # total_alerts = filtered_df["Anomaly_Flag"].sum()
    total_reports = filtered_df.drop_duplicates(["Partner_Name", "Model_ID", "Year", "Period"]).shape[0]
    # --- Filter for Last Quarter Anomalies ---
    # Get latest year and quarter
    recent_report = filtered_df["Report_Date"].max()
    latest_year = recent_report.year
    latest_quarter = (recent_report.month - 1) // 3 + 1
    quarter_start_date = pd.to_datetime(f"{latest_year}-{'%02d' % (3*(latest_quarter-1)+1)}-01")
    quarter_end_date = quarter_start_date + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    # Determine the latest available quarter
    if not filtered_df.empty:
        latest_report_date = filtered_df["Report_Date"].max()
        recent_quarter = latest_report_date.to_period("Q")
    else:
        recent_quarter = None
    # Filter to just that quarter
    recent_quarter_df = filtered_df[
        (filtered_df["Report_Date"] >= quarter_start_date) &
        (filtered_df["Report_Date"] <= quarter_end_date)
    ]

    # Use this for anomaly stats only
    recent_quarter_anomalies = recent_quarter_df[recent_quarter_df["Anomaly_Flag"] == True]
    total_alerts = recent_quarter_anomalies["Model_ID"].nunique()


    col1, col2, col3 = st.columns(3)
    col1.metric("Total Models", total_models)
    col2.metric("Total Partners", total_partners)
    col3.metric("No.of Models Flagged (Recent Quarter)", total_alerts)

    with st.expander("Model Information"):
        # Group to get model-level anomaly summary
        model_level_summary = (
            filtered_df
            .groupby(["Model_ID", "Model_Name", "Partner_Name", "Tier", "Frequency","Model_Info"])
            .agg(Anomaly_Flag=("Anomaly_Flag", "any"))  # True if any anomaly exists for that model
            .reset_index()
        ).drop(columns=["Anomaly_Flag"])

        # Display the result
        st.dataframe(model_level_summary, use_container_width=True)

    with st.expander("Models with Flagged Items (Recent Quarter)"):
        # Filter anomalies in the recent quarter
        anomaly_q_df = filtered_df[
            (filtered_df["Report_Date"].dt.to_period("Q") == recent_quarter) &
            (filtered_df["Anomaly_Flag"] == True)
        ]

        if not anomaly_q_df.empty:
            # Aggregate KPI(s) with anomalies per model
            anomaly_summary = (
                anomaly_q_df
                .groupby(["Model_ID", "Model_Name", "Partner_Name", "Tier"])
                .agg(Summary=("Report_Summary", "first"))  # Use first available summary
                .reset_index()
            )

            st.dataframe(anomaly_summary, use_container_width=True)
        else:
            st.info("No flagged items found for the recent quarter.")

    tier_counts = filtered_df.drop_duplicates("Model_ID")["Tier"].value_counts()
    freq_counts = filtered_df.drop_duplicates(["Model_ID", "Frequency"])["Frequency"].value_counts()
    report_freq_df = filtered_df.drop_duplicates(["Partner_Name", "Model_ID", "Year", "Period"])
    report_frequencies2 = report_freq_df["Frequency"].value_counts()
    reports_by_partner = report_freq_df["Partner_Name"].value_counts()
    models_by_partner = filtered_df.drop_duplicates(["Partner_Name", "Model_ID"])["Partner_Name"].value_counts()
    # --- New: Pie Charts for Last Quarter Anomalies ---

    # Filtered anomalies for last quarter already created earlier: `recent_quarter_anomalies`

    # Models by Last Quarter Anomalies
    models_by_anomaly = (
        recent_quarter_anomalies
        .groupby("Model_Name")
        .size()
        .sort_values(ascending=False)
    )

    # Reports by Last Quarter Anomalies
    reports_by_anomaly = (
        recent_quarter_anomalies
        .groupby("Partner_Name")
        .size()
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.pie(tier_counts, names=tier_counts.index, values=tier_counts.values, 
                               title="Models by Tier").update_traces(textinfo='label+value', hole=0.2), use_container_width=True)

    with col2:
        st.plotly_chart(px.pie(models_by_partner, names=models_by_partner.index, values=models_by_partner.values, 
                               title="Models by Partner").update_traces(textinfo='label+value', hole=0.2), use_container_width=True)
        



    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.plotly_chart(px.pie(report_frequencies2, names=report_frequencies2.index, values=report_frequencies2.values, 
    #                            title="Reports by Frequency").update_traces(textinfo='label+value', hole=0.2), use_container_width=True)

    # with col2:
    #     st.plotly_chart(px.pie(reports_by_partner, names=reports_by_partner.index, values=reports_by_partner.values, 
    #                            title="Reports by Partner").update_traces(textinfo='label+value', hole=0.2), use_container_width=True)

    # with col3:
    #     fig_anomaly_reports = px.pie(
    #         reports_by_anomaly,
    #         names=reports_by_anomaly.index,
    #         values=reports_by_anomaly.values,
    #         title="Reports with Anomalies (Last Quarter)"
    #     )
    #     fig_anomaly_reports.update_traces(textinfo='label+value', hole=0.3)
    #     st.plotly_chart(fig_anomaly_reports, use_container_width=True)

    with st.expander("Summary"):
        tier_summary = ", ".join([f"{tier}: {count}" for tier, count in tier_counts.items()])
        freq_summary = ", ".join([f"{freq}: {count}" for freq, count in report_frequencies2.items()])
        latest_upload = filtered_df["Uploaded_On"].max()
        latest_upload_str = latest_upload.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(latest_upload) else "N/A"

        st.markdown(f"""
        - In the **most recent quarter**, we identified **{int(total_alerts)} flagged items**.
        - Tier Distribution: {tier_summary}
        - Frequency Distribution: {freq_summary}
        """)

with tab2:
    kpi_groups = {
        "Power": ["AUC", "KS", "Gini"],
        "Stability": ["PSI", "CSI"],
        "Accuracy": ["PDO"]
    }

    # Model selection and KPI selection in a single row
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        unique_models = filtered_df["Model_Name"].dropna().unique().tolist()
        if not st.session_state.selected_model and not filtered_df.empty:
            st.session_state.selected_model = filtered_df.sort_values("Uploaded_On", ascending=False).iloc[0]["Model_Name"]
        selected_model = st.selectbox("Select Model for Trend View", options=unique_models, index=unique_models.index(st.session_state.selected_model) if st.session_state.selected_model in unique_models else 0)
        st.session_state.selected_model = selected_model
    with row1_col2:
        # KPI group selection
        group_options = ["-- Select Group --"] + list(kpi_groups.keys())
        default_group = group_options.index("Power")  # Set "Power" as default
        selected_group = st.selectbox("KPI Bundle", group_options, index=default_group)

    with row1_col3:
        model_kpi_options = filtered_df[filtered_df["Model_Name"] == selected_model]["KPI"].dropna().unique().tolist()
        if selected_group != "-- Select Group --":
            # Auto-select KPIs from group if available in current model
            grouped_kpis = [kpi for kpi in kpi_groups[selected_group] if kpi in model_kpi_options]
            selected_kpis = st.multiselect("KPIs", options=model_kpi_options, default=grouped_kpis)
        else:
            selected_kpis = st.multiselect("KPIs", options=model_kpi_options, default=st.session_state.selected_kpis)

        st.session_state.selected_kpis = selected_kpis

    if not selected_kpis:
        st.info("Please select at least one KPI to view trends.")
    else:
        for kpi in selected_kpis:
            st.markdown(f"##### {kpi} Trend")
            kpi_df = filtered_df[(filtered_df["Model_Name"] == selected_model) & (filtered_df["KPI"] == kpi)].copy()

            kpi_df["Quarter_Num"] = kpi_df["Period"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
            kpi_df["Sort_Key"] = kpi_df["Year"].astype(str) + "-" + kpi_df["Quarter_Num"].astype(str)
            kpi_df["Period_Label"] = kpi_df["Year"].astype(str) + " " + kpi_df["Period"]
            kpi_df = kpi_df.sort_values("Sort_Key")

            chart_col, summary_col = st.columns([0.7, 0.3])
            with chart_col:
                # 1. Main KPI trend line
                fig = px.line(
                    kpi_df,
                    x="Period_Label",
                    y="Value",
                    color="Model_Name",
                    markers=True,
                    line_group="Model_Name",
                    hover_data=["Partner_Name", "Tier", "Frequency"]
                )

                # # 2. Overlay anomaly points (red circles)
                # if "Anomaly_Flag" in kpi_df.columns:
                #     anomaly_points = kpi_df[kpi_df["Anomaly_Flag"] == True]
                #     if not anomaly_points.empty:
                #         fig.add_scatter(
                #             x=anomaly_points["Period_Label"],
                #             y=anomaly_points["Value"],
                #             mode="markers",
                #             name="Anomaly",
                #             marker=dict(size=10, color="red", symbol="circle"),
                #             hovertext=anomaly_points["Model_Name"],
                #             showlegend=True
                #         )

                # 3. Add yellow threshold lines (dashed)
                for model_name, group in kpi_df.groupby("Model_Name"):
                    if "Threshold" in group.columns and group["Threshold"].notna().any():
                        fig.add_scatter(
                            x=group["Period_Label"],
                            y=group["Threshold"],
                            mode="lines+markers",
                            name=f"{model_name} Threshold",
                            line=dict(dash="dash", color="#FFA500"),
                            marker=dict(color="#FFA500"),
                            showlegend=True
                        )

                # 4. Final chart styling
                fig.update_layout(
                    xaxis_title="Period",
                    yaxis_title=kpi,
                    legend_title="Model",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

            with summary_col:
                total_models = kpi_df["Model_ID"].nunique()
                total_partners = kpi_df["Partner_Name"].nunique()
                volatility = round(kpi_df["Value"].std(), 3)
                recent_value = (
                    kpi_df[kpi_df["Report_Date"] == kpi_df["Report_Date"].max()]
                    .sort_values("Uploaded_On", ascending=False)["Value"]
                    .mean()
                )
                st.markdown("**Summary**")
                st.markdown(f"""
                - **Volatility (Std):** {volatility}  
                - **Recent Avg Value:** {recent_value:.3f}
                """)

with tab3:
    # Show anomaly records, trends in anomaly counts etc.
    # Anomaly Records Table
    anomaly_df = recent_quarter_df[recent_quarter_df["Anomaly_Flag"] == True]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("No.of Models Flagged (Recent Quarter)", total_alerts)

#     with col2:
#         st.metric("Affected Models", anomaly_df['Model_ID'].nunique())

#     with col3:
#       st.metric("Affected KPIs", anomaly_df['KPI'].nunique())

    heatmap_df = (
        anomaly_df.groupby(["Model_Name", "KPI"]).size().unstack(fill_value=0)
    )
    st.dataframe(heatmap_df)
 
    st.markdown('<h5 style="margin-bottom:10px;">Flagged Records</h5>', unsafe_allow_html=True)

    if not anomaly_df.empty:
        st.dataframe(anomaly_df.drop(columns=["Anomaly_Flag"]).sort_values("Uploaded_On", ascending=False), use_container_width=True)
    else:
        st.info("No anomalies found in selected filters.")

    with st.expander("Summary / Recommendations"):
        st.markdown(f"""
        - **{anomaly_df['Model_ID'].nunique()} models** had at least one Flagged KPI.
        - **{anomaly_df['KPI'].nunique()} KPIs** were involved in flagged reports.
        - Consider deep-diving into models with recurring anomalies.
        """)

with tab4:
    # Raw Data Table (Expandable)
    #with st.expander("🧾 View Full Data Table"):
    raw_df = filtered_df.rename(columns={"Anomaly_Flag": "Flag"})  # Don't use inplace=True here
    st.dataframe(raw_df.sort_values("Uploaded_On", ascending=False), use_container_width=True)
    st.markdown("**Note:** This is the raw data used for all visualizations and analyses in this dashboard.")