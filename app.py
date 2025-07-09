import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ---------- Configuration ----------
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("sample_dataset.csv", parse_dates=["Uploaded_On"])
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary", "Model across KPI's", "KPI's across Model", "Flagged Items", "Raw Data", "Missing Reports"
])

# ---------- Overview Tab ----------
with tab1:
    total_partners = filtered_df["Partner_Name"].nunique()
    total_models = filtered_df["Model_ID"].nunique()
    total_active_models = filtered_df[filtered_df["Model_Status"] == "Active"]["Model_ID"].nunique()
    unique_metrics = filtered_df["KPI"].nunique()
    # total_alerts = filtered_df["Anomaly_Flag"].sum()
    total_reports = filtered_df.drop_duplicates(["Partner_Name", "Model_ID", "Year", "Period"]).shape[0]
    # --- Filter for Last Quarter Anomalies ---
    # Get latest year and quarter
    recent_report = filtered_df["Report_Date"].max()
    latest_year = recent_report.year
    latest_quarter = (recent_report.month - 1) // 3 + 1
    # Label for recent quarter (e.g., "Q1 2025")
    recent_quarter_label = f"Q{latest_quarter} {latest_year}"
    

    

    quarter_start_date = pd.to_datetime(f"{latest_year}-{'%02d' % (3*(latest_quarter-1)+1)}-01")
    quarter_end_date = quarter_start_date + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    # Determine the latest available quarter
    if not filtered_df.empty:
        latest_report_date = filtered_df["Report_Date"].max()
        recent_quarter = latest_report_date.to_period("Q")
        # Determine the last full year (based on latest date)
        last_year = latest_report_date.year - 1
        last_year_label = f"{last_year}"
        # --- Filter for Last Year's Reports ---
        last_year_df = filtered_df[filtered_df["Year"] == last_year]
        last_year_anomalies = last_year_df[last_year_df["Anomaly_Flag"] == True]
        total_anomalies_last_year = last_year_anomalies["Model_ID"].nunique()
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


    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Models", total_models)
    col2.metric("Total Active Models", total_active_models)
    col3.metric("Total Partners", total_partners)
    col4.metric(f"Flagged Models in {recent_quarter_label}", total_alerts)
    col5.metric(f"Flagged Models in {last_year_label}", total_anomalies_last_year)

    with st.expander("Model Information"):
        # Group to get model-level anomaly summary
        model_level_summary = (
            filtered_df
            .groupby(["Model_ID", "Model_Name", "Partner_Name", "Tier", "Frequency","Model_Info","Production_Date"])
            .agg(Anomaly_Flag=("Anomaly_Flag", "any"))  # True if any anomaly exists for that model
            .reset_index()
        ).drop(columns=["Anomaly_Flag"])

        # Display the result
        st.dataframe(model_level_summary, use_container_width=True)

    with st.expander(f"Flagged Models in {recent_quarter_label}"):
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

    last_year_df = filtered_df[filtered_df["Year"] == last_year]
    last_year_anomalies = last_year_df[last_year_df["Anomaly_Flag"] == True]
    total_anomalies_last_year = last_year_anomalies["Model_ID"].nunique()

    with st.expander(f"Flagged Models in {last_year_label}"):
        if not last_year_anomalies.empty:
            st.dataframe(
                last_year_anomalies
                .groupby(["Model_ID", "Model_Name", "Partner_Name"])
                .agg(Summary=("Report_Summary", "first"))
                .reset_index(),
                use_container_width=True
            )
        else:
            st.info(f"No flagged items found for {last_year_label}.")


    tier_counts = filtered_df.drop_duplicates("Model_ID")["Tier"].value_counts()
    freq_counts = filtered_df.drop_duplicates(["Model_ID", "Frequency"])["Frequency"].value_counts()
    report_freq_df = filtered_df.drop_duplicates(["Partner_Name", "Model_ID", "Year", "Period"])
    report_frequencies2 = report_freq_df["Frequency"].value_counts()
    reports_by_partner = report_freq_df["Partner_Name"].value_counts()
    models_by_partner = filtered_df.drop_duplicates(["Partner_Name", "Model_ID"])["Partner_Name"].value_counts()
    # Get unique models with their status
    model_status_counts = (
        filtered_df.drop_duplicates("Model_ID")["Model_Status"]
        .value_counts()
    )

    # Drop duplicates to avoid duplicate report entries
    report_type_counts = (
        report_freq_df["Frequency"]
        .value_counts()
    )


    # --- New: Pie Charts for Last Quarter Anomalies ---

    # Filtered anomalies for last quarter already created earlier: `recent_quarter_anomalies`

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
        px.pie(
            model_status_counts,
            names=model_status_counts.index,
            values=model_status_counts.values,
            title="Models by Status"
        ).update_traces(textinfo="label+value", hole=0.2),
        use_container_width=True
    )

    with col2:
        #with st.expander("📌 Missing Reports by Partner"):
        #st.markdown("This table highlights model reports that were **expected but not submitted** for specific periods.")

        # Setup: Define period range
        years = list(range(2023, 2026))
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        yearly = ["Y1"]

        # Create expected schedule for each model based on its frequency
        expected_reports = []

        for _, row in df.drop_duplicates(["Partner_Name", "Model_ID"]).iterrows():
            frequency = row["Frequency"]
            partner = row["Partner_Name"]
            model_id = row["Model_ID"]
            model_name = row["Model_Name"]

            for year in years:
                periods = quarters if frequency == "Quarterly" else yearly
                for period in periods:
                    expected_reports.append({
                        "Model_ID": model_id,
                        "Model_Name": model_name,
                        "Partner_Name": partner,
                        "Year": year,
                        "Period": period
                    })

        expected_df = pd.DataFrame(expected_reports)

        # Actual submissions
        submitted_df = df.drop_duplicates(["Model_ID", "Year", "Period"])[["Model_ID", "Year", "Period"]]
        submitted_df["Submitted"] = True

        # Merge to flag missing
        merged_df = expected_df.merge(submitted_df, on=["Model_ID", "Year", "Period"], how="left")
        missing_df = merged_df[merged_df["Submitted"].isna()].drop(columns=["Submitted"])

        # Count missing reports by partner
        missing_by_partner = missing_df["Partner_Name"].value_counts()

        st.plotly_chart(
            px.pie(
                names=missing_by_partner.index,
                values=missing_by_partner.values,
                title="Missing Reports by Partner"
            ).update_traces(textinfo="label+value", hole=0.2),
            use_container_width=True
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
            kpi_df["Model_Product"] = kpi_df.apply(
                lambda row: f"{row['Model_Name']} - {row['Product']}" if pd.notnull(row.get("Product")) and row["Product"] else row["Model_Name"],
                axis=1
            )
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
                    color="Model_Product",        # <-- Updated
                    markers=True,
                    line_group="Model_Product",  # <-- Updated
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

                import re

                # 3. Add yellow threshold lines (dashed)
                for model_product, group in kpi_df.groupby("Model_Product"):
                    if "Threshold" in group.columns and group["Threshold"].notna().any():
                        threshold_str = group["Threshold"].iloc[0]

                        match = re.match(r"(<=|>=|<|>|==|=)?\s*([\d.]+)", str(threshold_str))
                        if match:
                            op, threshold_val = match.groups()
                            try:
                                threshold_val = float(threshold_val)
                                fig.add_hline(
                                    y=threshold_val,
                                    line_dash="dash",
                                    line_color="#FFA500",
                                    annotation_text=f"Threshold ({op} {threshold_val})",
                                    annotation_position="top left"
                                )
                            except ValueError:
                                pass



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
    #st.markdown("#### KPI Bundle Overview (Latest Quarter Only)")
    st.markdown("Analyze how each model performed for each KPI category in the most recent reporting quarter.")

    # Step 1: Define KPI Bundles
    kpi_bundles = {
        "Power": ["AUC", "KS", "Gini"],
        "Stability": ["PSI", "CSI"],
        "Accuracy": ["PDO"]
    }

    # Step 2: Determine latest available quarter
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        latest_q = filtered_df["Report_Date"].max().to_period("Q")
        latest_q_df = filtered_df[filtered_df["Report_Date"].dt.to_period("Q") == latest_q]

        if latest_q_df.empty:
            st.info("No KPI data available for the latest quarter.")
        else:
            for bundle_name, kpis in kpi_bundles.items():
                bundle_df = latest_q_df[latest_q_df["KPI"].isin(kpis)]
                if bundle_df.empty:
                    continue

                st.markdown(f"##### {bundle_name} KPIs")

                pivot_df = (
                    bundle_df
                    .pivot_table(
                        index=["Model_ID", "Model_Name", "Partner_Name", "Tier"],
                        columns="KPI",
                        values="Anomaly_Flag",
                        aggfunc="first"
                    )
                    .fillna(False)
                    .astype(bool)
                    .reset_index()
                )

                # Add a column for Total KPIs Flagged
                kpi_cols = [col for col in pivot_df.columns if col in kpis]
                pivot_df["Total_Flagged"] = pivot_df[kpi_cols].sum(axis=1)

                # Sort by total flagged
                pivot_df = pivot_df.sort_values("Total_Flagged", ascending=False)

                st.dataframe(pivot_df, use_container_width=True)

            with st.expander("Summary View"):
                summary_data = []
                for bundle_name, kpis in kpi_bundles.items():
                    bundle_df = latest_q_df[latest_q_df["KPI"].isin(kpis)]
                    flagged_models = bundle_df[bundle_df["Anomaly_Flag"] == True]["Model_ID"].nunique()
                    total_models = bundle_df["Model_ID"].nunique()
                    summary_data.append({
                        "KPI Bundle": bundle_name,
                        "Total Models": total_models,
                        "Models with Flagged KPIs": flagged_models
                    })
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df)


with tab4:
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

    # Safely create a combined column with fallback if Product is missing
    anomaly_df["Model_Product"] = anomaly_df.apply(
        lambda row: row["Model_Name"] if pd.isna(row["Product"]) or row["Product"] == ''
        else f"{row['Model_Name']} - {row['Product']}", axis=1
    )

    # Then group and pivot
    heatmap_df = (
        anomaly_df.groupby(["Model_Product", "KPI"])
        .size()
        .unstack(fill_value=0)
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

with tab5:
    # Raw Data Table (Expandable)
    #with st.expander("🧾 View Full Data Table"):
    raw_df = filtered_df.rename(columns={"Anomaly_Flag": "Flag"})  # Don't use inplace=True here
    st.dataframe(raw_df.sort_values("Uploaded_On", ascending=False), use_container_width=True)
    st.markdown("**Note:** This is the raw data used for all visualizations and analyses in this dashboard.")

with tab6:
    #with st.expander("Missing Reports by Partner"):
    st.markdown("This table highlights model reports that were **expected but not submitted** for specific periods.")

    # Setup: Define period range
    years = list(range(2023, 2026))
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    yearly = ["Y1"]

    # Create expected schedule for each model based on its frequency
    expected_reports = []

    for _, row in df.drop_duplicates(["Partner_Name", "Model_ID"]).iterrows():
        frequency = row["Frequency"]
        partner = row["Partner_Name"]
        model_id = row["Model_ID"]
        model_name = row["Model_Name"]

        for year in years:
            periods = quarters if frequency == "Quarterly" else yearly
            for period in periods:
                expected_reports.append({
                    "Model_ID": model_id,
                    "Model_Name": model_name,
                    "Partner_Name": partner,
                    "Year": year,
                    "Period": period
                })

    expected_df = pd.DataFrame(expected_reports)

    # Actual submissions
    submitted_df = df.drop_duplicates(["Model_ID", "Year", "Period"])[["Model_ID", "Year", "Period"]]
    submitted_df["Submitted"] = True

    # Merge to flag missing
    merged_df = expected_df.merge(submitted_df, on=["Model_ID", "Year", "Period"], how="left")
    missing_df = merged_df[merged_df["Submitted"].isna()].drop(columns=["Submitted"])

    if not missing_df.empty:
        st.info(f"{len(missing_df)} missing reports found across models.")
        st.dataframe(missing_df.sort_values(["Partner_Name", "Model_Name", "Year", "Period"]), use_container_width=True)
    else:
        st.success("✅ All expected reports have been submitted.")

        # --- Summary for Missing Reports ---
    total_expected = len(expected_df)
    total_missing = len(missing_df)
    total_submitted = total_expected - total_missing
    affected_partners = missing_df["Partner_Name"].nunique()
    affected_models = missing_df["Model_Name"].nunique()

    with st.expander("Missing Reports Summary"):
        st.markdown(f"""
        - **Total Expected Reports**: {total_expected}  
        - **Submitted Reports**: {total_submitted}  
        - **Missing Reports**: {total_missing}  
        - **Partners Affected**: {affected_partners}  
        - **Models Affected**: {affected_models}
        """)

        # Optional: Partner-wise summary
        #st.markdown("Partner-wise Missing Report Summary")
        partner_summary = missing_df.groupby("Partner_Name").agg({
            "Model_Name": "nunique",
            "Model_ID": "count"
        }).rename(columns={"Model_Name": "Models Affected", "Model_ID": "Missing Reports"}).reset_index()
        st.dataframe(partner_summary, use_container_width=True)

