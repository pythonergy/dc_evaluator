import streamlit as st
import pandas as pd
import plotly.express as px
from nsrdb_sam_analysis import NSRDBSAMAnalyzer
import os
from datetime import datetime
import json
import logging
import numpy as np
from dispatch_optimizer import DispatchOptimizer, create_results_dataframe

# Simplify page config
st.set_page_config(page_title="NSRDB SAM Analysis Tool", page_icon="☀️", layout="wide")

# Keep only essential CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stNumberInput { width: 100%; }
    </style>
""", unsafe_allow_html=True)

def create_download_zip(results_dir: str) -> bytes:
    """Create a zip file containing all results."""
    import zipfile
    import io
    import os
    
    # Create a BytesIO object to store the zip file
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the results directory
        for folder_path, _, filenames in os.walk(results_dir):
            for filename in filenames:
                # Get the full file path
                file_path = os.path.join(folder_path, filename)
                # Get the relative path for the zip file
                rel_path = os.path.relpath(file_path, results_dir)
                # Add file to zip
                zip_file.write(file_path, rel_path)
    
    return zip_buffer.getvalue()

def main():
    # Add this near the top of main(), after setting up session state for terminal output
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

    st.title("☀️ NSRDB SAM Analysis Tool")
    
    # Move all configuration to sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # NSRDB API Settings in collapsible section
        with st.expander("NSRDB API Settings", expanded=False):
            api_key = st.text_input(
                "API Key",
                value="UJKXTKMu2Pd3W97ASDKjYMAt3xc3OI4LtURDSl9a",
                help="Enter your NREL API key. Default key is provided but may have usage limits."
            )
            
            # User Information
            st.subheader("User Information")
            name = st.text_input(
                "Full Name",
                value="cleanai energy",
                help="Your full name for NSRDB records"
            )
            
            email = st.text_input(
                "Email",
                value="tech@cleanaienergy.com",
                help="Your email address for NSRDB records"
            )
            
            affiliation = st.text_input(
                "Affiliation",
                value="Research Institution",
                help="Your organization or institution"
            )
            
            reason = st.text_input(
                "Reason for Use",
                value="Research and Analysis",
                help="Purpose for accessing NSRDB data"
            )
            
            mailing_list = st.checkbox(
                "Join Mailing List",
                value=False,
                help="Receive updates about NSRDB"
            )
        
        # Location inputs
        st.subheader("Location")
        lat_col, lon_col = st.columns(2)
        
        lat = lat_col.number_input(
            "Latitude",
            value=31.431940,
            min_value=-90.0,
            max_value=90.0,
            format="%.6f",
            help="Enter latitude (-90 to 90) up to 6 decimal places"
        )
        lon = lon_col.number_input(
            "Longitude",
            value=-97.425000,
            min_value=-180.0,
            max_value=180.0,
            format="%.6f",
            help="Enter longitude (-180 to 180) up to 6 decimal places"
        )
        
        # System capacity
        st.subheader("System Size")
        capacity = st.number_input(
            "System Capacity (MW AC)",
            value=1000.0,
            min_value=0.1,
            help="Enter system capacity in MW AC"
        )
        
        # Year selection (moved up)
        st.subheader("Analysis Period")
        year_range = st.slider(
            "Select Year Range",
            min_value=1998,
            max_value=2022,
            value=(1998, 2022),
            help="Select the range of years to analyze"
        )
        selected_years = list(range(year_range[0], year_range[1] + 1))

        # Add TMY options right after year selection
        include_tmy = st.checkbox(
            "Include TMY Analysis",
            value=True,
            help="Include Typical Meteorological Year analysis"
        )

        if include_tmy:
            tmy_method = st.selectbox(
                "TMY Method",
                options=['calculated', 'api'],  # Changed order to make 'calculated' default
                help="Choose between calculated TMY from historical data or NSRDB API TMY"
            )
            
            if tmy_method == 'api':
                tmy_version = st.selectbox(
                    "TMY Version",
                    options=['tmy', 'tmy-2023'],
                    help="Select TMY version from NSRDB API"
                )
        
        # SAM Configuration
        st.subheader("SAM Configuration")
        
        # System Configuration
        with st.expander("System Configuration", expanded=False):
            dc_ac_ratio = st.number_input(
                "DC/AC Ratio",
                value=1.3,
                min_value=1.0,
                max_value=2.0,
                help="Ratio of DC capacity to AC capacity"
            )
            
            gcr = st.number_input(
                "Ground Coverage Ratio",
                value=0.35,
                min_value=0.1,
                max_value=0.9,
                help="Ground Coverage Ratio for tracker spacing"
            )
            
            inv_eff = st.number_input(
                "Inverter Efficiency (%)",
                value=98.5,
                min_value=90.0,
                max_value=100.0,
                help="Modern inverter efficiency"
            )
        
        # Bifacial Configuration
        with st.expander("Bifacial Settings", expanded=False):
            bifaciality = st.number_input(
                "Bifaciality Factor",
                value=0.7,
                min_value=0.5,
                max_value=0.9,
                help="Bifacial factor (typically 0.65-0.75)"
            )
            
            ground_albedo = st.number_input(
                "Ground Albedo",
                value=0.25,
                min_value=0.1,
                max_value=0.5,
                help="Ground reflectance"
            )
            
            bifacial_height = st.number_input(
                "Height Above Ground (m)",
                value=1.5,
                min_value=0.5,
                max_value=3.0,
                help="Module height above ground"
            )
        
        # System Losses
        with st.expander("System Losses", expanded=False):
            soiling = st.number_input("Soiling (%)", value=2.0, min_value=0.0, max_value=10.0)
            shading = st.number_input("Shading (%)", value=0.5, min_value=0.0, max_value=5.0)
            snow = st.number_input("Snow (%)", value=0.0, min_value=0.0, max_value=5.0)
            mismatch = st.number_input("Mismatch (%)", value=1.0, min_value=0.0, max_value=5.0)
            wiring_dc = st.number_input("DC Wiring (%)", value=1.5, min_value=0.0, max_value=5.0)
            connections_dc = st.number_input("DC Connections (%)", value=0.5, min_value=0.0, max_value=5.0)
            lid = st.number_input("Light Induced Degradation (%)", value=1.0, min_value=0.0, max_value=5.0)
            nameplate = st.number_input("Nameplate Rating (%)", value=0.5, min_value=0.0, max_value=5.0)
            availability = st.number_input("Availability (%)", value=1.0, min_value=0.0, max_value=5.0)
        
        # Output directory
        output_dir = st.text_input(
            "Output Directory",
            value="results",
            help="Directory where results will be saved"
        )
    
    # Main area for results and terminal output
    # Create containers for terminal output and results
    terminal_expander = st.expander("Terminal Output", expanded=False)
    with terminal_expander:
        terminal_placeholder = st.empty()  # Single placeholder for terminal output
    results_container = st.empty()
    
    # Initialize terminal output in session state if it doesn't exist
    if 'terminal_output' not in st.session_state:
        st.session_state.terminal_output = ""
    
    # Custom handler to capture log messages
    class StreamlitHandler(logging.Handler):
        def emit(self, record):
            try:
                log_entry = self.format(record)
                # Append new log entry to existing logs
                st.session_state.terminal_output += log_entry + "\n"
                # Update the terminal output using the placeholder
                terminal_placeholder.text_area(
                    label="",
                    value=st.session_state.terminal_output,
                    height=300,
                    disabled=True
                )
            except Exception as e:
                print(f"Error in StreamlitHandler: {e}")
    
    # Add the Streamlit handler to the logger
    streamlit_handler = StreamlitHandler()
    streamlit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('nsrdb_sam_analysis')
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(streamlit_handler)
    
    # Modify the Run Analysis and add Stop button section
    col1, col2 = st.sidebar.columns(2)
    
    if not st.session_state.is_running:
        if col1.button("Run Analysis", type="primary"):
            st.session_state.is_running = True
            st.rerun()
    else:
        if col1.button("Stop Analysis", type="secondary"):
            st.session_state.is_running = False
            st.rerun()

    # Modify your analysis section
    if st.session_state.is_running:
        # Clear previous terminal output
        st.session_state.terminal_output = ""
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize analyzer with custom settings
            analyzer = NSRDBSAMAnalyzer(
                api_key=api_key,
                lat=lat,
                lon=lon,
                system_capacity_mw=capacity,
                years=selected_years,
                include_tmy=include_tmy,
                tmy_method=tmy_method if include_tmy else 'api',
                tmy_version=tmy_version if (include_tmy and tmy_method == 'api') else 'tmy',
                dc_ac_ratio=dc_ac_ratio,
                gcr=gcr,
                inv_eff=inv_eff,
                bifaciality=bifaciality,
                ground_albedo=ground_albedo,
                bifacial_height=bifacial_height,
                losses={
                    'soiling': soiling,
                    'shading': shading,
                    'snow': snow,
                    'mismatch': mismatch,
                    'wiring_dc': wiring_dc,
                    'connections_dc': connections_dc,
                    'light_induced_degradation': lid,
                    'nameplate_rating': nameplate,
                    'availability': availability
                }
            )

            # Add check for stop condition throughout the analysis
            if not st.session_state.is_running:
                st.warning("Analysis stopped by user")
                return

            # Update analyzer's NSRDB settings
            analyzer.your_name = name
            analyzer.your_email = email
            analyzer.your_affiliation = affiliation
            analyzer.reason_for_use = reason
            analyzer.mailing_list = str(mailing_list).lower()
            
            # Create timestamp for this run
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(output_dir, f"run_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Check for stop condition
            if not st.session_state.is_running:
                st.warning("Analysis stopped by user")
                return

            # Dump system settings
            status_text.text("Saving system settings...")
            system_settings = analyzer.dump_system_settings(run_dir)
            progress_bar.progress(0.1)
            
            # Check for stop condition
            if not st.session_state.is_running:
                st.warning("Analysis stopped by user")
                return

            # Run analysis
            status_text.text("Running analysis...")
            results = analyzer.analyze_all_years()
            progress_bar.progress(0.8)
            
            # Check for stop condition
            if not st.session_state.is_running:
                st.warning("Analysis stopped by user")
                return

            # Save results
            status_text.text("Saving results...")
            analyzer.save_results_to_csv(results, run_dir)
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
            # Reset running state after completion
            st.session_state.is_running = False

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Summary", "TMY Analysis", "Detailed Results", 
                "System Settings", "Dispatch Optimization", "Downloads"
            ])
            
            with tab1:
                st.header("Summary Statistics")
                
                # Overall Performance section
                st.subheader("Overall Performance")
                overall_cols = st.columns(2)
                
                # Calculate average metrics from historical data
                if selected_years:  # Only calculate if we have historical years
                    historical_data = [results[year] for year in selected_years if year in results]
                    if historical_data:
                        avg_cf = np.mean([data['capacity_factor'] for data in historical_data]) * 100
                        avg_gen = np.mean([data['annual_generation_mwh'] for data in historical_data])
                        
                        with overall_cols[0]:
                            st.metric(
                                "Average Historical Capacity Factor",
                                f"{avg_cf:.1f}%",
                                help="Mean capacity factor across all historical years"
                            )
                        with overall_cols[1]:
                            st.metric(
                                "Average Annual Generation",
                                f"{avg_gen/1000:.1f} GWh",
                                help="Mean annual generation across all historical years"
                            )
                
                # Historical Analysis Results and Charts
                annual_data = []
                for year in selected_years:
                    if year in results:
                        annual_data.append({
                            'Year': year,
                            'Annual Generation (MWh)': results[year]['annual_generation_mwh'],
                            'Capacity Factor (%)': results[year]['capacity_factor'] * 100
                        })
                
                if annual_data:
                    st.subheader("Historical Analysis")
                    df_annual = pd.DataFrame(annual_data)
                    
                    # Add TMY data to plots
                    if 'tmy' in results:
                        tmy_data = []
                        if results['tmy']['method'] == 'calculated':
                            # Add P50/P10/P90 scenarios
                            for scenario in ['p50', 'p10', 'p90']:
                                scenario_data = results['tmy'][scenario]
                                tmy_data.append({
                                    'Year': f"TMY ({scenario.upper()})",
                                    'Annual Generation (MWh)': scenario_data['annual_generation_mwh'],
                                    'Capacity Factor (%)': scenario_data['capacity_factor'] * 100
                                })
                        else:
                            # Add API TMY data
                            tmy_data.append({
                                'Year': f"TMY ({results['tmy']['version']})",
                                'Annual Generation (MWh)': results['tmy']['annual_generation_mwh'],
                                'Capacity Factor (%)': results['tmy']['capacity_factor'] * 100
                            })
                        
                        # Add TMY data to the dataframe
                        df_tmy = pd.DataFrame(tmy_data)
                        df_annual = pd.concat([df_annual, df_tmy], ignore_index=True)
                    
                    # Update plots to highlight TMY points
                    tmy_points = df_annual[df_annual['Year'].astype(str).str.contains('TMY')]
                    
                    # Generation plot
                    fig1 = px.line(
                        df_annual[~df_annual['Year'].astype(str).str.contains('TMY')],  # Historical data only
                        x='Year',
                        y='Annual Generation (MWh)',
                        title='Annual Generation Over Time'
                    )
                    
                    # Add TMY points
                    fig1.add_scatter(
                        x=tmy_points['Year'],
                        y=tmy_points['Annual Generation (MWh)'],
                        mode='markers',
                        marker=dict(size=12, symbol='star'),
                        name='TMY Values'
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Capacity factor plot
                    fig2 = px.line(
                        df_annual[~df_annual['Year'].astype(str).str.contains('TMY')],  # Historical data only
                        x='Year',
                        y='Capacity Factor (%)',
                        title='Capacity Factor Over Time'
                    )
                    
                    # Add TMY points
                    fig2.add_scatter(
                        x=tmy_points['Year'],
                        y=tmy_points['Capacity Factor (%)'],
                        mode='markers',
                        marker=dict(size=12, symbol='star'),
                        name='TMY Values'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
                st.header("TMY Analysis")
                
                if 'tmy' in results:
                    if results['tmy']['method'] == 'calculated':
                        # Summary metrics
                        st.subheader("TMY Scenarios")
                        metrics_cols = st.columns(3)
                        
                        with metrics_cols[0]:
                            st.metric(
                                "P50 (Median)",
                                f"{results['tmy']['p50']['capacity_factor']*100:.1f}%",
                                f"{results['tmy']['p50']['annual_generation_mwh']/1000:.1f} GWh",
                                help="Median scenario - 50% chance of exceeding"
                            )
                        with metrics_cols[1]:
                            st.metric(
                                "P10 (Optimistic)",
                                f"{results['tmy']['p10']['capacity_factor']*100:.1f}%",
                                f"{results['tmy']['p10']['annual_generation_mwh']/1000:.1f} GWh",
                                help="Optimistic scenario - 10% chance of exceeding"
                            )
                        with metrics_cols[2]:
                            st.metric(
                                "P90 (Conservative)",
                                f"{results['tmy']['p90']['capacity_factor']*100:.1f}%",
                                f"{results['tmy']['p90']['annual_generation_mwh']/1000:.1f} GWh",
                                help="Conservative scenario - 90% chance of exceeding"
                            )
                        
                        # Create TMY comparison chart
                        st.subheader("TMY Comparison")
                        tmy_data = []
                        for scenario in ['p50', 'p10', 'p90']:
                            # Get hourly data for each scenario
                            scenario_data = results['tmy'][scenario]
                            
                            # Create proper timestamps for a non-leap year
                            timestamps = pd.date_range(
                                start=f"{datetime.now().year}-01-01",
                                periods=8760,
                                freq='H'
                            )
                            
                            df_hourly = pd.DataFrame({
                                'Timestamp': timestamps,
                                'Generation (MW)': scenario_data['data']['generation'],
                                'Scenario': f'TMY-{scenario.upper()}'
                            })
                            tmy_data.append(df_hourly)
                        
                        # Combine all scenarios
                        df_tmy_hourly = pd.concat(tmy_data)
                        
                        # Create line charts
                        fig_gen = px.line(
                            df_tmy_hourly, 
                            x='Timestamp', 
                            y='Generation (MW)',
                            color='Scenario',
                            title='TMY Generation Profiles'
                        )
                        fig_gen.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Generation (MW)",
                            hovermode='x unified',
                            xaxis=dict(
                                tickformat="%b %d",
                                dtick="M1",  # Show tick for each month
                                tickangle=45
                            )
                        )
                        st.plotly_chart(fig_gen, use_container_width=True)
                        
                        # Monthly averages (update to use timestamps)
                        df_monthly = df_tmy_hourly.copy()
                        df_monthly['Month'] = df_monthly['Timestamp'].dt.month
                        monthly_avg = df_monthly.groupby(['Month', 'Scenario'])['Generation (MW)'].mean().reset_index()
                        
                        # Add month names for better readability
                        month_names = {i: datetime(2000, i, 1).strftime('%B') for i in range(1, 13)}
                        monthly_avg['Month_Name'] = monthly_avg['Month'].map(month_names)
                        
                        fig_monthly = px.line(
                            monthly_avg,
                            x='Month_Name',
                            y='Generation (MW)',
                            color='Scenario',
                            title='TMY Monthly Average Generation',
                            markers=True
                        )
                        fig_monthly.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Average Generation (MW)",
                            xaxis=dict(
                                tickmode='array',
                                ticktext=list(month_names.values()),
                                tickvals=list(range(len(month_names))),
                                tickangle=45
                            ),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_monthly, use_container_width=True)
                        
                        # Detailed metrics in expander
                        with st.expander("Detailed TMY Metrics"):
                            st.dataframe(df_tmy_hourly)
                            
                    else:
                        # API TMY results
                        st.subheader("TMY Results")
                        metrics_cols = st.columns(3)
                        with metrics_cols[0]:
                            st.metric(
                                "TMY Capacity Factor",
                                f"{results['tmy']['capacity_factor']*100:.1f}%",
                                help="TMY capacity factor"
                            )
                        with metrics_cols[1]:
                            st.metric(
                                "Annual Generation",
                                f"{results['tmy']['annual_generation_mwh']/1000:.1f} GWh",
                                help="TMY annual generation"
                            )
                        with metrics_cols[2]:
                            st.metric(
                                "Min Week Generation",
                                f"{results['tmy']['min_week_generation_mwh']:.1f} MWh",
                                help="Minimum weekly generation"
                            )
                    
                    st.info(f"TMY Method: {results['tmy']['method'].upper()}" + 
                            (f" ({results['tmy']['version']})" if results['tmy']['method'] == 'api' else ""))
                else:
                    st.warning("No TMY analysis was performed. Enable TMY analysis in the configuration to see results.")
            
            with tab3:
                st.header("Detailed Results")
                if 'combined_data' in results:
                    df = results['combined_data']
                    
                    # Monthly averages
                    df['Month'] = df.index.month
                    monthly_avg = df.groupby('Month')['generation'].mean()
                    
                    fig3 = px.line(
                        monthly_avg,
                        title='Average Monthly Generation'
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Show raw data
                    st.subheader("Raw Data Sample")
                    st.dataframe(df.head())
                    
                    # Download button for full dataset
                    csv = df.to_csv(index=True)
                    st.download_button(
                        "Download Full Dataset",
                        csv,
                        "nsrdb_sam_analysis_results.csv",
                        "text/csv",
                        key='download-csv-results'
                    )
            
            with tab4:
                st.header("System Settings")
                st.json(system_settings)
                
                st.subheader("Download Individual Files")
                col1, col2 = st.columns(2)
                
                # List all files in the results directory
                files = {}
                for root, _, filenames in os.walk(run_dir):
                    for filename in filenames:
                        filepath = os.path.join(root, filename)
                        rel_path = os.path.relpath(filepath, run_dir)
                        with open(filepath, 'rb') as f:
                            files[rel_path] = f.read()
                
                # Create download buttons for each file
                for i, (filename, data) in enumerate(files.items()):
                    with col1 if i % 2 == 0 else col2:
                        st.download_button(
                            label=f"📄 {filename}",
                            data=data,
                            file_name=filename,
                            mime="application/octet-stream",
                            key=f"download_{i}"
                        )
            
            with tab5:
                st.header("Dispatch Optimization")
                
                # Battery configuration inputs
                st.subheader("Battery Configuration")
                col1, col2 = st.columns(2)
                with col1:
                    battery_power = st.number_input(
                        "Battery Power (MW)",
                        value=1000.0,
                        min_value=0.1,
                        help="Battery power capacity in MW"
                    )
                    battery_duration = st.number_input(
                        "Battery Duration (hours)",
                        value=4.0,
                        min_value=0.5,
                        help="Battery duration in hours"
                    )
                    battery_efficiency = st.number_input(
                        "Battery Round-trip Efficiency (%)",
                        value=90.0,
                        min_value=50.0,
                        max_value=100.0,
                        help="Battery round-trip efficiency"
                    )
                
                with col2:
                    generator_max = st.number_input(
                        "Generator Maximum Power (MW)",
                        value=500.0,
                        min_value=0.1,
                        help="Maximum generator power in MW"
                    )
                    generator_min = st.number_input(
                        "Generator Minimum Power (MW)",
                        value=50.0,
                        min_value=0.0,
                        help="Minimum generator power in MW"
                    )

                # Add load profile configuration
                st.subheader("Load Profile Configuration")
                load_percentage = st.slider(
                    "Load as percentage of max PV (%)",
                    min_value=0,
                    max_value=200,
                    value=80,
                    help="Set load as a percentage of maximum PV generation"
                )

                # Add a debug message to verify data
                if 'combined_data' in results:
                    st.info(f"PV data available: {len(results['combined_data'])} hours")
                else:
                    st.warning("No PV data available. Please run the analysis first.")
                
                dispatch_button = st.button("Run Dispatch Optimization", key="dispatch_button")
                
                if dispatch_button and 'combined_data' in results:
                    try:
                        with st.spinner("Running dispatch optimization..."):
                            # Convert values and run optimization
                            battery_power_kw = battery_power * 1000
                            battery_capacity_kwh = battery_power_kw * battery_duration
                            generator_max_kw = generator_max * 1000
                            generator_min_kw = generator_min * 1000
                            
                            pv_generation = results['combined_data']['generation'].values * 1000
                            max_pv = np.max(pv_generation)
                            load_profile = np.ones_like(pv_generation) * (load_percentage/100 * max_pv)
                            
                            optimizer = DispatchOptimizer(
                                time_steps=len(pv_generation),
                                battery_capacity_kwh=battery_capacity_kwh,
                                battery_power_kw=battery_power_kw,
                                battery_efficiency=battery_efficiency / 100,
                                generator_max_power_kw=generator_max_kw,
                                generator_min_power_kw=generator_min_kw
                            )
                            
                            dispatch_results = optimizer.optimize(
                                pv_generation=pv_generation,
                                load_profile=load_profile,
                                dt_hours=1.0
                            )
                            
                            # Create results DataFrame
                            df_dispatch = create_results_dataframe(
                                optimization_results=dispatch_results,
                                pv_generation=pv_generation,
                                original_load=load_profile,
                                timestamp_index=results['combined_data'].index
                            )
                            
                            # Display results
                            st.subheader("Optimization Results")
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Load Served (%)", 
                                    f"{100 * df_dispatch['Load_Served_kW'].sum() / df_dispatch['Load_Original_kW'].sum():.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Total PV Generation (GWh)", 
                                    f"{df_dispatch['PV_Generation_kW'].sum() / 1e6:.1f}"
                                )
                            with col3:
                                st.metric(
                                    "Total Battery Discharge (GWh)", 
                                    f"{df_dispatch['Battery_Discharge_kW'].sum() / 1e6:.1f}"
                                )
                            
                            # Plot results
                            fig = px.line(
                                df_dispatch,
                                y=['Load_Served_kW', 'PV_Generation_kW', 'Generator_Power_kW', 
                                   'Battery_Discharge_kW', 'Battery_Charge_kW'],
                                title='Dispatch Optimization Results'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Battery state of charge
                            fig_soc = px.line(
                                df_dispatch,
                                y='Battery_Energy_kWh',
                                title='Battery State of Charge'
                            )
                            st.plotly_chart(fig_soc, use_container_width=True)
                            
                            # Download button for dispatch results
                            csv = df_dispatch.to_csv(index=True)
                            st.download_button(
                                "Download Dispatch Results",
                                csv,
                                "dispatch_optimization_results.csv",
                                "text/csv",
                                key='download-csv-dispatch'
                            )
                            
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
                elif dispatch_button:
                    st.error("Please run the main analysis first to generate PV data.")
        
            with tab6:
                st.header("Download Results")
                
                # Create sections for different types of downloads
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Analysis Results")
                    if 'combined_data' in results:
                        # Main results dataset
                        csv = results['combined_data'].to_csv(index=True)
                        st.download_button(
                            "📊 Download Full Dataset",
                            csv,
                            "nsrdb_sam_analysis_results.csv",
                            "text/csv",
                            help="Complete hourly generation data for all analyzed periods"
                        )
                    
                    # Summary statistics
                    if 'summary_statistics.csv' in files:
                        st.download_button(
                            "📈 Download Summary Statistics",
                            files['summary_statistics.csv'],
                            "summary_statistics.csv",
                            "text/csv",
                            help="Summary metrics for all analyzed periods"
                        )
                
                with col2:
                    st.subheader("TMY Data")
                    # Filter TMY-related files
                    tmy_files = {k: v for k, v in files.items() if 'tmy' in k.lower()}
                    for filename, data in tmy_files.items():
                        st.download_button(
                            f"☀️ {filename}",
                            data,
                            filename,
                            "text/csv",
                            help="TMY scenario data and comparisons"
                        )
                
                # System configuration and reports section
                st.subheader("Configuration & Reports")
                report_col1, report_col2 = st.columns(2)
                
                with report_col1:
                    # System settings
                    st.download_button(
                        "⚙️ System Settings",
                        files.get('pv_system_settings.json', '{}'),
                        "pv_system_settings.json",
                        "application/json",
                        help="Complete system configuration"
                    )
                    
                    # Cleaning report
                    if 'cleaning_report.txt' in files:
                        st.download_button(
                            "🧹 Data Cleaning Report",
                            files['cleaning_report.txt'],
                            "cleaning_report.txt",
                            "text/plain",
                            help="Details about data cleaning process"
                        )
                
                with report_col2:
                    # Other reports and files
                    other_files = {k: v for k, v in files.items() 
                                  if not any(x in k.lower() for x in ['tmy', 'settings', 'cleaning', 'summary'])}
                    for filename, data in other_files.items():
                        st.download_button(
                            f"📄 {filename}",
                            data,
                            filename,
                            "application/octet-stream"
                        )
                
                # Download all button at the bottom
                st.markdown("---")
                st.subheader("Download Everything")
                zip_data = create_download_zip(run_dir)
                st.download_button(
                    label="📥 Download All Results as ZIP",
                    data=zip_data,
                    file_name=f"nsrdb_analysis_{timestamp}.zip",
                    mime="application/zip",
                    help="Download all analysis results, including TMY data and charts"
                )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.is_running = False  # Reset running state on error

    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Made with ❤️ using Streamlit | "
        "[Documentation](https://github.com/NREL/nsrdb) | "
        "[Report Issues](https://github.com/NREL/nsrdb/issues)"
    )

if __name__ == "__main__":
    main() 