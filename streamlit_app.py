import streamlit as st
import pandas as pd
import plotly.express as px
from nsrdb_sam_analysis import NSRDBSAMAnalyzer
import os
from datetime import datetime
import json
import logging

# Set page config
st.set_page_config(
    page_title="NSRDB SAM Analysis Tool",
    page_icon="☀️",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("☀️ NSRDB SAM Analysis Tool")
    
    # Move all configuration to sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # NSRDB API Settings in collapsible section
        with st.expander("NSRDB API Settings", expanded=False):
            api_key = st.text_input(
                "API Key",
                value="j2onuxe80weyakaW10yryNHuXMTfFHaYMqRYhK57",
                help="Enter your NREL API key. Default key is provided but may have usage limits."
            )
            
            # User Information
            st.subheader("User Information")
            name = st.text_input(
                "Full Name",
                value="Gautham Ramesh",
                help="Your full name for NSRDB records"
            )
            
            email = st.text_input(
                "Email",
                value="gauthamramesh0@gmail.com",
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
            value=31.43194,
            min_value=-90.0,
            max_value=90.0,
            help="Enter latitude (-90 to 90)"
        )
        lon = lon_col.number_input(
            "Longitude",
            value=-97.42500,
            min_value=-180.0,
            max_value=180.0,
            help="Enter longitude (-180 to 180)"
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
            min_value=1998,  # Updated to 1998
            max_value=2022,
            value=(1998, 2022),  # Updated default range
            help="Select the range of years to analyze"
        )
        selected_years = list(range(year_range[0], year_range[1] + 1))
        
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
    terminal_container = st.expander("Terminal Output", expanded=False)  # Changed to expander
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
                # Update the terminal output in the expander
                terminal_container.text_area(
                    label="",  # Remove label since we have the expander header
                    value=st.session_state.terminal_output,
                    height=300,
                    disabled=True  # Make it read-only
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
    
    # Run analysis button (in sidebar)
    if st.sidebar.button("Run Analysis", type="primary"):
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
            
            # Dump system settings
            status_text.text("Saving system settings...")
            system_settings = analyzer.dump_system_settings(run_dir)
            progress_bar.progress(0.1)
            
            # Run analysis
            status_text.text("Running analysis...")
            results = analyzer.analyze_all_years()
            progress_bar.progress(0.8)
            
            # Save results
            status_text.text("Saving results...")
            analyzer.save_results_to_csv(results, run_dir)
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Results", "System Settings"])
            
            with tab1:
                st.header("Summary Statistics")
                if 'overall_min_mwh_per_mw' in results:
                    st.metric(
                        "Minimum MWh per MW-ac",
                        f"{results['overall_min_mwh_per_mw']:.2f}"
                    )
                
                if 'absolute_worst_week' in results:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Worst Week Generation (MWh)",
                            f"{results['absolute_worst_week']['generation_mwh']:.2f}"
                        )
                    with col2:
                        st.metric(
                            "Worst Week MWh per MW-ac",
                            f"{results['absolute_worst_week']['mwh_per_mw']:.2f}"
                        )
                
                # Plot annual generation
                annual_data = []
                for year in selected_years:
                    if year in results:
                        annual_data.append({
                            'Year': year,
                            'Annual Generation (MWh)': results[year]['annual_generation_mwh'],
                            'Capacity Factor (%)': results[year]['capacity_factor'] * 100
                        })
                
                if annual_data:
                    df_annual = pd.DataFrame(annual_data)
                    
                    # Generation plot
                    fig1 = px.line(
                        df_annual,
                        x='Year',
                        y='Annual Generation (MWh)',
                        title='Annual Generation Over Time'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Capacity factor plot
                    fig2 = px.line(
                        df_annual,
                        x='Year',
                        y='Capacity Factor (%)',
                        title='Capacity Factor Over Time'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
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
            
            with tab3:
                st.header("System Settings")
                st.json(system_settings)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Made with ❤️ using Streamlit | "
        "[Documentation](https://github.com/NREL/nsrdb) | "
        "[Report Issues](https://github.com/NREL/nsrdb/issues)"
    )

if __name__ == "__main__":
    main() 