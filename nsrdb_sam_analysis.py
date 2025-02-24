import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Tuple, Dict
from pathlib import Path
import urllib.parse
import hashlib
import json
import argparse
import ssl
import urllib.request
from io import StringIO
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NSRDBSAMAnalyzer:
    def __init__(
        self,
        api_key: str,
        lat: float,
        lon: float,
        system_capacity_mw: float = 1000.0,
        years: List[int] = None,
        cache_dir: str = "nsrdb_cache",
        dc_ac_ratio: float = 1.3,
        gcr: float = 0.35,
        inv_eff: float = 98.5,
        bifaciality: float = 0.7,
        ground_albedo: float = 0.25,
        bifacial_height: float = 1.5,
        losses: Dict[str, float] = None
    ):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.system_capacity = system_capacity_mw * 1000  # Convert MW to kW for SAM
        self.years = years or list(range(1998, 2023))  # 1998-2022
        self.cache_dir = cache_dir
        
        # SAM configuration
        self.dc_ac_ratio = dc_ac_ratio
        self.gcr = gcr
        self.inv_eff = inv_eff
        self.bifaciality = bifaciality
        self.ground_albedo = ground_albedo
        self.bifacial_height = bifacial_height
        
        # System losses
        default_losses = {
            "soiling": 2.0,
            "shading": 0.5,
            "snow": 0.0,
            "mismatch": 1.0,
            "wiring_dc": 1.5,
            "connections_dc": 0.5,
            "light_induced_degradation": 1.0,
            "nameplate_rating": 0.5,
            "availability": 1.0
        }
        self.dc_losses = losses if losses is not None else default_losses
        
        # Create simulation timestamp and output directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.simulation_dir = None  # Will be set when output directory is created
        
        # System losses breakdown (percentages)
        # DC-side losses that go into PySAM's 'losses' parameter
        self.other_losses = {
            "wiring_ac": 0.5,          # AC wiring losses
            "connections_ac": 0.5,      # AC connections
            "transformer": 1.0,         # Transformer losses
            "age": 0.5,                # First-year degradation
        }
        
        # Calculate total DC-side losses using series loss calculation
        total_dc_efficiency = 1.0
        for loss_value in self.dc_losses.values():
            total_dc_efficiency *= (1 - loss_value/100)
        self.total_dc_losses = (1 - total_dc_efficiency) * 100  # Convert to percentage
        
        # Calculate total system losses for reporting only (not passed to PySAM)
        total_system_efficiency = total_dc_efficiency
        # Apply inverter efficiency (handled separately by PySAM)
        inverter_efficiency = 0.985  # 98.5%
        total_system_efficiency *= inverter_efficiency
        # Apply other AC losses (not passed to PySAM)
        for loss_value in self.other_losses.values():
            total_system_efficiency *= (1 - loss_value/100)
        self.total_system_losses = (1 - total_system_efficiency) * 100
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Log the actual losses being used
        logger.info("DC-side losses (applied in series, passed to PySAM's 'losses' parameter):")
        for loss_name, loss_value in self.dc_losses.items():
            logger.info(f"  - {loss_name}: {loss_value:.1f}%")
        logger.info(f"Total DC losses passed to PySAM: {self.total_dc_losses:.4f}%")
        
        logger.info("\nInverter efficiency (passed separately to PySAM's 'inv_eff' parameter): 98.5%")
        
        logger.info("\nOther losses (tracked for reporting only, not passed to PySAM):")
        for loss_name, loss_value in self.other_losses.items():
            logger.info(f"  - {loss_name}: {loss_value:.1f}%")
        logger.info(f"Total system losses (for reporting only): {self.total_system_losses:.4f}%")
        
        # NSRDB API parameters
        self.attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
        self.interval = '60'  # 60-minute intervals
        self.utc = 'false'    # Use local time
        self.your_name = 'Gautham Ramesh'
        self.reason_for_use = 'Research and Analysis'
        self.your_affiliation = 'Research Institution'
        self.your_email = 'gauthamramesh0@gmail.com'
        self.mailing_list = 'false'

    def create_simulation_dir(self, base_output_dir: str = "results") -> str:
        """Create a unique directory for this simulation run."""
        # Create base output directory if it doesn't exist
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create unique simulation directory with timestamp and location
        sim_dir_name = f"simulation_{self.lat}_{self.lon}_{self.timestamp}"
        sim_dir = os.path.join(base_output_dir, sim_dir_name)
        Path(sim_dir).mkdir(parents=True, exist_ok=True)
        
        self.simulation_dir = sim_dir
        logger.info(f"Created simulation directory: {sim_dir}")
        return sim_dir

    def _get_cache_filename(self, year: int) -> str:
        """Generate a unique cache filename based on location and year."""
        # Create a unique identifier based on location and parameters
        params = f"{self.lat}_{self.lon}_{year}_{self.interval}_{self.attributes}"
        hash_str = hashlib.md5(params.encode()).hexdigest()[:10]
        return os.path.join(self.cache_dir, f"nsrdb_data_{year}_{hash_str}.csv")

    def _save_to_cache(self, df: pd.DataFrame, metadata: pd.DataFrame, year: int):
        """Save data to cache."""
        cache_file = self._get_cache_filename(year)
        
        # Save both metadata and data to the same file
        with open(cache_file, 'w') as f:
            # Save metadata first
            metadata.to_csv(f, index=False)
            f.write('\n')  # Add separator line
            # Save actual data
            df.to_csv(f, index=False)
        
        logger.info(f"Cached data saved to {cache_file}")

    def _load_from_cache(self, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from cache if available."""
        cache_file = self._get_cache_filename(year)
        
        if os.path.exists(cache_file):
            # Read the first line to get metadata
            metadata = pd.read_csv(cache_file, nrows=1)
            # Read the rest of the file for actual data
            df = pd.read_csv(cache_file, skiprows=2)
            logger.info(f"Loaded data from cache: {cache_file}")
            return df, metadata
        
        return None, None

    def _construct_url(self, year: int) -> str:
        """Construct the NSRDB API URL for a given year."""
        # Updated to use PSM v3.2.2 endpoint
        base_url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv'
        
        # Properly encode all parameters
        params = {
            'wkt': f'POINT({self.lon} {self.lat})',
            'names': str(year),
            'leap_day': 'false',
            'interval': self.interval,
            'utc': self.utc,
            'full_name': self.your_name,
            'email': self.your_email,
            'affiliation': self.your_affiliation,
            'mailing_list': self.mailing_list,
            'reason': self.reason_for_use,
            'api_key': self.api_key,
            'attributes': self.attributes
        }
        
        # URL encode each parameter value
        encoded_params = {k: urllib.parse.quote(str(v)) for k, v in params.items()}
        url_params = '&'.join([f'{k}={v}' for k, v in encoded_params.items()])
        
        full_url = f'{base_url}?{url_params}'
        logger.info(f"Constructed URL for year {year}: {full_url}")
        return full_url

    def fetch_nsrdb_data(self, year: int) -> Tuple[pd.DataFrame, float, float]:
        """Fetch NSRDB data for a specific year."""
        logger.info(f'Fetching NSRDB data for year {year}')
        
        # Try to load from cache first
        df, metadata = self._load_from_cache(year)
        
        if df is None or metadata is None:
            # If not in cache, fetch from API
            url = self._construct_url(year)
            try:
                # Configure requests session
                session = requests.Session()
                session.verify = False  # Disable SSL verification
                
                # Fetch data using requests
                response = session.get(url)
                response.raise_for_status()  # Raise an error for bad status codes
                
                # Convert response content to StringIO for pandas
                content = StringIO(response.text)
                
                # Get metadata
                metadata = pd.read_csv(content, nrows=1)
                if metadata.empty:
                    raise ValueError(f"No data returned for year {year}")
                
                # Reset content to start
                content.seek(0)
                
                # Get actual data
                df = pd.read_csv(content, skiprows=2)
                if df.empty:
                    raise ValueError(f"No data returned for year {year}")
                
                # Cache the data
                self._save_to_cache(df, metadata, year)
                
            except Exception as e:
                logger.error(f'Error fetching data for year {year}: {str(e)}')
                raise
        
        # Process the data
        timezone = float(metadata['Local Time Zone'].iloc[0])
        elevation = float(metadata['Elevation'].iloc[0])
        
        # Create datetime index from the Year, Month, Day, Hour columns with proper zero-padding
        df['datetime'] = pd.to_datetime({
            'year': df['Year'],
            'month': df['Month'],
            'day': df['Day'],
            'hour': df['Hour'] - 1  # Adjust hour from 1-24 to 0-23
        })
        
        # Remove leap day if present
        df = df[~((df['Month'] == 2) & (df['Day'] == 29))]
        
        # Set the datetime as index
        df = df.set_index('datetime')
        
        # Sort index to ensure chronological order
        df = df.sort_index()
        
        # Verify we have the correct number of hours
        if len(df) != 8760:
            logger.warning(f"Year {year} has {len(df)} hours instead of 8760")
            if len(df) < 8760:
                raise ValueError(f"Insufficient data for year {year}: only {len(df)} hours")
            else:
                # Trim to exactly 8760 hours if we have extra
                df = df.iloc[:8760]
        
        logger.info(f'Successfully loaded data for year {year} with {len(df)} hourly periods')
        return df, timezone, elevation

    def run_sam_simulation(self, df: pd.DataFrame, timezone: float, elevation: float) -> pd.DataFrame:
        """Run SAM simulation for the given data."""
        try:
            import PySAM.PySSC as pssc
        except ImportError:
            logger.error("PySAM not installed. Please install it using: pip install NREL-PySAM")
            sys.exit(1)

        logger.info('Running SAM simulation with bifacial panels')
        ssc = pssc.PySSC()
        
        # Create weather data
        wfd = ssc.data_create()
        ssc.data_set_number(wfd, b'lat', self.lat)
        ssc.data_set_number(wfd, b'lon', self.lon)
        ssc.data_set_number(wfd, b'tz', timezone)
        ssc.data_set_number(wfd, b'elev', elevation)
        ssc.data_set_array(wfd, b'year', df.index.year)
        ssc.data_set_array(wfd, b'month', df.index.month)
        ssc.data_set_array(wfd, b'day', df.index.day)
        ssc.data_set_array(wfd, b'hour', df.index.hour)
        ssc.data_set_array(wfd, b'minute', df.index.minute)
        ssc.data_set_array(wfd, b'dn', df['DNI'])
        ssc.data_set_array(wfd, b'df', df['DHI'])
        ssc.data_set_array(wfd, b'wspd', df['Wind Speed'])
        ssc.data_set_array(wfd, b'tdry', df['Temperature'])

        # Create SAM compliant object
        dat = ssc.data_create()
        ssc.data_set_table(dat, b'solar_resource_data', wfd)
        ssc.data_free(wfd)

        # Adjust system capacity to account for DC/AC ratio to achieve desired AC output
        dc_capacity = self.system_capacity * self.dc_ac_ratio  # This makes the AC output match the desired capacity
        
        # System Configuration for Single Axis Tracker with Bifacial Panels
        ssc.data_set_number(dat, b'system_capacity', dc_capacity)
        ssc.data_set_number(dat, b'dc_ac_ratio', self.dc_ac_ratio)
        ssc.data_set_number(dat, b'tilt', 0)
        ssc.data_set_number(dat, b'azimuth', 180)
        ssc.data_set_number(dat, b'inv_eff', self.inv_eff)
        ssc.data_set_number(dat, b'losses', self.total_dc_losses)
        ssc.data_set_number(dat, b'array_type', 2)
        ssc.data_set_number(dat, b'gcr', self.gcr)
        ssc.data_set_number(dat, b'adjust:constant', 0)
        
        # Bifacial specific parameters
        ssc.data_set_number(dat, b'bifaciality', self.bifaciality)
        ssc.data_set_number(dat, b'ground_albedo', self.ground_albedo)
        ssc.data_set_number(dat, b'bifacial_height', self.bifacial_height)
        
        # Execute simulation with bifacial model
        mod = ssc.module_create(b'pvwattsv7')  # Using v7 for bifacial support
        ssc.module_exec(mod, dat)
        
        # Get generation in kW and convert to MW
        df['generation'] = np.array(ssc.data_get_array(dat, b'gen')) / 1000.0

        # Clean up
        ssc.data_free(dat)
        ssc.module_free(mod)

        return df

    def analyze_all_years(self) -> Dict:
        """Run analysis for all specified years."""
        results = {}
        all_data = []  # List to store DataFrames for each year
        
        for year in self.years:
            try:
                # Fetch data
                df, timezone, elevation = self.fetch_nsrdb_data(year)
                
                # Verify we have exactly 8760 hours (365 days)
                if len(df) != 8760:
                    logger.warning(f"Year {year} has {len(df)} hours instead of 8760")
                    continue
                
                # Run simulation
                df_with_gen = self.run_sam_simulation(df, timezone, elevation)
                
                # Calculate metrics (generation is now in MW)
                annual_generation = df_with_gen['generation'].sum()  # Already in MWh since it's hourly data
                capacity_factor = annual_generation / (8760 * (self.system_capacity/1000))  # 8760 hours per year
                
                # Calculate MWh per MW-ac
                mwh_per_mw = annual_generation / (self.system_capacity/1000)
                
                # Find minimum week generation
                df_with_gen['datetime'] = df_with_gen.index
                weekly_gen = df_with_gen.set_index('datetime').resample('7D')['generation'].sum()  # Already in MWh
                min_week = weekly_gen.min()
                min_week_start = weekly_gen.idxmin()
                min_week_mwh_per_mw = min_week / (self.system_capacity/1000)
                
                # Prepare time series data
                ts_data = df_with_gen.copy()
                ts_data['year'] = ts_data.index.year
                ts_data['month'] = ts_data.index.month
                ts_data['day'] = ts_data.index.day
                ts_data['hour'] = ts_data.index.hour
                ts_data['minute'] = ts_data.index.minute
                
                # Store results
                results[year] = {
                    'annual_generation_mwh': annual_generation,
                    'capacity_factor': capacity_factor,
                    'mwh_per_mw': mwh_per_mw,
                    'min_week_generation_mwh': min_week,
                    'min_week_start_date': min_week_start,
                    'min_week_mwh_per_mw': min_week_mwh_per_mw,
                    'data': ts_data
                }
                
                # Add to combined data
                all_data.append(ts_data)
                
                logger.info(f'Year {year} analysis complete:')
                logger.info(f'  - Annual Generation: {annual_generation:.2f} MWh')
                logger.info(f'  - Capacity Factor: {capacity_factor:.3%}')
                logger.info(f'  - MWh per MW-ac: {mwh_per_mw:.2f}')
                logger.info(f'  - Min Week Generation: {min_week:.2f} MWh (starting {min_week_start.strftime("%Y-%m-%d")})')
                
            except Exception as e:
                logger.error(f'Error processing year {year}: {str(e)}')
                continue
        
        # Combine all years' data
        if all_data:
            combined_df = pd.concat(all_data)
            results['combined_data'] = combined_df
            
            # Calculate overall minimum MWh per MW-ac
            all_years_min_mwh_per_mw = min(results[year]['mwh_per_mw'] for year in self.years if year in results)
            results['overall_min_mwh_per_mw'] = all_years_min_mwh_per_mw
            
            # Find absolute worst week across all years
            combined_df['datetime'] = combined_df.index
            all_weekly_gen = combined_df.set_index('datetime').resample('7D')['generation'].sum()  # Already in MWh
            worst_week = all_weekly_gen.min()
            worst_week_start = all_weekly_gen.idxmin()
            worst_week_mwh_per_mw = worst_week / (self.system_capacity/1000)
            
            results['absolute_worst_week'] = {
                'generation_mwh': worst_week,
                'start_date': worst_week_start,
                'mwh_per_mw': worst_week_mwh_per_mw
            }
            
        return results

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean the data by identifying and removing days with missing data.
        Returns cleaned DataFrame and dictionary with cleaning info.
        """
        logger.info("Starting data cleaning process...")
        
        # Store original shape
        original_rows = len(df)
        
        # Create expected datetime range at hourly intervals
        expected_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='H'
        )
        
        # Find missing timestamps
        missing_times = expected_range.difference(df.index)
        
        # Get days with missing data
        days_with_missing = pd.Series(missing_times.date).unique()
        
        # Create cleaning report
        cleaning_info = {
            'original_rows': original_rows,
            'missing_timestamps': missing_times.tolist(),
            'days_with_missing': [d.strftime('%Y-%m-%d') for d in days_with_missing],
            'total_missing_hours': len(missing_times),
            'total_days_removed': len(days_with_missing)
        }
        
        # Remove days with missing data
        if len(days_with_missing) > 0:
            clean_df = df.copy()
            for day in days_with_missing:
                clean_df = clean_df[clean_df.index.date != day]
            
            cleaning_info['final_rows'] = len(clean_df)
            cleaning_info['removed_rows'] = original_rows - len(clean_df)
            
            logger.info(f"Found {len(missing_times)} missing hours across {len(days_with_missing)} days")
            logger.info(f"Removed {cleaning_info['removed_rows']} rows from dataset")
            
            return clean_df, cleaning_info
        
        cleaning_info['final_rows'] = original_rows
        cleaning_info['removed_rows'] = 0
        logger.info("No missing data found")
        return df, cleaning_info

    def save_results_to_csv(self, results: Dict, output_dir: str = "results"):
        """Save the simulation results to CSV files."""
        # Create simulation directory if not already created
        if self.simulation_dir is None:
            self.create_simulation_dir(output_dir)
        
        # Save combined time series data
        if 'combined_data' in results:
            combined_df = results['combined_data']
            
            # Calculate AC efficiency (all AC-side losses including inverter)
            ac_efficiency = 0.985  # Start with inverter efficiency
            for loss_value in self.other_losses.values():
                ac_efficiency *= (1 - loss_value/100)
            
            # Select and rename columns for clarity
            output_columns = {
                'datetime': 'Timestamp',
                'GHI': 'Global Horizontal Irradiance (W/m2)',
                'DNI': 'Direct Normal Irradiance (W/m2)',
                'DHI': 'Diffuse Horizontal Irradiance (W/m2)',
                'Temperature': 'Ambient Temperature (C)',
                'Wind Speed': 'Wind Speed (m/s)',
                'generation': 'DC to AC Power (MW)',
            }
            
            # Process unclean data
            output_df = combined_df[list(output_columns.keys())].copy()
            output_df.columns = list(output_columns.values())
            
            # Add column for power after AC losses
            output_df['Final AC Power (MW)'] = output_df['DC to AC Power (MW)'] * ac_efficiency
            
            # Save unclean data
            unclean_path = os.path.join(self.simulation_dir, 'simulation_results_unclean.csv')
            output_df.to_csv(unclean_path, index=False)
            logger.info(f'Unclean time series data saved to: {unclean_path}')
            
            # Clean the data
            clean_df, cleaning_info = self.clean_data(output_df)
            
            # Save clean data
            clean_path = os.path.join(self.simulation_dir, 'simulation_results_clean.csv')
            clean_df.to_csv(clean_path, index=False)
            logger.info(f'Clean time series data saved to: {clean_path}')
            
            # Save cleaning report
            cleaning_report_path = os.path.join(self.simulation_dir, 'cleaning_report.txt')
            with open(cleaning_report_path, 'w') as f:
                f.write("Data Cleaning Report\n")
                f.write("===================\n\n")
                f.write(f"Original number of rows: {cleaning_info['original_rows']}\n")
                f.write(f"Final number of rows: {cleaning_info['final_rows']}\n")
                f.write(f"Total rows removed: {cleaning_info['removed_rows']}\n")
                f.write(f"Total missing hours: {cleaning_info['total_missing_hours']}\n")
                f.write(f"Total days removed: {cleaning_info['total_days_removed']}\n\n")
                
                if cleaning_info['days_with_missing']:
                    f.write("Days removed due to missing data:\n")
                    for day in cleaning_info['days_with_missing']:
                        f.write(f"- {day}\n")
                    
                    f.write("\nMissing timestamps:\n")
                    for ts in cleaning_info['missing_timestamps']:
                        f.write(f"- {ts}\n")
            
            logger.info(f'Cleaning report saved to: {cleaning_report_path}')
            
            # Save detailed summary statistics
            summary_data = []
            for year in self.years:
                if year in results:
                    summary_data.append({
                        'Year': year,
                        'Annual Generation (MWh)': results[year]['annual_generation_mwh'],
                        'Capacity Factor (%)': results[year]['capacity_factor'] * 100,
                        'MWh per MW-ac': results[year]['mwh_per_mw'],
                        'Min Week Generation (MWh)': results[year]['min_week_generation_mwh'],
                        'Min Week Start Date': results[year]['min_week_start_date'].strftime('%Y-%m-%d'),
                        'Min Week MWh per MW-ac': results[year]['min_week_mwh_per_mw']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Add overall statistics
            summary_df.loc['Overall Min'] = {
                'Year': 'All Years',
                'MWh per MW-ac': results['overall_min_mwh_per_mw'],
                'Min Week Generation (MWh)': results['absolute_worst_week']['generation_mwh'],
                'Min Week Start Date': results['absolute_worst_week']['start_date'].strftime('%Y-%m-%d'),
                'Min Week MWh per MW-ac': results['absolute_worst_week']['mwh_per_mw']
            }
            
            summary_path = os.path.join(self.simulation_dir, 'summary_statistics.csv')
            summary_df.to_csv(summary_path, index=False)
            logger.info(f'Summary statistics saved to: {summary_path}')
            
            # Log key findings
            logger.info("\nKey Findings:")
            logger.info(f"Minimum MWh per MW-ac across all years: {results['overall_min_mwh_per_mw']:.2f}")
            logger.info(f"Worst week generation: {results['absolute_worst_week']['generation_mwh']:.2f} MWh")
            logger.info(f"Worst week start date: {results['absolute_worst_week']['start_date'].strftime('%Y-%m-%d')}")
            logger.info(f"Worst week MWh per MW-ac: {results['absolute_worst_week']['mwh_per_mw']:.2f}")

    def dump_system_settings(self, output_dir: str = "results") -> Dict:
        """Dump all PV system settings and configuration details to a JSON file."""
        # Create simulation directory if not already created
        if self.simulation_dir is None:
            self.create_simulation_dir(output_dir)
        
        # Calculate series efficiency for verification
        total_efficiency = 1.0
        for loss_value in self.dc_losses.values():
            total_efficiency *= (1 - loss_value/100)
        total_dc_loss_verification = (1 - total_efficiency) * 100
        
        # Compile system settings
        settings = {
            "Location": {
                "latitude": self.lat,
                "longitude": self.lon,
                "years_analyzed": self.years
            },
            "Data Source": {
                "api": "NREL NSRDB PSM v3.2.2",
                "attributes": self.attributes.split(','),
                "interval": f"{self.interval} minutes",
                "timezone": "Local time" if self.utc == 'false' else 'UTC'
            },
            "System Configuration": {
                "system_capacity_ac_mw": self.system_capacity / 1000,  # Convert kW to MW
                "dc_ac_ratio": self.dc_ac_ratio,
                "system_capacity_dc_mw": (self.system_capacity * self.dc_ac_ratio) / 1000,  # Include DC/AC ratio
                "module_type": "Bifacial",
                "tracking": {
                    "type": "Single Axis",
                    "tilt": 0,  # Tilt is handled by tracking
                    "azimuth": 180,
                    "gcr": self.gcr  # Ground Coverage Ratio
                }
            },
            "Bifacial Configuration": {
                "bifaciality_factor": self.bifaciality,
                "ground_albedo": self.ground_albedo,
                "rear_irradiance_loss": 5,  # 5%
                "transmission_factor": 0.9,  # 90%
                "height_above_ground_m": self.bifacial_height
            },
            "System Performance": {
                "inverter_efficiency": self.inv_eff,  # %
                "total_system_losses": self.total_system_losses,  # %
                "loss_calculation": "Series (multiplicative)",
                "loss_formula": "(1-L1)(1-L2)...(1-Ln)",
                "system_efficiency": total_efficiency * 100,  # %
                "loss_breakdown": {
                    "soiling_loss": {
                        "value": self.dc_losses["soiling"],
                        "description": "Dirt and dust accumulation on panels"
                    },
                    "shading_loss": {
                        "value": self.dc_losses["shading"],
                        "description": "External shading effects"
                    },
                    "snow_loss": {
                        "value": self.dc_losses["snow"],
                        "description": "Snow coverage impacts"
                    },
                    "mismatch_loss": {
                        "value": self.dc_losses["mismatch"],
                        "description": "Module manufacturing tolerance mismatch"
                    },
                    "wiring_loss": {
                        "value": self.dc_losses["wiring_dc"],
                        "description": "DC wiring losses"
                    },
                    "connections_loss": {
                        "value": self.dc_losses["connections_dc"],
                        "description": "DC connections"
                    },
                    "light_induced_degradation": {
                        "value": self.dc_losses["light_induced_degradation"],
                        "description": "Initial light exposure degradation"
                    },
                    "nameplate_rating_loss": {
                        "value": self.dc_losses["nameplate_rating"],
                        "description": "Module nameplate rating tolerance"
                    }
                }
            },
            "Simulation Details": {
                "sam_version": "PVWatts v7",
                "model_type": "Bifacial-enabled hourly simulation",
                "time_resolution": "Hourly",
                "analysis_period": f"{min(self.years)}-{max(self.years)}"
            }
        }
        
        # Save to JSON file
        settings_file = os.path.join(self.simulation_dir, 'pv_system_settings.json')
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        
        # Create and save losses CSV
        self.dump_losses_csv(self.simulation_dir)
        
        logger.info(f'System settings saved to: {settings_file}')
        return settings

    def dump_losses_csv(self, output_dir: str, timestamp: str = None) -> None:
        """Create a detailed CSV file of all system losses."""
        # Calculate cumulative efficiencies
        dc_efficiency = 1.0
        dc_losses_cumulative = []
        for name, value in self.dc_losses.items():
            dc_efficiency *= (1 - value/100)
            dc_losses_cumulative.append({
                'Category': 'DC Losses',
                'Component': name,
                'Loss (%)': value,
                'Cumulative Efficiency (%)': dc_efficiency * 100,
                'Description': self.get_loss_description(name)
            })

        # Start AC losses with inverter efficiency loss
        inverter_loss = 100 - self.inv_eff  # Inverter efficiency is 98.5%
        ac_efficiency = dc_efficiency * (self.inv_eff/100)  # Apply inverter efficiency first
        ac_losses_cumulative = [{
            'Category': 'AC/Other Losses',
            'Component': 'inverter_efficiency',
            'Loss (%)': inverter_loss,
            'Cumulative Efficiency (%)': ac_efficiency * 100,
            'Description': self.get_loss_description('inverter_efficiency')
        }]

        # Add remaining AC losses
        for name, value in self.other_losses.items():
            ac_efficiency *= (1 - value/100)
            ac_losses_cumulative.append({
                'Category': 'AC/Other Losses',
                'Component': name,
                'Loss (%)': value,
                'Cumulative Efficiency (%)': ac_efficiency * 100,
                'Description': self.get_loss_description(name)
            })

        # Calculate total AC losses (including inverter)
        total_ac_losses = (1 - ac_efficiency/dc_efficiency) * 100

        # Create summary rows
        summary_rows = [
            {
                'Category': 'TOTALS',
                'Component': 'Total DC Losses',
                'Loss (%)': self.total_dc_losses,
                'Cumulative Efficiency (%)': dc_efficiency * 100,
                'Description': 'Total losses on DC side (passed to PySAM)'
            },
            {
                'Category': 'TOTALS',
                'Component': 'Total AC Losses',
                'Loss (%)': total_ac_losses,
                'Cumulative Efficiency (%)': ac_efficiency * 100,
                'Description': 'Total losses on AC side (including inverter efficiency)'
            },
            {
                'Category': 'TOTALS',
                'Component': 'Total System Losses',
                'Loss (%)': self.total_system_losses,
                'Cumulative Efficiency (%)': ac_efficiency * 100,
                'Description': 'Total system losses (DC + AC + Other)'
            }
        ]

        # Combine all rows
        all_rows = dc_losses_cumulative + ac_losses_cumulative + summary_rows

        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_rows)
        csv_file = os.path.join(output_dir, 'system_losses.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f'System losses saved to: {csv_file}')

        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_rows)
        csv_file = os.path.join(output_dir, 'system_losses.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f'System losses saved to: {csv_file}')

    def get_loss_description(self, loss_name: str) -> str:
        """Get description for a loss component."""
        descriptions = {
            'soiling': 'Dirt and dust accumulation on panels',
            'shading': 'External shading effects',
            'snow': 'Snow coverage impacts',
            'mismatch': 'Module manufacturing tolerance mismatch',
            'wiring_dc': 'DC wiring losses',
            'connections_dc': 'DC connection losses',
            'light_induced_degradation': 'Initial light exposure degradation',
            'nameplate_rating': 'Module nameplate rating tolerance',
            'wiring_ac': 'AC wiring losses',
            'connections_ac': 'AC connection losses',
            'transformer': 'Transformer losses',
            'age': 'First-year degradation',
            'availability': 'Scheduled maintenance downtime',
            'inverter_efficiency': 'DC to AC conversion losses in the inverter'
        }
        return descriptions.get(loss_name, 'No description available')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='NSRDB SAM Analysis Tool')
    parser.add_argument('--api-key', type=str, default='j2onuxe80weyakaW10yryNHuXMTfFHaYMqRYhK57',
                        help='NREL API key')
    parser.add_argument('--lat', type=float, default=31.43194,
                        help='Latitude of the location')
    parser.add_argument('--lon', type=float, default=-97.42500,
                        help='Longitude of the location')
    parser.add_argument('--capacity', type=float, default=1000.0,
                        help='System capacity in MW AC')
    parser.add_argument('--years', type=int, nargs='+',
                        help='Years to analyze (default: 1998-2022)')
    parser.add_argument('--dump-only', action='store_true',
                        help='Only dump system settings JSON file without running analysis')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = NSRDBSAMAnalyzer(
        api_key=args.api_key,
        lat=args.lat,
        lon=args.lon,
        system_capacity_mw=args.capacity,
        years=args.years
    )
    
    # Dump system settings
    system_settings = analyzer.dump_system_settings(args.output_dir)
    
    # If dump-only flag is set, exit here
    if args.dump_only:
        logger.info("Settings dumped successfully. Exiting as requested.")
        return
    
    # Run analysis
    results = analyzer.analyze_all_years()
    
    # Save results to CSV
    analyzer.save_results_to_csv(results, args.output_dir)
    
if __name__ == "__main__":
    main()