import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import calendar
from datetime import datetime
import logging
import os
from pathlib import Path

# Simplify logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TMYCalculator:
    """
    Calculator for generating Typical Meteorological Year (TMY) data sets based on historical weather data.
    Calculates P50 (median), P10 (optimistic), and P90 (conservative) scenarios.
    """
    
    def __init__(self, historical_data: pd.DataFrame, date_column: str = 'datetime',
                 required_columns: Optional[List[str]] = None):
        """
        Initialize the TMY calculator with historical weather data.
        
        Args:
            historical_data: DataFrame containing historical weather data
            date_column: Name of column containing datetime information
            required_columns: List of weather variable columns that must be present
        """
        self.data = historical_data.copy()
        self.date_column = date_column
        
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        self.data['month'] = self.data[date_column].dt.month
        self.data['day'] = self.data[date_column].dt.day
        self.data['hour'] = self.data[date_column].dt.hour
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.weather_params = [col for col in self.data.columns 
                             if col not in [date_column, 'month', 'day', 'hour']]
        
        logger.info(f"TMY Calculator initialized with {len(self.data)} records and "
                   f"{len(self.weather_params)} weather parameters")
    
    def _calculate_percentile_by_month_hour(self, param: str, percentile: float) -> pd.DataFrame:
        """
        Calculate the specified percentile for a weather parameter for each month and hour.
        
        Args:
            param: Weather parameter column name
            percentile: Percentile to calculate (0-100)
            
        Returns:
            DataFrame with percentile values by month and hour
        """
        if param not in self.weather_params:
            raise ValueError(f"Parameter {param} not found in data")
            
        return self.data.groupby(['month', 'hour'])[param].quantile(percentile/100).reset_index()
    
    def generate_tmy(self, percentile: float = 50) -> pd.DataFrame:
        """
        Generate a Typical Meteorological Year dataset based on the specified percentile.
        Selects representative months from different years based on the percentile value.
        
        Args:
            percentile: Percentile to use (default: 50 for P50)
            
        Returns:
            DataFrame containing TMY data
        """
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        # Initialize empty TMY dataframe
        tmy_data = pd.DataFrame()
        
        # Process each month separately
        for month in range(1, 13):
            # Get data for this month across all years
            month_data = self.data[self.data['month'] == month].copy()
            
            # Calculate monthly sums/averages for each year
            yearly_stats = {}
            for year in month_data[self.date_column].dt.year.unique():
                year_month_data = month_data[month_data[self.date_column].dt.year == year]
                
                # Calculate statistics for each weather parameter
                stats = {}
                for param in self.weather_params:
                    if 'generation' in param.lower():
                        stats[param] = year_month_data[param].sum()
                    else:
                        stats[param] = year_month_data[param].mean()
                yearly_stats[year] = stats
            
            # Find the year closest to the target percentile for this month
            target_year = None
            min_distance = float('inf')
            
            ranking_param = next((p for p in self.weather_params if 'generation' in p.lower()), 
                               next((p for p in self.weather_params if 'ghi' in p.lower()), 
                                    self.weather_params[0]))
            
            values = [stats[ranking_param] for stats in yearly_stats.values()]
            target_value = np.percentile(values, percentile)
            
            for year, stats in yearly_stats.items():
                distance = abs(stats[ranking_param] - target_value)
                if distance < min_distance:
                    min_distance = distance
                    target_year = year
            
            # Get the data for the selected year/month
            selected_data = month_data[month_data[self.date_column].dt.year == target_year].copy()
            
            # Store source year before changing the year
            selected_data['source_year'] = target_year
            
            # Append to TMY dataset
            tmy_data = pd.concat([tmy_data, selected_data])
        
        # Reset the year to a non-leap year while preserving month/day/hour
        current_year = datetime.now().year
        if calendar.isleap(current_year):
            current_year += 1
        
        # Handle the datetime conversion safely
        def safe_year_replace(ts):
            try:
                # Try to replace the year directly
                return ts.replace(year=current_year)
            except ValueError:
                # If it's February 29, change to February 28
                if ts.month == 2 and ts.day == 29:
                    return ts.replace(year=current_year, day=28)
                raise  # Re-raise any other ValueError
        
        # Apply the safe year replacement
        tmy_data[self.date_column] = tmy_data[self.date_column].map(safe_year_replace)
        
        # Sort chronologically
        tmy_data = tmy_data.sort_values(self.date_column)
        
        logger.info(f"Generated P{percentile} TMY dataset with {len(tmy_data)} records")
        return tmy_data
    
    def generate_all_scenarios(self) -> Dict[str, pd.DataFrame]:
        """
        Generate P50, P10, and P90 TMY datasets.
        
        Returns:
            Dictionary containing 'P50', 'P10', and 'P90' DataFrames
        """
        scenarios = {
            'P50': self.generate_tmy(50),  # Median case
            'P10': self.generate_tmy(90),  # Optimistic case (only 10% chance of exceeding)
            'P90': self.generate_tmy(10)   # Conservative case (90% chance of exceeding)
        }
        
        logger.info(f"Generated all TMY scenarios (P50, P10, P90)")
        return scenarios
    
    def compare_scenarios(self, params: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare key statistics across P50, P10, and P90 scenarios.
        
        Args:
            params: List of parameters to compare (default: all weather parameters)
            
        Returns:
            DataFrame with comparison statistics
        """
        if not params:
            params = self.weather_params
            
        scenarios = self.generate_all_scenarios()
        comparison_data = []
        
        for param in params:
            for scenario, data in scenarios.items():
                comparison_data.append({
                    'Parameter': param,
                    'Scenario': scenario,
                    'Mean': data[param].mean(),
                    'Min': data[param].min(),
                    'Max': data[param].max(),
                    'Std Dev': data[param].std()
                })
        
        comparison = pd.DataFrame(comparison_data)
        return comparison

    def save_tmy_data(self, output_dir: str, prefix: str = "tmy") -> Dict[str, str]:
        """
        Save TMY data for all scenarios to CSV files.
        
        Args:
            output_dir: Directory to save files
            prefix: Prefix for filenames
            
        Returns:
            Dictionary mapping scenario names to file paths
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate all scenarios
        scenarios = self.generate_all_scenarios()
        saved_files = {}
        
        # Save each scenario
        for scenario_name, data in scenarios.items():
            filename = f"{prefix}_{scenario_name.lower()}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Add metadata as comments in the first rows
            with open(filepath, 'w') as f:
                f.write(f"# TMY Data - {scenario_name} Scenario\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Weather Parameters: {', '.join(self.weather_params)}\n")
                f.write(f"# Records: {len(data)}\n")
                f.write("#\n")  # Empty line after metadata
            
            # Save data
            data.to_csv(filepath, mode='a', index=False)
            saved_files[scenario_name] = filepath
            logger.info(f"Saved {scenario_name} TMY data to {filepath}")
        
        # Save comparison statistics
        comparison_file = os.path.join(output_dir, f"{prefix}_comparison.csv")
        self.compare_scenarios().to_csv(comparison_file, index=False)
        saved_files['comparison'] = comparison_file
        
        return saved_files 