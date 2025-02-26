import cvxpy as cp
import numpy as np
import pandas as pd
import os
import sys

class DispatchOptimizer:
    def __init__(self, 
                 time_steps: int,
                 battery_capacity_kwh: float,
                 battery_power_kw: float,
                 battery_efficiency: float,
                 generator_max_power_kw: float,
                 generator_min_power_kw: float):
        """
        Initialize the dispatch optimizer
        
        Args:
            time_steps: Number of time periods to optimize
            battery_capacity_kwh: Battery energy capacity in kWh
            battery_power_kw: Battery power capacity in kW
            battery_efficiency: Round-trip efficiency of the battery (0-1)
            generator_max_power_kw: Maximum generator power output in kW
            generator_min_power_kw: Minimum generator power output in kW
        """
        self.time_steps = time_steps
        self.battery_capacity = battery_capacity_kwh
        self.battery_power = battery_power_kw
        self.battery_efficiency = battery_efficiency
        self.generator_max_power = generator_max_power_kw
        self.generator_min_power = generator_min_power_kw
        
    def optimize(self, 
                pv_generation: np.ndarray,
                load_profile: np.ndarray,
                dt_hours: float = 1.0) -> dict:
        """
        Optimize the dispatch of all resources to maximize load serving
        
        Args:
            pv_generation: Array of PV generation values (kW)
            load_profile: Array of load values (kW)
            dt_hours: Time step duration in hours
            
        Returns:
            Dictionary containing optimization results
        """
        # Decision variables
        battery_charge = cp.Variable(self.time_steps)  # Battery charging power (kW)
        battery_discharge = cp.Variable(self.time_steps)  # Battery discharging power (kW)
        battery_energy = cp.Variable(self.time_steps)  # Battery energy state (kWh)
        generator_power = cp.Variable(self.time_steps)  # Generator output (kW)
        load_served = cp.Variable(self.time_steps)  # Amount of load actually served (kW)
        
        # Constraints list
        constraints = []
        
        # Power balance constraint
        for t in range(self.time_steps):
            constraints.append(
                load_served[t] == pv_generation[t] + 
                generator_power[t] + 
                battery_discharge[t] - 
                battery_charge[t]
            )
        
        # Battery constraints
        for t in range(self.time_steps):
            # Power limits
            constraints.append(battery_charge[t] >= 0)
            constraints.append(battery_discharge[t] >= 0)
            constraints.append(battery_charge[t] <= self.battery_power)
            constraints.append(battery_discharge[t] <= self.battery_power)
            
            # Energy state evolution
            if t == 0:
                constraints.append(
                    battery_energy[t] == self.battery_capacity * 0.5 +  # Start at 50% SOC
                    (battery_charge[t] * self.battery_efficiency - 
                     battery_discharge[t] / self.battery_efficiency) * dt_hours
                )
            else:
                constraints.append(
                    battery_energy[t] == battery_energy[t-1] +
                    (battery_charge[t] * self.battery_efficiency - 
                     battery_discharge[t] / self.battery_efficiency) * dt_hours
                )
            
            # Energy capacity limits
            constraints.append(battery_energy[t] >= 0)
            constraints.append(battery_energy[t] <= self.battery_capacity)
        
        # Generator constraints
        for t in range(self.time_steps):
            constraints.append(generator_power[t] >= self.generator_min_power)
            constraints.append(generator_power[t] <= self.generator_max_power)
        
        # Load constraints
        for t in range(self.time_steps):
            constraints.append(load_served[t] >= 0)
            constraints.append(load_served[t] <= load_profile[t])
        
        # Objective: Maximize load served
        objective = cp.Maximize(cp.sum(load_served))
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            raise ValueError(f"Problem failed to solve optimally. Status: {problem.status}")
        
        # Return results
        return {
            'load_served': load_served.value,
            'battery_charge': battery_charge.value,
            'battery_discharge': battery_discharge.value,
            'battery_energy': battery_energy.value,
            'generator_power': generator_power.value,
            'objective_value': problem.value
        }

def create_results_dataframe(
    optimization_results: dict,
    pv_generation: np.ndarray,
    original_load: np.ndarray,
    timestamp_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Create a DataFrame with all the optimization results
    
    Args:
        optimization_results: Dictionary of optimization results
        pv_generation: Original PV generation profile
        original_load: Original load profile
        timestamp_index: DatetimeIndex for the results
        
    Returns:
        DataFrame containing all results
    """
    df = pd.DataFrame({
        'Load_Original_kW': original_load,
        'Load_Served_kW': optimization_results['load_served'],
        'PV_Generation_kW': pv_generation,
        'Generator_Power_kW': optimization_results['generator_power'],
        'Battery_Charge_kW': optimization_results['battery_charge'],
        'Battery_Discharge_kW': optimization_results['battery_discharge'],
        'Battery_Energy_kWh': optimization_results['battery_energy']
    }, index=timestamp_index)
    
    return df

def test_optimizer_with_pysam_results(results_dir: str = None):
    """
    Test the dispatch optimizer using PySAM simulation results, processing one year at a time
    
    Args:
        results_dir: Path to the simulation results directory. If None, uses default path.
    """
    if results_dir is None:
        results_dir = "results/run_20250223_204931/simulation_31.43194_-97.425_20250223_204931"
    
    # Load PV generation data from results
    pv_data = pd.read_csv(f"{results_dir}/simulation_results_clean.csv", index_col=0, parse_dates=True)
    
    # Process each year separately
    yearly_results = []
    years = pv_data.index.year.unique()
    
    for year in years:
        print(f"\nProcessing year {year}...")
        
        # Filter data for current year
        year_data = pv_data[pv_data.index.year == year]
        
        # Convert MW to kW and ensure non-negative values
        pv_generation = np.maximum(year_data['Final AC Power (MW)'].values * 1000, 0)  # Convert MW to kW
        
        # Skip if no PV generation
        if np.sum(pv_generation) == 0:
            print(f"Skipping {year} - no PV generation data")
            continue
            
        # Create a sample load profile (for testing, we'll use 80% of max PV generation)
        max_pv = np.max(pv_generation)
        if max_pv == 0:
            print(f"Skipping {year} - max PV generation is 0")
            continue
            
        load_profile = np.ones_like(pv_generation) * (0.8 * max_pv)
        
        # Initialize optimizer with sample parameters
        optimizer = DispatchOptimizer(
            time_steps=len(pv_generation),
            battery_capacity_kwh=4000000.0,  # 4000 MWh battery (4-hour duration at 1000MW)
            battery_power_kw=1000000.0,      # 1000 MW power rating
            battery_efficiency=0.90,          # 90% round-trip efficiency
            generator_max_power_kw=500000.0,  # 500 MW generator
            generator_min_power_kw=50000.0    # 50 MW minimum generation
        )
        
        try:
            # Run optimization
            results = optimizer.optimize(
                pv_generation=pv_generation,
                load_profile=load_profile,
                dt_hours=1.0
            )
            
            # Verify results are valid
            if np.any(np.isnan(results['load_served'])) or np.sum(results['load_served']) <= 0:
                print(f"Skipping {year} - invalid optimization results")
                continue
                
            # Create results DataFrame for this year
            df_year_results = create_results_dataframe(
                optimization_results=results,
                pv_generation=pv_generation,
                original_load=load_profile,
                timestamp_index=year_data.index
            )
            
            yearly_results.append(df_year_results)
            
            # Print summary statistics for this year
            print(f"\nOptimization Results Summary for {year}:")
            print(f"Number of hours: {len(pv_generation)}")
            print(f"Total Load: {np.sum(load_profile)/1000:.2f} MWh")
            print(f"Load Served: {np.sum(results['load_served'])/1000:.2f} MWh")
            print(f"Load Served Percentage: {100 * np.sum(results['load_served']) / np.sum(load_profile):.2f}%")
            print(f"Total PV Generation: {np.sum(pv_generation)/1000:.2f} MWh")
            print(f"Total Generator Energy: {np.sum(results['generator_power'])/1000:.2f} MWh")
            print(f"Total Battery Discharge: {np.sum(results['battery_discharge'])/1000:.2f} MWh")
            print(f"Total Battery Charge: {np.sum(results['battery_charge'])/1000:.2f} MWh")
            print(f"Final Battery Energy: {results['battery_energy'][-1]/1000:.2f} MWh")
            
        except Exception as e:
            print(f"Optimization failed for year {year}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Combine all years' results
    if yearly_results:
        df_all_results = pd.concat(yearly_results)
        
        # Save combined results
        output_dir = f"{results_dir}/dispatch_optimization"
        os.makedirs(output_dir, exist_ok=True)
        df_all_results.to_csv(f"{output_dir}/optimization_results.csv")
        print(f"\nCombined results saved to: {output_dir}/optimization_results.csv")
        
        # Print overall statistics
        print("\nOverall Results Summary:")
        print(f"Number of hours: {len(df_all_results)}")
        print(f"Total Load: {df_all_results['Load_Original_kW'].sum()/1000:.2f} MWh")
        print(f"Load Served: {df_all_results['Load_Served_kW'].sum()/1000:.2f} MWh")
        print(f"Load Served Percentage: {100 * df_all_results['Load_Served_kW'].sum() / df_all_results['Load_Original_kW'].sum():.2f}%")
        print(f"Total PV Generation: {df_all_results['PV_Generation_kW'].sum()/1000:.2f} MWh")
        print(f"Total Generator Energy: {df_all_results['Generator_Power_kW'].sum()/1000:.2f} MWh")
        print(f"Total Battery Discharge: {df_all_results['Battery_Discharge_kW'].sum()/1000:.2f} MWh")
        print(f"Total Battery Charge: {df_all_results['Battery_Charge_kW'].sum()/1000:.2f} MWh")

if __name__ == "__main__":
    # Use default simulation results path
    test_optimizer_with_pysam_results()