import random
import pandas as pd
import yaml
from multiprocessing import Pool, cpu_count
import sys
import os
import time
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from monte_carlo_tools  import perturb_df_energies,residual_params_dict
from monte_carlo_analysis import analyze_monte_carlo_results,plot_selected_columns_script,interactive_plot_script

# Check if the user provided a path for the User.yaml file
if len(sys.argv) > 1:
    user_path = sys.argv[1]
else:
    sys.exit("User.yaml path not provided.")

# Relative file path to be used
current_directory = os.path.dirname(__file__)

# YAML file with user instructions and consuptions database
database_path = os.path.join(current_directory, 'Database.yaml')
with open(user_path, "r") as user_yaml:
    User = yaml.safe_load(user_yaml)
with open(database_path, "r") as database_yaml:
    Database = yaml.safe_load(database_yaml)

# Read Forecast
matching_files = [filename for filename in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, filename))]
forecast_file = next((file for file in matching_files if file.lower().endswith('.xlsx')), None)

if forecast_file:
    df_energies = pd.read_excel(os.path.join(current_directory, forecast_file))
elif os.path.isfile(User['Path_Database']):
    df_energies = pd.read_excel(User['Path_Database'])
else:
    csv_file = next((file for file in matching_files if file.lower().endswith('.csv')), None)
    if csv_file:
        csv_path = os.path.join(current_directory, csv_file)
        xlsx_path = os.path.join(current_directory, os.path.splitext(csv_file)[0] + '.xlsx')
        pd.read_csv(csv_path).to_excel(xlsx_path, index=False)
        df_energies = pd.read_excel(xlsx_path)
    else:
        sys.exit("No suitable data file found.")

# Function to obtain the difference between Production minus Demand
def surplus(perturbed_df_energies_data):
    """Creates a dataframe with Date and the difference between
    demand and total produced, showing electrical surplus or deficit in kW for each date"""
    
    df = pd.DataFrame()
    df['Date'] = perturbed_df_energies_data['ds']

    data_unit = User['Database_Unit'].upper()
    if data_unit == 'GW':
        unit = 1000000
    elif data_unit == 'MW':
        unit = 1000
    elif data_unit == 'KW':
        unit = 1
    else:
        sys.exit("Invalid database unit in User.yaml")

    df['Prod_Dem_Diff (kW)'] = (perturbed_df_energies_data['Total'] - perturbed_df_energies_data['Demand']) * unit
    df['Accum_Diff (kW)'] = df['Prod_Dem_Diff (kW)'].cumsum()
    
    return df

# Function to collect technology consumption values
def technology_values():
    """Stores the names of production technologies selected
    in User.yaml as well as their power consumption"""
    tech_prod = []
    for value in User['Production_Share'].values():
        techs = []
        for item in value[1:]:
            if isinstance(item, str):
                techs.append(item)
            elif isinstance(item, dict):
                techs.extend(item.values())
        tech_prod.append(techs)

    values = []
    
    for tech_list in tech_prod:
        tech_values = []
        for tech in tech_list:
            for key1, value1 in Database['Electricity_Consumption'].items():
                for key2, value2 in value1.items():
                    if tech == key2:
                        ran = User['Stochastic_Processes']
                        
                        if isinstance(value2, list) and len(value2) == 2 and ran== True:
                            tech_values.append(round(random.uniform(value2[0], value2[1]), 4))
                        else:
                            tech_values.append(mean(value2))
        values.append(tech_values)

    result = []
    for sublist in values:
        nested = []
        for item in sublist:
            if isinstance(item, list):
                nested.extend(item)
            else:
                nested.append(item)
        result.append(nested)

    single_comm = []
    dependent_comm = []
    for item in result:
        if len(item) == 1:
            single_comm.append(item[0])
        else:
            dependent_comm.append(item)

    return single_comm, dependent_comm

# Function to collect production share
def production_share():
    """Stores the percentages described in User.yaml"""
    single_perc = []
    dependent_perc = []
    for _, perc in User['Production_Share'].items():
        if len(perc) == 2:
            single_perc.append(float(perc[0]))
        else:
            for value in perc:
                if isinstance(value, (float, int)):
                    dependent_perc.append(float(value))
    if sum(single_perc + dependent_perc) == 1.0:
        return single_perc, dependent_perc
    else:
        print("The sum of percentages must equal 1. Please check User.yaml.")
        sys.exit()

# Function to generate production of commodities
def final_production(perturbed_df_energies_data):
    sur = surplus(perturbed_df_energies_data)

    single_cons, dependent_cons = technology_values()

    single_perc, dependent_perc = production_share()
    dep_values = []
    for sust in User.get('Production_Share'):
        if sust in Database.get('Mass_Balance'):
            metric_val = Database['Mass_Balance'][sust]
            temp_list = []
            for row in metric_val.values():
                
                ran = User['Stochastic_Processes']
                if isinstance(row, list) and len(row) == 2 and ran:
                    random_value = round(random.uniform(row[0], row[1]), 4)
                    temp_list.append(random_value)
                else:
                    temp_list.append(mean(row))
            dep_values.append(temp_list)

    single_names = []
    dependent_names = []
    for key, value in User['Production_Share'].items():
        if len(value) == 2:
            single_names.append(key)
        elif len(value) > 2:
            dependent_names.append(key)
    sum_val = []
    for sublist in dependent_cons:
        sum_val.append(sublist.pop(0))
    result_list = []
    for i in range(len(dep_values)):
        result = 0
        for j in range(len(dep_values[i])):
            result += dep_values[i][j] * dependent_cons[i][j]
        result += sum_val[i]
        result_list.append(result)

    for col in single_names + dependent_names:
        sur[col] = 0.0
        sur[col]=sur[col].astype('float64')
    for index, row in sur.iterrows():
        if row['Prod_Dem_Diff (kW)'] > 0:
            for i, col in enumerate(single_names):
                if col in single_names:
                    sur.at[index, col] = row['Prod_Dem_Diff (kW)'] * single_perc[i] / single_cons[i] if single_cons[i] != 0 else 0
            for i, col in enumerate(dependent_names):
                if col in dependent_names:
                    sur.at[index, col] = row['Prod_Dem_Diff (kW)'] * dependent_perc[i] / result_list[i]
    return sur

# Function to transform commodities produced into electrical energy
def to_electricity(perturbed_df_energies_data):
    surplus = final_production(perturbed_df_energies_data)
    result_dict = {}
    names = surplus.columns.tolist()
   
    for names in Database['Commodities_to_Electricity']:
        values = Database['Commodities_to_Electricity'][names]
        for key, val in values.items():
            ran = User['Stochastic_Processes']
            efficiency_values = val[0]
            energy_density_values = val[1]
            if len(efficiency_values) == 2 and ran:
                efficiency = random.uniform(*efficiency_values)
            else:
                efficiency = mean(efficiency_values)
            if len(energy_density_values) == 2 and ran:
                energy_density = random.uniform(*energy_density_values)
            else:
                energy_density = mean(energy_density_values)

            result = round(efficiency * energy_density, 4)
        result_dict[names] = result
        

    fuels = User['Fuels_Burning']
    for key, perc in fuels.items():
        if key in result_dict and perc <= 1.0 and perc > 0.0:
            col_name_accum = f'Accum_{key}'
            surplus[col_name_accum] = surplus[key].cumsum()
            col_name = f'{key} (kW)'
            surplus[col_name] = surplus[key] * perc * result_dict[key]
    columns_to_sum = surplus.filter(regex='(kW)').columns.difference(
        surplus.filter(like='Prod_Dem_Diff').columns).difference(
        surplus.filter(like='Accum_Diff').columns)
    surplus['Total_Potential (kW)'] = surplus[columns_to_sum].sum(axis=1)
    surplus['Accum_Total_Potential (kW)'] = surplus['Total_Potential (kW)'].cumsum()
    surplus['Accum_Burned_Potential (kW)'] = surplus['Total_Potential (kW)'].cumsum()
    surplus['Burned_Diff (kW)'] = surplus['Prod_Dem_Diff (kW)']
    reset_accum = False
    for index, row in surplus.iterrows():
        if index == 0:
            continue
        if row['Prod_Dem_Diff (kW)'] < 0:
            if reset_accum:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = 0
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)'] + row[
                    'Prod_Dem_Diff (kW)']
            else:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = surplus.at[index - 1, 'Accum_Burned_Potential (kW)'] + \
                                                                    row['Prod_Dem_Diff (kW)']
            if surplus.at[index, 'Accum_Burned_Potential (kW)'] < 0:
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)']
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = 0
                reset_accum = True
            else:
                reset_accum = False
        else:
            if reset_accum:
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)'] + row[
                    'Prod_Dem_Diff (kW)']
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = row['Total_Potential (kW)']
                reset_accum = False
            else:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = surplus.at[index - 1, 'Accum_Burned_Potential (kW)'] + \
                                                                    row['Total_Potential (kW)']
    for index, row in surplus.iterrows():
        if index == 0:
            continue
        if row['Prod_Dem_Diff (kW)'] < 0 and row['Accum_Burned_Potential (kW)'] > 0:
            surplus.at[index, 'Burned_Diff (kW)'] = 0
        if row['Prod_Dem_Diff (kW)'] > 0 and surplus.at[index - 1, 'Accum_Burned_Potential (kW)'] == 0:
            surplus.at[index, 'Burned_Diff (kW)'] = row['Prod_Dem_Diff (kW)']
    surplus['Accum_Burned_Diff (kW)'] = surplus['Burned_Diff (kW)'].cumsum()
    surplus['Covered_Deficit (kW)'] = surplus['Prod_Dem_Diff (kW)'] - surplus['Burned_Diff (kW)']
    surplus['Deficit_to_Cover(kW)'] = surplus['Burned_Diff (kW)'].apply(lambda x: x if x < 0 else 0)
    surplus['Accum_Def_to_Cover(kW)'] = surplus['Deficit_to_Cover(kW)'].cumsum()
    return surplus

# Function to export data to an Excel file
def excel(perturbed_df_energies_data,j):
    
    sur = final_production(perturbed_df_energies_data)
    surplus = to_electricity(perturbed_df_energies_data)

    sur['Date'] = pd.to_datetime(sur['Date']).dt.date
    surplus['Date'] = pd.to_datetime(sur['Date']).dt.date

    elec_unit = User['Electricity_Output'].upper()
    if elec_unit == 'GW':
        unit = 1000000
    elif elec_unit == 'MW':
        unit = 1000
    elif elec_unit == 'KW':
        unit = 1
    else:
        sys.exit("Invalid electricity unit in User.yaml")

    columns_sur = [col for col in sur.columns if 'kW' in col]
    columns_surplus = [col for col in surplus.columns if 'kW' in col]

    sur[columns_sur] = sur[columns_sur].div(unit)
    surplus[columns_surplus] = surplus[columns_surplus].div(unit)

    sur.rename(columns=lambda x: x.replace('kW', elec_unit), inplace=True)
    surplus.rename(columns=lambda x: x.replace('kW', elec_unit), inplace=True)

    for df in [sur, surplus]:
        for column in df.columns:
            if elec_unit not in column:
                if column != 'H2O' and column != 'Date':
                    df.rename(columns={column: f"{column} (Kg)"}, inplace=True)
                elif column == 'H2O':
                    df.rename(columns={column: f"{column} (Lt)"}, inplace=True)

    file_names = ['Commodities_Production_{}.xlsx'.format(j), 'Commodities_to_Electricity_{}.xlsx'.format(j)]
    for i, df in enumerate([sur, surplus]):
        path = os.path.join(current_directory, 'Output', file_names[i])
        df.to_excel(path, index=False)
    
    print("Simulation completed successfully")
def run_simulation(i, df_energies, residual_params, User, current_directory):
    forecast_df = df_energies.copy()
    start_time = time.time()
    print(f"Iteration {i + 1}:")

    # Generate the perturbed DataFrame
    if User['Stochastic_ER_Generation'] == True:
        print("Generating perturbed data for iteration", i + 1)
        perturbed_df_energies_data = perturb_df_energies(forecast_df, residual_params,User)
    else:
        perturbed_df_energies_data = forecast_df

    # Run the simulation and save the results to Excel
    excel(perturbed_df_energies_data, i)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for iteration {i + 1}: {elapsed_time:.2f} seconds\n")

def monte_carlo_simulation(num_iterations, User):
    # Load the parameters of the residual distributions
    residual_params_filepath = 'Residual_distribution.csv'
    residual_params = residual_params_dict(residual_params_filepath)
    
    if num_iterations > 1:
        print(f"Running {num_iterations} iterations of Monte Carlo simulation...\n")
        
        # Determine the number of cores to use (maximum number of available CPUs)
        num_cores = min(cpu_count(), num_iterations)
        print(f"Using {num_cores} CPU cores...\n")
        
        # Create a pool of processes to distribute the iterations
        with Pool(processes=num_cores) as pool:
            # Distribute the iterations among the cores
            pool.starmap(run_simulation, [
                (i, df_energies, residual_params, User, current_directory) 
                for i in range(num_iterations)
            ])
        
        # Output directory
        output_directory = os.path.join(current_directory, 'Output')
        
        # Analyze results
        start_time = time.time()
        analyze_monte_carlo_results(num_iterations, output_directory)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for analyze_monte_carlo_results: {elapsed_time:.2f} seconds\n")
        
        # Generate the analysis file
        analysis_file = os.path.join(output_directory, 'Monte_Carlo_Analysis.xlsx')
        analysis_df = pd.read_excel(analysis_file, index_col='Date', parse_dates=True)
        interactive_plot_script(analysis_df, output_directory, num_iterations)
    
    else:
        # If there is only one iteration, run the simulation once
        run = excel(df_energies, 1)
        
if __name__ == "__main__":
    try:
        # Load YAML configuration
        with open('User.yaml', 'r') as file:
            User = yaml.safe_load(file)
        
        # Number of simulations
        num_iterations = User.get('Number_of_simulation')
        
        # Call the Monte Carlo simulation function
        monte_carlo_simulation(num_iterations, User)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)