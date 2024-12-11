import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

# Set a visual style for Matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

# # Function to format axes without scientific notation and with correct dates
def format_axis(ax):
    # Format for the Y-axis numeric values (without scientific notation)
    formatter = FuncFormatter(lambda x, _: f'{x:,.0f}')  
    ax.yaxis.set_major_formatter(formatter)

    # Format for the X-axis dates (only show years)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))  # Labels every year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Show only the year
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor labels by month
    
# function to plot all iterations for selected columns
def plot_all_iterations_for_columns(num_iterations, output_dir, selected_columns):
    plt.figure(figsize=(12, 6))
    colors = plt.get_cmap('tab20')  # Paleta de colores más amplia

    for i in range(num_iterations):
        file_path = os.path.join(output_dir, f'Commodities_to_Electricity_{i}.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            for idx, col in enumerate(selected_columns):
                if col in df.columns:
                    color = colors((i * len(selected_columns) + idx) % 20)  # Asignar un color único
                    plt.plot(df['Date'], df[col], label=f'Iteration {i} - {col}', color=color, linewidth=1.5)
        else:
            print(f"File {file_path} does not exist. Skipping iteration {i}.")

    # Adjust the title of the plot
    plt.title('All Iterations for Selected Columns: \n' + ', '.join(selected_columns), fontsize=18, fontweight='bold')

    # Adjust the axes
    ax = plt.gca()  # Get the current axes
    format_axis(ax)

    # Rotate the X-axis labels if they are dates for better readability
    plt.xticks(rotation=45, ha='right')

    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, borderpad=1, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_filename = '_'.join(selected_columns) + '_all_iterations_plot.svg'  # Save in vector format
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, format='svg', dpi=300)  # Save in SVG
    plt.show()
    plt.close()

    print(f"Plot for all iterations saved to {plot_path}")


# Function to analyze Monte Carlo results
def analyze_monte_carlo_results(num_iterations, output_dir):
    simulation_results = []

    for i in range(num_iterations):
        file_path = os.path.join(output_dir, f'Commodities_to_Electricity_{i}.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            simulation_results.append(df)
        else:
            print(f"File {file_path} does not exist. Skipping iteration {i}.")

    if not simulation_results:
        print("No simulation results found. Exiting.")
        return None

    combined_df = pd.concat(simulation_results, keys=range(num_iterations), names=['Iteration', 'Index'])
    grouped = combined_df.groupby('Date')
    means = grouped.mean()
    stds = grouped.std()
    result_df = pd.concat([means.add_suffix('_mean'), stds.add_suffix('_std')], axis=1)
    output_file = os.path.join(output_dir, 'Monte_Carlo_Analysis.xlsx')
    result_df.to_excel(output_file)

    print(f"Monte Carlo analysis completed. Results saved to {output_file}")
    return result_df

# Function to plot selected columns from the analysis
def plot_selected_columns_script(analysis_df, selected_columns, output_dir):
    plt.figure(figsize=(12, 6))
    colors = plt.get_cmap('Set2')

    for idx, col in enumerate(selected_columns):
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col in analysis_df.columns and std_col in analysis_df.columns:
            color = colors(idx)  # Assign color from the palette

            plt.plot(analysis_df.index, analysis_df[mean_col], label=f'{col} Mean', color=color, linewidth=2.5)
            plt.fill_between(
                analysis_df.index,
                analysis_df[mean_col] - analysis_df[std_col],
                analysis_df[mean_col] + analysis_df[std_col],
                color=color, alpha=0.2, label=f'{col} Std Dev'
            )

    # Adjust the title of the plot
    plt.title('Monte Carlo Simulation Results: \n' + ', '.join(selected_columns), fontsize=18, fontweight='bold')

    # Adjust the axes
    ax = plt.gca()  # Get the current axes
    format_axis(ax)

     # Rotate the X-axis labels if they are dates for better readability
    plt.xticks(rotation=45, ha='right')

    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_filename = '_'.join(selected_columns) + '_plot.svg'  # Save in vector format
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, format='svg', dpi=300)  # Save in SVG
    plt.show()
    plt.close()

    print(f"Plot saved to {plot_path}")

# Interactive function
def interactive_plot_script(analysis_df, output_dir, num_iterations):
    mean_columns = [col for col in analysis_df.columns if col.endswith('_mean')]
    base_columns = [col.replace('_mean', '') for col in mean_columns]

    while True:
        print("\nAvailable options:")
        print("1. Plot selected columns (mean and std dev)")
        print("2. Plot all iterations for selected columns")
        option = input("Enter the number of the option you want (or type 'exit' to finish): ").strip()

        if option.lower() == 'exit':
            break

        if option == '1':
            print("\nAvailable columns:")
            for idx, col in enumerate(base_columns, 1):
                print(f"{idx}. {col}")

            selected_numbers = input("Enter the numbers of columns to plot, separated by commas: ").strip()
            selected_indices = [int(num.strip()) - 1 for num in selected_numbers.split(',') if num.strip().isdigit()]
            selected_columns = [base_columns[idx] for idx in selected_indices if 0 <= idx < len(base_columns)]

            if selected_columns:
                plot_selected_columns_script(analysis_df, selected_columns, output_dir)
            else:
                print("Invalid selection. Please try again.")
        elif option == '2':
            print("\nAvailable columns:")
            for idx, col in enumerate(base_columns, 1):
                print(f"{idx}. {col}")

            selected_numbers = input("Enter the numbers of columns to plot, separated by commas: ").strip()
            selected_indices = [int(num.strip()) - 1 for num in selected_numbers.split(',') if num.strip().isdigit()]
            selected_columns = [base_columns[idx] for idx in selected_indices if 0 <= idx < len(base_columns)]

            if selected_columns:
                plot_all_iterations_for_columns(num_iterations, output_dir, selected_columns)
            else:
                print("Invalid selection. Please try again.")
        else:
            print("Invalid option. Please try again.")
