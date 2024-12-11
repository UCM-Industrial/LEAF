import pandas as pd
from statsmodels.tsa.seasonal import STL
from fitter import Fitter
import matplotlib.pyplot as plt
import os

def create_output_directory():
    # Create the 'output' folder if it does not exist
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def stl_decomposition(input_csv_list):
    combined_stl_results = {}

    for input_csv in input_csv_list:
        df = pd.read_csv(input_csv, parse_dates=['Date'], index_col='Date')
        
        for column in df.columns:
            stl = STL(df[column], seasonal=13, robust=True)
            result = stl.fit()
            
            if column not in combined_stl_results:
                combined_stl_results[column] = {
                    'trend': [],
                    'seasonal': [],
                    'residual': []
                }
            
            combined_stl_results[column]['trend'].append(result.trend)
            combined_stl_results[column]['seasonal'].append(result.seasonal)
            combined_stl_results[column]['residual'].append(result.resid)
    
    return combined_stl_results

def fit_distribution_with_fitter(data, column_name, component_name, output_dir):
    # Create a Fitter object and fit distributions to the data
    f = Fitter(data, distributions=[
        'alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_l', 'frechet_r', 'gamma', 'gausshyper', 'genexpon', 'genextreme', 'gengamma', 'genhalflogistic', 'geninvgauss', 'genlogistic', 'gennorm', 'genpareto', 'gilbrat', 'gompertz', 'gumbel_l', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm',
        'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa3', 'kappa4', 'ksone', 'kstwo', 'kstwobign', 'laplace', 'levy', 'levy_l', 'levy_stable', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform', 'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncf', 'nct', 'ncx2',
        'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rayleigh', 'rdist', 'recipinvgauss', 'reciprocal', 'rice', 'rv_continuous', 'rv_histogram', 'semicircular', 'skewnorm', 't', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_max', 'weibull_min', 'wrapcauchy'])
    f.fit()
    
    # Show summary and best distributions
    print(f"\nBest distributions for {column_name} - {component_name}:")
    print(f.summary())
    
    # Plot the results and save the figure in the 'output' folder
    plt.figure(figsize=(10,6))
    plt.hist(data, bins=50, density=True, alpha=0.5, label=component_name)
    f.plot_pdf()
    plt.plot([], [], ' ', label='Fit')  # Create an empty legend for the fit
    plt.title(f'{column_name} - {component_name}')
    plt.legend()
    
    # Save the figure in the 'output' folder
    plt.savefig(os.path.join(output_dir, f'{column_name}_{component_name}_distribution_fit.png'))
    plt.close()  # Close the figure to not show it immediately

    # Get the best distribution
    best_dist_dict = f.get_best(method='sumsquare_error')
    
    # Get the name of the distribution and the parameters
    best_dist_name = list(best_dist_dict.keys())[0]
    best_dist_params = list(best_dist_dict.values())[0]
    
    return best_dist_name, best_dist_params

def analyze_distributions_with_fitter(stl_results):
    output_dir = create_output_directory()  # Create the 'output' folder
    best_fits = {}
    residual_parameters = []

    for column, components in stl_results.items():
        best_fits[column] = {}
        
        for component_name, data_list in components.items():
            combined_data = pd.concat(data_list)  # Combine results from multiple CSVs
            
            best_dist_name, best_dist = fit_distribution_with_fitter(combined_data, column, component_name, output_dir)
            best_fits[column][component_name] = (best_dist_name, best_dist)
            
            # If it's a residual component, store the parameters and the distribution name
            if component_name == 'residual':
                params_dict = {'column': column, 'distribution': best_dist_name}
                params_dict.update(best_dist)
                residual_parameters.append(params_dict)
    
    # Export the residual parameters to a CSV in the 'output' folder
    residual_df = pd.DataFrame(residual_parameters)
    residual_df.to_csv(os.path.join(output_dir, 'residual_distribution.csv'), index=False)
    
    return best_fits