import pandas as pd
from statsmodels.tsa.seasonal import STL
from fitter import Fitter
import matplotlib.pyplot as plt
import os
from copulas.multivariate import GaussianMultivariate
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def create_output_directory():
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
            
            # Almacenar los residuos en el DataFrame
            if column not in combined_stl_results:
                combined_stl_results[column] = result.resid  # Almacenar directamente los residuos como Series
            # Graficar los componentes
            plt.figure(figsize=(12, 8))
            plt.subplot(4, 1, 1)
            plt.plot(df[column], label='Original')
            plt.title(f'{column} - Descomposición STL)')
            plt.legend()
            
            plt.subplot(4, 1, 2)
            plt.plot(result.trend, label='Tendencia')
            plt.legend()
            
            plt.subplot(4, 1, 3)
            plt.plot(result.seasonal, label='Estacionalidad')
            plt.legend()
            
            plt.subplot(4, 1, 4)
            plt.plot(result.resid, label='Residual')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig (f'{column}label.png')
            plt.close()
    # Convertir el diccionario a un DataFrame
    residuals_df = pd.DataFrame(combined_stl_results)
    tukey_data = pd.DataFrame(columns=['Residual', 'Group'])
    
    for column in residuals_df.columns:
        temp_df = pd.DataFrame({
            'Residual': residuals_df[column].dropna(),
            'Group': column
        })
        tukey_data = pd.concat([tukey_data, temp_df], ignore_index=True)
    # Realizar el análisis de Tukey
    tukey_results = pairwise_tukeyhsd(endog=tukey_data['Residual'], groups=tukey_data['Group'], alpha=0.05)
    
    # Imprimir los resultados
    print(tukey_results)
    
    # También puedes graficar los resultados
    tukey_results.plot_simultaneous()
    plt.show()
        
    return residuals_df

def fit_distribution_with_fitter(data, column_name, component_name, output_dir):
    f = Fitter(data, distributions=['t','tukeylambda'])  # Simplificado para el ejemplo
    f.fit()
    
    print(f"\nBest distributions for {column_name} - {component_name}:")
    print(f.summary())
    
    plt.figure(figsize=(10,6))
    plt.hist(data, bins=50, density=True, alpha=0.5, label=component_name)
    f.plot_pdf()
    plt.plot([], [], ' ', label='Fit')
    plt.title(f'{column_name} - {component_name}')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f'{column_name}_{component_name}_distribution_fit.png'))
    plt.close()

    best_dist_dict = f.get_best(method='sumsquare_error')
    best_dist_name = list(best_dist_dict.keys())[0]
    best_dist_params = list(best_dist_dict.values())[0]
    
    return best_dist_name, best_dist_params

def analyze_distributions_with_fitter(residuals_df):
    output_dir = create_output_directory()
    best_fits = {}
    residual_parameters = []

    for column in residuals_df.columns:
        best_fits[column] = {}
        
        # Asegúrate de que los residuos sean numéricos
        combined_data = residuals_df[column].dropna()  # Eliminar NaN antes de ajustar la distribución

        # Ajustar la distribución
        best_dist_name, best_dist = fit_distribution_with_fitter(combined_data, column, 'residual', output_dir)
        best_fits[column]['residual'] = (best_dist_name, best_dist)
        
        # Almacenar parámetros residuales
        params_dict = {'column': column, 'distribution': best_dist_name}
        params_dict.update(best_dist)
        residual_parameters.append(params_dict)
    
    # Exportar parámetros residuales a CSV
    residual_df = pd.DataFrame(residual_parameters)
    residual_df.to_csv(os.path.join(output_dir, 'residual_distribution.csv'), index=False)
    
    return best_fits

def fit_copula_to_residuals(residuals_df):
    # Ajustar la copula Gaussiana
    copula = GaussianMultivariate()
    copula.fit(residuals_df)
    # Revisar la matriz de correlación original
    model_dict = copula.to_dict()
    print(model_dict)
    original_correlation_matrix = residuals_df.corr()
    print(original_correlation_matrix)
    # Simular nuevos datos
    simulated_data = copula.sample(10)  # Simular 1000 muestras
    print("Distribuciones univariadas ajustadas y sus parámetros:")
    for i, distribution in enumerate(copula.univariates):
        column = residuals_df.columns[i]  # Obtener el nombre de la columna correspondiente
        print(f"{column}: {distribution}")  # Imprimir la distribución ajustada
        print(f"  Tipo de distribución: {type(distribution)}")
        # Obtener parámetros de la distribución
        if hasattr(distribution, 'params'):
            print(f"  Parámetros: {distribution.params}")  # Imprimir parámetros si están disponibles
        else:
            print("  No se encontraron parámetros disponibles.")
    simulated_correlation_matrix = simulated_data.corr()
    print(simulated_correlation_matrix)
    return simulated_data