import numpy as np
import pandas as pd
from scipy.stats import *
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
# Create a dictionary of distribution functions
distribution_functions = {
    'alpha': alpha.rvs,
    'anglit': anglit.rvs,
    'arcsine': arcsine.rvs,
    'argus': argus.rvs,
    'beta': beta.rvs,
    'betaprime': betaprime.rvs,
    'bradford': bradford.rvs,
    'burr': burr.rvs,
    'cauchy': cauchy.rvs,
    'chi': chi.rvs,
    'chi2': chi2.rvs,
    'cosine': cosine.rvs,
    'crystalball': crystalball.rvs,
    'dgamma': dgamma.rvs,
    'dweibull': dweibull.rvs,
    'erlang': erlang.rvs,
    'expon': expon.rvs,
    'exponnorm': exponnorm.rvs,
    'exponpow': exponpow.rvs,
    'exponweib': exponweib.rvs,
    'f': f.rvs,
    'fatiguelife': fatiguelife.rvs,
    'fisk': fisk.rvs,
    'foldcauchy': foldcauchy.rvs,
    'foldnorm': foldnorm.rvs,
    'gamma': gamma.rvs,
    'gausshyper': gausshyper.rvs,
    'genexpon': genexpon.rvs,
    'genextreme': genextreme.rvs,
    'gengamma': gengamma.rvs,
    'genhalflogistic': genhalflogistic.rvs,
    'geninvgauss': geninvgauss.rvs,
    'genlogistic': genlogistic.rvs,
    'gennorm': gennorm.rvs,
    'genpareto': genpareto.rvs,
    'gompertz': gompertz.rvs,
    'gumbel_l': gumbel_l.rvs,
    'gumbel_r': gumbel_r.rvs,
    'halfc cauchy': halfcauchy.rvs,
    'halfgennorm': halfgennorm.rvs,
    'halflogistic': halflogistic.rvs,
    'halfnorm': halfnorm.rvs,
    'hypsecant': hypsecant.rvs,
    'invgamma': invgamma.rvs,
    'invgauss': invgauss.rvs,
    'invweibull': invweibull.rvs,
    'johnsonsb': johnsonsb.rvs,
    'johnsonsu': johnsonsu.rvs,
    'kappa3': kappa3.rvs,
    'kappa4': kappa4.rvs,
    'ksone': ksone.rvs,
    'kstwo': kstwo.rvs,
    'kstwobign': kstwobign.rvs,
    'laplace': laplace.rvs,
    'levy': levy.rvs,
    'levy_l': levy_l.rvs,
    'levy_stable': levy_stable.rvs,
    'loggamma': loggamma.rvs,
    'logistic': logistic.rvs,
    'loglaplace': loglaplace.rvs,
    'lognorm': lognorm.rvs,
    'loguniform': loguniform.rvs,
    'lomax': lomax.rvs,
    'maxwell': maxwell.rvs,
    'mielke': mielke.rvs,
    'moyal': moyal.rvs,
    'nakagami': nakagami.rvs,
    'ncf': ncf.rvs,
    'nct': nct.rvs,
    'ncx2': ncx2.rvs,
    'norm': norm.rvs,
    'norminvgauss': norminvgauss.rvs,
    'pareto': pareto.rvs,
    'pearson3': pearson3.rvs,
    'powerlaw': powerlaw.rvs,
    'powerlognorm': powerlognorm.rvs,
    'powernorm': powernorm.rvs,
    'rayleigh': rayleigh.rvs,
    'rdist': rdist.rvs,
    'recipinvgauss': recipinvgauss.rvs,
    'reciprocal': reciprocal.rvs,
    'rice': rice.rvs,
    'rv_continuous': rv_continuous.rvs,
    'rv_histogram': rv_histogram.rvs,
    'semicircular': semicircular.rvs,
    'skewnorm': skewnorm.rvs,
    't': t.rvs,
    'trapz': trapz.rvs,
    'triang': triang.rvs,
    'truncexpon': truncexpon.rvs,
    'truncnorm': truncnorm.rvs,
    'tukeylambda': tukeylambda.rvs,
    'uniform': uniform.rvs,
    'vonmises': vonmises.rvs,
    'vonmises_line': vonmises_line.rvs,
    'wald': wald.rvs,
    'weibull_max': weibull_max.rvs,
    'weibull_min': weibull_min.rvs,
    'wrapcauchy': wrapcauchy.rvs
}

def residual_params_dict(params_csv):
    # Read the CSV file
    params_df = pd.read_csv(params_csv)

    # Initialize the residual parameters dictionary
    residual_params_dict = {}

    # Iterate over each row of the DataFrame and build the dictionary
    for index, row in params_df.iterrows():
        column = row['column']
        distribution = row['distribution']

        # Create a dictionary for the distribution parameters
        params = {}

        # Add the parameters that are present and not null, omitting the first two columns
        for col in params_df.columns[2:]:  
            if pd.notna(row[col]):
                params[col] = row[col]

        # Add the distribution and its parameters to the dictionary
        residual_params_dict[column] = {
            'distribution': distribution,
            **params  # Unpack the parameters
        }

    
    return residual_params_dict


def perturb_df_energies(forecast_df, residual_params_dict, User):
    Sigma = User.get('Variation_range')
    technologys = [col for col in forecast_df.columns if col in residual_params_dict and 'trend' not in col.lower()]
    num_samples = len(forecast_df)

    # Verificar si todas las tecnologías tienen distribución normal
    all_norm = all(residual_params_dict[tech]['distribution'] == 'norm' for tech in technologys)
    all_t_student = all(residual_params_dict[tech]['distribution'] == 't' for tech in technologys)
    if all_norm:
        # Crear un array de medias y una matriz de covarianza
        means = np.array([residual_params_dict[tech]['loc'] for tech in technologys])
        cov_matrix = np.diag([residual_params_dict[tech]['scale']**2 for tech in technologys])
        print("n")
        # Generar muestras aleatorias usando multivariate_normal
        perturbations = multivariate_normal.rvs(mean=means, cov=cov_matrix, size=num_samples)
        perturbations = pd.DataFrame(perturbations, columns=technologys, index=forecast_df.index)

        # Aplicar el clipping basado en Sigma
        for tech in technologys:
            mean = residual_params_dict[tech]['loc']
            std_dev = residual_params_dict[tech]['scale']
            lower_bound = mean - Sigma * std_dev
            upper_bound = mean + Sigma * std_dev
            perturbations[tech] = np.clip(perturbations[tech], lower_bound, upper_bound)
    elif all_t_student:
        print("t")
        means = np.array([residual_params_dict[tech]['loc'] for tech in technologys])
        cov_matrix = np.diag([residual_params_dict[tech]['scale']**2 for tech in technologys])
        print(cov_matrix)
        # Generar muestras aleatorias usando multivariate_t
        perturbations = multivariate_t.rvs(loc=means, shape=cov_matrix, df=5, size=num_samples)
        perturbations = pd.DataFrame(perturbations, columns=technologys, index=forecast_df.index)

        # Aplicar el clipping basado en Sigma
        for tech in technologys:
            mean = residual_params_dict[tech]['loc']
            std_dev = residual_params_dict[tech]['scale']
            lower_bound = mean - Sigma * std_dev
            upper_bound = mean + Sigma * std_dev
            perturbations[tech] = np.clip(perturbations[tech], lower_bound, upper_bound)

    else:
        print("o")
        # Crear un DataFrame para almacenar las perturbaciones
        perturbations = pd.DataFrame(index=forecast_df.index, columns=technologys)

        for technology in technologys:
            # Obtener la distribución y sus parámetros
            dist_name = residual_params_dict[technology]['distribution']
            params = {k: v for k, v in residual_params_dict[technology].items() if k != 'distribution'}

            # Obtener la función de muestreo de la distribución
            perturbation_func = distribution_functions.get(dist_name)
            if perturbation_func:
                try:
                    # Generar la perturbación usando la función de muestreo
                    perturbation = perturbation_func(size=num_samples, **params)
                except Exception as e:
                    print(f"Error generating perturbation for {technology}: {e}")
                    perturbation = np.zeros(num_samples)  # Por defecto a cero si hay un error

                # Aplicar el clipping basado en Sigma
                mean = params['loc']
                std_dev = params['scale']
                lower_bound = mean - Sigma * std_dev
                upper_bound = mean + Sigma * std_dev
                perturbation = np.clip(perturbation, lower_bound, upper_bound)
                perturbations[technology] = perturbation
            else:
                print(f"Distribution '{dist_name}' not found for technology '{technology}'.")
                perturbations[technology] = np.zeros(num_samples)  # Por defecto a cero si la distribución no se encuentra

    # Aplicar las perturbaciones al DataFrame de pronóstico
    for technology in technologys:
        forecast_df[technology] = forecast_df[technology] + perturbations[technology]
        forecast_df[technology] = forecast_df[technology].apply(lambda x: max(0, x))  # Asegurar que los valores no sean negativos

    # Calcular la columna 'Total' si existe
    technologys_para_total = [col for col in technologys if col not in ['Nuclear_Centrals_trend', 'Demand']]
    if 'Nuclear' in forecast_df.columns:
        technologys_para_total.append('Nuclear')
    if 'Total' in forecast_df.columns:
        forecast_df['Total'] = forecast_df[technologys_para_total].sum(axis=1)

    return forecast_df