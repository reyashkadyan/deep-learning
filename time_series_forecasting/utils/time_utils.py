import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan

def boxcox_transform(series):
    # Ensure data is positive
    if any(series <= 0):
        shift = abs(series.min()) + 1
        series_shifted = series + shift
        transformed_data, lambda_value = boxcox(series_shifted)
        return transformed_data, lambda_value, shift
    else:
        transformed_data, lambda_value = boxcox(series)
        return transformed_data, lambda_value, 0

def test_stationarity(series, series_name):
    result = adfuller(series)
    print(f'\nADF Statistic for {series_name}: {result[0]}')
    print(f'p-value for {series_name}: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
    if result[1] < 0.05:
        print(f"The {series_name} series is stationary.")
    else:
        print(f"The {series_name} series is non-stationary.")

def test_homoscedasticity(y, X):
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    bp_test = het_breuschpagan(residuals, model.model.exog)
    labels = ['LM Statistic', 'LM Test p-value', 'F-Statistic', 'F-Test p-value']
    test_results = dict(zip(labels, bp_test))
    print('\nBreusch-Pagan Test Results:')
    for key in test_results:
        print(f'{key}: {test_results[key]}')
    if test_results['LM Test p-value'] < 0.05:
        print("Residuals are heteroscedastic.") # reject the null-hypothesis of homoscedasticity
    else:
        print("Residuals are homoscedastic.")
    return residuals

def plot_residuals_vs_fitted(y, residuals, xlabel='Fitted Values'):
    fitted_values = y - residuals
    plt.scatter(fitted_values, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs {xlabel}')
    plt.show()