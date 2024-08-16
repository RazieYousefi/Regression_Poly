
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from scipy.special import inv_boxcox
from scipy.stats import anderson

from statsmodels.stats.outliers_influence import variance_inflation_factor




def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_score = metrics.r2_score(y_test, y_pred)
    
    results = [mae, mse, rmse, r2_score]
    return pd.DataFrame(results, index=['MAE', 'MSE', 'RMSE', 'R2 Score'], columns=[model_name])



def calculate_residuals(model, X_test, y_test):
    """
    Generates predictions using the model and calculates residuals.
    """
    y_pred = model.predict(X_test)
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results



def check_linearity_assumption(model, X_test, y_test):
    """
    Visually inspects the linearity assumption in a linear regression model.
    """
    df_results = calculate_residuals(model, X_test, y_test)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=80)
    
    # Plotting Actual vs. Predicted Values
    sns.regplot(x='Predicted', y='Actual', data=df_results,  ax=ax[0],
                color='#0055ff', line_kws={'color':'#ff7000', 'ls':'--', 'lw':2.5})
    ax[0].set_title('Actual vs. Predicted Values', fontsize=15)
    ax[0].set_xlabel('Predicted', fontsize=12)
    ax[0].set_ylabel('Actual', fontsize=12)
    
    # Plotting Residuals vs. Predicted Values
    sns.regplot(x='Predicted', y='Residuals', data=df_results,  ax=ax[1],
                color='#0055ff', line_kws={'color':'#ff7000', 'ls':'--', 'lw':2.5})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=15)
    ax[1].set_xlabel('Predicted', fontsize=12)
    ax[1].set_ylabel('Residuals', fontsize=12)


def autocorrelation_assumption(model, X_test, y_test):
    '''
    Checks the assumption of no autocorrelation in the residuals.
    Autocorrelation indicates that residuals are not independent, which suggests that the model
    may be missing important information.
    '''
    df_results = calculate_residuals(model, X_test, y_test)

    # Calculate Durbin-Watson statistic
    durbin_watson_stat = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson Statistic:', round(durbin_watson_stat, 3))
    
    # Interpret Durbin-Watson statistic
    if durbin_watson_stat < 1.5:
        print('Possible positive autocorrelation detected. Assumption not satisfied.')
    elif durbin_watson_stat > 2.5:
        print('Possible negative autocorrelation detected. Assumption not satisfied.')
    else:
        print('No significant autocorrelation detected. Assumption satisfied.')



def homoscedasticity_assumption(model, X_test, y_test):
    """
    Checks the assumption of homoscedasticity, which states that residuals should have constant variance.
    This function plots residuals versus predicted values to visually inspect for constant variance.
    """
    print('The horizontal line should be flat if homoscedasticity assumption is met.')
    
    # Calculate residuals
    df_results = calculate_residuals(model, X_test, y_test)
    
    # Create the plot
    plt.figure(figsize=(6, 6), dpi=80)
    sns.regplot(x='Predicted', y='Residuals', data=df_results, 
                color='b', line_kws={'color':'r', 'ls':'--', 'lw':2.5})
    plt.axhline(y=0, color='k', lw=1)
    plt.title('Residuals vs. Predicted Values', fontsize=15)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.show()



def normal_anderson(residuals):
    """
    Perform the Anderson-Darling test for normality on the residuals.
    Returns the test statistic and p-value.
    """
    result = anderson(calculate_residuals, dist='norm')
    
    # Convert the significance level into a p-value-like format
    p_value = np.interp(result.statistic, result.critical_values, [0.15, 0.10, 0.05, 0.025, 0.01])
    
    # Return the test statistic and the p-value
    return result.statistic, p_value


def check_normality_of_residuals(model, X_test, y_test, p_value_threshold=0.05):
    """
    Inspects the normality assumption of residuals using the Anderson-Darling test and visualizations.
    """
    df_results = calculate_residuals(model, X_test, y_test)
    
    # Anderson-Darling Test for Normality
    p_value = normal_anderson(df_results['Residuals'])[1]
    
    print(f'\nP-value from the test (below {p_value_threshold} generally indicates non-normality): {np.round(p_value, 6)}')
    if p_value < p_value_threshold:
        print('Residuals are not normally distributed. Assumption not satisfied.') 
    else:
        print('Residuals are normally distributed. Assumption satisfied.')
      
    # Visualizations: Residuals Histogram and Q-Q Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=80)
    
    # Plotting the distribution of residuals
    sns.histplot(data=df_results, x='Residuals', kde=True, ax=ax[0], bins=15, 
                 color='#0055ff', edgecolor='none', alpha=0.4, line_kws={'lw': 2.5})
    ax[0].set_xlabel('Residuals', fontsize=12)
    ax[0].set_ylabel('Count', fontsize=12)
    ax[0].set_title('Distribution of Residuals', fontsize=15)
    
    # Displaying mean and standard deviation of residuals
    textstr = f'$\mu={np.mean(df_results["Residuals"]):.2f}$\n$\sigma={np.std(df_results["Residuals"]):.2f}$'
    ax[0].text(0.7, 0.9, textstr, transform=ax[0].transAxes, fontsize=15, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#509aff', edgecolor='black', pad=0.5))
    
    # Q-Q Probability Plot
    stats.probplot(df_results['Residuals'], dist="norm", plot=ax[1])
    ax[1].set_title("Residuals Q-Q Plot", fontsize=15)
    ax[1].set_xlabel('Theoretical Quantiles', fontsize=12)
    ax[1].set_ylabel('Ordered Values', fontsize=12)
    ax[1].get_lines()[0].set_markerfacecolor('#509aff')
    ax[1].get_lines()[1].set_color('#ff7000')
    ax[1].get_lines()[1].set_linewidth(2.5)
    ax[1].get_lines()[1].set_linestyle('--')
    ax[1].legend(['Actual', 'Theoretical'])
    
    plt.show()




def check_multicollinearity(X):
    """
    Evaluates multicollinearity among predictors by calculating Variance Inflation Factor (VIF) values.
    """
    # Calculate VIF for each predictor
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Identify possible and definite multicollinearity cases
    possible_multicollinearity = sum(v > 10 for v in vif)
    definite_multicollinearity = sum(v > 100 for v in vif)
    
    # Output results
    print(f'{possible_multicollinearity} cases of possible multicollinearity.')
    print(f'{definite_multicollinearity} cases of definite multicollinearity.')
    
    if definite_multicollinearity == 0:
        if possible_multicollinearity == 0:
            print('Assumption satisfied.')
        else:
            print('Assumption possibly satisfied.')
    else:
        print('Assumption not satisfied.')
    
    # Return VIF values in a DataFrame
    return pd.DataFrame({'VIF': vif}, index=X.columns).round(2)



