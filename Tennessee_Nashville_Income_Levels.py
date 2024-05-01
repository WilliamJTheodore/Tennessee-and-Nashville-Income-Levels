# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:43:25 2024

@author: William Theodore
"""

#%%
import pandas as pd
import numpy as np 
import statsmodels.formula.api as smf 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import f
from scipy.stats import t

#%%
df = pd.read_csv('TN_Exercise.csv')

# Drops duplicte observations based on the 'SERIAL' variable
df = df.drop_duplicates(subset=['SERIAL'])
# Drops obervations where total income is 0
df = df[df['INCTOT'] != 0]

#%%
# Creates new dataframe to only Nasvhille-Davidson County-Murfreesboro
nashville_df = df[df['METAREA'] == 5361]
print("Number of observations in Nashville dataframe:", len(nashville_df))

#%%
# Calculates median income and both upper and lower quartile incomes of Nashville
nashville_median_income = nashville_df['INCTOT'].median()
nashville_q1 = nashville_df['INCTOT'].quantile(0.25)
nashville_q3 = nashville_df['INCTOT'].quantile(0.75)

# Calculates median income and both upper and lower quartile incomes of Tennessee
tennessee_median_income = df['INCTOT'].median()
tennessee_q1 = df['INCTOT'].quantile(0.25)
tennessee_q3 = df['INCTOT'].quantile(0.75)

print("Median yearly income for Nashville:", nashville_median_income)
print("1st quartile for Nashville:", nashville_q1)
print("3rd quartile for Nashville:", nashville_q3)

print("Median yearly income for Tennessee:", tennessee_median_income)
print("1st quartile for Tennessee:", tennessee_q1)
print("3rd quartile for Tennessee:", tennessee_q3)

print(f'The median yearly income of Nashivlle is ${nashville_median_income - tennessee_median_income} greater than that of Tennessee. Additionaly, the first quartile of income for Nashville is ${nashville_q1 - tennessee_q1} greater than Tennessee and the third quartile of income for Nasvhille is ${nashville_q3 - tennessee_q3} greater than that of Tennessee. Therefore, on average, workers in Nashville make more money than workers across Tennessee.')

#%%
df = df[df['INCTOT'] > 0]
nashville_df = nashville_df[nashville_df['INCTOT'] > 0]

# Creates logged income variable for both dataframes
df['ln_INCTOT'] = np.log(df['INCTOT'])
nashville_df['ln_INCTOT'] = np.log(nashville_df['INCTOT'])

#%%
# Creates dummy variables for race and gender for Tennessee
df = pd.get_dummies(df, columns=['RACE'])
df['Female'] = (df['SEX'] == 2).astype(int)

# Creates dummy variables for race and gender for Nashville
nashville_df = pd.get_dummies(nashville_df, columns = ['RACE'])
nashville_df['Female'] = (nashville_df['SEX'] == 2).astype(int)

#%%
# Regression of logged income on female dummy variable in Tennessee
reg = smf.ols('ln_INCTOT ~ Female', data = df, missing='drop').fit()
print(reg.summary2())

# Regression of logged income on female dummy variable in Nashville
reg2 = smf.ols('ln_INCTOT ~ Female', data = nashville_df, missing='drop').fit()
print(reg2.summary2())  

print('In Tennesse, females have 71.56% lower wages than males. In Nashville, females have 82.22 lower wages than males.')

#%%
# Including education, gender dummy variable, industry, age dummy variable, and race dummy variable
# Tennessee Regression
reg3 = smf.ols('ln_INCTOT ~ Female + EDUC + RACE_100 + IND + AGE', data = df, missing='drop').fit()
print(reg3.summary2())

# Nashville Regression
reg4 = smf.ols('ln_INCTOT ~ Female + EDUC + RACE_100 + IND + AGE', data = nashville_df, missing='drop').fit()
print(reg4.summary2())

print('In Tennessee, the coefficient on the Female variable changed to -.7722, which is more negative than the previous regression. Therefore, when other variables are added to the model, females are predicted to have a lower wage compared to males than the model without those variables.')
print('In Nasvhille, the coefficient on the Female variable changed to -.8651, which is more negative than the previous regression. Therefore, when other variables are added to the model, females are predicted to have a lower wage compared to males than the modl without those variables.')
print('The coefficients on the Female variable are more negative because the relationship between sex and wage is stronger when considering the effects of other variables like age. This is because excluding those variables from the regression might have caused omitted variable bias, therefore atering the slope coefficient of the Female variable.')
#%%
# Calculting correlation between the variables of interest in the regression
corr_tn = df[['ln_INCTOT', 'Female', 'EDUC', 'RACE_100', 'IND', 'AGE']].corr()

# Heatmap displaying the strength of correlation between the variables
sns.heatmap(corr_tn, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix - Tennessee")
plt.show()

print('The variables that exhibit the strongest correlation are IND (industry) and AGE. They have a negative correlation of -.44.')


