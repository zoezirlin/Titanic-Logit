#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:52:43 2020

@author: zoezirlin
"""

### Package and Data Importation ###



## Importing libraries of use
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import norm
import statsmodels.api as sm



## Importing the dataset "train"
df_train = pd.read_csv("/Users/zoezirlin/Desktop/Datasets/titanic/train.csv")
## Printing the first ten observations of the dataset "train"
df_train[:10]



## Importing the dataset "test
df_test = pd.read_csv('/Users/zoezirlin/Desktop/Datasets/titanic/test.csv')
## Printing the first ten observations of the dataset "test"
df_test[:10]






### Data Exploration ###


## Learning that there are 891 passengers in this set, with 12 variables assigned to each passenger
df_train.shape



## Learning that there are 418 passengers in this set, with 12 variables assigned to each passenger
df_test.shape



## Learning the descriptive statistics for the continuous variables
df_train.describe()


## Creating frequency dist. bar graph for survivorship
df_train["Survived"].value_counts().plot(kind = 'bar')
plt.title('Survivorship of the Titanic Frequency Distribution')
# We can see that more people died than survived in this training set



## Finding the proportion of those who survived and those who did not
df_train["Survived"].value_counts() 
survived = (342 / (549 + 342) ) * 100
print(survived)
# 38% of the observations in this dataset survived



## Creating bar chart for class frequency distribution
df_train["Pclass"].value_counts().plot(kind = 'bar')
plt.title('Titanic Class Frequency Distribution')
# We can see that class 3 had the most people, followed by 2 and 1 which were about equal
# Class 3 accounted for 55.1% of the observations
# Class 2 accounted for 20.6% of the observations
# Class 1 accounted for 24.2% of the observations



## Gathering value counts for class frequency
plcass_value = df_train["Pclass"].value_counts()
# Class 3 = 491 obs
# Class 2 = 184 obs
# Class 1 = 216 obs
# Total = 891



## Looking at average survival rates for female and male, "what are the odds?"
survived_sex_grouping = df_train.groupby('Sex')[['Survived']].mean()            
# Female:  0.742038: 74% of those who survived were female
# Male:    0.188908: 19% of those who survived were male



## Looking at survival by sex and class
survived_sex_pclass_pivot = df_train.pivot_table('Survived', 
                     index = 'Sex' , 
                     columns = 'Pclass')
#          	Class 1	            Class 2	            Class 3
#   female	0.9680851063829787	0.9210526315789473	0.5
#   male	0.36885245901639346	0.1574074074074074	0.13544668587896252

# Females in 1st class have a 96% chance of survival
# Males in 3rd class have a 13% chance of survival, less than 1 in 5 chance of survival
# Females in 3rd class have a 1 in 2 chance of survival



## Visualizing the pivot table above
df_train.pivot_table('Survived', 
                     index = 'Sex' , 
                     columns = 'Pclass').plot()



## Creating freq. dist. for gender
df_train["Sex"].value_counts().plot(kind = "bar")
plt.title('Titanic Sex Frequency Distribution')
# We can see that there were more men than women on the Titanic
sex_value = df_train["Sex"].value_counts()



## Creating freq. dist for survivorship by class
df_train.hist(by = "Pclass",
       column = "Survived")
# We can see that people in the highest class (1) survived proportionally...
# more than those in the lowest class (3)


## Matrix of bar plots for continuous variables
cols = ['Pclass','Sex','Parch','Embarked']

n_rows = 2
n_cols = 2

figs, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 5, n_rows * 5)) #the 5 denotes size of the graph

for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r * n_cols + c
        ax = axs[r][c]
        sns.countplot(df_train[cols[i]], hue=df_train['Survived'], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='Survived', loc = 'upper right')

plt.tight_layout()



## Plotting the survival rate of each class
sns.barplot(x = 'Pclass', y = 'Survived', data = df_train )



## Plotting the survival rate of sex
sns.barplot(x = 'Survived', y = 'Sex', data = df_train )



## Survival rate by both sex, age and class
# We are going to be segmenting age into multiple groups

Age = pd.cut(df_train['Age'],[0,18,80])
age_pivot = df_train.pivot_table('Survived',['Sex', Age],'Pclass')
# Women in 2nd class age 0-18 have a 100% chance of survival (!)
# Women in 1st class age 18-80 have a 97% chance of survival
# Men in 2nd class age 18-80 have a 7% chance of survival







### Feature Selection and Engineering ###



## Creating dummy variables for sex
# Using pandas "get_dummies" because variable is binary
df_train['Sex'] = pd.get_dummies(df_train['Sex'])
df_test['Sex'] = pd.get_dummies(df_test['Sex'])



## Creating dummy variables for embarkment
# Using map because variable is nonbinary but still categorical
df_train['Embarked'] = df_train.Embarked.map({'Q':1,
                                              'S':2,
                                              'C':3
                                              })
df_test['Embarked'] = df_test.Embarked.map({'Q':1,
                                              'S':2,
                                              'C':3
                                              })



## Correlation Matrix
corr_matrix = df_train.corr(method='pearson')
# Variable correlations of interest:
# Survived x Sex : 0.5433513806577526
# Survived x PClass : -0.33848103596101586
# Survived x Fare : 0.2573065223849618
# Survived x Embarked : 0.12675346550352637
# Survived x Parch : 0.08162940708348222
# Survived x Age : -0.07722109457217737

# Keep PClass, Sex, Fare, Embarked



## Correlation Matrix Heatmap
corr_heatmap = df_train.corr()
heatmap = sns.heatmap(
    corr_heatmap, 
    square=True
)



## Creating new dataset for regression including correlated variables
df_train = df_train.drop(columns = ['Name','Age','SibSp','Ticket','Cabin'])
df_test = df_test.drop(columns = ['Name','Age','SibSp','Ticket','Cabin'])

df_train = df_train.dropna()
df_test = df_test.dropna()






### Binary Logistic Regression Modeling and Plotting ###


## Model 1
y = df_train['Survived']
X = df_train[['Pclass','Sex','Fare','Embarked']]
X = sm.add_constant(X)

model_1 = sm.Logit(y,X)
result_1 = model_1.fit()
result_1.summary()
# Pseudo R-squ.:                  0.3025
# ------------------------------------------------------------------------------
#                 coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
#const          0.0358      0.517      0.069      0.945      -0.977       1.049
#Pclass        -0.8887      0.124     -7.164      0.000      -1.132      -0.646
#Sex            2.6391      0.186     14.217      0.000       2.275       3.003
#Fare           0.0011      0.002      0.543      0.587      -0.003       0.005
#Embarked       0.1998      0.176      1.137      0.256      -0.145       0.544

# Remove Fare and Embarked from the model (p-values > .05)
# Very high p-value for the constant



## Model 2
y = df_train['Survived']
X = df_train[['Pclass','Sex']]
X = sm.add_constant(X)

model_2 = sm.Logit(y,X)
result_2 = model_2.fit()
result_2.summary()
# Pseudo R-squ.:                  0.3010
# ------------------------------------------------------------------------------
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.6452      0.241      2.675      0.007       0.172       1.118
# Pclass        -0.9576      0.106     -9.020      0.000      -1.166      -0.749
# Sex            2.6387      0.184     14.346      0.000       2.278       2.999



# Print a confusion matrix
















