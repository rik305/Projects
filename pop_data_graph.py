import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from math import e
from math import log as ln

df = read_csv("/Users/rikkumar/python/ind_pop_data.csv")

df = df.iloc[:54,:]

def logistics_eq(x):
    return (1515000000/(1 + 1.717482448 * e ** (-0.04404049946 * x)))
def gompertz_eq(x):
    return (1515000000 * e ** (-ln(2.717482448)* e ** (-0.0317697536 * x)))
def per_error(actual, predicted):
    return (actual - predicted)/actual * 100

df['predicted population for gompertz model'] = df['years from 1970'].apply(gompertz_eq)

df['predicted population for logistic model'] = df['years from 1970'].apply(logistics_eq)

df = df.drop(columns= ['predicted for logistics '])
plt.scatter(df['year'], df['population'], label = "population")
plt.plot(df['year'], df['predicted population for linear model'], label = "Linear model prediction")
plt.legend()
plt.show()


def per_error(actual, predicted):
    return (actual - predicted)/actual * 100

df['error for gompertz model'] = per_error(df['population'], df['predicted population for gompertz model'])
df['error for logistic model'] = per_error(df['population'], df['predicted population for logistic model'])

df['residual_gompertz'] = df['population'] - df['predicted population for gompertz model']
df['residual_logistic'] = df['population'] - df['predicted population for logistic model']
df['residual_linear'] = df['population'] - df['predicted population for linear model']


df.to_csv("/Users/rikkumar/desktop/ind_pop_data.csv", index = False)

from sklearn.metrics import r2_score
print(df)
print(r2_score(df['population'], df['predicted population for gompertz model']))
print(r2_score(df['population'], df['predicted population for logistic model']))
print(r2_score(df['population'], df['predicted population for linear model']))