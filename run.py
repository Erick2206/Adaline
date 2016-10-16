from adaline import Adaline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/erick/Repo/Machine Learning/Adaline/iris.csv', header=None)

y=df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

adn= Adaline()
adn.fit(X,y)

setosa_example=[5.2,1.8]
versicolor_example=[6.4,4.6]

print adn.predict(setosa_example)
print adn.predict(versicolor_example)
