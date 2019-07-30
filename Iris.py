import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

#slicing and dicing operation
X = dataset.data 
y = dataset.target

#Using sepal length and sepal width
plt.scatter(X[y == 0,0] ,X[y == 0,1] , c = 'r' , label = 'Setosa')
plt.scatter(X[y == 1,0] ,X[y == 1,1] , c = 'b' , label = 'versicolour')
plt.scatter(X[y == 2,0] ,X[y == 2,1] , c = 'g' , label = 'verginica')
plt.legend()
plt.xlabel('Sepal Length' )
plt.ylabel('Sepal Width')
plt.title('Analysis on the Iris data set')
plt.show()



#Using petal length and petal width
plt.scatter(X[y == 0,2] ,X[y == 0,3] , c = 'r' , label = 'Setosa')
plt.scatter(X[y == 1,2] ,X[y == 1,3] , c = 'b' , label = 'versicolour')
plt.scatter(X[y == 2,0] ,X[y == 2,3] , c = 'g' , label = 'verginica')
plt.legend()
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Analysis ont the Iris data set')
plt.show()