# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph

## Program:
```
Developed by: S.Kishore
RegisterNumber:  212222240050

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data =np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0], X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0], X[y==0][:,1],label="Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()
def costFuction(theta, X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return j,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFuction(theta,X_train,y)
print(j)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFuction(theta,X_train,y)
print(j)
print(grad)
def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
printf=(res.fun)
print(res.x)
def plotDecisionBoundary(theta, x, y):
  x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
  y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
  x_plot = np.c_[xx.ravel(), yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0], x[y==1][:,1], label= "admitted")
  plt.scatter(x[y==0][:,0], x[y==0][:,1], label= "not admitted")
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/653e47bc-6072-461d-9e0b-bc342a147e87)

![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/aea94908-e62f-4f89-b507-fa133e023c4c)

![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/f48866eb-d81c-4284-a3b7-4b0f01d3f022)

![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/6a4458a2-6acf-4277-9b73-3e18821bfa38)

![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/8e3192cd-a92c-4a76-84e9-86d9f387c0d3)

![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/6fba9fab-88b4-4ecd-ae98-13fb1879d343)

![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/dd5d9aae-d641-497b-93ae-9c33aadd8cec)

![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/a8e7deb7-a9d6-40b1-9b13-e7ee32978d9b)

![image](https://github.com/Kishore2o/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679883/15b1668e-ef9c-4cb5-b2a2-a6bae2cbac4d)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

