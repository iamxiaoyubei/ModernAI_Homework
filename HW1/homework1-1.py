#####run this command to use######
# python3 homework1-1.py
##################################
# (1) generate sin(2*pi*x)
import numpy as np
import matplotlib.pyplot as plt
size = 10
x = np.linspace(0., 1., size)
s = np.sin(2*np.pi*x)
plt.figure(1)
plt.subplot(221)
plt.scatter(x, s)


# (2) add gaussian noise on s(n)
mu = 0
sigma = 0.2
w = np.random.normal(mu, sigma, size)
s = s + w
plt.subplot(222)
plt.scatter(x, s)
plt.show()

# (3) polynomials fitting
# Figure 1.4 Plots of polynomials having various orders M
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
x_test = np.linspace(0, 1, 100)
y_test = np.sin(2*np.pi*x_test)
X_train = x[:, np.newaxis]
X_test = x_test[:, np.newaxis]
plt.figure(2)
for i, degree in enumerate([0,1,3,9]):
    plt.subplot(2, 2, i + 1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, s)
    predict = model.predict(X_test)
    plt.scatter(x, s)
    plt.plot(x_test, y_test, c="g")
    plt.plot(x_test, predict, c="r")
    plt.ylim(-1.5, 1.5)
    plt.annotate("M={}".format(degree), xy=(0.8, 1))
plt.show()

# Figure 1.6 using the M = 9 polynomial for N = 15 data points (left plot) and N = 100 data points
def generate_data(x, size):
    return np.sin(2*np.pi*x) + np.random.normal(mu, sigma, size)

x_train_A = np.linspace(0, 1, 15)
x_train_B = np.linspace(0, 1, 100)
y_train_A = generate_data(x_train_A, 15)
y_train_B = generate_data(x_train_B, 100)
X_train_A = x_train_A[:, np.newaxis]
X_train_B = x_train_B[:, np.newaxis]
plt.figure(3)
plt.subplot(221)
model = make_pipeline(PolynomialFeatures(9), LinearRegression())
model.fit(X_train_A, y_train_A)
predict = model.predict(X_test)
plt.scatter(x_train_A, y_train_A)
plt.plot(x_test, y_test, c="g")
plt.plot(x_test, predict, c="r")
plt.ylim(-1.5, 1.5)
plt.subplot(222)
model.fit(X_train_B, y_train_B)
predict = model.predict(X_test)
plt.scatter(x_train_B, y_train_B)
plt.plot(x_test, y_test, c="g")
plt.plot(x_test, predict, c="r")
plt.ylim(-1.5, 1.5)
plt.show()


# Figure 1.7 Plots of M = 9 polynomials for two values of the regularization parameter \ 
# λ corresponding to ln λ = −18 and ln λ = 0.
import math
from sklearn.linear_model import Ridge
plt.figure(4)
plt.subplot(221)
model = make_pipeline(PolynomialFeatures(9), Ridge(solver='lsqr',alpha=math.exp(-18)))
model.fit(X_train, s)
predict = model.predict(X_test)
plt.scatter(x, s)
plt.plot(x_test, y_test, c="g")
plt.plot(x_test, predict, c="r")
plt.ylim(-1.5, 1.5)
plt.subplot(222)
model = make_pipeline(PolynomialFeatures(9), Ridge(solver='lsqr',alpha=math.exp(0)))
model.fit(X_train, s)
predict = model.predict(X_test)
plt.scatter(x, s)
plt.plot(x_test, y_test, c="g")
plt.plot(x_test, predict, c="r")
plt.ylim(-1.5, 1.5)
plt.show()

# Reference
# https://scikit-learn.org/stable/modules/linear_model.html