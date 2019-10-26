#####run this command to use######
# python3 homework1-2.py
##################################
# Generate n = 2,000 points uniformly at random in the two-dimensional unit square.
import matplotlib.pyplot as plt
import numpy as np
plt.figure(1)
points = np.random.rand(2000,2)
plt.plot(points[:,0],points[:,1],'.',color='b')
plt.show()

# What objective does the centroid of the points optimize
def cost(centroid):
    sum = 0
    for i in range(2000):
        sum += ((centroid[0]-points[i,0])**2+(centroid[1]-points[i,1])**2)**0.5
    return sum

# Apply gradient descent (GD) to find the centroid
def gradient_descent(centroid):
    sum_dx = 0
    sum_dy = 0
    for i in range(2000):
        divided = ((centroid[0]-points[i,0])**2+(centroid[1]-points[i,1])**2)**0.5
        sum_dx += ((centroid[0]-points[i,0])/divided)
        sum_dy += ((centroid[1]-points[i,1])/divided)
    sum = (sum_dx**2+sum_dy**2)**0.5
    dx = sum_dx/sum
    dy = sum_dy/sum
    return dx, dy
    
centroid = np.random.rand(2)
theta = 0.01
max_loop = 100
plt.figure(2)
plt.plot(points[:,0],points[:,1],'.',color='b')
for i in range(max_loop):
    print("Centroid is:", centroid)
    print("Cost is:", cost(centroid))
    plt.plot(centroid[0],centroid[1],'.', color='g')
    dx, dy = gradient_descent(centroid)
    centroid[0] = centroid[0] - theta * dx
    centroid[1] = centroid[1] - theta * dy

print("Final Centroid is:", centroid)
print("Final Cost is:", cost(centroid))
plt.plot(centroid[0],centroid[1],'.', color='r')
plt.show()

# Apply stochastic gradient descent (SGD) to find the centroid
from random import choice
def stochastic_gradient_descent(centroid):
    point = choice(points)
    divided = ((centroid[0]-point[0])**2+(centroid[1]-point[1])**2)**0.5
    sum_dx = ((centroid[0]-point[0])/divided)
    sum_dy = ((centroid[1]-point[1])/divided)
    sum = (sum_dx**2+sum_dy**2)**0.5
    dx = sum_dx/sum
    dy = sum_dy/sum
    return dx, dy

centroid = np.random.rand(2)
theta = 0.01
max_loop = 100
plt.figure(3)
plt.plot(points[:,0],points[:,1],'.',color='b')
for i in range(max_loop):
    print("Centroid is:", centroid)
    print("Cost is:", cost(centroid))
    plt.plot(centroid[0],centroid[1],'.', color='g')
    dx, dy = stochastic_gradient_descent(centroid)
    centroid[0] = centroid[0] - theta * dx
    centroid[1] = centroid[1] - theta * dy

print("Final Centroid is:", centroid)
print("Final Cost is:", cost(centroid))
plt.plot(centroid[0],centroid[1],'.', color='r')
plt.show()