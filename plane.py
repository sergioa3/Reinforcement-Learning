import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from framework import distribution, kernel, rewards, environment, agent




n = 15
m = 10


reward_dict = {}

distribution_dict = {}

import math

def objective_region(i,j):
    objective = n*(17/20) <= i <= n
    objective = objective and m*(12/15) <= j <= m
    return objective

def danger_region(i,j):
    danger1 = n*(13/20) <= i <= n*(15/20)
    danger1 = danger1 and m*(8/15) <= j <= m

    danger2 = n*(6/20) <= i <= n*(8/20)
    danger2 = danger2 and m*(2/15) <= j <= m*(6/15)

    danger = danger1 or danger2

    return danger
    #return diamond(i,j, -n/5, -m/5) or diamond(i,j, n/10, m/5)


def circular(i, j):
    centers = [(n//4, m//4), (3*n//4, 3*m//4), (n//2, m//2)]
    radius = min(n, m) // 8
    for cx, cy in centers:
        if (i - cx)**2 + (j - cy)**2 < radius**2:
            return True
    return False

def checkerboard(i, j):
    block_size = min(n, m) // 10
    return ((i // block_size) % 2) == ((j // block_size) % 2)

def wavy(i, j):
    amplitude = m // 10
    wavelength = n // 5
    wave = amplitude * math.sin(2 * math.pi * i / wavelength) + m // 2
    return abs(j - wave) < amplitude

def concentric(i, j):
    cx, cy = n//2, m//2
    distance = math.sqrt((i - cx)**2 + (j - cy)**2)
    ring_width = n // 10
    return int(distance // ring_width) % 2 == 0

def ellipse(i, j):
    ellipses = [
        (n // 4, m // 4, n // 8, m // 12),
        (3 * n // 4, 3 * m // 4, n // 6, m // 10),
        (n // 2, m // 2, n // 5, m // 8)
    ]
    for cx, cy, rx, ry in ellipses:
        if ((i - cx) / rx)**2 + ((j - cy) / ry)**2 < 1:
            return True
    return False

def diamond(i, j, x, y):
    cx, cy = n // 2 + x, m // 2 + y  # Center of the grid
    return abs(i - cx) + abs(j - cy) < n // 4







for i in range(n):
    for j in range(m):
        #print(i*m+j)
        #print(i,j)
        s = i*m + j

        up = (i+1)*m + j
        down = (i-1)*m + j
        left = i*m + (j - 1)
        right = i*m + (j + 1)
        if objective_region(i,j):
            reward_dict[(s,'u')] = 0
            u_dist = {s:1}
            distribution_dict[(s,'u')] = distribution(u_dist)
        else:

            if i+1 < n:
                if objective_region(i+1,j):
                    reward_dict[(s,'u')] = 10000000
                    u_dist = {up:1}
                    distribution_dict[(s,'u')] = distribution(u_dist)
                elif danger_region(i+1,j):
                    reward_dict[(s,'u')] = -10
                    u_dist = {up:1}
                    distribution_dict[(s,'u')] = distribution(u_dist)
                else:
                    reward_dict[(s,'u')] = -5
                    u_dist = {up:1}
                    distribution_dict[(s,'u')] = distribution(u_dist)
            if i-1 >= 0:
                if objective_region(i-1,j):
                    reward_dict[(s,'d')] = 10000000
                    d_dist = {down:1}
                    distribution_dict[(s,'d')] = distribution(d_dist)
                elif danger_region(i-1,j):
                    reward_dict[(s,'d')] = -10
                    d_dist = {down:1}
                    distribution_dict[(s,'d')] = distribution(d_dist)
                else:
                    reward_dict[(s,'d')] = -5
                    d_dist = {down:1}
                    distribution_dict[(s,'d')] = distribution(d_dist)

            if j+1 < m:
                if objective_region(i,j+1):
                    reward_dict[(s,'r')] = 10000000
                    r_dist = {right:1}
                    distribution_dict[(s,'r')] = distribution(r_dist)
                elif danger_region(i,j+1):
                    reward_dict[(s,'r')] = -10
                    r_dist = {right:1}
                    distribution_dict[(s,'r')] = distribution(r_dist)
                else:
                    reward_dict[(s,'r')] = -5
                    r_dist = {right:1}
                    distribution_dict[(s,'r')] = distribution(r_dist)
            
            if j-1 >= 0:
                if objective_region(i,j-1):
                    reward_dict[(s,'l')] = 10000000
                    l_dist = {left:1}
                    distribution_dict[(s,'l')] = distribution(l_dist)
                elif danger_region(i,j-1):
                    reward_dict[(s,'l')] = -10
                    l_dist = {left:1}
                    distribution_dict[(s,'l')] = distribution(l_dist)
                else:
                    reward_dict[(s,'l')] = -5
                    l_dist = {left:1}
                    distribution_dict[(s,'l')] = distribution(l_dist)


def plot(trajectory, score):
    # Create the grid
    grid = np.zeros((n, m, 3))  # RGB grid initialized to black

    # Fill in the grid based on the regions
    for i in range(n):
        for j in range(m):
            if danger_region(i, j):
                grid[n-i-1, j] = [1, 0, 0]  # Red for danger region
            elif objective_region(i, j):
                grid[n-i-1, j] = [0, 1, 0]  # Green for objective region

    # Example list of points to connect with a blue line
    points = [(p//m, p%m) for p in trajectory]
    #points = [(0, 0), (1, 2), (3, 4), (6, 5), (9, 9)]  # Format: (i, j)

    # Extract x and y coordinates from points
    x_coords = [point[1] for point in points]
    y_coords = [n - point[0] - 1 for point in points]  # Adjust for grid orientation

    # Plotting the grid with correct orientation
    plt.imshow(grid, interpolation='none', origin='upper')
    #plt.grid(True, which='both', color='black', linestyle='-', linewidth=2)

    # Adjust gridlines and ticks
    plt.xticks(np.arange(-0.5, m, 1), labels=[])
    plt.yticks(np.arange(-0.5, n, 1), labels=[])
    plt.gca().set_xticks(np.arange(0.5, m, 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, n, 1), minor=True)
    plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=1)

    # comment codeblock above and use this to remove grid from border 
    #plt.gca().set_xticks([])
    #plt.gca().set_yticks([])


    # Remove axis labels
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    # Plot the blue line connecting the points
    plt.plot(x_coords, y_coords, color='blue', marker='')
    # Plot the latest location
    plt.plot(x_coords[-1], y_coords[-1], color='yellow', marker='.')

    #score = 95  # Example score value
    plt.text(1.02, 0.95, "Score: {}".format(score), transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='left')


    #plt.show()
    #plt.clf()
    plt.show()
    plt.pause(0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001)
    plt.clf()




K = kernel(distribution_dict)
R = rewards(reward_dict)
d0 = distribution({0:1})
env = environment(K,R,d0)
#agent1 = agent(env,discount=0.9, history_callback=plot, action0=['u','r'])
agent1 = agent(env,discount=0.9, history_callback=plot, action0=None)

#agent1.sarsa(alpha=0.1, epsilon=0.01)
agent1.q_learning(alpha=0.1, epsilon=0.01)