import numpy as np
import matplotlib as plt
import random as r
import time

class Student:
    def __init__(self):
        # state 0 = susceptible/uninfected
        # state 1 = susceptible + sick people around
        # state 2 = sick
        # state 3 = recoverd (immune)
        self.state = 0
        self.days_infected = 0
        self.neighbors = set()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def add_neighbor(self, other):
        self.neighbors.add(self, other)

    def get_neighbors(self):
        return self.neighbors

    # toString method
    def __repr__(self):
        return str(self.state)


row_dim = 5
col_dim = 10
time_steps = 100 # days
results = np.ndarray(shape=(time_steps, row_dim, col_dim), dtype=object)

transition_matrix = np.matrix('')  #rows = curr state, col = next state

# Initialization (can include vaccinations/other assumptions here)
for row in range(row_dim):
    for col in range(col_dim):
        results[0, row, col] = Student()

# infect random student: patient zero
results[0, np.random.randint(0, row_dim),np.random.randint(0,col_dim)].set_state(1)

def update_states(room):
    shape = np.shape(room)
    for row in range(shape[0]):
        for col in range(shape[1]):
           neighbors = room[row,col].get_neighbors()
            # do stuff to update state
            # i.e. if state = infected, increase self.days_infected

    #placeholder update
    room[np.random.randint(0, row_dim), np.random.randint(0, col_dim)].set_state(1)

print "day: 0"
print results[0, :, :] #initial room state

for day in range(1, time_steps):
    update_states(results[day - 1, :, :])
    results[day, :, :] = results[day -1, :, :]
    print "day: ", day
    print results[day, :, :]


