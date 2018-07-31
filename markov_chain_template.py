import numpy as np
import matplotlib as plt
import random as r
import time
import itertools as itr

class Student:
    def __init__(self):
        # state 0 = susceptible/uninfected
        # state 1 = susceptible + sick people around
        # state 2 = sick
        # state 3 = recovered (immune)
        self.state = 0
        self.days_infected = 0
        self.neighbors = set()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def add_neighbor(self, other):
        self.neighbors.add(other)

    def get_neighbors(self):
        return self.neighbors

    # toString method
    def __repr__(self):
        return str(self.state)



# classroom dimensions
row_dim = 5
col_dim = 5

time_steps = 100 # days to run simulation for

# probabilities of transitioning between states; not used much yet, but I
# put it here in case we need it in the future

# right now, 0.2 is the chance of going from state: near sick person -> sick
transition_matrix = np.matrix('0.95 0 0.05; 0 0.8 0.2; 0 0 1')  #rows = curr state, col = next state

def update_states(room, tran_mat):
    shape = np.shape(room)
    for row in range(shape[0]):
        for col in range(shape[1]):
            student = room[row, col]
            neighbors = student.get_neighbors()
            # do stuff to update state

            # if state = infected, increase count for days infected
            if student.get_state() == 2:
               student.days_infected += 1

            # if there is a sick person nearby and I am not sick, I become at risk
            elif student.get_state() == 0 and 2 in [x.get_state() for x in neighbors]:
               student.set_state(1)

            # if I am at risk, there is a chance I become infected
            elif student.get_state() == 1 and r.random() <= tran_mat[1,2]:
                student.set_state(2)




def run_simulation(tran_mat):
    #room information stored here; who is sick when and what not
    results = np.ndarray(shape=(time_steps, row_dim, col_dim), dtype=object)

    # Initialization (can include vaccinations/other assumptions here)
    for row in range(row_dim):
        for col in range(col_dim):
            results[0, row, col] = Student()

    for row in range(row_dim):
        for col in range(col_dim):
            student = results[0, row, col]

            # all possible combos of row +-1 and col +- 1 (with edge cases)
            positions = set(itr.product( list(range(max(0, row - 1), min(row + 1, row_dim - 1) + 1)), list(range(max(0, col - 1), min(col + 1, col_dim - 1) + 1))))
            positions.remove((row,col))

            # adding all neighbors to given student
            for pos in positions:
                student.add_neighbor(results[0, pos[0], pos[1]])

    # infect random student: patient zero
    results[0, np.random.randint(0, row_dim),np.random.randint(0,col_dim)].set_state(2)

    print "day: 0"
    print results[0, :, :] #initial room state

    for day in range(1, time_steps):
        update_states(results[day - 1, :, :], tran_mat)
        results[day, :, :] = results[day -1, :, :]
        print "day: ", day
        print results[day, :, :]





run_simulation(transition_matrix)