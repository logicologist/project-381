import numpy as np
import matplotlib as plt
import random as r
import time
import itertools as itr

from Student import *

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




# initializes a single classroom for a given simulation
# parameters:
#      class_size is tuple (rows, cols)
#       time_steps = days to run simulation for
def initialize_sim(class_size, time_steps):
    #room information stored here; who is sick when and what not
    row_dim = class_size[0]
    col_dim = class_size[1]
    results = np.ndarray(shape=(time_steps, row_dim, col_dim), dtype=object)

    # again, in the future we might consider creating all our students first
    # so we can randomly distribute them throughout the classrooms
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
    # we can change/expand upon this with future ideas (i.e. vaccinations)
    results[0, np.random.randint(0, row_dim),np.random.randint(0,col_dim)].set_state(2)
    return results



# runs a single simulation for the spread of the flu
# parameters:
#       tran_mat = transition matrix between states
#       class_sizes = set of 2-member tuples (rows, columns)
#       time_steps = days to run simulation for
def run_simulation(tran_mat, class_sizes, time_steps):

    # list of results for all classrooms (list of our ndarrays)
    classrooms = list()

    for i, cs in enumerate(class_sizes):
        # we also may want to consider creating a vector of students, and passing
        # a random subset of these to each initialize call in an attempt to
        # randomly populate our classes with a shared collection of students
        classrooms.append(initialize_sim(cs, time_steps))

    # prints initial room state for each classroom
    for results in classrooms:
        print("day: 0")
        print(results[0, :, :]) #initial room state

    # runs simulation across all classrooms
    for day in range(1, time_steps):
        for results in classrooms:
            update_states(results[day - 1, :, :], tran_mat)
            results[day, :, :] = results[day -1, :, :]
            print("day: ", day)
            print(results[day, :, :])








# classroom dimensions: each tuple = 1 classroom
class_sizes = {(5,5), (10,5)}

time_steps = 100 # days to run simulation for

# probabilities of transitioning between states; not used much yet, but I
# put it here in case we need it in the future

# right now, 0.2 is the chance of going from state: near sick person -> sick
transition_matrix = np.matrix('0.95 0 0.05; 0 0.8 0.2; 0 0 1')  #rows = curr state, col = next state

run_simulation(transition_matrix, class_sizes, time_steps)