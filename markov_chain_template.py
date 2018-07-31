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
            # do stuff to update state

            # Probablistic Updates: Using our markov chain

            rand_val = r.random()

            # extracts a single row from transition matrix
            new_state_probs = tran_mat.tolist()[student.get_state()]
            cummulative_sum = 0

            # determines end state for the given start state of
            # student by generating random number and finding which bin
            # it lies in
            for index, prob in enumerate(new_state_probs):
                cummulative_sum += prob
                if rand_val <= cummulative_sum:
                    student.set_state(index)
                    break

    # It looks weird having these loops separate, but it deals with the issue that
    # sometimes someone gets infected AFTER you look past some people. So you have
    # to go back retroactively to update the people who are now sick, and now next to
    # sick people
    # alternatively, we could just look through all the neighbors of people who become
    # sick and assign from there, but I thought we might have future uses for this
    # manual update loop
    for row in range(shape[0]):
        for col in range(shape[1]):
            student = room[row, col]
            neighbors = student.get_neighbors()

            # Manual Updates: not part of our markov chain probabilities

            # if state = infected, increase count for days infected
            if student.get_state() == 2:
               student.days_infected += 1

            # if there is a sick person nearby and I am not sick, I become at risk
            elif student.get_state() == 0 and 2 in [x.get_state() for x in neighbors]:
               student.set_state(1)




# initializes a single classroom for a given simulation (populates with students)
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
            neigh_rows = list(range(max(0, row - 1), min(row + 1, row_dim - 1) + 1))
            neigh_cols = list(range(max(0, col - 1), min(col + 1, col_dim - 1) + 1))
            positions = set(itr.product(neigh_rows, neigh_cols))
            positions.remove((row, col))   # remove student (student isn't neighbor of themselves)

            # adding all neighbors to given student
            for pos in positions:
                student.add_neighbor(results[0, pos[0], pos[1]])

    # infect random student: patient zero
    # we can change/expand upon this with future ideas (i.e. vaccinations)
    results[0, np.random.randint(0, row_dim), np.random.randint(0, col_dim)].set_state(2)
    return results



# runs a single simulation for the spread of the flu
# parameters:
#       tran_mat = transition matrix between states
#       class_sizes = set of 2-member tuples (rows, columns)
#       time_steps = days to run simulation for
def run_simulation(tran_mat, class_sizes, time_steps):

    # list of results for all classrooms (list of our ndarrays)
    classrooms = list()

    for cs in class_sizes:
        # we also may want to consider creating a vector of students, and passing
        # a random subset of these to each initialize call in an attempt to
        # randomly populate our classes with a shared collection of students
        classrooms.append(initialize_sim(cs, time_steps))

    # prints initial room state for each classroom
    for i, results in enumerate(classrooms):
        print("day: 0, classroom: " + str(i + 1))
        print(results[0, :, :]) #initial room state

    # runs simulation across all classrooms
    for day in range(1, time_steps):
        for i, results in enumerate(classrooms):
            update_states(results[day - 1, :, :], tran_mat)
            results[day, :, :] = results[day -1, :, :]
            print("day: " + str(day) + ", classroom: " + str(i + 1))
            print(results[day, :, :])








# classroom dimensions: each tuple = 1 classroom
class_sizes = {(5,5), (10,2)}

time_steps = 100 # days to run simulation for

# probabilities of transitioning between states; not used much yet, but I
# put it here in case we need it in the future

# right now, 0.1 is the chance of going from state: near sick person -> sick
transition_matrix = np.matrix('0.99 0 0.01; 0 0.9 0.1; 0 0 1')  #rows = curr state, col = next state
# 0.99  0       0.01     state 0: not sick and not at risk
# 0     0.9     0.1      state 1: not sick but at risk
# 0     0       1        state 2: sick


run_simulation(transition_matrix, class_sizes, time_steps)