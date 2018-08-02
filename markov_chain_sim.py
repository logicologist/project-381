import numpy as np
import matplotlib.pyplot as plt
import random as r
import time
import math
import itertools as itr

from Student import *

def update_states(room, tran_mat, day):
    shape = np.shape(room)
    for row in range(shape[0]):
        for col in range(shape[1]):
            student = room[row, col]

            # Probabilistic Updates: Using our Markov Chain

            rand_val = r.random()

            # extracts a single row from transition matrix
            # i.e. if student is at state i, this row i represents the
            # probabilities they transition to any of the other states
            new_state_probs = tran_mat.tolist()[student.get_state()]

            cumulative_sum = 0

            # determines end state for the given start state of
            # student by generating random number and finding which bin
            # it lies in
            for index, prob in enumerate(new_state_probs):
                cumulative_sum += prob
                if rand_val <= cumulative_sum:
                    student.set_state(index)
                    break

    # It looks weird having these loops separate, but it deals with the issue that
    # sometimes someone gets infected AFTER you look past some people. So you have
    # to go back retroactively to update the people who are now sick, and now next to
    # sick people
    # alternatively, we could just look through all the neighbors of people who become
    # sick and assign from there, but I thought we might have future uses for this
    # manual update loop. I think it may be helpful to keep track of metadata within students
    for row in range(shape[0]):
        for col in range(shape[1]):
            student = room[row, col]
            neighbors = student.get_neighbors()

            # Manual Updates: not part of our markov chain probabilities

            # if state = infected, increase count for days infected
            if student.get_state() == 2:
               student.add_day_infected(day)

            # if there is a sick person nearby and I am not sick, I become at risk
            elif student.get_state() == 0 and 2 in [x.get_state() for x in neighbors]:
               student.set_state(1)




# initializes a single classroom for a given simulation (populates with students)
# parameters:
#      class_size is tuple (rows, cols)
#       time_steps = days to run simulation for
def initialize_class(class_size, time_steps):
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
    patient_zero = results[0, np.random.randint(0, row_dim), np.random.randint(0, col_dim)]
    patient_zero.set_state(2)
    patient_zero.add_day_infected(0)
    return results


# Plots some information about simulation such as:
#     fraction sick each day (line)
#     total days spent sick for each individual (histogram)
def graph_results(classrooms, time_steps):
    t_vals = list(range(time_steps))
    for i, cs in enumerate(classrooms):
        end_results = cs[-1,: :]
        counts = list()
        days_sick = list()
        for row in end_results:
            for student in row:
                counts = counts + student.days_infected
                days_sick += [len(student.days_infected)]

        plt.figure(i + 2)

        binWidth = 5
        edges = list(range(0, time_steps + 1, binWidth))
        plt.hist(days_sick, bins=edges, rwidth=0.9)
        plt.title("Classroom " + str(i+1))
        plt.xlabel("Days Spent Sick")
        plt.ylabel("# Students")
        plt.xticks(edges)


        class_shape = np.shape(end_results)
        class_size = class_shape[0]*class_shape[1]
        frac_sick = [counts.count(x) * 1.0 / class_size for x in t_vals]


        plt.figure(1)
        plt.plot(t_vals, frac_sick, label="Classroom " + str(i+1) + " shape: " + str(class_shape))
    plt.figure(1)
    plt.legend(loc='best')
    plt.ylabel("Fraction Classroom Infected")
    plt.xlabel("Day")
    plt.xticks(list(range(0, time_steps, 2)))

    plt.show()





# runs a single simulation for the spread of the flu
# preconditions: no classroom has more than 1 / classes_per_student of the total number
#                of seats
# parameters:
#       tran_mat = transition matrix between states
#       class_sizes = set of 2-member tuples (rows, columns)
#       time_steps = days to run simulation for
def run_simulation(tran_mat, class_sizes, time_steps, classes_per_student = 1):

    # list of results for all classrooms (list of our ndarrays)
    classrooms = list()

    total_class_seats = sum(map(lambda x: x[0]*x[1], class_sizes))
    student_count = math.ceil(total_class_seats * 1.0 / classes_per_student)

    #TO DO: find way to distribute students
    #   do we want each student to be in N classes
    #   do we want a set # of students to be distributed through the classes, with some taking more than others?


    for cs in class_sizes:
        # we also may want to consider creating a vector of students, and passing
        # a random subset of these to each initialize call in an attempt to
        # randomly populate our classes with a shared collection of students
        classrooms.append(initialize_class(cs, time_steps))

        # i.e. if we want to assign each student to 3 classes, we could create
        # sum(class_room_sizes) / 3 students --> [S1 S2 S3 S4 ...]
        # then we could duplicate these references doing [S1 S2 S3 S4 ...] * 3
        # and sample the proper number of students for each class; This wont be an issue
        # as long as no classroom is > 1/3 the sum of all class room sizes.

    # prints initial room state for each classroom
    print("Day: 0 \n")
    for i, results in enumerate(classrooms):
        print("Classroom: " + str(i + 1))
        print(str(results[0, :, :]) + "\n") #initial room state

    # runs simulation across all classrooms
    for day in range(1, time_steps):
        print("-"*40)
        print("Day: " + str(day) + "\n")
        for i, results in enumerate(classrooms):
            update_states(results[day - 1, :, :], tran_mat, day)
            results[day, :, :] = results[day - 1, :, :]
            print("Classroom: " + str(i + 1))
            print(str(results[day, :, :]) + "\n")


    graph_results(classrooms, time_steps)









# classroom dimensions: each tuple = 1 classroom (rows, columns)
class_sizes = {(5,5), (2,2)}

time_steps = 20 # days to run simulation for

# probabilities of transitioning between states; not used much yet, but I
# put it here in case we need it in the future

# right now, 0.1 is the chance of going from state: near sick person -> sick
transition_matrix = np.matrix('0.99 0 0.01 0; 0 0.9 0.1 0; 0 0 0.98 0.02; 0 0 0 1')  #rows = curr state, col = next state
# 0.99  0       0.01       0      state 0: not sick and not at risk
# 0     0.9     0.1        0      state 1: not sick but at risk
# 0     0       0.98       0.02   state 2: sick
# 0     0       0          1      state 3: recovered (immune)


run_simulation(transition_matrix, class_sizes, time_steps)