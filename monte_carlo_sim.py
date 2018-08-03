import numpy as np
import matplotlib.pyplot as plt
import random as r
import time
import math
import itertools as itr

from Student import *
from classes import classes

def update_states(room, infect_rate, day, weekends = False):
    shape = np.shape(room)
    for row in range(shape[0]):
        for col in range(shape[1]):
            student = room[row, col]

            # Probabilistic Updates: Using our Markov Chain

            # implementing "weekend": only run this section if
            #       student.get_state() = 2 (i.e. we are sick = chance to recover over weekend)
            #       not day % 7 in [5,6] --> we are not on a weekend; 5,6 arbitrary
            if not weekends or (student.get_state() == 2 or not day % 7 in [5, 6]):
                rand_val = r.random()

                if student.get_state() == 2:
                    student.days_infected.append(day)

                    # determine if recovers
                    if student.stays_sick_for <= len(student.days_infected):
                        student.set_state(3)

                if student.get_state() == 0:
                    sick_neighbors = [n.get_state() for n in student.get_neighbors()].count(2)
                    prob_infected = 1 - (1 - infect_rate)**sick_neighbors
                    if rand_val <= prob_infected:
                        student.set_state(2)




# initializes a single classroom for a given simulation (populates with students)
# parameters:
#      class_size is tuple (rows, cols)
#       time_steps = days to run simulation for
def initialize_class(class_size, num_days):
    # room information stored here; who is sick when and what not
    row_dim = class_size[0]
    col_dim = class_size[1]
    results = np.ndarray(shape=(num_days, row_dim, col_dim), dtype=object)

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

    # Recovery times: constant 8 days + geometric distribution with mean 2 days
    recovery_times = (np.random.geometric(0.5, size=row_dim * col_dim)) + 8
    for row in range(row_dim):
        for col in range(col_dim):
            student = results[0, row, col]
            student.set_days_sick(recovery_times[row * row_dim + col])

    # infect random student: patient zero
    # we can change/expand upon this with future ideas (i.e. vaccinations)
    patient_zero = results[0, np.random.randint(0, row_dim), np.random.randint(0, col_dim)]
    patient_zero.set_state(2)
    patient_zero.add_day_infected(0)
    return results


# Plots some information about simulation such as:
#     fraction sick each day (line)
#     total days spent sick for each individual (histogram) in each classroom
def graph_results(classrooms, num_days):
    t_vals = list(range(num_days))
    for i, cs in enumerate(classrooms):
        end_results = cs[-1, :, :]
        counts = list()
        days_sick = list()
        for row in end_results:
            for student in row:
                counts = counts + student.days_infected
                days_sick += [len(student.days_infected)]

        plt.figure(i + 2)

        class_shape = np.shape(end_results)

        bin_width = 5
        edges = list(range(0, num_days + 1, bin_width))
        plt.hist(days_sick, bins=edges, rwidth=0.9)
        plt.title("Classroom " + str(i+1) + " shape: " + str(class_shape))
        plt.xlabel("Days Spent Sick")
        plt.ylabel("# Students")
        plt.xticks(edges)

        class_size = class_shape[0]*class_shape[1]
        frac_sick = [counts.count(x) * 1.0 / class_size for x in t_vals]

        plt.figure(1)
        plt.plot(t_vals, frac_sick, label="Classroom " + str(i+1) + " shape: " + str(class_shape))

    plt.figure(1)
    plt.legend(loc='best')
    plt.ylabel("Fraction Classroom Infected")
    plt.xlabel("Day")
    plt.xticks(list(range(0, num_days, 5)))

    plt.show()


# runs a single simulation for the spread of the flu
# preconditions: no classroom has more than 1 / classes_per_student of the total number
#                of seats
# parameters:
#       tran_mat = transition matrix between states
#       class_sizes = set of 2-member tuples (rows, columns)
#       time_steps = days to run simulation for
#       classes_per_student = allows a student to be placed in multiple classrooms; allows cross-infection
#       weekends = True to include weekends (i.e. no class Sat/Sun; cannot spread flu) but allows sick to recover
def run_simulation(infect_rate, class_sizes, time_steps, classes_per_student = 1, weekends=False):

    # list of results for all classrooms (list of our ndarrays)
    classrooms = list()

    # TO DO: find way to distribute students
    #   do we want each student to be in N classes
    #   do we want a set # of students to be distributed through the classes, with some taking more than others?

    # classes.py

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
        print(str(results[0, :, :]) + "\n")

    # runs simulation across all classrooms for "time_steps" days
    for day in range(1, time_steps):
        print("-"*40)
        print("Day: " + str(day) + "\n")
        for i, results in enumerate(classrooms):
            update_states(results[day - 1, :, :], infect_rate, day, weekends)
            results[day, :, :] = results[day - 1, :, :]
            print("Classroom: " + str(i + 1))
            print(str(results[day, :, :]) + "\n")

    graph_results(classrooms, time_steps)


# classroom dimensions: each tuple = 1 classroom (rows, columns)
class_sizes = [(5, 5), (10, 10)]

time_steps = 100  # days to run simulation for
infection_rate = 0.1 # chance per sick neighbor of spreading infection

run_simulation(infection_rate, class_sizes, time_steps, weekends=False)