import numpy as np
import matplotlib.pyplot as plt
import random as r
import time
import math
import itertools as itr

from Student import *
from room_assign import *

r.seed(42) # for reproducability

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


def initialize_classrooms(class_sizes, num_days, classes_per_student):
    '''Initializes a set of classrooms (populated with students) given room sizes
    Params:
        class_sizes - list of tuples containing (row, column) dimensions of rooms
        num_days - days to run simulation for
        classes_per_student - int for number of class periods'''
    classrooms = list()
    student_list = np.empty((0), dtype=object)
    for class_size in class_sizes:
        # room information stored here; who is sick when and what not
        row_dim = class_size[0]
        col_dim = class_size[1]
        results = np.ndarray(shape=(num_days, row_dim, col_dim), dtype=object)
        # create students
        for row in range(row_dim):
            for col in range(col_dim):
                results[0, row, col] = Student()
        classrooms.append(results)
        student_list = np.append(student_list, results[0].flatten())
    
    # Assign neighbors
    room_assign(class_sizes, student_list, classes_per_student)
    
    for i, class_size in enumerate(class_sizes):
        row_dim = class_size[0]
        col_dim = class_size[1]
    
        # Recovery times: constant 8 days + geometric distribution with mean 2 days
        recovery_times = (np.random.geometric(recovery_time_dropoff_rate, size=row_dim * col_dim)) + recovery_time_fixed_days
        for row in range(row_dim):
            for col in range(col_dim):
                results = classrooms[i]
                student = results[0, row, col]
                student = results[0, row, col]
                student.set_days_sick(recovery_times[row * row_dim + col])
    
    # infect random student: patient zero
    # we can change/expand upon this with future ideas (i.e. vaccinations)
    patient_zero = student_list[np.random.randint(0, len(student_list))]
    patient_zero.set_state(2)
    patient_zero.add_day_infected(0)
    # to approximate what the code was doing before (these three lines will go away when room-assign.py code is fully integrated)
    patient_one = student_list[0]
    patient_one.set_state(2)
    patient_one.add_day_infected(0)
    
    return classrooms



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
    classrooms = initialize_classrooms(class_sizes, time_steps, classes_per_student)

#    for cs in class_sizes:
        # we also may want to consider creating a vector of students, and passing
        # a random subset of these to each initialize call in an attempt to
        # randomly populate our classes with a shared collection of students
#        classrooms.append(initialize_class(cs, time_steps))

        # i.e. if we want to assign each student to 3 classes, we could create
        # sum(class_room_sizes) / 3 students --> [S1 S2 S3 S4 ...]
        # then we could duplicate these references doing [S1 S2 S3 S4 ...] * 3
        # and sample the proper number of students for each class; This wont be an issue
        # as long as no classroom is > 1/3 the sum of all class room sizes.

    # prints initial room state for each classroom
#    print("Day: 0 \n")
#    for i, results in enumerate(classrooms):
#        print("Classroom: " + str(i + 1))
#        print(str(results[0, :, :]) + "\n")

    # runs simulation across all classrooms for "time_steps" days
    for day in range(1, time_steps):
#        print("-"*40)
#        print("Day: " + str(day) + "\n")
        for i, results in enumerate(classrooms):
            update_states(results[day - 1, :, :], infect_rate, day, weekends)
            results[day, :, :] = results[day - 1, :, :]
#            print("Classroom: " + str(i + 1))
#            print(str(results[day, :, :]) + "\n")

    return classrooms


# classroom dimensions: each tuple = 1 classroom (rows, columns)
class_sizes = [(5, 5), (10, 10)]

trials = 5 # number of times to run simulation
time_steps = 100  # days to run simulation for
infection_rate = 0.1 # chance per sick neighbor of spreading infection
recovery_time_fixed_days = 8 # constant number of days that an infected student is sick at minimum
recovery_time_dropoff_rate = 0.5 # after fixed days, student recovers with this probability each day

classrooms_list = []
for trial in range(trials):
    classrooms = run_simulation(infection_rate, class_sizes, time_steps, weekends=False)
    classrooms_list.append(classrooms)
    # What the classrooms data structure looks like:
    # classrooms[which_classroom][which_time_step][row][column]

f1 = plt.figure(1)
graph_frac_infected(classrooms_list, time_steps)
f2 = plt.figure(2)
graph_days_infected(classrooms_list, time_steps)

f1.savefig('sim-data/frac_infected.pdf')
f2.savefig('sim-data/days_infected.pdf')

plt.show()

