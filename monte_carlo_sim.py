import numpy as np
import matplotlib.pyplot as plt
import random as r
import time
import math
import itertools as itr

from Student import *
from room_assign import *
from graphing import *

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


def initialize_classrooms(vaccination_rate, vaccination_effectiveness, class_sizes, num_days, classes_per_student):
    '''Initializes a set of classrooms (populated with students) given room sizes
    Params:
        vaccination_rate - fraction of students who get vaccinated
        vaccination_effectiveness - percent effectiveness of vaccine
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
    
    # Assign some students to be vaccinated
    student_nums = list(range(len(student_list)))
    r.shuffle(student_nums)
    vacc_cutoff = math.floor(len(student_nums) * vaccination_rate)
    for i in range(vacc_cutoff):
        s_num = student_nums[i]
        student_list[s_num].set_vaccinated()
        # With probability vaccination_effectiveness, the student is actually immune
        rand = r.random()
        if (rand < vaccination_effectiveness):
            student_list[s_num].set_state(3)
            
    # Recovery times: constant 8 days + geometric distribution with mean 2 days
    recovery_times = (np.random.geometric(recovery_time_dropoff_rate, size=len(student_list))) + recovery_time_fixed_days
    for i in range(len(student_list)):
        student_list[i].set_recovery_time(recovery_times[i])
    
    # infect random student: patient zero
    # we can change/expand upon this with future ideas (i.e. vaccinations)
    patient_zero = student_list[np.random.randint(0, len(student_list))]
    patient_zero.set_state(2)
    patient_zero.add_day_infected(0)
    
    return (classrooms, student_list)



# runs a single simulation for the spread of the flu
# preconditions: no classroom has more than 1 / classes_per_student of the total number
#                of seats
# parameters:
#       infect_rate = probability of infected student infecting susceptible adjacent student (float)
#       vaccination_rate = fraction of students who get vaccinated against the flu (float)
#       vaccination_effectiveness = percent effectiveness of vaccine (float)
#       class_sizes = set of 2-member tuples (rows, columns)
#       time_steps = days to run simulation for (int)
#       classes_per_student = allows a student to be placed in multiple classrooms; allows cross-infection (int)
#       weekends = True to include weekends (i.e. no class Sat/Sun; cannot spread flu) but allows sick to recover
def run_simulation(infect_rate, vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student = 1, weekends=False):

    # list of results for all classrooms (list of our ndarrays)
    (classrooms, students) = initialize_classrooms(vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student)

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

    return (classrooms, students)


# classroom dimensions: each tuple = 1 classroom (rows, columns)
#class_sizes = [(5, 5), (10, 10)]
class_sizes = [(4, 6), # BAG 106
               (4, 7), # BAG 108
               (13, 24), # BAG 131
               (10, 12), # BAG 154
               (8, 10), # BAG 260
               (8, 10), # BAG 261
               (5, 4), # BAG 331A
               (3, 10), # JHN 022
               (3, 10), # JHN 026
               (7, 12), # JHN 075
               (12, 16), # JHN 102
               (3, 14), # JHN 111
               (4, 16), # JHN 175
               (11, 21), # KNE 110
               (15, 30), # KNE 120
               (15, 36), # KNE 130 first floor
               (5, 26), # KNE 130 balcony
               (8, 28), # KNE 210
               (8, 31), # KNE 220
               ]

trials = 5 # number of times to run simulation
time_steps = 100  # days to run simulation for
num_periods = 2 # number of class periods in the day
infection_rate = 0.15 # chance per sick neighbor of spreading infection
vaccination_rate = 0.46 # percentage of students who get vaccinated
vaccination_effectiveness = 0.39 # percent effectiveness of vaccine
# Note: recovery time parameters are tuned for infectiousness
# CDC reports infectiousness lasts from 1 day before symptoms to 5-7 days after,
# for a total of 6-8 days infectious.
recovery_time_fixed_days = 6 # constant number of days that an infected student is sick and infectious at minimum
recovery_time_dropoff_rate = 0.5 # after fixed days, student recovers (stops being infectious) with this probability each day

# classrooms[which_trial][which_classroom][which_time_step][row][column]
classrooms_list = []
students_list = []
for trial in range(trials):
    # classrooms[which_classroom][which_time_step][row][column]
    # students[which_student]
    (classrooms, students) = run_simulation(infection_rate, vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student=num_periods, weekends=False)
    classrooms_list.append(classrooms)
    students_list.append(students)

f1 = plt.figure(1)
graph_frac_infected(classrooms_list, time_steps)
f2 = plt.figure(2)
graph_days_infected(classrooms_list, time_steps)
f3 = plt.figure(3)
graph_disease_burden(students_list, time_steps)

f4 = plt.figure(4)
v_rates = (0.46, 0.56, 0.66, 0.76, 0.86, 0.96)
for v_rate in v_rates:
    classrooms_list = []
    students_list = []
    for trial in range(trials):
        (classrooms, students) = run_simulation(infection_rate, v_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student=num_periods, weekends=False)
        classrooms_list.append(classrooms)
        students_list.append(students)
    graph_disease_burden(students_list, time_steps, v_rate, legend=True)
# ... and label the figure
plt.legend(loc='best')

f1.savefig('sim-data/frac_infected.pdf')
f2.savefig('sim-data/days_infected.pdf')
f3.savefig('sim-data/disease_burden.pdf')
f4.savefig('sim-data/disease_burden_varying_vrate.pdf')

plt.show()

