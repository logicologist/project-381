import numpy as np
import matplotlib.pyplot as plt
import random as r
import time
import math
import itertools as itr

from graphing import *

from Student import *
from room_assign import *
from graphing import *

r.seed(42) # for reproducability

def update_states(room, infect_rate, day, weekends = False):
    shape = np.shape(room)
    for row in range(shape[0]):
        for col in range(shape[1]):
            student = room[row, col]

            # implementing "weekend": only run this section if
            #       student.get_state() = 2 (i.e. we are sick = chance to recover over weekend)
            #       not day % 7 in [5,6] --> we are not on a weekend; 5,6 arbitrary
            if not weekends or (student.get_state() == 2 or not day % 7 in [5, 6]):
                rand_val = r.random()

                if student.get_state() == 2:
                    student.days_infected.append(day)

                    # determine if sick student recovers
                    if student.stays_sick_for <= len(student.days_infected):
                        student.set_state(3)

                # determine if uninfected student gets infected; more sick people around them
                # leads to higher infection rate
                if student.get_state() == 0:
                    sick_neighbors = [n.get_state() for n in student.get_neighbors()].count(2)
                    prob_infected = 1 - (1 - infect_rate)**sick_neighbors
                    if rand_val <= prob_infected:
                        student.set_state(2)


def initialize_classrooms(vaccination_rate, vaccination_effectiveness, class_sizes, num_days, classes_per_student, init_patients):
    '''Initializes a set of classrooms (populated with students) given room sizes
    Params:
        vaccination_rate - fraction of students who get vaccinated
        vaccination_effectiveness - percent effectiveness of vaccine
        class_sizes - list of tuples containing (row, column) dimensions of rooms
        num_days - days to run simulation for
        classes_per_student - int for number of class periods
        init_patients - int for number of students who initially have the flu'''
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
    vacc_cutoff = int(math.floor(len(student_nums) * vaccination_rate))
    for i in range(vacc_cutoff):
        s_num = student_nums[i]
        student_list[s_num].set_vaccinated()
        # With probability vaccination_effectiveness, the student is actually immune
        rand = r.random()
        if (rand < vaccination_effectiveness):
            student_list[s_num].set_state(3)
    
    for i, class_size in enumerate(class_sizes):
        row_dim = class_size[0]
        col_dim = class_size[1]
    
        # Recovery times: constant 8 days + geometric distribution with mean 2 days
        recovery_times = (np.random.geometric(recovery_time_dropoff_rate, size=row_dim * col_dim)) + recovery_time_fixed_days
        for row in range(row_dim):
            for col in range(col_dim):
                results = classrooms[i]
                student = results[0, row, col]
                student.set_recovery_time(recovery_times[row * col_dim + col])

            
    # Recovery times: constant 8 days + geometric distribution with mean 2 days
    recovery_times = (np.random.geometric(recovery_time_dropoff_rate, size=len(student_list))) + recovery_time_fixed_days
    for i in range(len(student_list)):
        student_list[i].set_recovery_time(recovery_times[i])

    # infect init_patients random students
    susceptible_students = [student for student in student_list if student.get_state() == 0]
    r.shuffle(susceptible_students)
    for i in range(init_patients):
        if (i < len(susceptible_students)):
            patient_zero = susceptible_students[i]
            patient_zero.set_state(2)
            patient_zero.add_day_infected(0)
        else:
            print("WARNING: more initial patients than susceptible students")
#    patient_zero = student_list[np.random.randint(0, len(student_list))]
#    patient_zero.set_state(2)
#    patient_zero.add_day_infected(0)
    
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
def run_simulation(infect_rate, vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student = 1, weekends=False, init_patients=1):

    # list of results for all classrooms (list of our ndarrays)
    (classrooms, students) = initialize_classrooms(vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student, init_patients)

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

trials = 10 # number of times to run simulation
time_steps = 100  # days to run simulation for
num_periods = 3 # number of class periods in the day
R_0 = 1.3 # virulence of flu: reproductive number
lambda_interactions = 13.4 # average number of interactions during the day for an average person
infection_rate = (R_0 / lambda_interactions) # chance per sick neighbor of spreading infection
vaccination_rate = 0.46 # percentage of students who get vaccinated
vaccination_effectiveness = 0.39 # percent effectiveness of vaccine
# Note: recovery time parameters are tuned for infectiousness
# CDC reports infectiousness lasts from 1 day before symptoms to 5-7 days after,
# for a total of 6-8 days infectious.
recovery_time_fixed_days = 6 # constant number of days that an infected student is sick and infectious at minimum
recovery_time_dropoff_rate = 0.5 # after fixed days, student recovers (stops being infectious) with this probability each day

# EXPERIMENT 1: analyses with standard parameters
# classrooms_list[which_trial][which_classroom][which_time_step][row][column]
# students_list[which_trial][which_student]
classrooms_list = []
students_list = []
for trial in range(trials):
    # classrooms[which_classroom][which_time_step][row][column]
    # students[which_student]
    (classrooms, students) = run_simulation(infection_rate, vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student=num_periods, weekends=True)
    classrooms_list.append(classrooms)
    students_list.append(students)
f1 = plt.figure(1)
graph_frac_room_infected(classrooms_list, time_steps)
f2 = plt.figure(2)
graph_days_infected(classrooms_list, time_steps)
f3 = plt.figure(3)
graph_disease_burden(students_list, time_steps)
f10 = plt.figure(10)
graph_disease_burden(students_list, int(0.2 * time_steps))

# EXPERIMENT 1.1: analysis with standard parameters except weekends turned off
classrooms_list = []
students_list = []
for trial in range(trials):
    # classrooms[which_classroom][which_time_step][row][column]
    # students[which_student]
    (classrooms, students) = run_simulation(infection_rate, vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student=num_periods, weekends=False)
    classrooms_list.append(classrooms)
    students_list.append(students)
f11 = plt.figure(11)
graph_disease_burden(students_list, time_steps)
f12 = plt.figure(12)
graph_disease_burden(students_list, int(0.2 * time_steps))

# EXPERIMENT 2: varying vaccination rate, holding all other params at standard
f4 = plt.figure(4)
f5 = plt.figure(5)
v_rates = (0.46, 0.56, 0.66, 0.76, 0.86, 0.96)
for v_rate in v_rates:
    classrooms_list = []
    students_list = []
    for trial in range(trials):
        (classrooms, students) = run_simulation(infection_rate, v_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student=num_periods, weekends=True)
        classrooms_list.append(classrooms)
        students_list.append(students)
    plt.figure(4)
    graph_disease_burden(students_list, time_steps, lbl=str(round(v_rate*100,1))+"% vaccinated", legend=True)
    plt.figure(5)
    graph_frac_students_infected(students_list, time_steps, lbl=str(round(v_rate*100,1))+"% vaccinated")
# ... and put legends on the figures
plt.figure(4)
plt.legend(loc='best')
plt.figure(5)
plt.legend(loc='best')

# EXPERIMENT 3: varying infection rate, holding all other params at standard
f6 = plt.figure(6)
f7 = plt.figure(7)
infect_rates = (0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01)
for i_rate in infect_rates:
    classrooms_list = []
    students_list = []
    for trial in range(trials):
        (classrooms, students) = run_simulation(i_rate, vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student=num_periods, weekends=True)
        classrooms_list.append(classrooms)
        students_list.append(students)
    plt.figure(6)
    graph_disease_burden(students_list, time_steps, lbl=str(round(i_rate*100,1))+"% infection rate", legend=True)
    plt.figure(7)
    graph_frac_students_infected(students_list, time_steps, lbl=str(round(i_rate*100,1))+"% infection rate")
# ... and put legends on the figures
plt.figure(6)
plt.legend(loc='best')
plt.figure(7)
plt.legend(loc='best')

# EXPERIMENT 4: different constant for infection rate, varying vaccination rate, holding all other params at standard
f8 = plt.figure(8)
f9 = plt.figure(9)
new_infect_rate = 0.03
v_rates = (0.46, 0.56, 0.66, 0.76, 0.86, 0.96)
for v_rate in v_rates:
    classrooms_list = []
    students_list = []
    for trial in range(trials):
        (classrooms, students) = run_simulation(new_infect_rate, v_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student=num_periods, weekends=True)
        classrooms_list.append(classrooms)
        students_list.append(students)
    plt.figure(8)
    graph_disease_burden(students_list, time_steps, lbl=str(round(v_rate*100,1))+"% vaccinated", legend=True)
    plt.figure(9)
    graph_frac_students_infected(students_list, time_steps, lbl=str(round(v_rate*100,1))+"% vaccinated")
# ... and put legends on the figures
plt.figure(8)
plt.legend(loc='best')
plt.figure(9)
plt.legend(loc='best')

# EXPERIMENT 5: standard constants, varying initial number of patients with the flu
f13 = plt.figure(13)
f14 = plt.figure(14)
n_init = (1, 2, 3, 4, 5, 6)
for n in n_init:
    classrooms_list = []
    students_list = []
    for trial in range(trials):
        (classrooms, students) = run_simulation(infection_rate, vaccination_rate, vaccination_effectiveness, class_sizes, time_steps, classes_per_student=num_periods, weekends=True, init_patients=n)
        classrooms_list.append(classrooms)
        students_list.append(students)
    plt.figure(13)
    graph_disease_burden(students_list, time_steps, lbl=str(n)+" initial patients", legend=True)
    plt.figure(14)
    graph_frac_students_infected(students_list, time_steps, lbl=str(n)+" initial patients")
# ... and put legends on the figures
plt.figure(13)
plt.legend(loc='best')
plt.figure(14)
plt.legend(loc='best')

f1.savefig('sim-data/frac_infected.pdf')
f2.savefig('sim-data/days_infected.pdf')
f3.savefig('sim-data/disease_burden.pdf')
f10.savefig('sim-data/disease_burden_closeup.pdf')
f11.savefig('sim-data/disease_burden_nowkend.pdf')
f12.savefig('sim-data/disease_burden_nowkend_closeup.pdf')
f4.savefig('sim-data/disease_burden_varying_vrate.pdf')
f5.savefig('sim-data/frac_students_infected_varying_vrate.pdf')
f6.savefig('sim-data/disease_burden_varying_p.pdf')
f7.savefig('sim-data/frac_students_infected_varying_p.pdf')
f8.savefig('sim-data/disease_burden_diff_p_vary_vrate.pdf')
f9.savefig('sim-data/frac_students_infected_diff_p_vary_vrate.pdf')
f13.savefig('sim-data/disease_burden_vary_init_patients.pdf')
f14.savefig('sim-data/frac_students_infected_vary_init_patients.pdf')

plt.show()

