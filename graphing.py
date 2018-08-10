import numpy as np
import matplotlib.pyplot as plt
import math

from Student import *
from room_assign import *


def graph_frac_infected(classrooms_list, num_days):
    ''' Graphs the average fraction of classroom infected for each classroom,
    averaged over all trials run. Params:
        classrooms_list[which_trial][which_classroom][which_time_step][row][column]
        num_days: number of time steps'''
    t_vals = list(range(num_days))
    # Data: frac_sick_data[which_trial][which_classroom]
    frac_sick_data = list()
    
    # Compute fraction of students sick for each trial for each classroom
    for trial, classrooms in enumerate(classrooms_list):
        frac_sick_data.append(list())
        for i, cs in enumerate(classrooms):
            end_results = cs[-1, :, :]
            counts = list()
            for row in end_results:
                for student in row:
                    counts = counts + student.days_infected
    
            class_shape = np.shape(end_results)
            class_size = class_shape[0]*class_shape[1]
            frac_sick = [counts.count(x) * 1.0 / class_size for x in t_vals]
            frac_sick_data[trial].append(frac_sick)
            
    # Compute average fraction of students sick for each classroom across all trials
    n_trials = len(frac_sick_data)
    for classroom_index in range(len(frac_sick_data[0])):
        avg_frac_sick = []
        for t in range(num_days):
            frac_sick_t = [frac_sick_data[trial][classroom_index][t] for trial in range(n_trials)]
            avg_frac_sick.append(sum(frac_sick_t) / len(frac_sick_t))
        plt.plot(t_vals, avg_frac_sick) #, label="Classroom " + str(classroom_index+1))

    # Label figure
#    plt.legend(loc='best')
    plt.ylabel("Fraction Classroom Infected")
    plt.xlabel("Day")
    plt.xticks(list(range(0, num_days, 5)))


def graph_days_infected(classrooms_list, num_days):
    ''' Graphs the distribution of the number of days that students are
    infected with the flu. Params:
        classrooms_list[which_trial][which_classroom][which_time_step][row][column]
        num_days: number of time steps'''
    n_trials = len(classrooms_list)
    
    days_sick = list()
    for trial, classrooms in enumerate(classrooms_list):
        for i, cs in enumerate(classrooms):
            end_results = cs[-1, :, :]
            for row in end_results:
                for student in row:
                    days_sick += [len(student.days_infected)]
    
    bin_width = 5
    edges = list(range(0, num_days + 1, bin_width))
    ax = plt.gca()
    plt.hist(days_sick, bins=edges, rwidth=0.9)
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:1.0f}'.format(x // n_trials) for x in y_vals])
    plt.title("Number of days spent sick (averaged across all trials)")
    plt.xlabel("Days Spent Sick")
    plt.ylabel("# Students")
    plt.xticks(edges)


def graph_disease_burden(students_list, num_days, lbl='', legend=False):
    '''Graphs, for each time step, the number of students who are or have been
    sick with the flu; averaged over all trials. Params:
        students_list[which_trial][which_student]
        num_days: number of time steps
        v_rate: vaccination rate for this experiment
        legend: whether to include a legend in the graph'''
    n_trials = len(students_list)
    n_students = len(students_list[0])
    sick_counts = [0 for i in range(num_days)]
    for t in range(n_trials):
        for student in students_list[t]:
            if len(student.days_infected) > 0:
                day_one = student.days_infected[0]
                for day in range(day_one, num_days):
                    sick_counts[day] += 1
    for i in range(len(sick_counts)):
        sick_counts[i] /= 1.0*(n_students * n_trials)
    if (legend):
        plt.plot(list(range(num_days)), sick_counts, label=lbl)
    else:
        plt.plot(list(range(num_days)), sick_counts)
    plt.ylabel("Fraction of Students Who Have Caught the Flu")
    plt.xlabel("Day")

