import random
import numpy as np

from Student import *

def room_assign(room_dim):
#room_dim = list of room dimensions for each room (e.g. seat length x seat width)
#periods = number of time periods (student has n periods (classes) a day)

    # determine number of students (seats) in each room
    room_size = []
    for dim in room_dim:  # for each set of dim in room_dim
        room_size.append(dim[0] * dim[1])  # size = r * c

    print(room_size)

    #determine total number of students, generate list of students
    num_students = sum(room_size)  # total number of students
    students = range(num_students)  # list of student numbers


    random.shuffle(students) #shuffle list of students randomly

    #assigns students to classrooms
    c = 0 #index for where in list room starts
    d = 0 #index for which room
    room_list = []
    for size in room_size:
        room_list.append(np.reshape((students[c:c + size]), room_dim[d]))
        c += size #adds previous room size to room start index
        d += 1 #moves to next room

    return room_list


print(room_assign([[4,2],[2,3],[5,4]])) #example






