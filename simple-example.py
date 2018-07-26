import random

# For reproducability
random.seed(100)

# (row, column) dimensions of room
r_dim = 4
c_dim = 5

# Constants
d = 5 # number of days to run simulation
p = 0.25 # probability of transmission to adjacent student

# 0 = uninfected, 1 = sick
room_log = [[[0 for j in range(c_dim)] for i in range(r_dim)] for t in range(d+1)]

# Pick one student at random to catch the flu

source = (random.randrange(0, r_dim), random.randrange(0, c_dim))
room_log[0][source[0]][source[1]] = 1

def pretty_print(room):
    '''Generates printable string of room layout and infection status'''
    result = ''
    for row in range(len(room)):
        result += '\t'.join([str(seat) for seat in room[row]]) + '\n'
    return result



# Simulation
print("t = 0:")
print(pretty_print(room_log[0]))

current_room = room_log[0]
for t in range(1, d+1):
    # Get state of each seat for time step t
    new_room = room_log[t]
    for row in range(r_dim):
        for col in range(c_dim):
            if current_room[row][col] == 0: # student uninfected
                num_sick = 0
                # count sick neighbors
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        # don't count yourself, stay within the grid,
                        # and neigbor is sick
                        if (dr, dc) != (0, 0) \
                                and row + dr >= 0 and row + dr < r_dim \
                                and col + dc >= 0 and col + dc < c_dim \
                                and current_room[row + dr][col + dc] == 1:
                            num_sick += 1
                if num_sick > 0:
                    # Calculate transmission probability
                    p_transmission = 1 - ((1-p)**num_sick)
                    rand = random.random()
                    if (rand < p_transmission):
                        new_room[row][col] = 1
            else: # student infected
                new_room[row][col] = 1
    
    # Update room to this new state
    current_room = room_log[t]
    
    # Print result
    print("t = " + str(t) + ":")
    print(pretty_print(room_log[t]))


# Stats

# Average number of days uninfected
uninfected_days = 0;
for row in range(r_dim):
    for col in range(c_dim):
        history = [room_log[t][row][col] for t in range(d+1)]
        uninfected_days += history.count(0)
uninfected_days /= (r_dim * c_dim) # average over total number of students
print("Average number of days uninfected: " + str(uninfected_days))

