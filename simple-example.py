import random

# For reproducability
random.seed(100)

# (row, column) dimensions of room
r_dim = 4
c_dim = 5

# 0 = uninfected, 1 = sick
room = [[0 for j in range(c_dim)] for i in range(r_dim)]

# Pick one student at random to catch the flu

source = (random.randrange(0, r_dim), random.randrange(0, c_dim))
room[source[0]][source[1]] = 1

# Constants
d = 5 # number of days to run simulation
p = 0.25 # probability of transmission to adjacent student

def pretty_print(room):
    '''Generates printable string of room layout and infection status'''
    result = ''
    for row in range(len(room)):
        result += '\t'.join([str(seat) for seat in room[row]]) + '\n'
    return result



# Simulation
print("t = 0:")
print(pretty_print(room))

for t in range(1, d+1):
    # Get state of each seat for time step t
    new_room = [[0 for j in range(c_dim)] for i in range(r_dim)]
    for row in range(r_dim):
        for col in range(c_dim):
            if room[row][col] == 0: # student uninfected
                num_sick = 0
                # count sick neighbors
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        # don't count yourself, stay within the grid,
                        # and neigbor is sick
                        if (dr, dc) != (0, 0) \
                                and row + dr >= 0 and row + dr < r_dim \
                                and col + dc >= 0 and col + dc < c_dim \
                                and room[row + dr][col + dc] == 1:
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
    room = [[new_room[r][c] for c in range(c_dim)] for r in range(r_dim)]
    
    # Print result
    print("t = " + str(t) + ":")
    print(pretty_print(room))
