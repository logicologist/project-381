import random
import matplotlib.pyplot as plt

# For reproducability
random.seed(100)


# For debugging

def pretty_print(room):
    '''Generates printable string of room layout and infection status'''
    result = ''
    for row in range(len(room)):
        result += '\t'.join([str(seat) for seat in room[row]]) + '\n'
    return result


# Simulation function

def room_flu_simulate(r_dim, c_dim, d, p):
    '''Run simulation on (r_dim x c_dim) room
    for d days with transmission probability p.'''
    
    # 0 = uninfected, 1 = sick
    room_log = [[[0 for j in range(c_dim)] for i in range(r_dim)] for t in range(d+1)]

    # Pick one student at random to catch the flu
    source = (random.randrange(0, r_dim), random.randrange(0, c_dim))
    room_log[0][source[0]][source[1]] = 1

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
                            # and neighbor is sick
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
        
        # Print result (for debugging)
        print("t = " + str(t) + ":")
        print(pretty_print(room_log[t]))

    # Return result (for stat analysis)
    return room_log


# Constants

# (row, column) dimensions of room
r_dim = 4
c_dim = 5
# number of days to run simulation
d = 5
# list of probabilities of transmission to adjacent student
p_list = (0.05, 0.1, 0.25, 0.5, 0.75, 1)


# Run the simulation!

f1 = plt.figure(1) # For plotting infections over time
plt.xlabel("Time step")
plt.ylabel("Fraction of students infected")
f2 = plt.figure(2) # For plotting average number of days uninfected
plt.xlabel("Probability of transmission")
plt.ylabel("Average number of days uninfected")

uninfected_avgs = []
for p in p_list:
    print('\n\n*** p = ' + str(p) + ' ***\n')
    room_log = room_flu_simulate(r_dim, c_dim, d, p)

    # Stats
    
    # Total infected at each time step
    infected = [0 for i in range(d+1)]
    for t in range(d+1):
        for row in range(r_dim):
            for col in range(c_dim):
                if room_log[t][row][col] == 1:
                    infected[t] += 1
    infected_frac = [1.0*infected[i] / (r_dim * c_dim) for i in range(len(infected))]
    plt.figure(1)
    plt.plot(range(d+1), infected_frac, label=('p = ' + str(p)))
    print("Total number of infections over time: " + str(infected))
    print("Total fraction of infections over time: " + str(infected_frac))

    # Average number of days uninfected
    uninfected_days = 0;
    for row in range(r_dim):
        for col in range(c_dim):
            history = [room_log[t][row][col] for t in range(d+1)]
            uninfected_days += history.count(0)
    uninfected_days /= (r_dim * c_dim) # average over total number of students
    print("Average number of days uninfected: " + str(uninfected_days))
    uninfected_avgs.append(uninfected_days)

plt.figure(1)
plt.legend()
plt.figure(2)
plt.plot(p_list, uninfected_avgs)

# Save figures
f1.savefig('simple-example-data/infections_over_time.pdf')
f2.savefig('simple-example-data/avg_days_uninfected.pdf')