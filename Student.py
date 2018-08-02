#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:42:07 2018

@author: KCR
"""

class Student:
    def __init__(self):
        # all possible states (subject to change)
        # state 0 = susceptible/uninfected

        # state 2 = sick
        # state 3 = recovered (immune)
        self.state = 0
        self.days_infected = list()
        self.neighbors = set()

        # determined using Poisson Distribution with lambda =
        self.stays_sick_for = 10  # 10 is placeholder for now

    # toString method
    def __repr__(self):
        return str(self.state)

    def set_days_sick(self, days_sick):
        self.stays_sick_for = days_sick

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def add_neighbor(self, other):
        self.neighbors.add(other)

    def get_neighbors(self):
        return self.neighbors

    def add_day_infected(self, day):
        self.days_infected.append(day)

    def get_days_infected(self):
        return self.days_infected
