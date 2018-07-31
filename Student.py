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
        # state 1 = susceptible + sick people around
        # state 2 = sick
        # state 3 = recovered (immune)
        self.state = 0
        self.days_infected = 0
        self.neighbors = set()

    # toString method
    def __repr__(self):
        return str(self.state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def add_neighbor(self, other):
        self.neighbors.add(other)

    def get_neighbors(self):
        return self.neighbors

    def get_days_infected(self):
        return self.days_infected
