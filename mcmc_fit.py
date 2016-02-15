'''
'''
# Author: Andrew Nadolski
# Filename: mcmc_fit.py

import numpy as np
import matplotlib.pyplot as plt
import emcee
import random

class MCMC(object):
    '''
    '''
    def __init__(self, name='MCMC'):
        self.name = name
        self.num_wagers = []
        self.val_wagers = []

    def __repr__(self):
        return '%r' % self.name

    def roll_dice(self):
        self.roll = random.randint(1,100)
        if self.roll == 100 or self.roll <= 50:
            return False
        else:
            return True

    def simple_bettor(self, funds, initial_wager, wager_count):
        self.value = funds
        self.wager = initial_wager
        self.current_wager = 1
        while self.current_wager <= wager_count:
            if self.roll_dice():
                self.value += self.wager
                self.num_wagers.append(self.current_wager)
                self.val_wagers.append(self.value)
                self.current_wager += 1
            else:
                self.value -= self.wager
                self.num_wagers.append(self.current_wager)
                self.val_wagers.append(self.value)
                self.current_wager += 1

    def show_mc(self):
        self.fig = plt.Figure()
        self.fig.add_subplot(111)
        plt.plot(self.num_wagers, self.val_wagers, alpha=.15)
        plt.show()
