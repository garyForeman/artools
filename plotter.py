''' Contains convenience functions for plotting AR simulation results such as transmission and reflection.
'''

#Filename: plotter.py
#Author: Andrew Nadolski

import os
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np

class Plot:
    ''' Contains the generic elements needed for an AR simulation plot

    Attributes
    ----------
    title : string
        The title of the plot
    type : string
        The type of plot
    '''
    def __init__(self):
        self.title = 'Generic plot'
        self.type = 'Generic'

    def __repr__(self):
        return '{type} plot'.format(type=self.type)
        

class ReflectionPlot(Plot):
    ''' Contains elements needed for a reflection plot

    Attributes
    ----------
    title : string
        The title of the plot
    type : string
        The type of plot
    '''
    def __init__(self):
        self.title = 'Reflection plot'
        self.type = 'Reflection'

    def __repr__(self):
        return '{type} plot'.format(type=self.type)

class TransmissionPlot(Plot):
    ''' Contains elements needed for a transmission plot

    Attributes
    ----------
    title : string
        The title of the plot
    type : string
        The type of plot
    '''
    def __init__(self):
        self.title = 'Transmission plot'
        self.type = 'Transmission'

    def __repr__(self):
        return '{type} plot'.format(type=self.type)

class MCPlot(Plot):
    ''' Contains elements needed for a Monte Carlo plot

    Attributes
    ----------
    title : string
        The title of the plot
    type : string
        The type of plot
    '''
    def __init__(self):
        self.title = 'MCMC plot'
        self.type = 'MCMC'

    def __repr__(self):
        return '{type} plot'.format(type=self.type)
