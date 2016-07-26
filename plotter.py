''' Contains convenience functions for plotting AR simulation results such as transmission and reflection.
'''

#Filename: plotter.py
#Author: Andrew Nadolski

import os
import pprint
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np

'''
TODO

7/26
    * Debug _convert_to_wavelength(). The plot output looks funny....

'''


class Plot:
    ''' Contains the generic elements needed for an AR simulation plot

    Attributes
    ----------
    data : numpy array
        Defaults to 'None' type until the data to be plotted are loaded.
        Once data are loaded, any operations on the data happen to this instance.
        Any call to 'load_data()' will overwrite this instance.
    frequency_units : string
        The units to plot on the frequency axis, if it exists. Must be one of:
            'Hz'
            'KHz'
            'MHz'
            'GHz'
            'THz'
    raw_data : numpy array
        Defaults to 'None' type until the data to be plotted are loaded.
        Once the data are loaded, this copy of the data are kept in the
        'as-loaded' state so they may be reverted to easily. Any call to
        'load_data()' will overwrite this copy.
    title : string
        The title of the plot
    type : string
        The type of plot
    wavelength_units : string
        The units to plot on the wavelength axis, if it exists. Must be one of:
            'm'
            'cm'
            'mm'
            'um'
            'micron'
    xlabel : string
        The x-axis label
    ylabel : string
        The y-axis label
    '''
    def __init__(self):
        self.data = None
        self.frequency_units = 'GHz'
        self.raw_data = None
        self.title = 'Generic plot'
        self.type = 'Generic'
        self.vs_frequency = True
        self.wavelength_units = 'mm'
        self.xlabel = 'X-axis'
        self.ylabel = 'Y-axis'

    def __repr__(self):
        return '{type} plot'.format(type=self.type)

    def _convert_to_wavelength(self, frequencies):
        ''' Converts frequencies to wavelength. Ignores division by zero
        errors and sets results of division by zero to 0.
        
        Arguments
        ---------
        frequencies : array
            An array of frequencies given in hertz

        Returns
        -------
        wavelengths : array
            An array of wavelengths computed from the input frequency array
        '''
        with np.errstate(divide='ignore', invalid='ignore'):
            wavelengths = np.true_divide(3e8, frequencies)
            wavelengths[np.isinf(wavelengths)] = 0.
        return wavelengths
        
    def _shape_data(self):
        ''' Does some basic data manipulation based on plot attributes
        such as preferred units
        '''
        freq_units = {'Hz':1, 'KHz':10**3, 'MHz':10**6, 'GHz':10**9, 'THz':10**12}
        wave_units = {'m':1, 'cm':10**-2, 'mm':10**-3, 'um':10**-6, 'micron':10**-6}
        if self.vs_frequency:
            try:
                self.data[0] = self.data[0]/freq_units[self.frequency_units]
            except:
                raise ValueError('Unrecognized frequency units. See plotter.Plot() docstring for accepted units.')
        else:
            try:
                self.data[0] = self._convert_to_wavelength(self.data[0])
                self.data[0] = self.data[0]/wave_units[self.wavelength_units]
            except:
                raise ValueError('Unrecognized wavelength units. See plotter.Plot() docstring for accepted units.')
        return
        
    def load_data(self, data):
        ''' Load a new set of data while retaining other plot
        characteristics

        Arguments
        ---------
        data : numpy array
            The data to be plotted. Replaces any existing data in
            the 'data' and 'raw_data' attributes.
        '''
        self.data = data
        self.raw_data = data
        return

    def make_plot(self):
        ''' Draws a plot of the loaded data

        Arguments
        ---------
        data : array
            A 2-element array where the first element is a set of
            frequencies (or wavelengths) and the second elements
            is a set of transmissions (or reflections)
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(self.title)
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel(self.xlabel)
        self._shape_data()
        ax.plot(self.data[0], self.data[1])
        plt.savefig('artools/plots/my_plot_{t}.pdf'.format(t=time.ctime(time.time())), \
                        bbox_inches='tight')

    def plot_vs_freq(self):
        ''' Plot the data vs frequency
        '''
        self.vs_frequency = True
        return

    def plot_vs_wavelength(self):
        ''' Plot the data vs wavelength
        '''
        self.vs_frequency = False
        return

    def revert_data(self):
        ''' Resets the data to its original, 'as-loaded' form
        '''
        self.data = self.raw_data
        return

    def set_title(self, title):
        ''' Set the plot title

        Arguments
        ---------
        title : string
            The title of the plot
        '''
        self.title = title
        return

    def set_xlabel(self, xlabel):
        ''' Set the x-axis label

        Arguments
        ---------
        xlabel : string
            The label for the x-axis
        '''
        self.xlabel = xlabel
        return

    def set_ylabel(self, ylabel):
        ''' Set the y-axis label

        Arguments
        ---------
        ylabel : string
            The label for the y-axis
        '''
        self.ylabel = ylabel
        return

    def show_attributes(self):
        ''' Convenience function to display all the attributes of the plot
        '''
        print'The plot attributes are:\n'
        pprint.pprint(vars(self))
        return


class ReflectionPlot(Plot):
    ''' Contains elements needed for a reflection plot
    '''
    def __init__(self):
        Plot.__init__(self)    # Inherit attributes from generic 'Plot' class
        self.title = 'Reflection plot'
        self.type = 'Reflection'
        self.ylabel = 'Reflection (%)'


class TransmissionPlot(Plot):
    ''' Contains elements needed for a transmission plot
    '''
    def __init__(self):
        Plot.__init__(self)    # Inherit attributes from generic 'Plot' class
        self.title = 'Transmission plot'
        self.type = 'Transmission'
        self.ylabel = 'Transmission (%)'


class MCPlot(Plot):
    ''' Contains elements needed for a Monte Carlo plot
    '''
    def __init__(self):
        Plot.__init__(self)    # Inherit attributes from generic 'Plot' class
        self.title = 'MCMC plot'
        self.type = 'MCMC'
