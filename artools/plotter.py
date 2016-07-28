"""Contains convenience functions for plotting AR simulation results such as transmission and reflection.
"""

#Filename: plotter.py
#Author: Andrew Nadolski

import os
import pprint
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np

"""
TODO

7/26
    * Debug _convert_to_wavelength(). The plot output looks funny....
    * write a bandpass drawing function that take upper and lower limits
    as input and draws a semi-opaque, colored rectangular region
"""


class Plot:
    """Contains the generic elements needed for an AR simulation plot

    Attributes
    ----------
    bandpasses : list
        A list of bandpasses (tuples), where each element contains a lower and
        upper bound, a color, a name, and an opacity. Bandpasses can be added
        using ``add_bandpass()``.
    data : array
        Defaults to 'None' type until the data to be plotted are loaded.
        Once data are loaded, any operations on the data happen to this instance.
        Any call to ``load_data()`` will overwrite this instance.
    draw_bandpasses : boolean
        If `True`, the contents of ``bandpasses`` is drawn on the plot. If 
        `False`, the contents of ``bandpasses`` is ignored when drawing the
        plot. Defaults to `False`.
    frequency_units : string
        The units to plot on the frequency axis, if it exists. Must be one of:
            'Hz',
            'KHz',
            'MHz',
            'GHz',
            'THz'.
    legend : boolean
        If `True`, draws a legend on the plot. Defaults to `False`.
    raw_data : array
        Defaults to 'None' type until the data to be plotted are loaded.
        Once the data are loaded, this copy of the data are kept in the
        'as-loaded' state so they may be reverted to easily. Any call to
        ``load_data()`` will overwrite this copy.
    save_name : string
        The name under which the output plot is saved. Defaults to
        'my_plot_XXXXX.pdf' where `XXXXX` is a time-stamp to avoid overwriting
        previous plots.
    save_path : string
        The path to which the output plot will be saved. Defaults to the current
        working directory
    title : string
        The title of the plot
    type : string
        The type of plot
    wavelength_units : string
        The units to plot on the wavelength axis, if it exists. Must be one of:
            'm',
            'cm',
            'mm',
            'um',
            'micron'.
    xlabel : string
        The x-axis label
    ylabel : string
        The y-axis label
    """
    def __init__(self):
        self.bandpasses = []
        self.data = None
        self.draw_bandpasses = False
        self.draw_legend = False
        self.frequency_units = 'GHz'
        self.raw_data = None
        self.save_name = 'my_plot_{t}.pdf'.format(t=time.ctime(time.time()))
        self.save_path = '.'
        self.title = 'Generic plot'
        self.type = 'Generic'
        self.vs_frequency = True
        self.wavelength_units = 'mm'
        self.xlabel = None
        self.ylabel = None

    def __repr__(self):
        return '{type} plot'.format(type=self.type)

    def _convert_to_wavelength(self, frequencies):
        """Converts frequencies to wavelength. Ignores division by zero
        errors and sets results of division by zero to 0.
        
        Arguments
        ---------
        frequencies : array
            An array of frequencies given in hertz

        Returns
        -------
        wavelengths : array
            An array of wavelengths computed from the input frequency array
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            wavelengths = np.true_divide(3e8, frequencies)
            wavelengths[np.isinf(wavelengths)] = 0.
        return wavelengths

    def _draw_bandpasses(self):
        """Draws the contents of ``bandpasses`` attribute on the plot
        """
        for bandpass in self.bandpasses:
            low = bandpass[0]
            high = bandpass[1]
            color = bandpass[2]
            label = bandpass[3]
            opacity = bandpass[4]
            plt.axvspan(low, high, fc=color, ec='none', alpha=opacity, label=label)
        return
        
    def _draw_legend(self):
        """Draws a legend on the plot at the position matplotlib deems best
        """
        plt.legend(fontsize='x-small')
        return

    def _make_save_path(self):
        """Assembles the full save path for the output plot
        
        Returns
        -------
        path : string
            The full path to which the output plot will be saved
        """
        if self.save_name.endswith('.pdf'): 
            path = os.path.join(self.save_path, self.save_name)
        else:
            self.save_name = self.save_name+'.pdf'
            path = os.path.join(self.save_path, self.save_name)
        return path

    def _shape_data(self):
        """Does some basic data manipulation based on plot attributes
        such as preferred units
        """
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
        
    def add_bandpass(self, lower_bound, upper_bound, color=None, label=None, opacity=0.1):
        """Adds a bandpass region to the plot. The region is a shaded rectangle 
        spanning the full height of the plot.

        Arguments
        ---------
        lower_bound : float
            The lower edge of the bandpass, given in x-axis units.
        upper_bound : float
            The upper edge of the bandpass, given in x-axis units.
        color : string, optional
            The color of the bandpass region. Can be any color string
            recognized by matplotlib. Defaults to 'None', which means a
            random color will be chosen for the bandpass shading.
        label : string, optional
            The name that will appear in the legend, if a legend is used.
            Deafults to 'None', which means no name will be displayed in
            the legend.
        opacity : float, optional
            The opacity of the shaded region. Must be between 0 and 1, inclusive.
            1 is completely opaque, and 0 is completely transparent.
        """
        bandpass = (lower_bound, upper_bound, color, label, opacity)
        self.bandpasses.append(bandpass)
        return

    def load_data(self, data):
        """Load a new set of data while retaining other plot
        characteristics

        Arguments
        ---------
        data : numpy array
            The data to be plotted. Replaces any existing data in
            the 'data' and 'raw_data' attributes.
        """
        self.data = data
        self.raw_data = data
        return

    def make_plot(self):
        """Draws a plot of the loaded data

        Arguments
        ---------
        data : array
            A 2-element array where the first element is a set of
            frequencies (or wavelengths) and the second elements
            is a set of transmissions (or reflections)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.set_xlabel()
        ax.set_title(self.title)
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel(self.xlabel)
        self._shape_data()
        if self.type == 'Transmission':
            ax.plot(self.data[0], self.data[1])
        elif self.type == 'Reflection':
            ax.plot(self.data[0], self.data[2])
        else:
            ax.plot(self.data[0], self.data[0])
        if self.draw_bandpasses:
            self._draw_bandpasses()
        if self.draw_legend:
            self._draw_legend()
        path = self._make_save_path()
        plt.savefig(path, bbox_inches='tight')

    def plot_vs_freq(self):
        """Plot the data vs frequency
        """
        self.vs_frequency = True
        return

    def plot_vs_wavelength(self):
        """Plot the data vs wavelength
        """
        self.vs_frequency = False
        return

    def revert_data(self):
        """Resets the data to its original, 'as-loaded' form
        """
        self.data = self.raw_data
        return

    def set_title(self, title):
        """Set the plot title

        Arguments
        ---------
        title : string
            The title of the plot
        """
        self.title = title
        return

    def set_xlabel(self, xlabel=None):
        """Set the x-axis label

        Arguments
        ---------
        xlabel : string, optional
            The label for the x-axis. Defaults to `None`. If `None`, x-axis
            label is chosen based on the x-axis units
        """
        if xlabel is None:
            if self.vs_frequency:
                self.xlabel = r'$\nu$'+' [{}]'.format(self.frequency_units)
            else:
                self.xlabel = r'$\lambda$'+' [{}]'.format(self.wavelength_units)
        else:
            self.xlabel = xlabel
        return

    def set_ylabel(self, ylabel):
        """Set the y-axis label

        Arguments
        ---------
        ylabel : string
            The label for the y-axis
        """
        self.ylabel = ylabel
        return

    def show_attributes(self):
        """Convenience function to display all the attributes of the plot
        """
        print('The plot attributes are:\n')
        pprint.pprint(vars(self))
        return

    def toggle_bandpasses(self):
        """Toggles the value of ``draw_bandpasses`` attribute between
        `False` and `True`. If set to `False` bandpasses will be ignored. If
        `True`, bandpasses will be drawn on the plot.
        """
        if type(self.draw_bandpasses) == type(True):
            if self.draw_bandpasses:
                self.draw_bandpasses = False
            elif not self.draw_bandpasses:
                self.draw_bandpasses = True
        else:
            raise TypeError("'draw_bandpasses' must be boolean")
        return

    def toggle_legend(self):
        """Toggles the value of ``draw_legend`` attribute between `False` and 
        `True`. If set to `False` the legend will be ignored. If `True`, 
        the legend will be drawn on the plot.
        """
        if type(self.draw_legend) == type(True):
            if self.draw_legend:
                self.draw_legend = False
            elif not self.draw_legend:
                self.draw_legend = True
        else:
            raise TypeError("'draw_legend' must be boolean")
        return


class ReflectionPlot(Plot):
    """Contains elements needed for a reflection plot
    """
    def __init__(self):
        Plot.__init__(self)    # Inherit attributes from generic 'Plot' class
        self.title = 'Reflection plot'
        self.type = 'Reflection'
        self.ylabel = 'Reflection (%)'


class TransmissionPlot(Plot):
    """Contains elements needed for a transmission plot
    """
    def __init__(self):
        Plot.__init__(self)    # Inherit attributes from generic 'Plot' class
        self.title = 'Transmission plot'
        self.type = 'Transmission'
        self.ylabel = 'Transmission (%)'


class MCPlot(Plot):
    """Contains elements needed for a Monte Carlo plot
    """
    def __init__(self):
        Plot.__init__(self)    # Inherit attributes from generic 'Plot' class
        self.title = 'MCMC plot'
        self.type = 'MCMC'
