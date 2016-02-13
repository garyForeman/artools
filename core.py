''' Contains the tools needed to set up a multilayer antireflection coating
simulation.

Based on transfer matrix method outlined in Hou, S. 1974.
'''
# Author: Andrew Nadolski
# Filename: infrastructure.py

"""
###### TODO ######
10 Feb 16
    * Improve the docstrings! ALL THE DOCSTRINGS!!!
11 Feb 16
    * Un-hardcode the optimization freq in ideal_thickness(), or maybe in the
      Builder __init__()? Track it down.
    * Add functionality that automatically sets up an FTS sim. It just needs to 
      automatically set the substrate, then reverse the given layers, then stick
      a vacuum layer on the end.
    * 

"""

import glob
import os
import pprint
import time
import _materials as mats
import numpy as np
import scipy as sp


class Layer(object):
    ''' Represents a layer in the AR coating.

    Arguments
    ----------
    material: string
        A key in the dictionary of materials.
    '''
    def __init__(self, material):
        self.name = material.lower()
        self.thickness = 5.
        self.type = 'Layer'
        try:
            self.dielectric = mats.Electrical.DIELECTRIC[self.name]
        except:
            raise KeyError('I don\'t know that material!')
        try:
            self.losstangent = mats.Electrical.LOSS_TAN[self.name]
        except:
            self.losstangent = 0
            print '\nI don\'t know this loss tangent. Setting loss to 0!'

    def __repr__(self):
        return '%r (AR layer)' % self.name

    def get_index(self):
        return (np.sqrt(self.dielectric))

    def ideal_thickness(self):
        return (1/np.sqrt(self.dielectric)*3e8/(4*160e9))


class BondingLayer(Layer):
    def __init__(self, material):
        Layer.__init__(self, material)
        self.thickness = 40.e-6 # typical Stycast 1266 thickness
        self.type = 'Bonding layer'

    def __repr__(self):
        return '%r (bonding layer)' % self.name


class SourceLayer(Layer):
    ''' A special case of Layer. The layer from which the simulated wave emanates.

    Arguments
    ---------
    material: string
        A key in the materials dictionary.
    '''
    def __init__(self, material):
        Layer.__init__(self, material)
        self.thickness = np.inf
        self.type = 'Source'

    def __repr__(self):
        return '%r (source layer)' % self.name


class SubstrateLayer(Layer):
    '''
    '''
    def __init__(self, material):
        Layer.__init__(self, material)
        self.thickness = 6.35e-3
        self.type = 'Substrate'
        
    def __repr__(self):
        return '%r (substrate)' % self.name


class TerminatorLayer(Layer):
    ''' A special case of Layer. The layer upon which the simulated wave terminates.

    Arguments
    ---------
    material: string
        A key in the materials dictionary.
    '''
    def __init__(self, material):
        Layer.__init__(self, material)
        self.thickness = np.inf
        self.type = 'Terminator'

    def __repr__(self):
        return '%r (terminator layer)' % self.name


class Builder(object):
    ''' In progress. Should assemble all the layer objects and set up all the simulation
    parameters.
    '''
    def __init__(self):
        self.source = SourceLayer('vacuum')
        self.terminator = TerminatorLayer('alumina')
        self.structure = []
        self.freq_sweep = 0.
        self.optimization_frequency = 160e9 # 160 GHz

    def _get_R(self, net_r_amp):
        return np.abs(net_r_amp)**2

    def _get_T(self, polarization, net_t_amp, n_i, n_f, theta_i=0., theta_f=0.):
        if (polarization=='s'):
            return np.abs(net_t_amp**2) * (n_f/n_i)
        elif (polarization=='p'):
            return np.abs(net_t_amp**2) * (n_f/n_i)
        else:
            raise ValueError("Polarization must be 's' or 'p'")

    def _make_2x2(self, A11, A12, A21, A22, dtype=float):
        array = np.empty((2,2), dtype=dtype)
        array[0,0] = A11
        array[0,1] = A12
        array[1,0] = A21
        array[1,1] = A22
        return array

    def _make_meters(self):
        ''' Convert the thickness from given units to meters. Assumes the thickness is
        given in mils.
        '''
        for layer in self.structure:
            layer.thickness = layer.thickness * 2.54e-5 # convert from mils to meters
        return

    def _r_at_interface(self, polarization, n_1, n_2):
        ''' Description
        '''
        if polarization == 's':
            return ((n_1 - n_2)/(n_1 + n_2))
        elif polarization == 'p':
            return ((n_1 - n_2)/(n_1 + n_2))
        else:
            raise ValueError("Polarization must be 's' or 'p'")

    def _t_at_interface(self, polarization, n_1, n_2):
        ''' Description
        '''
        if polarization == 's':
            return 2*n_1/(n_1 + n_2)
        elif polarization == 'p':
            return 2*n_1/(n_1 + n_2)
        else:
            raise ValueError("Polarization must be 's' or 'p'")

    def _unpolarized_simulation(self, frequency, theta_0=0):
        s_data = self.simulate(frequency, 's', theta_0)
        p_data = self.simulate(frequency, 'p', theta_0)
        T = (s_data + p_data)/2
        return T

    def clear_stack(self):
        ''' Remove all elements from the current AR stack.
        '''
        self.structure = []
        return

    def interconnect(self, stack):
        ''' Connect all the AR coating layer objects.

        Arguments
        ---------
        stack: list
            Contains the dictionary keys of the layers to be simulated.
        '''
        self.clear_stack()
        self.structure.append(self.source)
        for i in range(len(stack)):
            self.structure.append(stack[i])
        self.structure.append(self.terminator)
        return

    def set_freq_sweep(self, lower_bound, upper_bound, resolution=1):
        ''' Set the frequency range over which the simulation will run.
        
        Arguments
        ---------
        lower_bound: float
            The low end of the frequency range, given in GHz.
        upper_bound: float
            The high end of the frequency range, given in GHz.
        reolution: float, optional
            The interval at which to sample the frequency range, given in GHz.
            If 'resolution' is not specified, defaults to 1 GHz.
        '''
        self.freq_sweep = np.linspace(lower_bound*1e9, upper_bound*1e9, (upper_bound-lower_bound)/resolution)
        return

    def set_source_layer(self, material='vacuum'):
        ''' Change the default source layer from vacuum to something else. I'm not 
        sure why you would want to do this.

        Arguments
        ---------
        material: string, optional
            A key in the dielectrics dictionary. If 'material' is not specified,
            defaults to 'vacuum'.
        '''
        self.source = SourceLayer(material)
        return

    def set_terminator_layer(self, material='alumina'):
        ''' Change the default layer from alumina to something else. If running an FTS
        simulation, the terminator layer will be set as 'vacuum'.

        Arguments
        ---------
        material: string, optional
            A key in the dielectrics dictionary. If 'material' is not specified,
            defaults to 'alumina'.
        '''
        self.terminator = TerminatorLayer(material)
        return

    def show_available_materials(self):
        ''' List the materials with known properties. The simulator can handle these
        materials.

        The listed material names are keys in the materials properties dictionary. 
        '''
        print '\nThe materials with known dielectric properties are:\n'
        pprint.pprint(mats.Electrical.DIELECTRIC)
        print '\nThe materials with known loss tangents are:\n'
        pprint.pprint(mats.Electrical.LOSS_TAN)
        return

    def simulate(self, frequency, polarization, theta_0=0):
        ''' Description

        Arguments
        ---------

        Returns
        -------
        '''
        # check the desired polarization
        if polarization == 'u':
            return self._unpolarized_simulation(frequency)

        # get all the indices of refraction in one place
        n = []
        [n.append(layer.get_index()) for layer in self.structure]

        print self.structure
        # get all thicknesses in one place
        d = []
        print [layer.thickness for layer in self.structure]
        [d.append(layer.thickness) for layer in self.structure if (layer.type == 'Layer' or layer.type == 'Substrate')]
        d.insert(0, self.structure[0].thickness)
        d.append(self.structure[-1].thickness)
        
        # convert theicness and dielectric lists to numpy arrays
        n = np.asarray(n)
        d = np.asarray(d)

        # find the wavevectors, k
        k = 2*np.pi * n * frequency/3e8
        olderr = sp.seterr(invalid= 'ignore') # turn off 'invalid multiplication' error; it's just the 'inf' boundaries
        delta = k * d
        sp.seterr(**olderr) # turn the error back on

        # now get transmission and reflection amplitudes
        t_amp = np.zeros((len(self.structure), len(self.structure)), dtype=complex)
        r_amp = np.zeros((len(self.structure), len(self.structure)), dtype=complex)

        for i in range(len(self.structure)-1):
            t_amp[i,i+1] = self._t_at_interface(polarization, n[i], n[i+1])
            r_amp[i,i+1] = self._r_at_interface(polarization, n[i], n[i+1])

        M = np.zeros((len(self.structure),2,2),dtype=complex)
        for i in range(1,len(self.structure)-1):
            M[i] = 1/t_amp[i,i+1] * np.dot(self._make_2x2(np.exp(-1j*delta[i]),
                                                          0., 0., np.exp(1j*delta[i]),
                                                          dtype=complex),
                                           self._make_2x2(1., r_amp[i,i+1], r_amp[i,i+1], 1.,
                                                          dtype=complex))
        M_prime = self._make_2x2(1., 0., 0., 1., dtype=complex)
        for i in range(1, len(self.structure)-1):
            M_prime = np.dot(M_prime, M[i])
        M_prime = np.dot(self._make_2x2(1., r_amp[0,1], r_amp[0,1], 1., dtype=complex)/t_amp[0,1],
                         M_prime)
        
        # Now find the net transmission and reflection amplitudes
        t = 1/M_prime[0,0]
        r = M_prime[0,1]/M_prime[0,0]
        
        # Now find the net transmitted and reflected power
        T = self._get_T(polarization, t, n[0], n[-1])
        R = self._get_R(r)
        return T

    def snell(self, indices, theta_0):
        ''' Description
        '''
        return sp.arcsin(np.real_if_close(n_list[0]*np.sin(th_0) / n_list))
