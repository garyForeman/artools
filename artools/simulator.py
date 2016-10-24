"""
Simulator contains the tools needed to set up a multilayer antireflection
coating simulation.

Based on transfer matrix method outlined in Hou, H.S. 1974.
"""

# Author: Andrew Nadolski (with lots of help from previous work by Colin Merkel,
#         Steve Byrnes, and Aritoki Suzuki)
# Filename: simulator.py


import glob
import os
import pprint
import time
#import materials as mats
import numpy as np
import scipy as sp
import json
import logging

class Layer(object):
    """A layer in the AR coating.

    Attributes
    ----------
    name : string
        The name of the material comprising the layer. Default is 'Generic layer'
    thickness : float
        The thickness of the layer material. Default is 5 mil.
    type : string
        The type of layer. Default is `Layer`, which is an element of the AR
        coating. Other acceptable types are `Source` and `Terminator`.
    dielectric : float
        The dielectric constant of the layer material. Default is 1.
    losstangent : float
        The loss tangent of the material. Default is 0.
    """
    def __init__(self):
        self.name = 'Generic layer'
        self.thickness = 5.
        self.type = 'Layer'
        self.units = 'mil'
        self.dielectric = 1.
        self.losstangent = 0.

    def __repr__(self):
        """Return a nice string formatted representation of the layer."""
        return '{} (AR layer)'.format(self.name)

    def display_layer_parameters(self):
        """Display the attributes of the layer."""
        pprint.pprint(vars(self))
        return

    def get_index(self):
        """Return the refractive index of the layer."""
        return (np.sqrt(self.dielectric))

    def ideal_thickness(self, opt_freq=160e9):
        """Return the ideal quarter wavelength thickness of the AR coating layer
        at a given optimization frequency.
        
        Arguments
        ---------
        opt_freq : float, optional
            The optimization frequency (in Hz) for the layers thickness. Defaults 
            to 160 GHz.
        """
        return (1/np.sqrt(self.dielectric)*3e8/(4*opt_freq))


class SourceLayer(Layer):
    """A special case of ``Layer``; represents the layer from which the simulated wave 
    emanates.

    Attributes
    ----------
    thickness : float
        The thickness of the source layer. Defaults to ``numpy.inf`` since the model
        doesn't care about the thickness of source layer. The thickness of the
        source layer should not be changed under normal operations.
    type : string
        The type of layer. Default is `Source`, which is an element of the model,
        but not the coating. Other acceptable types are `Layer` and `Terminator`.
    """
    def __init__(self):
        Layer.__init__(self)
        self.thickness = np.inf
        self.type = 'Source'

    def __repr__(self):
        """Return a nice string formatted representation of the layer."""
        return '{} (source layer)'.format(self.name)


class SubstrateLayer(Layer):
    """A special case of ``Layer``; represents the layer to which the AR coating is 
    attached.

    Attributes
    ----------
    thickness : float
        The thickness of the substrate layer. Defaults to 250 mils, which is 
        the typical thickness of a sample puck used in the Berkeley FTS setup.
        This may be changed as is necessary, but the units must (eventually) be
        converted to meters before being fed to the simulator.
    type : string
        The type of layer
    """
    def __init__(self):
        Layer.__init__(self)
        self.thickness = 250.
        self.type = 'Substrate'
        
    def __repr__(self):
        return '{} (substrate)'.format(self.name)


class TerminatorLayer(Layer):
    """A special case of ``Layer``; represents the layer upon which the simulated wave 
    terminates.

    Attributes
    ----------
    thickness : float
        The thickness of the terminating layer. Defaults to ``numpy.inf`` since
        the model doesn't care about the thickness of the terminating layer. 
        The thickness of the terminating layer should not be changed under 
        normal operations.
    type : string
        The type of layer. Default is `Terminator`, which is an element of the model,
        but not the coating. Other acceptable types are `Source` and `Layer`.
    """
    def __init__(self):
        Layer.__init__(self)
        self.thickness = np.inf
        self.type = 'Terminator'

    def __repr__(self):
        """Return a nice string formatted representation of the layer."""
        return '{} (terminator layer)'.format(self.name)


class Builder(object):
    """The main body of the simulator code.

    Attributes
    ----------
    bands : list
        A list of n tuples, with each tuple composed of a lower and upper limit
        for a frequency band in units of hertz. Default is the SPT-3G bands.
    freq_sweep : array
        The range of frequencies to be simulated. Defaults to 0. Set a frequency
        sweep by calling ``set_freq_sweep()``.
    optimization_frequency : float
        The frequency (in Hz) at which to calculate the ideal thickness for a given
        material. Defaults to 160e9 Hz (160 GHz).
    save_name : string
        The name under which the results of the simulation are saved. Defaults to
        'transmission_data_XXXXX.txt' where `XXXXX` is a time-stamp to avoid
        overwriting previous simulation results.
    save_path : string
        The path to which the simulation results will be saved. Defaults to the 
        current working directory.
    source : object
        ``Layer`` object ``SourceLayer`` that defines where the wave emanates from.
        Default is `None`.
    stack : list
        The user-defined layers incorporated in the simulation EXCEPT the source
        and terminator layers. Default is empty list.
    structure : list
        The layers incorporated in the simulation INCLUDING the source and
        terminator layers. Default is empty list. The list is populated 
        by creating layers and calling ``_interconnect()``.
    terminator : object
        ``Layer`` object ``TerminatorLayer`` that defines where the wave terminates.
        Defaults is `None`.
    """
    def __init__(self):
        self.a_sweep_params = {'low':'unset', 'high':'unset', 'res':'unset',
                               'units':'unset'}
        self.angle_sweep = None
        self.bands = []
        self.f_sweep_params = {'low':'unset', 'high':'unset', 'res':'unset',
                               'units':'unset'}
        self.freq_sweep = None
        self.optimization_frequency = 160e9        # given in Hz, i.e. 160 GHz
        self.polarization = 's'
        self.save_name = 'artools_output_{t}'.format(t=time.ctime(time.time()))
        self.save_path = '.'
        self.source = None
        self.stack = []
        self.structure = []
        self.terminator = None

    def _calc_R_T_amp(self, polarization, n, delta, theta):
        """Calculate the reflected and transmitted amplitudes

        Arguments
        ---------
        polarization : string
            The polarization of the source wave. Must be one of: 's', 'p', or 'u'.
        n : array
            An array of refractive indices, ordered from source to terminator
        delta : array
            An array of wavevector offsets
        theta : array
            An array of Snell angles in radians. Expects the output of snell().
        
        Returns
        -------
        (r, t) : tuple
            A tuple where 'r' is the reflected amplitude, and 't' is the
            transmitted amplitude
        """
        t_amp = np.zeros((len(self.structure), len(self.structure)), dtype=complex)
        r_amp = np.zeros((len(self.structure), len(self.structure)), dtype=complex)
        for i in range(len(self.structure)-1):
            t_amp[i,i+1] = self._t_at_interface(polarization, n[i], n[i+1], theta[i], theta[i+1])
            r_amp[i,i+1] = self._r_at_interface(polarization, n[i], n[i+1], theta[i], theta[i+1])
        M = np.zeros((len(self.structure),2,2),dtype=complex)
        m_r_amp = np.zeros((len(self.structure),2,2), dtype=complex)
        m_t_amp = np.zeros((len(self.structure),2,2), dtype=complex)
        for i in range(1,len(self.structure)-1):
            m_t_amp[i] = self._make_2x2(np.exp(-1j*delta[i]), 0., 0., np.exp(1j*delta[i]), dtype=complex)
            m_r_amp[i] = self._make_2x2(1., r_amp[i,i+1], r_amp[i,i+1], 1., dtype=complex)
        m_temp = np.dot(m_t_amp, m_r_amp)
        for i in range(1,len(self.structure)-1):
            M[i] = 1/t_amp[i,i+1] * np.dot(self._make_2x2(np.exp(-1j*delta[i]),
                                                          0., 0., np.exp(1j*delta[i]),
                                                          dtype=complex),
                                           self._make_2x2(1., r_amp[i,i+1],
                                                              r_amp[i,i+1], 1.,
                                                              dtype=complex))
        M_prime = self._make_2x2(1., 0., 0., 1., dtype=complex)
        for i in range(1, len(self.structure)-1):
            M_prime = np.dot(M_prime, M[i])
        mod_M_prime = self._make_2x2(1.,r_amp[0,1], r_amp[0,1], 1., dtype=complex)/t_amp[0,1]
        M_prime = np.dot(self._make_2x2(1., r_amp[0,1], r_amp[0,1], 1.,
                                            dtype=complex)/t_amp[0,1], M_prime)
        t = 1/M_prime[0,0]
        r = M_prime[0,1]/M_prime[0,0]
        return (r, t)

    def _d_converter(self):
        """Check the units of all elements in the connected ar coating
        stack. Convert the lengths of the layers to meters if they are
        not already in meters.
        """
        units = {'um':1e-6, 'mm':1e-3, 'inch':2.54e-2, 'in':2.54e-2,
                 'micron':1e-6, 'mil':2.54e-5, 'm':1.0}
        for i in self.stack:
            i.thickness = i.thickness*units[i.units]
        return
        
    def _find_ks(self, n, frequency, tan, theta, lossy=True):
        """Calculate the wavenumbers.

        Arguments
        ---------
        n : array
            An array of refractive indices, ordered from source to
            terminator
        frequency : float
            The frequency at which to calculate the wavevector, k
        tan : array
            An array of loss tangents, ordered from vacuum to substrate
        theta : array
            An array of Snell angles for the model, in units of radians
        lossy : boolean, optional
            If `True` the wavevector will be found for a lossy material.
            If `False` the wavevector will be found for lossless material.
            Default is `True`.
        Returns
        -------
        k : complex
            The complex wavenumber, k
        """
        if lossy:
            k = 2*np.pi*n*frequency*np.cos(theta)*(1+0.5j*tan)/3e8
        else:
            k = 2*np.pi*n*frequency/3e8
        return k

    def _find_k_offsets(self, k, d):
        """Calculate the wavenumber offset, delta.

        Arguments
        ---------
        k : array
            The wavevector
        d : array
            An array of thicknesses, ordered from source to terminator

        Returns
        -------
        delta : array
            The phase difference
        """
        olderr = sp.seterr(invalid='ignore')  # turn off 'invalid multiplication' error;
                                              # it's just the 'inf' boundaries
        delta = k * d
        sp.seterr(**olderr)                   # turn the error back on
        return delta

    def _get_R(self, net_r_amp):
        """Return fraction of reflected power.

        Arguments
        ---------
        net_r_amp : float
            The net reflection amplitude after calculating the transfer matrix.
        """
        return np.abs(net_r_amp)**2

    def _get_T(self, net_t_amp, n_i, n_f, theta_i, theta_f):
        """Return the fraction of transmitted power.

        Arguments
        ---------
        polarization : string
            The polarization of the source wave. One of: 's' or 'p'.
        net_t_amp : float
            The net transmission amplitude after calculating the transfer matrix.
        n_i : float
            The index of refraction of material 'i'.
        n_f : float
            The index of refraction of material 'f'.
        theta_i : float
            The angle of incidence at interface 'i', in radians.
        theta_f : float
            The angle of incidence at interface 'f', in radians.
        """
        return np.abs(net_t_amp**2)*(n_f*np.cos(theta_f)/n_i*np.cos(theta_i))

    def _get_bandpass_stats(self, band, sim_output):
        """Compute basic descriptive statistics for a bandpass
        
        Arguments
        ---------
        band : tuple
            A tuple consisting of the lower and uppper bounds of the band
        sim_output : dict
            A dictionary of simulation output results. Must have keys:
                'freqs', 'T', 'R'

        Returns
        -------
        stats : list
            The basic descriptive statistics of the bandpass
        """
        units = {'Hz':1., 'hz':1., 'khz':1e3, 'KHz':1e3, 'mhz':1e6, 'MHz':1e6,
                 'ghz':1e9, 'GHz':1e9, 'thz':1e12, 'THz':1e12}
        convert = units[self.f_sweep_params['units']]
        freqs = sim_output['freqs']
        T = sim_output['T']
        R = sim_output['R']
        vals = np.ma.masked_inside(freqs, band[0]*convert, band[1]*convert)
        mask = np.ma.getmask(vals)
        masked_T = T[mask]
        masked_R = R[mask]
        T_avg = np.average(masked_T)
        R_avg = np.average(masked_R)
        stats = {'R_avg':R_avg, 'T_avg':T_avg}   
        return stats

    def _interconnect(self):
        """Connect all the AR coating layer objects, ensuring that the source
        and terminator layers come first and last, respectively.
        """
        self.clear_structure()
        self.structure.append(self.source)
        for i in range(len(self.stack)):
            self.structure.append(self.stack[i])
        self.structure.append(self.terminator)
        return

    def _make_2x2(self, A11, A12, A21, A22, dtype=float):
        """Return a 2x2 array quickly. (Thanks Steve!)

        Arguments
        ---------
        A11 : float
            Array element [0,0].
        A12 : float
            Array element [0,1].
        A21 : float
            Array element [1,0].
        A22 : float
            Array element [1,1].
        dtype : dtype, optional
            The datatype of the array. Defaults to float.
        """
        array = np.empty((2,2), dtype=dtype)
        array[0,0] = A11
        array[0,1] = A12
        array[1,0] = A21
        array[1,1] = A22
        return array

    def _make_header(self, sim_results):
        """Make the header for the output text file.

        Arguments
        ---------
        sim_input : dict
            A dictionary of simulation results

        Returns
        -------
        header : list
            A list of strings to write as the file header
        """
        f_input = sim_results['input']['frequency']
        a_input = sim_results['input']['angle']
        stats = sim_results['statistics']
        header = ['# Run date: {}\n'.format(time.ctime(time.time())),
                  '#\n',
                  '# Frequency sweep information\n',
                  '# low: {}, high: {}, res: {}, units: {}, polarization: {}\n'\
                      .format(f_input['f_low'], f_input['f_high'],
                              f_input['f_res'], f_input['f_units'],
                              f_input['pol']),
                  '#\n',
                  '# Angle sweep information\n',
                  '# low: {}, high: {}, res: {}, units: {}, this angle: {}\n'\
                      .format(a_input['a_low'], a_input['a_high'],
                              a_input['a_res'], a_input['a_units'],
                              a_input['a_input']),
                  '#\n',
                  '# Bandpass information\n']
        for i in range(len(self.bands)):
            header.append('# {}: {}\n'.format(self.bands[i], stats['band{}'.format(i)]))
        header.append('#\n')
        header.append('# Layer information\n')
        for layer in self.structure:
            header.append('# name: {}, thickness: {}, dielectric: {}, loss: {}, type: {}\n'.
                          format(layer.name, layer.thickness, layer.dielectric,
                                 layer.losstangent, layer.type))
        header.append('#\n')
        header.append('# Frequency (Hz)\t\t'
                      'Transmission\t\t\t'
                      'Reflection\t\t\t'
                      'Loss\n')
        return header

    def _make_save_path(self, save_path, save_name):
        """Assemble the file name and path to the results file.
        
        Returns
        -------
        path : string
            The full path to the save destination for the simulation results
        """
        if save_name.endswith('.txt'):
            path = os.path.join(save_path, save_name)
        else:
            save_name = save_name+'.txt'
            path = os.path.join(save_path, save_name)
        return path

    def _r_at_interface(self, polarization, n1, n2, theta1, theta2):
        """Calculate the reflected amplitude at an interface.

        Arguments
        ---------
        polarization : string
            The polarization of the source wave. Must be one of: 's' or 'p'.
        n1 : float
            The index of refraction of the first material.
        n2 : float
            The index of refraction of the second material.
        theta1 : float
            The angle of incidence at interface 1, in radians
        theta2 : float
            The angle of incidence at interface 2, in radians

        Returns
        -------
        reflected amplitude : float
            The amplitude of the reflected power
        """
        if polarization == 's':
            s_numerator = (n1*np.cos(theta1)-n2*np.cos(theta2))
            s_denominator = (n1*np.cos(theta1)+n2*np.cos(theta2))
            return s_numerator/s_denominator
        elif polarization == 'p':
            p_numerator = (n2*np.cos(theta1)-n1*np.cos(theta2))
            p_denominator = (n1*np.cos(theta2)+n2*np.cos(theta1))
            return p_numerator/p_denominator
        else:
            raise ValueError("Polarization must be 's' or 'p'")

    def _sort_ns(self):
        """Organize the refractive indices of the layers in the simulation.

        Returns
        -------
        n : array
            The ordered list of indices of refraction, from source to terminator
        """
        n = []
        for layer in self.structure:
            n.append(layer.get_index())
        n = np.asarray(n)
        return n

    def _sort_ds(self):
        """Organize the layers' thicknesses in the simulation.

        Returns
        -------
        d : array
            The ordered list of thicknesses, from source to terminator
        """
        d = []
        for layer in self.structure:
            if (layer.type == 'Layer' or layer.type == 'Substrate'):
                d.append(layer.thickness)
        d.insert(0, self.structure[0].thickness)
        d.append(self.structure[-1].thickness)
        d = np.asarray(d)
        return d

    def _sort_tans(self):
        """Organize the loss tangents of the layers in the simulation.

        Returns
        -------
        tan : array
            The ordered list of loss tangents, from source to terminator
        """
        tan = []
        for layer in self.structure:
            tan.append(layer.losstangent)
        tan = np.asarray(tan)
        return tan

    def _t_at_interface(self, polarization, n1, n2, theta1, theta2):
        """Calculate the transmission amplitude at an interface.

        Arguments
        ---------
        polarization : string
            The polarization of the source wave. Must be one of: 's' or 'p'.
        n1 : float
            The index of refraction of the first material.
        n2 : float
            The index of refraction of the second material.
        theta1 : float
            The angle of incidence at interface 1, in radians
        theta2 : float
            The angle of incidence at interface 2, in radians

        Returns
        -------
        transmitted_amplitude : float
            The amplitude of the transmitted power
        """
        if polarization == 's':
            s_numerator = 2*n1*np.cos(theta1)
            s_denominator = (n1*np.cos(theta1)+n2*np.cos(theta2))
            return s_numerator/s_denominator
        elif polarization == 'p':
            p_numerator = 2*n1*np.cos(theta1)
            p_denominator = (n1*np.cos(theta2)+n2*np.cos(theta1))
            return p_numerator/p_denominator
        else:
            raise ValueError("Polarization must be 's' or 'p'")

#     def _unpolarized_simulation(self, frequency, theta_0=0):
#         """Handle the special case of unpolarized light by running the model
#         for both 's' and 'p' polarizations and computing the mean of the two
#         results.

#         Arguments
#         ---------
#         frequency : float
#             The frequency (in Hz) at which to evaluate the model.
#         theta_0 : float, optional
#             The angle of incidence at the initial interface. Default is 0.
#         """
#         s_data = self.simulate(frequency, 's', theta_0)
#         p_data = self.simulate(frequency, 'p', theta_0)
#         T = (s_data + p_data)/2
#         return T
 
    def add_layer(self, material, thickness=5.0, units='mil', type='layer', \
                      stack_position=-1):
        """Create a layer from the set of pre-programmed materials and add it
        to the AR coating stack

        Arguments
        ---------
        material : string
            A key in the dictionary of materials found in materials.py.
            You can view these materials by calling
            'show_materials()'.
        thickness : float, optional
            The thickness of the AR coating layer material. Assumed to
            be given in 'mil' (i.e. thousandths of an inch) unless
            otherwise stated. Default is 5.
        units : string, optional
            The units of length for the AR coating layer. Default is 'mil'.
            Must be one of:
            { 'mil', 'inch', 'mm', 'm', 'um', 'in', 'micron' }
        type : string, optional
            The layer type. Default is 'layer', which corresponds to
            an AR layer. Other options are 'source' or 'terminator', which
            correspond to source and terminator layers, respectively.
        stack_position : int, optional
            The position of the layer in the AR coating stack, indexed
            from 0. Default is -1 (i.e., layer is automatically added
            to the end (bottom?) of the stack.
        """
        matpath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(matpath,'materials.json'), 'r') as f:
            mats = json.load(f)
            type = type.lower()
            if type == 'layer':
                layer = Layer()
                layer.name = material.lower()
                layer.thickness = thickness
                layer.units = units
                try:
                    layer.dielectric = mats[layer.name]['dielectric']
                except:
                    raise KeyError('I don\'t know that material!')
                try:
                    layer.losstangent = mats[layer.name]['tand']
                except:
                    layer.losstangent = 0
                    print('\nI don\'t know this loss tangent. Setting loss to 0!')
                if (stack_position == -1):
                    self.stack.append(layer)
                else:
                    self.stack.insert(stack_position, layer)
            elif type == 'source':
                self.source = SourceLayer()
                self.source.name = material.lower()
                try:
                    self.source.dielectric = mats[self.source.name]['dielectric']
                except:
                    raise KeyError('I don\'t know that material!')
                try:
                    self.source.losstangent = mats[self.source.name]['tand']
                except:
                    self.source.losstangent = 0
                    print('\nI don\'t know this loss tangent. Setting loss to 0!')
            elif type == 'terminator':
                self.terminator = TerminatorLayer()
                self.terminator.name = material.lower()
                try:
                    self.terminator.dielectric = mats[self.terminator.name]['dielectric']
                except:
                    raise KeyError('I don\'t know that material!')
                try:
                    self.terminator.losstangent = mats[self.terminator.name]['tand']
                except:
                    self.terminator.losstangent = 0
                    print('\nI don\'t know this loss tangent. Setting loss to 0!')
            else:
                raise ValueError('Type must be one of LAYER, SOURCE, or TERMINATOR')
        return

    def add_custom_layer(self, material, thickness, units, dielectric, loss_tangent, stack_position=-1):
        """Add a layer with custom properties to the AR stack.

        Arguments
        ---------
        material : string
            The name of the layer
        thickness : float
            The thickness of the layer
        units : string
            The units of length for the AR coating layer. Must be one of:
            { 'mil', 'inch', 'mm', 'm', 'um', 'in', 'micron' }
        dielectric : float
            The dielectric constant of the AR coating layer
        loss_tangent : float
            The loss tangent of the AR coating layer
        stack_position : int, optional
            The position of the layer in the AR coating stack, indexed
            from 0. Default is -1 (i.e., layer is automatically added
            to the end (bottom?) of the stack.
        """
        layer = Layer()
        layer.units = units
        layer.thickness = thickness
        layer.dielectric = dielectric
        layer.losstangent = loss_tangent
        if (stack_position == -1):
            self.stack.append(layer)
        else:
            self.stack.insert(stack_position, layer)
        return

    def show_sim_setup(self):
        """Display all the simulation parameters in one place."""
        pprint.pprint(vars(self))
        return

    def clear_structure(self):
        """Remove all elements from the current AR ``structure``."""
        self.structure = []
        return

    def remove_layer(self, layer_pos):
        """Remove the specified layer from the AR coating stack.

        Arguments
        ---------
        layer_pos : int
            The list index of the layer to remove from the AR coating stack
        """
        self.stack.pop(layer_pos)
        return

    def run_sim(self):
        """Take the attributes of the ``Builder()`` object and execute the
        simulation at each frequency in ``Builder().freq_sweep``. Save the
        output to a columnized, tab-separated text file.

        Returns
        -------
        transmission : array
            A three-element array. The first element is a list of
            frequencies, the second elements is a list of the
            transmissions at each frequency, and the third is a list of
            the reflections at each frequency.
        """
        t0 = time.time()
        print('Beginning AR coating simulation')
        self._d_converter()
        self._interconnect()
        if self.angle_sweep is None:
            f_list = []
            t_list = []
            r_list = []
            loss_list = []
            for f in self.freq_sweep:
                results = self.sim_single_freq(f, theta_0=0, pol=self.polarization)
                f_list.append(f)
                t_list.append(results['T'])
                r_list.append(results['R'])
                loss_list.append(results['loss'])
            fs = np.asarray(f_list)
            ts = np.asarray(t_list)
            rs = np.asarray(r_list)
            loss = np.asarray(loss_list)
            low = self.f_sweep_params['low']
            high = self.f_sweep_params['high']
            res = self.f_sweep_params['res']
            units = self.f_sweep_params['units']
            results = {}
            input = {}
            statistics = {}
            results['output'] = {'freqs':fs, 'T':ts, 'R':rs, 'loss':loss}
            input['frequency'] = {'f_low':low, 'f_high':high, 'f_res':res,
                                  'f_units':units, 'pol':self.polarization}
            input['angle'] = {'a_low':'None', 'a_high':'None', 'a_res':'None',
                              'a_units':'None', 'a_input':0.}
            if len(self.bands) == 0:
                statistics['band'] = 'No bandpasses set'
            else:
                for i in range(len(self.bands)):
                    statistics['band{}'.format(i)] = \
                        self._get_bandpass_stats(self.bands[i], results['output'])
            results['statistics'] = statistics
            results['input'] = input
            t = time.ctime(time.time())
            data_name = self._make_save_path(self.save_path, self.save_name)
            with open(data_name, 'wb') as f:
                header = self._make_header(results)
                f.writelines(header)
                np.savetxt(f, np.c_[fs, ts, rs, loss], delimiter='\t')
            print('Finished running AR coating simulation')
            t1 = time.time()
            t_elapsed = t1-t0
            print('Elapsed time: {t}s\n'.format(t=t_elapsed))
            return results
        else:
            if self.angle_sweep[0] != 0:
                self.angle_sweep = self.angle_sweep.tolist()
                self.angle_sweep.insert(0, 0)
                self.angle_sweep = np.asarray(self.angle_sweep)
            angles = []
            for angle in self.angle_sweep:
                f_list = []
                t_list = []
                r_list = []
                loss_list = []
                for f in self.freq_sweep:
                    results = self.sim_single_freq(f, theta_0=angle,
                                                   pol=self.polarization)
                    f_list.append(f)
                    t_list.append(results['T'])
                    r_list.append(results['R'])
                    loss_list.append(results['loss'])
                fs = np.asarray(f_list)
                ts = np.asarray(t_list)
                rs = np.asarray(r_list)
                loss = np.asarray(loss_list)
                f_low = self.f_sweep_params['low']
                f_high = self.f_sweep_params['high']
                f_res = self.f_sweep_params['res']
                f_units = self.f_sweep_params['units']
                a_low = self.a_sweep_params['low']
                a_high = self.a_sweep_params['high']
                a_res = self.a_sweep_params['res']
                a_units = self.a_sweep_params['units']
                theta = str(sp.rad2deg(angle))[:4]
                results = {}
                input = {}
                statistics = {}
                results['output'] = {'freqs':fs, 'T':ts, 'R':rs, 'loss':loss}
                input['frequency'] = {'f_low':f_low, 'f_high':f_high,
                                      'f_res':f_res, 'f_units':f_units,
                                      'pol':self.polarization}
                input['angle'] = {'a_low':a_low, 'a_high':a_high,
                                  'a_res':a_res, 'a_units':a_units,
                                  'a_input':theta}
                if len(self.bands) == 0:
                    statistics['band'] = 'No bandpasses set'
                else:
                    for i in range(len(self.bands)):
                        statistics['band{}'.format(i)] = \
                            self._get_bandpass_stats(self.bands[i], results['output'])
                results['statistics'] = statistics
                results['input'] = input
                t = time.ctime(time.time())
                data_name = self._make_save_path(self.save_path,
                                                 self.save_name+'_theta{}.txt'\
                                                     .format(theta))
                with open(data_name, 'wb') as f:
                    header = self._make_header(results)
                    f.writelines(header)
                    np.savetxt(f, np.c_[fs, ts, rs], delimiter='\t')
                print('Finished running AR coating simulation')
                t1 = time.time()
                t_elapsed = t1-t0
                print('Elapsed time: {t}s\n'.format(t=t_elapsed))
                angles.append(results)
            return np.asarray(angles)

    def set_angle_sweep(self, theta_min, theta_max, res=1., units='deg'):
        """Set the range of incident angles over which to run the simulation. 
        If you only want a single, non-normal incident angle, then set the same
        value for both theta_min and theta_max. The angle theta is defined as
        the deviation from normal incidence.

        The simulator will always include the results of normal incidence.
        Additional angles will be stored and recorded SOMEHOW.

        Arguments
        ---------
        theta_min : float
            The lower bound of the angle sweep. Must be greater than or equal
            to zero.
        theta_max : float
            The upper bound of the angle sweep. Must be less than or equal to
            90 degrees, or pi/2 radians.
        res : float, optional
            The resolution of the sweep. Default is 1.
        units : string, optional
            The units of the angle. Must be either 'deg' or 'rad'. Default is
            'rad'.
        """
        min = theta_min
        max = theta_max
        res = res
        self.a_sweep_params['low'] = min
        self.a_sweep_params['high'] = max
        self.a_sweep_params['res'] = res
        self.a_sweep_params['units'] = units
        if units == 'deg':
            min = sp.deg2rad(min)
            max = sp.deg2rad(max)
            res = sp.deg2rad(res)
        samples = (max-min)/res
        if samples == 0:
            self.angle_sweep = np.array([0., min])
            return
        self.angle_sweep = np.linspace(min, max, samples)
        return

    def set_bandpass(self, lower_bound, upper_bound):
        """Add a frequency range to the list of bands. You can get basic
        statistics for each range in the list of bands. Assumes the units of the
        band passes are those defined in ``set_freq_sweep()``.
        """
        self.bands.append((lower_bound, upper_bound))
        return

    def set_freq_sweep(self, lower_bound, upper_bound, resolution=1, units='ghz'):
        """Set the frequency range over which the simulation will run.
        
        Arguments
        ---------
        lower_bound : float
            The low end of the frequency range, given in GHz.
        upper_bound : float
            The high end of the frequency range, given in GHz.
        reolution : float, optional
            The interval at which to sample the frequency range, given in GHz.
            Defaults to 1 GHz.
        units : string
            The units of frequency. Must be one of:
            Hz, hz, KHz, khz, MHz, mhz, GHz, ghz
        """
        convert = {'Hz':1.0, 'hz':1.0, 'KHz':1e3, 'khz':1e3, 'MHz':1e6,
                   'mhz':1e6, 'GHz':1e9, 'ghz':1e9}
        low = lower_bound*convert[units]
        high = upper_bound*convert[units]
        samples = (high-low)/(resolution*convert[units])
        self.freq_sweep = np.linspace(low, high, samples)
        self.f_sweep_params['low'] = lower_bound
        self.f_sweep_params['high'] = upper_bound
        self.f_sweep_params['res'] = resolution
        self.f_sweep_params['units'] = units
        return

    def show_materials(self):
        """List the materials with known properties. The listed material names 
        are keys in the materials properties dictionary.
        """
        matpath = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(matpath, 'materials.json'), 'r') as f:
            mats = json.load(f)
            print('\nThe materials with known properties are:\n')
            pprint.pprint(mats.keys())
#            print('\nThe materials with known loss tangents are:\n')
#            pprint.pprint(mats.keys())
        return

    def sim_single_freq(self, frequency, theta_0, pol='s'):
        """Run the model simulation for a single frequency.

        Arguments
        ---------
        frequency : float
            The frequency at which to evaluate the model (in Hz).
        pol : string, optional
            The polarization of the source wave. Must be one of: 's', 
            'p', or 'u'. Default is 's'.
        theta_0 : float, optional
            The angle of incidence at the first interface.

        Returns
        -------
        result : dict
            result has two keys:
            1) 'T' : array; the total transmission through the model
            2) 'R' : array; the total reflection through the model
        """
        n = self._sort_ns()
        d = self._sort_ds()
        tan = self._sort_tans()
        theta = self.snell(n, theta_0)
        k = self._find_ks(n, frequency, tan, theta)
        delta = self._find_k_offsets(k, d)
        r, t = self._calc_R_T_amp(pol, n, delta, theta)
        T = self._get_T(t, n[0], n[-1], theta[0], theta[-1])
        R = self._get_R(r)
        loss = 1-T-R
        result = {'T':T, 'R':R, 'loss':loss}
        return result

    def snell(self, indices, theta_0):
        """Calculate the refraction angles for the entire model.

        Arguments
        ---------
        indices : array
            The array of indices of refraction for all elements in the model,
            ordered from source to terminator.
        theta_0 : float
            The angle of incidence at the first interface.
        """
        theta = [theta_0]
        for i in range(len(indices)-1):
            angle = sp.arcsin(np.real_if_close(indices[i]*np.sin(theta[i])/indices[i+1]))
            theta.append(angle)
        return theta
