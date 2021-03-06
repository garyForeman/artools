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
import materials as mats
import numpy as np
import scipy as sp


class Layer:
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


class Builder:
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
        self.bands = [(81.7e9, 107.5e9),(128.6e9, 167.2e9),(196.9e9, 249.2e9)]
        self.freq_sweep = 0.
        self.log_name = 'log_simulation_{t}.txt'.format(t=time.ctime(time.time()))
        self.optimization_frequency = 160e9        # given in Hz, i.e. 160 GHz
        self.save_name = 'transmission_data_{t}.txt'.format(t=time.ctime(time.time()))
        self.save_path = '.'
        self.source = None
        self.stack = []
        self.structure = []
        self.terminator = None

    def _calc_R_T_amp(self, polarization, n, delta):
        """Calculate the reflected and transmitted amplitudes

        Arguments
        ---------
        polarization : string
            The polarization of the source wave. Must be one of: 's', 'p', or 'u'.
        n : array
            An array of refractive indices, ordered from source to terminator
        delta : array
            An array of wavevector offsets
        
        Returns
        -------
        (r, t) : tuple
            A tuple where 'r' is the reflected amplitude, and 't' is the
            transmitted amplitude
        """
        t_amp = np.zeros((len(self.structure), len(self.structure)), dtype=complex)
        r_amp = np.zeros((len(self.structure), len(self.structure)), dtype=complex)
#         # debugging statement
#         print("\nr_amp is:")
#         for i in range(len(self.structure)):
#             for j in range(len(self.structure)):
#                 print("{}{} {}".format(i,j,r_amp[i][j]))
#         # debugging statement
#         print("\nt_amp is:")
#         for i in range(len(self.structure)):
#             for j in range(len(self.structure)):
#                 print("{}{} {}".format(i,j,t_amp[i][j]))

        for i in range(len(self.structure)-1):
            t_amp[i,i+1] = self._t_at_interface(polarization, n[i], n[i+1])
            r_amp[i,i+1] = self._r_at_interface(polarization, n[i], n[i+1])
#         # debugging statement
#         print("\nmod r_amp is:")
#         for i in range(len(self.structure)):
#             for j in range(len(self.structure)):
#                 print("{}{} {}".format(i,j,r_amp[i][j]))
#         # debugging statement
#         print("\nmod t_amp is:")
#         for i in range(len(self.structure)):
#             for j in range(len(self.structure)):
#                 print("{}{} {}".format(i,j,t_amp[i][j]))

        M = np.zeros((len(self.structure),2,2),dtype=complex)
#         # debugging statement
#         print("\nThe 'M' matrix is:")
#         for i in range(len(self.structure)):
#             for j in range(2):
#                 for k in range(2):
#                     print("M{}{}{} ---> {}".format(i,j,k,M[i][j][k]))

        m_r_amp = np.zeros((len(self.structure),2,2), dtype=complex)
        m_t_amp = np.zeros((len(self.structure),2,2), dtype=complex)
        for i in range(1,len(self.structure)-1):
            m_t_amp[i] = self._make_2x2(np.exp(-1j*delta[i]), 0., 0., np.exp(1j*delta[i]), dtype=complex)
            m_r_amp[i] = self._make_2x2(1., r_amp[i,i+1], r_amp[i,i+1], 1., dtype=complex)

#         # debugging statement
#         print("\nThe temporary 'm_r_amp' matrix is:")
#         for i in range(len(self.structure)):
#             for j in range(2):
#                 for k in range(2):
#                     print("m_r_amp{}{}{} ---> {}".format(i,j,k,m_r_amp[i][j][k]))

#         # debugging statement
#         print("\nThe temporary 'm_t_amp' matrix is:")
#         for i in range(len(self.structure)):
#             for j in range(2):
#                 for k in range(2):
#                     print("m_t_amp{}{}{} ---> {}".format(i,j,k,m_t_amp[i][j][k]))

        m_temp = np.dot(m_t_amp, m_r_amp)

#         # debugging statement
#         print("\nThe 'm_temp' matrix is:")
#         for i in m_temp:
#             print i
#         for i in range(len(self.structure)):
#             for j in range(2):
#                 for k in range(2):
#                     print("m_temp{}{}{} ---> {}".format(i,j,k,m_temp[i][j][k]))

        for i in range(1,len(self.structure)-1):
            M[i] = 1/t_amp[i,i+1] * np.dot(self._make_2x2(np.exp(-1j*delta[i]),
                                                          0., 0., np.exp(1j*delta[i]),
                                                          dtype=complex),
                                           self._make_2x2(1., r_amp[i,i+1], \
                                                              r_amp[i,i+1], 1., \
                                                              dtype=complex))
#         # debugging statement
#         print("\nThe modified 'M' matrix is:")
#         for i in range(len(self.structure)):
#             for j in range(2):
#                 for k in range(2):
#                     print("mod M{}{}{} ---> {}".format(i,j,k,M[i][j][k]))

        M_prime = self._make_2x2(1., 0., 0., 1., dtype=complex)

#         # debugging statement
#         print("\nThe first modified 'M_prime' matrix is:")
#         for i in range(2):
#             for j in range(2):
#                 print("1st mod M_prime{}{} ---> {}".format(i,j,M_prime[i][j]))

        for i in range(1, len(self.structure)-1):
#            print("\n'M_prime' #{} is:\n{}".format(i,M_prime))
            M_prime = np.dot(M_prime, M[i])

#         # debugging statement
#         print("\nThe second modified 'M_prime' matrix is:")
#         for i in range(2):
#             for j in range(2):
#                 print("2nd mod M_prime{}{} ---> {}".format(i,j,M_prime[i][j]))

#         print("\nr_amp01 is ---> {}".format(r_amp[0,1]))
#         print("t_amp01 is ---> {}".format(t_amp[0,1]))

        mod_M_prime = self._make_2x2(1.,r_amp[0,1], r_amp[0,1], 1., dtype=complex)/t_amp[0,1]

#         # debugging statement
#         print("\nThe third modified 'M_prime' matrix is:")
#         for i in range(2):
#             for j in range(2):
#                 print("3rd mod M_prime{}{} ---> {}".format(i, j, mod_M_prime[i][j]))

        M_prime = np.dot(self._make_2x2(1., r_amp[0,1], r_amp[0,1], 1., \
                                            dtype=complex)/t_amp[0,1], M_prime)

#         # debugging statement
#         print("\nThe 'M_final' matrix is:")
#         for i in range(2):
#             for j in range(2):
#                 print("M_final{}{} ---> {}".format(i, j, M_prime[i][j]))

        t = 1/M_prime[0,0]
        r = M_prime[0,1]/M_prime[0,0]

#         # debugging statement
#         print("\n't' ---> {}".format(t))
#         print("'r' ---> {}".format(r))

        return (r, t)

    def _d_converter(self):
        """Check the units of all elements in the connected ar coating
        stack. Convert the lengths of the layers to meters if they are
        not already in meters.
        """
        units = {'um':1e-6, 'mm':1e-3, 'inch':2.54e-2, 'in':2.54e-2,\
                     'micron':1e-6, 'mil':2.54e-5, 'm':1.0}
        for i in self.stack:
            i.thickness = i.thickness*units[i.units]
        return
        
    def _find_ks(self, n, frequency, tan, lossy=True):
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
            k = 2*np.pi*n*frequency*(1+0.5j*tan)/3e8 # New expression for loss (as of 9/13/16), this one is more physical (i.e. subtractive)
#             k = 2*np.pi*n*frequency*(1-0.5j*tan)/3e8 # Original expression for loss (pre 9/13/16), but it is incorrectly ADDITIVE
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
            The wavenumber offset
        """
        olderr = sp.seterr(invalid= 'ignore') # turn off 'invalid multiplication' error;
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

    def _get_T(self, polarization, net_t_amp, n_i, n_f, theta_i=0., theta_f=0.):
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
        theta_i : float, optional
            The angle of incidence at interface 'i'. Default is 0.
        theta_f : float, optional
            The angle of incidence at interface 'f'. Default is 0.
        """
        if (polarization=='s'):
            return np.abs(net_t_amp**2) * (n_f/n_i)
        elif (polarization=='p'):
            return np.abs(net_t_amp**2) * (n_f/n_i)
        else:
            raise ValueError("Polarization must be 's' or 'p'")

    def _get_bandpass_stats(self):
        mean = []
        for band in self.bands:
            pass
        pass

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
        """Return a 2x2 array quickly.

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

    def _make_log(self):
        pass

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
            self.save_name = save_name+'.txt'
            path = os.path.join(save_path, save_name)
        return path

    def _r_at_interface(self, polarization, n_1, n_2):
        """Calculate the reflected amplitude at an interface.

        Arguments
        ---------
        polarization : string
            The polarization of the source wave. Must be one of: 's' or 'p'.
        n_1 : float
            The index of refraction of the first material.
        n_2 : float
            The index of refraction of the second material.

        Returns
        -------
        reflected amplitude : float
            The amplitude of the reflected power
        """
        if polarization == 's':
            return ((n_1-n_2)/(n_1+n_2))
        elif polarization == 'p':
            return ((n_1-n_2)/(n_1+n_2))
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

    def _t_at_interface(self, polarization, n_1, n_2):
        """Calculate the transmission amplitude at an interface.

        Arguments
        ---------
        polarization : string
            The polarization of the source wave. Must be one of: 's' or 'p'.
        n_1 : float
            The index of refraction of the first material.
        n_2 : float
            The index of refraction of the second material.

        Returns
        -------
        transmitted_amplitude : float
            The amplitude of the transmitted power
        """
        if polarization == 's':
            return 2*n_1/(n_1 + n_2)
        elif polarization == 'p':
            return 2*n_1/(n_1 + n_2)
        else:
            raise ValueError("Polarization must be 's' or 'p'")

    def _unpolarized_simulation(self, frequency, theta_0=0):
        """Handle the special case of unpolarized light by running the model
        for both 's' and 'p' polarizations and computing the mean of the two
        results.

        Arguments
        ---------
        frequency : float
            The frequency (in Hz) at which to evaluate the model.
        theta_0 : float, optional
            The angle of incidence at the initial interface. Default is 0.
        """
        s_data = self.simulate(frequency, 's', theta_0)
        p_data = self.simulate(frequency, 'p', theta_0)
        T = (s_data + p_data)/2
        return T
 
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

        type = type.lower()
        if type == 'layer':
            layer = Layer()
            layer.name = material.lower()
            layer.thickness = thickness
            layer.units = units
            try:
#                 layer.dielectric = mats.Electrical.DIELECTRIC[layer.name]
                layer.dielectric = mats.Electrical.props[layer.name][0]
            except:
                raise KeyError('I don\'t know that material!')
            try:
#                 layer.losstangent = mats.Electrical.LOSS_TAN[layer.name]
                layer.losstangent = mats.Electrical.props[layer.name][1]
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
#                 self.source.dielectric = mats.Electrical.DIELECTRIC[self.source.name]
                self.source.dielectric = mats.Electrical.props[self.source.name][0]
            except:
                raise KeyError('I don\'t know that material!')
            try:
#                 self.source.losstangent = mats.Electrical.LOSS_TAN[self.source.name]
                self.source.losstangent = mats.Electrical.props[self.source.name][1]
            except:
                self.source.losstangent = 0
                print('\nI don\'t know this loss tangent. Setting loss to 0!')
        elif type == 'terminator':
            self.terminator = TerminatorLayer()
            self.terminator.name = material.lower()
            try:
#                 self.terminator.dielectric = mats.Electrical.DIELECTRIC[self.terminator.name]
                self.terminator.dielectric = mats.Electrical.props[self.terminator.name][0]
            except:
                raise KeyError('I don\'t know that material!')
            try:
#                 self.terminator.losstangent = mats.Electrical.LOSS_TAN[self.terminator.name]
                self.terminator.losstangent = mats.Electrical.props[self.terminator.name][1]
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

    def display_sim_parameters(self):
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
        f_list = []
        t_list = []
        r_list = []
        for f in self.freq_sweep:
            results = self.sim_single_freq(f)
            f_list.append(f)
            t_list.append(results['T'])
            r_list.append(results['R'])
        fs = np.asarray(f_list)
        ts = np.asarray(t_list)
        rs = np.asarray(r_list)
        results = np.array([fs, ts, rs])
        t = time.ctime(time.time())
        data_name = self._make_save_path(self.save_path, self.save_name)
        header = 'Frequency (Hz)\t\tTransmission amplitude\t\tReflection amplitude'
#         log_name = self._make_save_path(self.save_path, self.log_name)
#         log = self._make_log()
        with open(data_name, 'wb') as f:
            np.savetxt(f, np.c_[fs, ts, rs], delimiter='\t', header=header)
#         with open(log_name, 'wb') as f:
#             for line in log:
#                 f.writelines(line)
#                 f.write('\n')
        print('Finished running AR coating simulation')
        t1 = time.time()
        t_elapsed = t1-t0
        print('Elapsed time: {t}s\n'.format(t=t_elapsed))
        return results

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
        units : str
            The units of frequency. Must be one of:
            Hz, hz, KHz, khz, MHz, mhz, GHz, ghz
        """
        convert = {'Hz':1.0, 'hz':1.0, 'KHz':1e3, 'khz':1e3, 'MHz':1e6,
                   'mhz':1e6, 'GHz':1e9, 'ghz':1e9}
        low = lower_bound*convert[units]
        high = upper_bound*convert[units]
        samples = (high-low)/resolution
        self.freq_sweep = np.linspace(low, high, samples)
        return

#     def set_source_layer(self, material):
#         """Change the source layer.

#         Arguments
#         ---------
#         material : string
#             A key in the dielectrics dictionary.
#         """
#         self.source = SourceLayer(material)
#         return

#     def set_terminator_layer(self, material):
#         """Change the terminator layer.

#         Arguments
#         ---------
#         material : string
#             A key in the dielectrics dictionary.
#         """
#         self.terminator = TerminatorLayer(material)
#         return

    def show_materials(self):
        """List the materials with known properties. The listed material names 
        are keys in the materials properties dictionary. 
        """
        print('\nThe materials with known dielectric properties are:\n')
        pprint.pprint(mats.Electrical.props)
#         pprint.pprint(mats.Electrical.DIELECTRIC)
        print('\nThe materials with known loss tangents are:\n')
        pprint.pprint(mats.Electrical.props)
#         pprint.pprint(mats.Electrical.LOSS_TAN)
        return

    def sim_single_freq(self, frequency, polarization='s', theta_0=0):
        """Run the model simulation for a single frequency.

        Arguments
        ---------
        frequency : float
            The frequency at which to evaluate the model (in Hz).
        polarization : string, optional
            The polarization of the source wave. Must be one of: 's', 
            'p', or 'u'. Default is 's'.
            
            ### NOTE ###
            I've chosen 's' polarization as the default because this 
            simulator only handles normal incidence waves, and and at 
            normal incidence 's' and 'p' are equivalent.
        theta_0 : float, optional
            The angle of incidence at the first interface.

        Returns
        -------
        result : dict
            dict = {
                'T' : array; the total transmission through the model.
                'R' : array; the total reflection through the model.
                    }
        """
        # check the desired polarization
#        if polarization == 'u':
#            return self._unpolarized_simulation(frequency)
        n = self._sort_ns()                                 # get all refractive indices
        d = self._sort_ds()                                 # get all thicknesses
        tan = self._sort_tans()                             # get all loss tans
        k = self._find_ks(n, frequency, tan)                # find all wavevectors, k
        delta = self._find_k_offsets(k, d)                  # calculate all offsets
        r, t = self._calc_R_T_amp(polarization, n, delta)   # get trans, ref amps
        T = self._get_T(polarization, t, n[0], n[-1])       # find net trans, ref power
        R = self._get_R(r)
        result = {'T':T, 'R':R}
        return result

    def snell(self, indices, theta_0):
        """Caclulate the Snell angles for the entire model.

        Arguments
        ---------
        indices : list
            The list of indices of refraction for all elements in the model,
            ordered from source to terminator.
        theta_0 : float
            The angle of incidence at the first interface.
        """
        return sp.arcsin(np.real_if_close(n_list[0]*np.sin(th_0) / n_list))

class MCMC:
    """Contains the methods specific to ``emcee``, the MCMC Hammer, and helper
    methods to set up MCMC simulations and visualize the results.
    """
    def __init__(self):
        self.name = 'blah'
        self.priors = []

    def __repr__(self):
        return '{} (MCMC object)'.format(self.name)

    def add_prior(self, layer_number, prior_type, low_bound, hi_bound, units='mil'):
        """Add a prior to a part of the model in order to constrain the total
        simulation space. Can only place constraints on thickness and dielectric
        for now.

        Arguments
        ---------
        layer_number : int
            The position of the layer in the AR coating stack. Indexed from 1, so
            incident `vacuum` is 0 and first AR coating layer is 1.
        prior_type : string
            Flags the prior as either a cut to dielectric constant or thickness.
            One of 'thickness', 't', 'dielectric', or 'd'.
        low_bound : float
            The lower boundary of the range.
        hi_bound : float
            The higher boundary of the range.
        units : string, optional
            The units of the lower and upper bounds. Only applies to 'thickness'
            cuts because dielectric constants are unitless. Defaults to `mils`.
        """
        prior = {'layer_number':layer_number, 'prior_type':prior_type, \
                     'low_bound':low_bound, 'hi_bound':hi_bound, 'units':units}
        self.priors.append(prior)
        return

    def lnlikelihood(self):
        return

    def lnprior(self):
        """Define the known prior attributes of the model in order to constrain
        the simulation space.
        """
        
        return

    def lnprobability(self):
        """The logspace sum of ``lnprior`` and ``lnlikelihood``.
        """
        return

    def sort_priors(self):
        """Sort the contents of ``self.prior`` by layer number
        
        Returns
        -------
        sorted_priors : list
            A list of priors sorted by layer number. If a layer has both
            thickness and dielectric priors, the thickness dielectric is first
            and the dielectric is second.
        """
        return
