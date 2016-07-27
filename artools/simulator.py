"""Simulator contains the tools needed to set up a multilayer antireflection coating
simulation.

Based on transfer matrix method outlined in Hou, H.S. 1974.
"""

# Author: Andrew Nadolski (with lots of help from previous work by Colin Merkel, Steve Byrnes, and Toki Suzuki)
# Filename: simulator.py

"""
###### TODO ######
7/25

    * Write convenience function to display all attributes of a layer

########## OLD STUFF DOWN BELOW ###########
11 Feb 16
    * Add functionality that automatically sets up an FTS sim. It just needs to 
      automatically set the substrate, then reverse the given layers, then stick
      a vacuum layer on the end.
25 Feb 16
    * Implement threading in emcee
    * Implement maximum likelihood as first step in MCMC
    * Multplication is expensive computation. Turn matrix products into sums 
        (or something...anything...)
"""

#import emcee
import glob
import os
import pprint
import time
import materials as mats
import numpy as np
import scipy as sp


class Layer:
    """Represents a layer in the AR coating.

    Arguments
    ----------
    material : string
        A key in the dictionary of materials found in _materials.py. You can view
        these materials by calling 'show_available_materials()'.

    Attributes
    ----------
    name : string
        The name of the material as passed to the 'Layer' constructor. Converted
        to all lowercase.
    thickness : float
        The thickness of the layer material. Defaults to 5. meters.
    type : string
        A flag for the simulator. Defaults to 'Layer' and should not be changed,
        or things may break.
    dielectric : float
        The dielectric constant of the layer material as found in the dictionary
        of materials contained in _materials.py.
    losstangent : float
        The loss tangent of the material as found in the dictionary
        of materials contained in _materials.py.
    """
    def __init__(self, material):
        self.name = material.lower()
        self.thickness = 5.
        self.type = 'Layer'
        self.units = None
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
        """Return the index of refraction of a material given its dielectric
        constant.
        """
        return (np.sqrt(self.dielectric))

    def ideal_thickness(self, opt_freq=160e9):
        ''' Return the ideal thickness of an AR coating layer given its dielectric
        constant and some optimization frequency.
        
        Arguments
        ---------
        opt_freq : float, optional
            The optimization frequency (in Hz) for the layers thickness. Defaults 
            to 150 GHz.
        '''
        return (1/np.sqrt(self.dielectric)*3e8/(4*opt_freq))


class BondingLayer(Layer):
    """A special case of ``Layer``; represents the adhesive layer used in the AR stack.

    Arguments
    ---------
    material : string
        A key in the dictionary of materials found in _materials.py. You can view
        these materials by calling ``show_available_materials()``.

    Attributes
    ----------
    thickness : float
        The thickness of the bonding layer. Defaults to 100e-6 meters, which is
        the typical thickness of a Stycast 1266 layer. This may be changed as is 
        necessary, but the units must (eventually) be converted to meters before 
        being fed to the simulator.
    type : string
        A flag for the model. Defaults to 'Bonding layer'. Should not be changed, 
        or things may break.
    """
    def __init__(self, material):
        Layer.__init__(self, material)
        self.thickness = 100.e-6 # typical Stycast 1266 thickness
        self.type = 'Bonding layer'

    def __repr__(self):
        return '%r (bonding layer)' % self.name


class SourceLayer(Layer):
    """A special case of ``Layer``; represents the layer from which the simulated wave 
    emanates.

    Arguments
    ---------
    material : string
        A key in the materials dictionary found in _materials.py. You can view
        these materials by calling ``show_available_materials()``.

    Attributes
    ----------
    thickness : float
        The thickness of the source layer. Defaults to ``numpy.inf`` since the model
        doesn't care about the thickness of source layer. The thickness of the
        source layer should not be changed under normal operations.
    type : string
        A flag for the model. Defaults to 'Source'. Should not be changed, or things
        may break.
    """
    def __init__(self, material):
        Layer.__init__(self, material)
        self.thickness = np.inf
        self.type = 'Source'

    def __repr__(self):
        return '%r (source layer)' % self.name


class SubstrateLayer(Layer):
    """A special case of ``Layer``; represents the layer to which the AR coating is 
    attached.

    Arguments
    ---------
    material : string
        A key in the materials dictionary found in _materials.py. You can view
        these materials by calling ``show_materials()``.

    Attributes
    ----------
    thickness : float
        The thickness of the substrate layer. Defaults to 250 mils, which is 
        the typical thickness of a sample puck used in the Berkeley FTS setup.
        This may be changed as is necessary, but the units must (eventually) be
        converted to meters before being fed to the simulator.
    type : string
        A flag for the model. Defaults to 'Substrate'. Should not be changed,
        or things may break.
    """
    def __init__(self, material):
        Layer.__init__(self, material)
        self.thickness = 250. #mils
        self.type = 'Substrate'
        
    def __repr__(self):
        return '%r (substrate)' % self.name


class TerminatorLayer(Layer):
    """A special case of ``Layer``; represents the layer upon which the simulated wave 
    terminates.

    Arguments
    ---------
    material : string
        A key in the materials dictionary found in _materials.py. You can view
        these materials by calling ``show_available_materials()``.

    Attributes
    ----------
    thickness : float
        The thickness of the terminating layer. Defaults to ``numpy.inf`` since
        the model doesn't care about the thickness of the terminating layer. 
        The thickness of the terminating layer should not be changed under 
        normal operations.
    type : string
        A flag for the model. Defaults to 'Terminator'. Should not be changed, 
        or things may break.
    """
    def __init__(self, material):
        Layer.__init__(self, material)
        self.thickness = np.inf
        self.type = 'Terminator'

    def __repr__(self):
        return '%r (terminator layer)' % self.name


class Builder:
    """Represents the main body of the simulator code.

    Attributes
    ----------
    freq_sweep : numpy array
        The range of frequencies to be simulated. Defaults to 0. Set a frequency
        sweep by calling ``set_freq_sweep()``.
    optimization_frequency : float
        The frequency (in Hz) at which to calculate the ideal thickness for a given
        material. Defaults to 160e9 Hz (160 GHz).
    stack : list
        The user-defined layers incorporated in the simulation EXCEPT the source
        and terminator layers. Defaults to empty list.
    structure : list
        The layers incorporated in the simulation INCLUDING the source and
        terminator layers. Defaults to empty list. The list is populated 
        by creating layers and calling ``_interconnect()``.
    source : object
        ``Layer`` object ``SourceLayer`` that defines where the wave emanates from. 
        Defaults to ``SourceLayer('vacuum')``.
    terminator : object
        ``Layer`` object ``TerminatorLayer`` that defines where the wave terminates.
        Defaults to ``TerminatorLayer('alumina')``.
    """
    def __init__(self):
        self.freq_sweep = 0.
        self.optimization_frequency = 160e9        # given in Hz, i.e. 160 GHz
        self.source = SourceLayer('vacuum')
        self.stack = []
        self.structure = []
        self.terminator = TerminatorLayer('alumina')

    def _calc_R_T_amp(self, polarization, n, delta):
        """Calculates the reflected and transmitted amplitudes

        Arguments
        ---------
        polarization : string
            The polarization of the source wave. Must be one of: 's', 'p', or 'u'.
        n : array
            An array of refractive indices, ordered from vacuum to substrate
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

        for i in range(len(self.structure)-1):
            t_amp[i,i+1] = self._t_at_interface(polarization, n[i], n[i+1])
            r_amp[i,i+1] = self._r_at_interface(polarization, n[i], n[i+1])

        M = np.zeros((len(self.structure),2,2),dtype=complex)
        for i in range(1,len(self.structure)-1):
            M[i] = 1/t_amp[i,i+1] * np.dot(self._make_2x2(np.exp(-1j*delta[i]),
                                                          0., 0., np.exp(1j*delta[i]),
                                                          dtype=complex),
                                           self._make_2x2(1., r_amp[i,i+1], \
                                                              r_amp[i,i+1], 1., \
                                                              dtype=complex))
        M_prime = self._make_2x2(1., 0., 0., 1., dtype=complex)
        for i in range(1, len(self.structure)-1):
            M_prime = np.dot(M_prime, M[i])
        M_prime = np.dot(self._make_2x2(1., r_amp[0,1], r_amp[0,1], 1., \
                                            dtype=complex)/t_amp[0,1], M_prime)
        t = 1/M_prime[0,0]
        r = M_prime[0,1]/M_prime[0,0]
        return (r, t)

    def _d_converter(self):
        """Checks the units of all elements in the connected ar coating
        stack. Converts the lengths of the layers to meters if they are
        not already in meters.
        """
        units = {'um':1e-6, 'mm':1e-3, 'inch':2.54e-2, 'in':2.54e-2,\
                     'micron':1e-6, 'mil':2.54e-5, 'm':1.0}
        for i in self.stack:
            i.thickness = i.thickness*units[i.units]
        return
        
    def _find_ks(self, n, frequency, tan, lossy=True):
        """Calculate the wavevectors

        Arguments
        ---------
        n : array
            An array of refractive indices, ordered from vacuum to
            substrate
        frequency : float
            The frequency at which to calculate the wavevector, k
        tan : array
            An array of loss tangents, ordered from vacuum to substrate
        lossy : boolean
            If ``True`` the wavevector will be found for a lossy material.
            If ``False`` the wavevector will be found for lossless material.
        Returns
        -------
        k : complex
            The complex wavevector, k
        """
        if lossy:
            k = 2*np.pi*n*frequency*(1-0.5j*tan)/3e8
        else:
            k = 2*np.pi*n*frequency/3e8
        return k

    def _find_k_offsets(self, k, d):
        """Calculate the wavevector offset, delta

        Arguments
        ---------
        k : array
            The wavevector
        d : array
            An array of thicknesses, ordered from vacuum to substrate

        Returns
        -------
        delta : array
            The wavevector offset
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

    def _interconnect(self):
        """Connect all the AR coating layer objects.

        Arguments
        ---------
        stack : list
            Contains a list of ``Layer`` objects, ordered from source to terminator.
        """
        self.clear_stack()
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
        """
        if polarization == 's':
            return ((n_1 - n_2)/(n_1 + n_2))
        elif polarization == 'p':
            return ((n_1 - n_2)/(n_1 + n_2))
        else:
            raise ValueError("Polarization must be 's' or 'p'")

    def _sort_ns(self):
        """Organizes the refractive indices of the layers in the simulation

        Returns
        -------
        n : array
            The ordered list of indices of refraction, from incident
            medium to terminating medium
        """
        n = []
        [n.append(layer.get_index()) for layer in self.structure]
        n = np.asarray(n)
        return n

    def _sort_ds(self):
        """Organizes the layers' thicknesses in the simulation

        Returns
        -------
        d : array
            The ordered list of indices thicknesses, from incident
            medium to terminating medium
        """
        d = []
        [d.append(layer.thickness) for layer in self.structure if \
             (layer.type == 'Layer' or layer.type == 'Substrate')]
        d.insert(0, self.structure[0].thickness)
        d.append(self.structure[-1].thickness)
        d = np.asarray(d)
        return d

    def _sort_tans(self):
        """Organizes the loss tangents of the layers in the simulation

        Returns
        -------
        tan : array
            The ordered list of loss tangents, from incident
            medium to terminating medium
        """
        tan = []
        [tan.append(layer.losstangent) for layer in self.structure]
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
        transmitted amplitude : float
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

    def add_layer(self, material, thickness=5.0, units='mil', type=None, stack_position=-1):
        """Create a layer and add it to the AR coating stack

        Arguments
        ---------
        material : string
            A key in the dictionary of materials found in _materials.py.
            You can view these materials by calling 
            'show_materials()'.
        stack_position : int, optional
            The position of the layer in the AR coating stack, indexed
            from 0. Default is -1 (i.e., layer is automatically added
            to the end (bottom?) of the stack.
        thickness : float, optional
            The thickness of the AR coating layer material. Assumed to
            be given in 'mil' (i.e. thousandths of an inch) unless 
            otherwise stated
        type : string, optional
            The layer type. Defaults to 'None' type, which corresponds to
            an AR layer. Other options are 'substrate' or 'bonding', which
            correspond to substrate and bonding layers, respectively.
        units : string, optional
            The units of length for the AR coating layer. Must be one of:
                { 'mil'   ,
                  'inch'  ,
                  'mm'    ,
                  'm'     ,
                  'um'    ,
                  'in'    ,
                  'micron'}
        """
        if type == None:
            layer = Layer(material)
            layer.thickness = thickness
            layer.units = units
            pos_check = -1
            if (pos_check == stack_position):
                self.stack.append(layer)
            else:
                self.stack.insert(stack_position, layer)
        else:
            type = type.lower()
            if type == 'substrate':
                layer = SubstrateLayer(material)
                layer.thickness = thickness
                layer.units = units
                pos_check = -1
                if (pos_check == stack_position):
                    self.stack.append(layer)
                else:
                    self.stack.insert(stack_position, layer)
            elif type == 'bonding':
                layer = BondingLayer(material)
                layer.thickness = thickness
                layer.units = units
                pos_check = -1
                if (pos_check == stack_position):
                    self.stack.append(layer)
                else:
                    self.stack.insert(stack_position, layer)
        return

    def display_sim_parameters(self):
        """Displays all the simulation parameters in one place
        """
        pprint.pprint(vars(self))
        return

    def clear_stack(self):
        """Remove all elements from the current AR ``structure``.
        """
        self.structure = []
        return

    def remove_layer(self, layer_pos):
        """Removes the specified layer from the AR coating stack

        Arguments
        ---------
        layer_pos : int
            The index of the layer to remove from the AR coating stack
        """
        self.stack.pop(layer_pos)
        return

    def run_sim(self):
        """Takes the attributes of the ``Builder()`` object and executes
        the simulation at each frequency in ``Builder().freq_sweep``. The
        output is saved to a columnized, tab-separated text file.

        Returns
        -------
        transmission : numpy array
            A three-element array. The first element is a list of
            frequencies, the second elements is a list of the
            transmissions at each frequency, and the third is a list of
            the reflections at each frequency.
        """
        t0 = time.time()
        print 'Beginning AR coating simulation'
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
        fname = 'transmission_data_{t}.txt'.format(t=t)
        header = 'Frequency (Hz)\t\tTransmission amplitude\t\tReflection amplitude'
        with open(fname, 'wb') as f:
            np.savetxt(f, np.c_[fs, ts, rs], delimiter='\t', header=header)
        print 'Finished running AR coating simulation'
        t1 = time.time()
        t_elapsed = t1-t0
        print 'Elapsed time: {t}s\n'.format(t=t_elapsed)
        return results

    def set_freq_sweep(self, lower_bound, upper_bound, resolution=1):
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
        """
        self.freq_sweep = np.linspace(lower_bound*1e9, upper_bound*1e9, \
                                          (upper_bound-lower_bound)/resolution)
        return

    def set_source_layer(self, material='vacuum'):
        """Change the default source layer from vacuum to something else.

        Arguments
        ---------
        material : string, optional
            A key in the dielectrics dictionary. If 'material' is not specified,
            defaults to 'vacuum'.
        """
        self.source = SourceLayer(material)
        return

    def set_terminator_layer(self, material='alumina'):
        """Change the default layer from alumina to something else. If running an FTS
        simulation, the terminator layer should be set as 'vacuum'.

        Arguments
        ---------
        material : string, optional
            A key in the dielectrics dictionary. If 'material' is not specified,
            defaults to 'alumina'.
        """
        self.terminator = TerminatorLayer(material)
        return

    def show_materials(self):
        """List the materials with known properties. The simulator can handle these
        materials. The listed material names are keys in the materials properties 
        dictionary. 
        """
        print '\nThe materials with known dielectric properties are:\n'
        pprint.pprint(mats.Electrical.DIELECTRIC)
        print '\nThe materials with known loss tangents are:\n'
        pprint.pprint(mats.Electrical.LOSS_TAN)
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
                'T' : numpy array; the total transmission through the model.
                'R' : numpy array; the total reflection through the model.
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


# class FitFTS(object):
#     '''
#     '''
#     def __init__(self, real_data, n_walkers, n_dimensions, n_iterations):
#         self.name = real_data[:-4]
#         self.data_path = real_data
#         self.data = np.genfromtxt(real_data, unpack=True)
#         self.freqs = self.data[0]
#         self.data_trans = self.data[1]
#         self.sigma = self.data[2]
#         self.n_walkers = n_walkers
#         self.n_dim = n_dimensions
#         self.n_steps = n_iterations
#         self.sigma_n = .5
#         self.sigma_t = 2.5

#     def lnprior(self, params):
#         '''
#         # make cuts here: ex, if indices < 1: return -10000
#             if thicknesses > 50: return -10000
#         '''
#         index = params[:len(params)/2]
#         thick = params[len(params)/2:]
#         for n in index:
#             if (n < 1. or n > 10.5):
#                 return -np.inf
#         if (thick[0] < 10. or thick[0] > 18.5):
#             return -np.inf
#         if (thick[1] < 0. or thick[1] > 3.):
#             return -np.inf
#         if (thick[2] < 200. or thick[2] > 300.):
#             return -np.inf
#         if (thick[3] < 0. or thick[3] > 3.):
#             return -np.inf
#         if (thick[4] < 10. or thick[4] > 18.5):
#             return -np.inf
# #         for t in thick:
# #             if (t <= 0. or t > 300.):
# #                 return -np.inf
#         return 0.

#     def lnlikelihood(self, params):
#         '''
#         # read in FTS data here (the freqs, transmissions, and sigmas)
#         # compute the model at the frequencies in the FTS data
#         # make lnlikelihood return (-np.sum((model - data)**2*(1/(sigma)**2)) / 2)
#         '''
#         indices = params[:len(params)/2]
#         thicknesses = params[len(params)/2:]
#         model_transmission=[]
#         [model_transmission.append(self.simulate_for_mcmc(freq, indices, thicknesses)) for freq in self.freqs]
#         model_transmission = np.array(model_transmission)
#     #    return (-np.sum((model_transmission - self.data_trans)**2*(1/(self.sigma**2)) / 2))
#         inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
#         return -0.5*(np.sum((ymodel)**2*inv_sigma2 - np.log(inv_sigma2)))

#     def lnprobability(self, params):
#         '''
#         # the sum (in logspace) of lnprior and lnlikelihood
#         #
#         '''
#         lnprior = self.lnprior(params)
#         if not np.isfinite(lnprior):
#             return -np.inf
#         return (lnprior + self.lnlikelihood(params))

#     def run_mcmc(self, initial_guess):
#         p0 = []
#         for i in range(self.n_walkers):
#             p_intermediate = []
#             for n in initial_guess[:len(initial_guess)/2]:
#                 p_intermediate.append(n+np.random.normal(0, self.sigma_n))
#             for t in initial_guess[len(initial_guess)/2:]:
#                 p_intermediate.append(t+np.random.normal(0, self.sigma_t))
#             p0.append(p_intermediate)
         
#         sampler = emcee.EnsembleSampler(self.n_walkers, self.n_dim, self.lnprobability)
#         print '\nBeginning MCMC...'

#         position = []
#         lnprob = []
#         count = 1
#         for result in sampler.sample(p0, iterations=self.n_steps, storechain=True):
#             position.append(result[0])
#             lnprob.append(result[1])
#             print 'STEP NUMBER:', count
#             count += 1
#         pos = np.array(position)
#         prob = np.array(lnprob)
#         np.savez('{name}_{n_walkers}walkers_{n_steps}steps'.format(name=self.name, n_walkers=self.n_walkers, n_steps=self.n_steps), positions = [pos[x] for x in range(len(pos))], lnprob = [prob[x] for x in range(len(prob))])
#         print '\nAll done!'
#         return sampler

# #     def show_mcmc(self):
# #         fig = plt.Figure()
# #         ax = fig.add_subplot(111)
# #         plt.plot(self.data[0], self.data[1])
# #         plt.savefig('~/Desktop/testing_MCMC_plot.pdf')

#     def simulate_for_mcmc(self, frequency, indices, thicknesses, polarization='s', theta_0=0):
#         ''' Description

#         Arguments
#         ---------

#         Returns
#         -------
#         '''
#         # check the desired polarization
#         if polarization == 'u':
#             return Builder()._unpolarized_simulation(frequency)

#         # get all the indices of refraction in one place
#         n = []
#         [n.append(index) for index in indices]
#         n.insert(0, 1.)
#         n.append(1.)

#         # get all thicknesses in one place
#         d = []
#         [d.append(thickness) for thickness in thicknesses]
#         d.insert(0, np.inf)
#         d.append(np.inf)
        
#         # convert theicness and dielectric lists to numpy arrays
#         n = np.asarray(n)
#         d = np.asarray(d)
#         d *= 2.54e-5 #convert from mils to m

#         # find the wavevectors, k
#         k = 2*np.pi * n * frequency/3e8
#         olderr = sp.seterr(invalid= 'ignore') # turn off 'invalid multiplication' error; it's just the 'inf' boundaries
#         delta = k * d
#         sp.seterr(**olderr) # turn the error back on

#         # now get transmission and reflection amplitudes
#         t_amp = np.zeros((len(n), len(n)), dtype=complex)
#         r_amp = np.zeros((len(n), len(n)), dtype=complex)

#         for i in range(len(n)-1):
#             t_amp[i,i+1] = Builder()._t_at_interface(polarization, n[i], n[i+1])
#             r_amp[i,i+1] = Builder()._r_at_interface(polarization, n[i], n[i+1])

#         M = np.zeros((len(n),2,2),dtype=complex)
#         for i in range(1,len(n)-1):
#             M[i] = 1/t_amp[i,i+1] * np.dot(Builder()._make_2x2(np.exp(-1j*delta[i]),
#                                                           0., 0., np.exp(1j*delta[i]),
#                                                           dtype=complex),
#                                            Builder()._make_2x2(1., r_amp[i,i+1], r_amp[i,i+1], 1.,
#                                                           dtype=complex))
#         M_prime = Builder()._make_2x2(1., 0., 0., 1., dtype=complex)
#         for i in range(1, len(n)-1):
#             M_prime = np.dot(M_prime, M[i])
#         M_prime = np.dot(Builder()._make_2x2(1., r_amp[0,1], r_amp[0,1], 1., dtype=complex)/t_amp[0,1],
#                          M_prime)
        
#         # Now find the net transmission and reflection amplitudes
#         t = 1/M_prime[0,0]
#         r = M_prime[0,1]/M_prime[0,0]
        
#         # Now find the net transmitted and reflected power
#         T = Builder()._get_T(polarization, t, n[0], n[-1])
#         R = Builder()._get_R(r)
#         return T
