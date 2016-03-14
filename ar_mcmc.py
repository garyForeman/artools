
class MCSim(object):
    '''
    '''
    def __init__(self, real_data, n_walkers, n_dimensions, n_iterations):
        self.name = real_data[:-4]
        self.data_path = real_data
        self.data = np.genfromtxt(real_data, unpack=True)
        self.freqs = self.data[0]
        self.data_trans = self.data[1]
        self.sigma = self.data[2]
        self.n_walkers = n_walkers
        self.n_dim = n_dimensions
        self.n_steps = n_iterations
        self.sigma_n = .5
        self.sigma_t = 2.5

    def lnprior(self, params):
        '''
        # make cuts here: ex, if indices < 1: return -10000
            if thicknesses > 50: return -10000
        '''
        index = params[:len(params)/2]
        thick = params[len(params)/2:]
        for n in index:
            if (n < 1. or n > 10.5):
                return -np.inf
        if (thick[0] < 10. or thick[0] > 18.5):
            return -np.inf
        if (thick[1] < 0. or thick[1] > 3.):
            return -np.inf
        if (thick[2] < 200. or thick[2] > 300.):
            return -np.inf
        if (thick[3] < 0. or thick[3] > 3.):
            return -np.inf
        if (thick[4] < 10. or thick[4] > 18.5):
            return -np.inf
#         for t in thick:
#             if (t <= 0. or t > 300.):
#                 return -np.inf
        return 0.

    def lnlikelihood(self, params):
        '''
        # read in FTS data here (the freqs, transmissions, and sigmas)
        # compute the model at the frequencies in the FTS data
        # make lnlikelihood return (-np.sum((model - data)**2*(1/(sigma)**2)) / 2)
        '''
        indices = params[:len(params)/2]
        thicknesses = params[len(params)/2:]
        model_transmission=[]
        [model_transmission.append(self.simulate_for_mcmc(freq, indices, thicknesses)) for freq in self.freqs]
        model_transmission = np.array(model_transmission)
    #    return (-np.sum((model_transmission - self.data_trans)**2*(1/(self.sigma**2)) / 2))
        inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
        return -0.5*(np.sum((ymodel)**2*inv_sigma2 - np.log(inv_sigma2)))

    def lnprobability(self, params):
        '''
        # the sum (in logspace) of lnprior and lnlikelihood
        #
        '''
        lnprior = self.lnprior(params)
        if not np.isfinite(lnprior):
            return -np.inf
        return (lnprior + self.lnlikelihood(params))

    def run_mcmc(self, initial_guess):
        p0 = []
        for i in range(self.n_walkers):
            p_intermediate = []
            for n in initial_guess[:len(initial_guess)/2]:
                p_intermediate.append(n+np.random.normal(0, self.sigma_n))
            for t in initial_guess[len(initial_guess)/2:]:
                p_intermediate.append(t+np.random.normal(0, self.sigma_t))
            p0.append(p_intermediate)
         
        sampler = emcee.EnsembleSampler(self.n_walkers, self.n_dim, self.lnprobability)
        print '\nBeginning MCMC...'

        position = []
        lnprob = []
        count = 1
        for result in sampler.sample(p0, iterations=self.n_steps, storechain=True):
            position.append(result[0])
            lnprob.append(result[1])
            print 'STEP NUMBER:', count
            count += 1
        pos = np.array(position)
        prob = np.array(lnprob)
        np.savez('{name}_{n_walkers}walkers_{n_steps}steps'.format(name=self.name, n_walkers=self.n_walkers, n_steps=self.n_steps), positions = [pos[x] for x in range(len(pos))], lnprob = [prob[x] for x in range(len(prob))])
        print '\nAll done!'
        return sampler

#     def show_mcmc(self):
#         fig = plt.Figure()
#         ax = fig.add_subplot(111)
#         plt.plot(self.data[0], self.data[1])
#         plt.savefig('~/Desktop/testing_MCMC_plot.pdf')

    def simulate_for_mcmc(self, frequency, indices, thicknesses, polarization='s', theta_0=0):
        ''' Description

        Arguments
        ---------

        Returns
        -------
        '''
        # check the desired polarization
        if polarization == 'u':
            return Builder()._unpolarized_simulation(frequency)

        # get all the indices of refraction in one place
        n = []
        [n.append(index) for index in indices]
        n.insert(0, 1.)
        n.append(1.)

        # get all thicknesses in one place
        d = []
        [d.append(thickness) for thickness in thicknesses]
        d.insert(0, np.inf)
        d.append(np.inf)
        
        # convert theicness and dielectric lists to numpy arrays
        n = np.asarray(n)
        d = np.asarray(d)
        d *= 2.54e-5 #convert from mils to m

        # find the wavevectors, k
        k = 2*np.pi * n * frequency/3e8
        olderr = sp.seterr(invalid= 'ignore') # turn off 'invalid multiplication' error; it's just the 'inf' boundaries
        delta = k * d
        sp.seterr(**olderr) # turn the error back on

        # now get transmission and reflection amplitudes
        t_amp = np.zeros((len(n), len(n)), dtype=complex)
        r_amp = np.zeros((len(n), len(n)), dtype=complex)

        for i in range(len(n)-1):
            t_amp[i,i+1] = Builder()._t_at_interface(polarization, n[i], n[i+1])
            r_amp[i,i+1] = Builder()._r_at_interface(polarization, n[i], n[i+1])

        M = np.zeros((len(n),2,2),dtype=complex)
        for i in range(1,len(n)-1):
            M[i] = 1/t_amp[i,i+1] * np.dot(Builder()._make_2x2(np.exp(-1j*delta[i]),
                                                          0., 0., np.exp(1j*delta[i]),
                                                          dtype=complex),
                                           Builder()._make_2x2(1., r_amp[i,i+1], r_amp[i,i+1], 1.,
                                                          dtype=complex))
        M_prime = Builder()._make_2x2(1., 0., 0., 1., dtype=complex)
        for i in range(1, len(n)-1):
            M_prime = np.dot(M_prime, M[i])
        M_prime = np.dot(Builder()._make_2x2(1., r_amp[0,1], r_amp[0,1], 1., dtype=complex)/t_amp[0,1],
                         M_prime)
        
        # Now find the net transmission and reflection amplitudes
        t = 1/M_prime[0,0]
        r = M_prime[0,1]/M_prime[0,0]
        
        # Now find the net transmitted and reflected power
        T = Builder()._get_T(polarization, t, n[0], n[-1])
        R = Builder()._get_R(r)
        return T
