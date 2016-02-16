'''
'''
# Author: Andrew Nadolski
# Filename: mcmc_fit.py

import core

index = [2.4, 2.6, 9.7, 2.6, 2.4]
thick = [14.5, 1.5, 250., 1.5, 14.5]

p0 = index+thick

fit = core.FitFTS('./cut_eptfe_transmission.txt')
result = fit.run_mcmc(p0)

fit.show_mcmc()
