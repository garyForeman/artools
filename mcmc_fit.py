'''
'''
# Author: Andrew Nadolski
# Filename: mcmc_fit.py

import core
import matplotlib.pyplot as plt
import corner
import numpy as np

index = [2.4, 2.6, 9.7, 2.6, 2.4]
thick = [14.5, 1.5, 250., 1.5, 14.5]

p0 = index+thick

run_name = 'eptfe_50walkers_50steps'
fit = core.FitFTS('cut_eptfe_transmission.txt', 50, 10, 50)
result = fit.run_mcmc(p0)

samples = result.chain[:, 10:, :].reshape((-1, fit.n_dim))

fig = corner.corner(samples, labels=['$\epsilon_{r}$ eptfe.1','$\epsilon_{r}$ epoxy1', '$\epsilon_{r}$ alumina', '$\epsilon_{r}$ epoxy2', '$\epsilon_{r}$ eptfe.2', 't_eptfe.1', 't_epoxy1', 't_alumina', 't_epoxy2', 't_eptfe.2'], truths=[2.4, 2.6, 9.7, 2.6, 2.4, 14.5, 1.5, 250., 1.5, 14.5], show_titles=True)

fig.savefig('./{run_name}.pdf'.format(run_name=run_name))
