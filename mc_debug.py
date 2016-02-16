''' Debugger for mcmc_fit.py

Check out these resources:

http://stackoverflow.com/questions/9663562/what-is-difference-between-init-and-call-in-python
https://www.youtube.com/watch?v=0tYaMTK-1K0
https://github.com/dfm/emcee/blob/master/examples/rosenbrock.py
http://dan.iel.fm/emcee/current/user/line/
http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay8_MCMC_with_emcee.html
http://stats.stackexchange.com/questions/72511/difficulty-with-mcmc-implementation
http://www.ast.cam.ac.uk/~koposov/files/statpy_lectures2014/lecture3.pdf


'''

# Author: Andrew Nadolski
# Filename: mc_debug.py

import mcmc_fit as mc
import matplotlib.pyplot as plt
import time
import core
import emcee

starting_d = [14.5, 1.5, 250., 1.5, 14.5] #initial guess at thickness (mils) and dielectic
starting_n = [2.4, 2.6, 9.7, 2.6, 2.4]





# testmc = mc.MCMC()

# i=0
# while i < 100:
#     testmc.simple_bettor(funds=10000, initial_wager=100, wager_count=1000)
#     i += 1

# #print testmc.num_wagers
# testmc.show_mc()


