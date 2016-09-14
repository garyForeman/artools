''' A debugging script for artools.
Maybe also use this for test cases?
'''

#Filename: debug_artools.py
#Author: Andrew Nadolski

import artools as ar
import numpy as np

b = ar.simulator.Builder()
b.save_path = '/Users/E5mini/git/artools/scratch'
b.set_freq_sweep(0.,300.)
b.add_layer(material='eptfe', thickness=15.0)
b.add_custom_layer('mid_thermal_spray', thickness=10.0, units='mil', dielectric=3.4, loss_tangent=7.4e-4)
b.add_custom_layer('bot_thermal_spray', thickness=7.0, units='mil', dielectric=7.0, loss_tangent=7.4e-4)
b.add_layer(material='vacuum', type='source')
#b.add_layer(material='eptfe', thickness=15.0)
#b.add_layer(material='ro3035', thickness=5.0)
#b.add_layer(material='ro3006', thickness=5.0)
b.add_layer(material='alumina', type='terminator')
# b.add_layer(material='alumina', thickness=250.0)
# b.add_layer(material='ro3006', thickness=5.0)
# b.add_layer(material='ro3035', thickness=5.0)
# b.add_layer(material='eptfe', thickness=15.0)
# b.add_layer(material='vacuum', type='terminator')
#b.add_custom_layer('my_thing', thickness=22.1, units='mm', dielectric=8.2, loss_tangent=0.0025)


results = b.run_sim()

#my_plot = ar.plotter.MCPlot()
#my_plot = ar.plotter.ReflectionPlot()
my_plot = ar.plotter.TransmissionPlot()
my_plot.save_path = '/Users/E5mini/git/artools/scratch'


#my_plot.plot_vs_wavelength()
my_plot.toggle_bandpasses() #turn on bandpasses
my_plot.toggle_legend() #turn on the legend
my_plot.add_bandpass(81.7, 107.5, color='red', label='95 GHz band')
my_plot.add_bandpass(128.6, 167.2, color='green', label='150 GHz band')
my_plot.add_bandpass(196.9, 249.2, color='blue', label='220 GHz band')

my_plot.load_data(results)

my_plot.make_plot()

#############################
## DEBUGGING FOR C_ARTOOLS ##
############################


# b = ar.simulator.Builder()
# b.structure = [1,2,3,4,5]

# f = 150e9
# n = np.sqrt(np.array([1.0, 2.4, 3.5, 6.15, 9.7]))
# loss = np.array([0.0, 2.5e-4, 1.7e-3, 1.526e-3, 7.4e-4])
# d = np.array([1000.0, 15.0*2.54e-5, 5.0*2.54e-5, 5.0*2.54e-5, 1000.0])
# print "\n", f
# print n
# print loss
# print d, "\n"

# k = b._find_ks(n, f, loss)
# delta = b._find_k_offsets(k, d)

# print "\n", k
# print delta, "\n"

# r_amp = np.zeros((len(b.structure), len(b.structure)), dtype=complex)
# for i in range(len(b.structure)-1):
#     r_amp[i, i+1] = b._r_at_interface('s', n[i], n[i+1])

# rt = b._calc_R_T_amp('s', n, delta)

# T = b._get_T('s', rt[1], n[0], n[-1])
# R = b._get_R(rt[0])

# print("\nPercent transmission is ---> {}".format(T))
# print("Percent reflection is ---> {}".format(R))

# # M = np.zeros((len(b.structure), 2, 2), dtype=complex)

# # for i in range(1, len(b.structure)-1):
# #     M[i] = np.dot(b._make_2x2(np.exp(-1j*delta[i]), 0., 0., np.exp(1j*delta[i]), dtype=complex), b._make_2x2(1., r_amp[i,i+1], r_amp[i,i+1], 1., dtype=complex))

# # print("\n\nWeirdo nonsense\n\n")
# # for i in M:
# #     print i

# # testA = np.zeros((2,2), dtype=float)
# # testB = np.zeros((2,2), dtype=float)
# # testProd = np.zeros((2,2), dtype=float)

# # valA = 1
# # valB = 5
# # for i in range(2):
# #     for j in range(2):
# #         testA[i][j] = valA
# #         testB[i][j] = valB
# #         valA += 1
# #         valB += 1

# # print testA
# # print testB

# # for i in range(2):
# #     for j in range(2):
# #         sum = 0
# #         for l in range(2):
# #             print "A{}{} --->".format(i,l), testA[i][l], " * B{}{}".format(l,j), testB[l][j]
# #             sum = sum + testA[i][l]*testB[l][j]
# #         testProd[i][j] = sum

# # print testProd
