''' A debugging script for artools.
Maybe also use this for test cases?
'''

#Filename: debug_artools.py
#Author: Andrew Nadolski

import artools as ar
import numpy as np

# b = ar.simulator.Builder()
# b.save_path = '/Users/E5mini/git/artools/scratch'
# b.set_freq_sweep(0.,300.)
# b.add_layer(material='vacuum', type='source')
# b.add_layer(material='eptfe', thickness=15.0)
# b.add_layer(material='ro3035', thickness=5.0)
# b.add_layer(material='ro3006', thickness=5.0)
# b.add_layer(material='alumina', type='terminator')
# # b.add_layer(material='alumina', thickness=250.0)
# # b.add_layer(material='ro3006', thickness=5.0)
# # b.add_layer(material='ro3035', thickness=5.0)
# # b.add_layer(material='eptfe', thickness=15.0)
# # b.add_layer(material='vacuum', type='terminator')
# #b.add_custom_layer('my_thing', thickness=22.1, units='mm', dielectric=8.2, loss_tangent=0.0025)


# results = b.run_sim()

# #my_plot = ar.plotter.MCPlot()
# #my_plot = ar.plotter.ReflectionPlot()
# my_plot = ar.plotter.TransmissionPlot()
# my_plot.save_path = '/Users/E5mini/git/artools/scratch'


# #my_plot.plot_vs_wavelength()
# my_plot.toggle_bandpasses() #turn on bandpasses
# my_plot.toggle_legend() #turn on the legend
# my_plot.add_bandpass(75., 110., color='red', label='red region')
# my_plot.add_bandpass(135., 170., color='green', label='green region')
# my_plot.add_bandpass(200., 250., color='blue', label='blue region')

# my_plot.load_data(results)

# my_plot.make_plot()


b = ar.simulator.Builder()
b.structure = [1,2,3,4,5]

f = 150e9
n = np.sqrt(np.array([1.0, 2.4, 3.5, 6.15, 9.7]))
loss = np.array([0.0, 2.5e-4, 1.7e-3, 1.526e-3, 7.4e-4])
d = np.array([1000.0, 15.0*2.54e-5, 5.0*2.54e-5, 5.0*2.54e-5, 1000.0])
print "\n", f
print n
print loss
print d, "\n"

k = b._find_ks(n, f, loss)
delta = b._find_k_offsets(k, d)

print "\n", k
print delta, "\n"

rt = b._calc_R_T_amp('s', n, delta)
