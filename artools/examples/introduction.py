"""
This script is an introduction to artools. I'll comment it heavily, and try to
describe everything as I go.

In this example I will show you how to set up and run a simple simulation, and
then plot the results. I will simulate a trilayer PTFE anti-reflection coating.

To run this example:
    python introduction.py

All the output will be stashed in artools/examples/scratch .
"""

# Author: Andrew Nadolski
# Filename: introduction.py


""" First I import the two bits of machinery I'll need. """
import artools as ar
import numpy as np

"""
Now I create a 'Builder'. The Builder will move all the parts of the
simulation into the proper places, but I have to feed it a few things first:

     1) A frequency range to sweep through
     2) A source medium
     3) Some anti-reflective materials
     4) A terminating medium

Everything else is optional. (At least it is supposed to be.)
"""
b = ar.simulator.Builder()

"""
Now I add the 4 elements required by Builder. Take note of how I'm adding the AR
coating layers! I'm using materials that artools already knows about. These
materials live in 'materials.py'; you can check out it's contents and add to it,
if you're so inclined. I'll (slowly) continue to expand it.
"""
b.set_freq_sweep(0.,300.)                           # 1; Frequency in GHz
b.add_layer(material='vacuum', type='source')       # 2; Note the "type" arg
b.add_layer(material='eptfe', thickness=15.0)       # 3; Order matters when you
b.add_layer(material='ro3035', thickness=5.0)       #    add AR layers!
b.add_layer(material='ro3006', thickness=5.0)
b.add_layer(material='alumina', type='terminator')  # 4; Note the "type" arg
b.save_path = 'scratch'                             # OPTIONAL

""" And I run the sim, storing the results for quick plotting. """
results = b.run_sim()

"""
I want to look at the transmission of the coating I just simulated. I can do
that by creating a 'TransmissionPlot'. It isn't meant to be pretty plot; it is
meant to give me a quick way to check that the result are useful before going
into deeper analysis.
"""
my_plot = ar.plotter.TransmissionPlot()
my_plot.save_path = 'scratch'                       # OPTIONAL
my_plot.load_data(results)
my_plot.make_plot()

"""
The output of the simulation and the plot are both saved with a timestamps as
names in the 'scratch' directory.
"""
