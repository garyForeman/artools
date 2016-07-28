# artools
#### A simple anti-reflection coating simulator
***
***

### Purpose:
***
artools is designed to allow a user to easily simulate and visualize anti-reflection coating performance. The coating can be composed of a single material, or many layers of different materials. The program calculates reflection and transmission at each interface, accounting for dielectric loss as light propagates through the media. The results can be easily visualized with artools' plotting functionality. The artools plotting routines are matplotlib functions wrapped and combined to generate plots that are useful for assessing transmission or reflection of an electromagnetic wave.

The program is intended to be easy to use, so much of the "heavy-lifting" is hidden from the user in interactive mode (e.g. iPython) by using single underscore notation.

### Dependencies:
***
* glob
* matplotlib
* numpy
* os
* pprint
* scipy
* shutil
* time

### How it Works:
***
The results of the simulation are calculated using a transfer matrix method (TMM). This particular transfer matrix method was written by H.S. Hou and published in Applied Optics Volume 13 Number 8, 1974. The Hou TMM is a computationally efficient means of calculating reflection and transmission through multilayer media.

### Work Flow:
***
The 

An example follows below:

```python

import artools as ar

b = ar.simulator.Builder()

b.set_freq_sweep(0.,300.)
b.add_layer('eptfe', 15.0)
b.add_layer('ro3035', 5.0)
b.add_layer('ro3006', 5.0)

results = b.run_sim()

my_plot = ar.plotter.TransmissionPlot()

my_plot.load_data(results)
my_plot.make_plot()
```

### Future Improvements:
***
* Expand library of materials and material properties
* Implement temperature-dependent loss
* Implement non-normal starting angles
* Implement emcee
* GUI?

### Acknowledgements:
***
Thanks to Colin Merkel, Aritoki Suzuki, and Steve Byrnes 