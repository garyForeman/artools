# artools

### Dependencies:
***
*glob
*matplotlib
*numpy
*os
*pprint
*scipy
*shutil
*time

### Work flow:
***
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