artools
=======
#### A simple anti-reflection coating simulator


Purpose:
--------
artools is designed to allow a user to easily simulate and visualize anti-reflection coating performance. The coating can be composed of a single material, or many layers of different materials. The program calculates reflection and transmission at each interface, accounting for dielectric loss as light propagates through the media. The results can be easily visualized with artools' plotting functionality. The artools plotting routines are matplotlib functions wrapped and combined to generate plots that are useful for assessing transmission or reflection of an electromagnetic wave.

The program is intended to be easy to use--if it isn't, I would like to hear your feed back. That said, it would also be nice to hear what you like and what you would like to see. Please email me, stop me in the hall, make smoke signals, or send a telegram. Something.

How it Works:
-------------
The results of the simulation are calculated using a transfer matrix method (TMM). This particular transfer matrix method was written by H.S. Hou and published in Applied Optics Volume 13 Number 8, 1974. The Hou TMM is a computationally efficient means of calculating reflection and transmission through multilayer media.

Dependencies:
-------------
* matplotlib
* numpy
* scipy


Getting artools:
----------------
There are three ways to install artools:

1. pip
   `pip install artools`

2. Anaconda
   `conda install --channel https://conda.anaconda.org/anadolski artools`

3. python
   Download the repository and run:
   `python setup.py install`

You have to have matplotlib, numpy, and scipy installed before this program will work (see the "Dependencies" section above). If you use the Anaconda package manager, then all these packages are already installed. If you don't use Anaconda, then you may have to search them out on your own.

How to use artools:
-------------------
So far I've mostly run scripts, but I've also played with artools in iPython. I prefer scripts for work; iPython is nice for checking docstrings on the fly, and exploring functionality in the progam.

Work Flow:
----------
1. Create a "Builder" (the name is a relic from an earlier iteration of this code).

2. Choose a frequency range.

3. Set a source medium, i.e. the medium from which the wave it produced.

4. Add layers of anti-reflective material. You can either specify a name, thickness, dielectric constant, and loss tangent, or you can select a material from the list of known materials within the progrom.

5. Set a terminating medium.

6. Execute the simulation.

After the simulation runs you will be left with a columnized, tab-separated, text file. The columns are frequency, fractional transmission, and fractional reflection (and also labeled in a comment/header in the file). You can use the limited plotting functionality included in this program to view the results quickly to see if they're worth keeping.

Examples:
---------
See 'artools/examples' for details.

TODO:
-----
* Expand library of materials and material properties
* Implement temperature-dependent loss
* Implement non-normal incidence angles
* Implement emcee
* GUI?
* Better documentation!
* Make an examples folder

Acknowledgements:
-----------------
Thanks to Colin Merkel, Aritoki Suzuki, and Steve Byrnes 