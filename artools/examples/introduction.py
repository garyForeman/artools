"""
This script is an introduction to artools. I'll comment it heavily, and try to
describe everything as I go.

In this example I will show you how to set up and run a simple simulation, and
then plot the results. I will simulate a trilayer PTFE anti-reflection coating.

To run this example:
    python introduction.py

All the output will be stashed in artools/examples/scratch
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

""" Now I add the 4 elements required by Builder. """
b.save_path = 
