"""A reference module. It contains properties of materials
that are frequently used in AR coating simulations.
"""
# Author: Andrew Nadolski
# Filename: materials.py


class Electrical:	
	"""Contains the electrical properties of the materials that 
	are often used in AR coating simulations.
	"""
	# key is material name, first entry is dielectric constant, second entry is loss tangent
	props = {
                    'alumina' : (9.7, 7.4e-4),
                    'eptfe' : (1.6, 2.5e-4),
                    'ideal2' : (2., 0.),
                    'ideal4' : (4., 0.),
                    'ideal7' : (7., 0.),
                    'idealalumina' : (10., 0.),
                    'ro3006' : (6.15, 1.526e-3),
                    'ro3035' : (3.5, 1.7e-3),
                    'silicon' : (10.4, 1.5e-2),
                    'srtio3' : (300, 3.0e-4),
                    'stycast1266' : (2.6, 2.0e-2),
                    'stycast2850lv' : (5.36, 5.1e-2),
                    'ultem' : (3.15, 1.3e-3),
                    'vacuum' : (1., 0.),
                    }
	

# 	DIELECTRIC = {
#                     'alumina':9.7,
#                     'eptfe':2.4,
#                     'ideal2':2.,
#                     'ideal4':4.,
#                     'ideal7':7.,
# 		    'idealalumina': 10.,
#                     'ro3006':6.15,
#                     'ro3035':3.5,
#                     'silicon':10.4,
#                     'srtio3':300,
#                     'stycast1266':2.6,
#                     'stycast2850lv':5.36,
#                     'ultem':3.15,
# 		    'vacuum': 1.,
#                     }
                    
# 	LOSS_TAN = {
#                     'alumina':7.4e-4,
#                     'eptfe':2.5e-4,
#                     'ideal2':0.,
#                     'ideal4':0.,
#                     'ideal7':0.,
# 		    'idealalumina': 0.,
#                     'ro3006':1.562e-3,
#                     'ro3035':1.7e-3,
#                     'silicon':1.5e-2,
#                     'srtio3':3.0e-4,
#                     'stycast1266':2.0e-2,
#                     'stycast2850lv':5.1e-2,
#                     'ultem':1.3e-3,
# 		    'vacuum': 0.,
#                     }

		
class Thermal:
	"""Contains the thermal properties of the materials that 
	we use most often.
	"""
	pass
