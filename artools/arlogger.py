"""Configure logging for the artools application.

Right now this is just a test script to see if I can get things working. I'm
attempting to use the dictConfig method of setting up a logger. For now, I
just want it to print to the terminal. Perhaps later it will be useful to write
to a file.
"""

# Author: Andrew Nadolski
# Filename: arlogger.py

import logging
import logging.config


class ARLogger(object):
    """Contains all the logging set up in one place so I can jus import it
    into modules all willy-nilly.
    """

    def __init__(self):
        # First create the configuration dictionary
        self.arlog_config = {
            'version' : 1,
            'disable_existing_loggers' : True,
            'handlers' : {
                'console' : {
                    'level' : 'DEBUG',
                    'class' : 'logging.StreamHandler',
                    'formatter' : 'verbose',
                    },
                },
            'loggers' : {
                'artools' : {
                    'handlers' : ['console'],
                    'propagate' : True,
                    'level' : 'INFO',
                    },
                },
            'formatters' : {
                'verbose' : {
                    'format' : '%(asctime)s - %(module)s %(levelname)s:\t(%(funcName)s) %(message)s',
                    'datefmt' : '%H:%M:%S',
                    },
                },
            }

        # Pass the configuration dictionary to the logging module.
        logging.config.dictConfig(self.arlog_config)

        # Create a logger that will be passed to the different modules.
        # This logger should only output to the terminal.
        self.logger = logging.getLogger('artools')
