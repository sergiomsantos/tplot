# -*- coding: utf-8 -*-
"""
TPlot.py

A Python package for creating and displaying matplotlib plots in the console/terminal
"""


__copyright__ = "Copyright 2019, Sérgio Miguel Santos, Univ. Aveiro - Portugal"
__author__ = 'Sérgio Miguel Santos'
__version__ = '0.4.1'
__license__ = "MIT"


import sys
import os


IS_PYTHON3 = sys.version_info[0] == 3

MPL_DISABLED = 'TPLOT_NOGUI' in os.environ


if MPL_DISABLED:
    import matplotlib
    matplotlib.use('Agg')


from .tplot import TPlot, Format
from .utils import get_output_size
from .ansi import Ansi
from .cli import main

