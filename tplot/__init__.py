# -*- coding: utf-8 -*-
"""
TPlot.py

A Python package for creating and displaying matplotlib plots in the console/terminal
"""
from __future__ import print_function

__license__ = "MIT"
__version__ = '0.4.0'
__author__ = 'Sérgio Miguel Santos'
__copyright__ = "Copyright 2019, Sérgio Miguel Santos, Univ. Aveiro - Portugal"

# from itertools import cycle
import sys
import os

# IS_PYTHON3 = sys.version_info[0] == 3
# MPL_DISABLED = 'TPLOT_NOGUI' in os.environ

from . import constants
from .constants import Colors, Format

if constants.MPL_DISABLED:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .tplot import TPlot

__all__ = ['Colors', 'Format', 'TPlot', 'main', 'run']


if constants.IS_PYTHON3:
    def to_braille(m):
        # 10240 = int('2800', 16)
        return chr(m+10240)
else:
    def to_braille(m):
        return unichr(m+10240).encode('utf-8')

# try:
#     from scipy.signal import convolve2d
# except ImportError as ex:
#     # adapted from https://stackoverflow.com/a/43087771
#     def convolve2d(mat, kernel, **kwargs):
#         s = kernel.shape + tuple(np.subtract(mat.shape, kernel.shape) + 1)
#         strd = np.lib.stride_tricks.as_strided
#         subM = strd(mat, shape = s, strides = mat.strides * 2)
#         return np.einsum('ij,ijkl->kl', kernel, subM)


# class Colors:

#     RESET = '\033[0m'
#     BOLD  = '\033[1m'
#     ITALIC = '\033[3m'
#     UNDERLINE = '\033[4m'
#     _COLORS = None
#     _ENABLED = True
    
#     @staticmethod
#     def load():
#         def get(name, default):
#             # red, green, yellow, blue, magenta, cyan
#             # change (30+n) to (90+n) for light-color variants
#             color = os.getenv(name, None)
#             if color is None:
#                 color = default
#             else:
#                 if IS_PYTHON3:
#                     color = bytes(color, 'utf-8').decode('unicode_escape')
#                     # color = color.decode('unicode_escape')
#                 else:
#                     # color = color.decode('string_escape')
#                     color = color.decode('unicode_escape')
#             return color
        
#         Colors._COLORS = {
#             'COLOR%d'%n: get('TPLOT_COLOR%d'%n, '\033[%dm'%(30+n)) for n in range(1,7)
#         }
        
#         # light gray
#         Colors._COLORS['GRID'] = get('TPLOT_GRID', '\033[2m')
        
#     @staticmethod
#     def get(name):
#         if Colors._COLORS is None:
#             Colors.load()
#         return Colors._COLORS.get(name, '')
    
#     @staticmethod
#     def as_list():
#         if Colors._COLORS is None:
#             Colors.load()
#         colors = [Colors._COLORS[s]
#                     for s in sorted(Colors._COLORS.keys())
#                     if s.startswith('COLOR')]
#         return colors
    
#     @staticmethod
#     def disable():
#         Colors._ENABLED = False
    
#     @staticmethod
#     def enable():
#         Colors._ENABLED = True
    
#     @staticmethod
#     def format(s, *prefixes):
#         if Colors._ENABLED:
#             return ''.join(prefixes) + s + Colors.RESET
#         return s


# BRAILLE_KERNEL = np.array([
#     [  1,   8],
#     [  2,  16],
#     [  4,  32],
#     [ 64, 128]
# ])

# KERNEL41 = np.array([
#     [ 1],
#     [ 2],
#     [ 4],
#     [ 8],
# ])


# class Format:
#     NONE   = 0
#     LEFT   = 1
#     TOP    = 2
#     RIGHT  = 4
#     BOTTOM = 8
#     ALL    = 15
#     TOP_LEFT = 3
#     TOP_RIGHT = 6
#     BOTTOM_LEFT = 9
#     BOTTOM_RIGHT = 12


def get_unicode_array(size, fill=u''):
    ar = np.empty(size, dtype='U32')
    ar[:] = fill
    return ar




def run(args):
    
    # load data
    data = np.loadtxt(
                args.file,
                skiprows=args.skip,
                delimiter=args.delimiter,
                comments=args.comment)
    
    if len(data.shape) == 1:
        data = data.reshape(1,-1)
    else:
        data = data.T
    
    # instantiate TPlot
    plot = TPlot(args.width, args.height,
                #logx=args.logx,
                #logy=args.logy,
                padding=args.padding,
                # connect_points=args.lines
    )

    if args.logx:
        plot.set_xscale('log')
    if args.logy:
        plot.set_yscale('log')
    
    # configure output
    
    plot.show_grid(args.grid)
    plot.set_padding(*args.padding)
    
    plot.set_tick_position(getattr(Format, args.labels))
    plot.set_border(getattr(Format, args.border))
    plot.set_xtick_format(args.x_fmt)
    plot.set_ytick_format(args.y_fmt)

    if args.no_color:
        Colors.disable()

    # if no type is provided, simply plot all 
    # columns as series
    if not (args.c or args.xy or args.hist):
        for n,row in enumerate(data):
            plot.line(row, label='col-%d'%n)

    # add series
    for col,l in args.c:
        if l is None:
            l = 'col-%d'%col
        plot.line(data[col], label=l, connect=args.lines)
    
    # add histograms
    for col,l in args.hist:
        if l is None:
            l = 'hist-%d'%col
        limits = args.ax[-1] if args.ax else None
        plot.hist(data[col], bins=args.bins, range=limits, label=l)
        
    # add scatter plots
    for i,j,l in args.xy:
        if l is None:
            l = '%d-vs-%d'%(j,i)
        plot.plot(data[i], data[j], label=l)
    

    # finally set axis limits
    if args.ax:
        plot.xlim = args.ax[-1]
    if args.ay:
        plot.ylim = args.ay[-1]
    
    # and show output
    if args.mpl:
        if MPL_DISABLED:
            print('Matplotlib output is disabled!')
            print(' > unset environment variable TPLOT_NOGUI and try again.')
        else:
            plt.legend()
            plt.show()
    else:
        plot.show()


def get_output_size():
    from collections import namedtuple
    
    TSize = namedtuple('TSize', ['columns', 'lines'])

    # try to get default size from env variables
    try:
        size = os.getenv('TPLOT_SIZE', None)
        c,r = size.split(',')
        return TSize(int(c), int(r))
    except:
        pass
    
    # try shutil if py3
    try:
        import shutil
        return shutil.get_terminal_size()
    except:
        pass
    
    # try stty if py2
    try:
        r,c = os.popen('stty size', 'r').read().split()
        return TSize(int(c), int(r))
    except:
        pass

    # final fallback option
    return TSize(80, 24)


def main():
    
    import argparse
    
    tsize = get_output_size()
    print('DEFAULT SIZE =', tsize.lines, tsize.columns)

    def get_append_action(n):
        class CustomAppendAction(argparse._AppendAction):
            def __call__(self, parser, namespace, values, option_string=None):
                if len(values) == n:
                    values.append(None)
                elif len(values) != (n+1):
                    raise argparse.ArgumentError(self,
                        '%s requires %d int(s) and an optional label, %d values given. Consider splitting across multiple %s flags.' % (
                            option_string, n, len(values), option_string))
                try:
                    label = values[-1]
                    values = list(map(int, values[:-1]))
                except ValueError as ex:
                    raise argparse.ArgumentError(self,str(ex))
                else:
                    values.append(label)
                super(CustomAppendAction, self).__call__(parser, namespace, values, option_string)
        return CustomAppendAction
    
    desc = 'A Python package for creating and displaying matplotlib plots in the console/terminal'

    SRC = None if sys.stdin.isatty() else sys.stdin

    # Instantiate parser
    # ------------------------------------------- 
    parser = argparse.ArgumentParser('tplot', description=desc)
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)

    # parser: Input source
    # ------------------------------------------- 
    parser.add_argument('-f', '--file', default=SRC, help='source file. Use "-" to read from stdin')

    # parser: plot arguments
    # ------------------------------------------- 
    group = parser.add_argument_group('Plot arguments')
    group.add_argument('-c', action=get_append_action(1), nargs='+', type=str,
        metavar='C L?', help='series plot of column(s) C with optional label L', default=[])
    group.add_argument('-xy', action=get_append_action(2), nargs='+', type=str,
        metavar='X Y L?', help='scatter plot of column X vs Y with optional label L', default=[])
    group.add_argument('--hist', action=get_append_action(1), nargs='+', type=str,
        metavar='H L?', help='histogram of column(s) H with optional label L', default=[])
    group.add_argument('--bins', type=int,
        metavar='N', help='number of bins', default=10)
    group.add_argument('--lines', action='store_true',
        help='requires that the x-coordinate sequence is increasing '+
             'if the -xy option is specified')

    # parser: data parsing
    # ------------------------------------------- 
    group = parser.add_argument_group('Data parsing')
    group.add_argument('-d', '--delimiter', type=str,
        metavar='D', help='delimiter')
    group.add_argument('-s', '--skip', type=int, default=0,
        metavar='N', help='skip first N rows')
    group.add_argument('--comment', type=str, default=None, nargs='*',
        metavar='S', help='Characters used to indicate the start of a comment')

    # parser: axes configuration
    # ------------------------------------------- 
    group = parser.add_argument_group('Axis configuration')
    group.add_argument('-ax', action='append', nargs=2, type=float,
        metavar=('xmin','xmax'), help='x-axis limits')
    group.add_argument('-ay', action='append', nargs=2, type=float,
        metavar=('ymin','ymax'), help='y-axis limits')
    group.add_argument('--logx', action='store_true',
        help='set log-scale on the x-axis')
    group.add_argument('--logy', action='store_true',
        help='set log-scale on the y-axis')
    group.add_argument('--grid', action='store_true',
        help='show grid')
    
    choices=[k for k in vars(Format).keys() if not k.startswith('_')]
    group.add_argument('--border', choices=choices, default='BOTTOM_LEFT')
    group.add_argument('--labels', choices=choices, default='BOTTOM_LEFT')
    group.add_argument('--x-fmt', type=str, default='%r')
    group.add_argument('--y-fmt', type=str, default='%r')

    # parser: output configuration
    # ------------------------------------------- 
    group = parser.add_argument_group('Output configuration')
    group.add_argument('--width', type=int,
        metavar='W', help='output width', default=tsize.columns)
    group.add_argument('--height', type=int,
        metavar='H', help='output height', default=tsize.lines)
    group.add_argument('--padding', type=int, nargs='+',
        metavar='P', help='left padding', default=[2])
    group.add_argument('--mpl', action='store_true', help='show plot in matplotlib window')
    group.add_argument('--no-color', action='store_true', help='suppress colored output')


    # parser: run parser
    # ------------------------------------------- 
    args = parser.parse_args()

    if args.file is None:
        print('Error: Missing "file" (-f) argument.')
        exit(1)
    elif args.file == '-':
        args.file = sys.stdin

    #import pprint
    #pprint.pprint(vars(args), indent=4)
    
    # do some work
    run(args)

