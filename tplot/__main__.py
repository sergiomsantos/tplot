import argparse
import sys
import numpy as np
from __init__ import TPlot

def main(args):

    data = np.loadtxt(args.file, skiprows=args.skip, delimiter=args.delimiter)
    if len(data.shape) == 1:
        data = data.reshape(1,-1)
    else:
        data = data.T
    
    plot = TPlot(args.width, args.height, args.logx, args.logy)
    
    if not (args.c or args.xy or args.hist):
        for n,row in enumerate(data):
            plot.plot(row, label='col-%d'%n)

    for col in args.c:
        plot.plot(data[col], label='col-%d'%col)
    
    for col in args.hist:
        limits = args.ax[-1] if args.ax else None
        hist, bin_edges = np.histogram(data[col], bins=args.bins, range=limits)
        x = bin_edges[:-1] + bin_edges[1:]
        plot.plot(0.5*x, hist, label='hist-%d'%col, fill=True)
        plot.set_xticks(bin_edges)
    
    for i,j in args.xy:
        plot.plot(data[i], data[j], label='%d-vs-%d'%(j,i))
    
    if args.ax:
        plot.set_xlim(*args.ax[-1])
    if args.ay:
        plot.set_ylim(*args.ay[-1])
    
    if args.mpl:
        plt.show()
    else:
        print(plot)


try:
    import shutil
    tsize = shutil.get_terminal_size()
except:
    from collections import namedtuple
    import os
    TSize = namedtuple('TSize', ['columns', 'lines'])
    try:
        r,c = os.popen('stty size', 'r').read().split()
    except:
        r,c = 24,80
    tsize = TSize(int(c), int(r))


class TwoArgs(argparse._AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != 2:
            raise argparse.ArgumentError(self,
                '%s requires 2 values, %d given' % (option_string, len(values)))
        super(TwoArgs, self).__call__(parser, namespace, values, option_string)

class TwoOrThree(argparse._AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) not in (2,3):
            raise argparse.ArgumentError(self,
                '%s requires 2 or 3 values, %d given' % (option_string, len(values)))
        super(TwoArgsLabeled, self).__call__(parser, namespace, values, option_string)


SRC = None if sys.stdin.isatty() else sys.stdin

parser = argparse.ArgumentParser('tplot')
parser.add_argument('file', nargs='?', default=SRC)

group = parser.add_argument_group('Plot arguments')
group.add_argument('-xy', action=TwoArgs, nargs='+', type=int,
    metavar=('X','Y'), help='scatter plot of column X vs Y', default=[])
group.add_argument('-c', type=int, nargs='*',
    metavar='C', help='series plot of column(s) C', default=[])
group.add_argument('--hist', nargs='*', type=int,
    metavar='H', help='histogram of column(s) H', default=[])
group.add_argument('--bins', type=int,
    metavar='N', help='number of bins', default=10)

group = parser.add_argument_group('Data parsing')
group.add_argument('-d', '--delimiter', type=str,
    metavar='D', help='delimiter')
group.add_argument('-s', '--skip', type=int, default=0,
    metavar='N', help='skip first N rows')

group = parser.add_argument_group('Axis configuration')
group.add_argument('-ax', action=TwoArgs, nargs='+', type=float,
    metavar=('xmin','xmax'), help='x-axis limits')
group.add_argument('-ay', action=TwoArgs, nargs='+', type=float,
    metavar=('ymin','ymax'), help='y-axis limits')
group.add_argument('--logx', action='store_true',
    help='set log-scale on the x-axis')
group.add_argument('--logy', action='store_true',
    help='set log-scale on the y-axis')

group = parser.add_argument_group('Output configuration')
group.add_argument('--width', type=int,
    metavar='W', help='output width', default=tsize.columns)
group.add_argument('--height', type=int,
    metavar='H', help='output height', default=tsize.lines)
group.add_argument('--mpl', action='store_true', help='show plot in matplotlib window')

args = parser.parse_args()

if args.file is None:
    parser.print_help()
    print('\nError: Missing "file" argument.')
    exit(1)
    
# test()
# print(args)
main(args)
