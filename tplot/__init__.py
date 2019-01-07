# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


class Colors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



class TPlot(object):
    
    def __init__(self, columns, lines, logx=False, logy=False, padding=10, use_colors=True):
        self.columns = columns - padding - 5
        self.lines = lines - 5
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111)
        self.datasets = []
        self.padding = padding

        self.logx = logx
        if logx:
            self.ax.set_xscale('log')
        
        self.logy = logy
        if logy:
            self.ax.set_yscale('log')
        
        self._xticks = None
        if use_colors:
            self._colors  = cycle([Colors.BLUE, Colors.GREEN, Colors.RED, Colors.PURPLE])
        else:
            self._colors  = cycle(['',''])
        self._markers = cycle('ox+-.')
    
    def plot(self, x, y=None, color=None, marker=None, label=None, fill=False):
        if y is None:
            y = x
            x = np.arange(len(y))
        if color is None:
            color = next(self._colors)
        if marker is None:
            marker = next(self._markers)
        if label is None:
            label = 'dataset-%d' % len(self.datasets)
        self.datasets.append((x,y,color,marker,label,fill))
        self.ax.plot(x,y)


    def set_xlim(self, xmin, xmax):
        self.ax.set_xlim(xmin, xmax)


    def set_ylim(self, ymin, ymax):
        self.ax.set_ylim(ymin, ymax)        
    

    def transform(self, x, y):
        
        if self.logx:
            x = np.log10(x)
        if self.logy:
            y = np.log10(y)

        mapped = self.ax.transLimits.transform(list(zip(x,y)))
        x_in_range,y_in_range = np.logical_and(mapped>=0.0, mapped<=1.0).T
        idx, = np.nonzero(x_in_range & y_in_range)
        mapped = mapped[idx]
        mapped = np.round(mapped*[self.columns-1,self.lines-1])
        
        return mapped.astype(int), idx
    
    def get_canvas(self):
        
        ylim = self.ax.get_ylim()
        self.ax.set_ylim(reversed(ylim))
        
        canvas = [(self.columns)*[' '] for _ in range(self.lines)]
        
        # add curves
        for x,y,c,m,l,f in self.datasets:
            mapped,_ = self.transform(x, y)
            marker = c + m + Colors.ENDC
            for i,j in mapped:
                canvas[j][i] = marker
                if f:
                   for line in canvas[j:]:
                       line[i] = marker
        
        # add legends
        # -----------------------------
        for n,dataset in enumerate(self.datasets):
            _,_,color,marker,label,_ = dataset
            k = len(label)
            legend = color + label + ' ' + marker + Colors.ENDC
            canvas[n+1] = canvas[n+1][:-k-4] + [' ', legend]
        
        # add frame
        # -----------------------------
        padding = self.padding*' '
        for n,line in enumerate(canvas):
            line.insert(0, padding + '│')
        canvas.append(len(line)*['─'])

        # add y-ticks
        # -----------------------------
        fmt = '%%%d.2e ┤' % (self.padding-1)
        for i,label in self.get_yticks():
            canvas[i][0] = fmt%label

        # add x-ticks
        # -----------------------------
        xticks = self.get_xticks()
        fmt = '%%-%d.2e'%(xticks[1][0]-xticks[0][0])

        labels = ''
        for i,label in xticks:
            canvas[-1][i] = '┬'
            labels += fmt%label
        canvas[-1].insert(0, padding + '└')

        canvas.append([padding[:-3] + xticks[0][0]*' ' + labels.rstrip()])

        # reset y-limits        
        self.ax.set_ylim(ylim)
        
        return canvas
    

    def set_xticks(self, ticks):
        self._xticks = ticks

    def get_xticks(self):
        if self._xticks is None:
            ticks = self.ax.get_xticks()
        else:
            ticks = self._xticks
        xc,yc = self.ax.transLimits.inverted().transform((0.5,0.5))
        pos, idx = self.transform(ticks, yc*np.ones_like(ticks))
        return list(zip(pos[:,0], ticks[idx]))
    
    def get_yticks(self):
        ticks = self.ax.get_yticks()
        xc,yc = self.ax.transLimits.inverted().transform((0.5,0.5))
        pos, idx = self.transform(xc*np.ones_like(ticks), ticks)
        return list(zip(pos[:,1], ticks[idx]))

    def __str__(self):
        canvas = self.get_canvas()
        return '\n' + '\n'.join((''.join(line) for line in canvas)) + '\n'

    def __repr__(self):
        return str(self)


def run(args):

    data = np.loadtxt(args.file, skiprows=args.skip, delimiter=args.delimiter)
    if len(data.shape) == 1:
        data = data.reshape(1,-1)
    else:
        data = data.T
    
    plot = TPlot(args.width, args.height, args.logx, args.logy, use_colors=not args.no_color)
    
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
        s = str(plot)
        print(s)


def main():
    
    import argparse
    import sys
    
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

    # class TwoOrThree(argparse._AppendAction):
    #     def __call__(self, parser, namespace, values, option_string=None):
    #         if len(values) not in (2,3):
    #             raise argparse.ArgumentError(self,
    #                 '%s requires 2 or 3 values, %d given' % (option_string, len(values)))
    #         super(TwoArgsLabeled, self).__call__(parser, namespace, values, option_string)

    desc = 'A Python package for creating and displaying matplotlib plots in the console/terminal'

    SRC = None if sys.stdin.isatty() else sys.stdin

    parser = argparse.ArgumentParser('tplot', description=desc)
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
    group.add_argument('--no-color', action='store_true', help='suppress coloured output')

    args = parser.parse_args()

    if args.file is None:
        parser.print_help()
        print('\nError: Missing "file" argument.')
        exit(1)
        
    run(args)

