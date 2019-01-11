# -*- coding: utf-8 -*-
from __future__ import print_function

"""
TPlot.py

A Python package for creating and displaying matplotlib plots in the console/terminal
"""

__license__ = "MIT"
__version__ = '0.3.1'
__author__ = 'Sérgio Miguel Santos'
__copyright__ = "Copyright 2019, Sérgio Miguel Santos, Univ. Aveiro - Portugal"


from itertools import cycle
import sys
import os

IS_PY_VERSION_3 = sys.version_info[0] == 3
MPL_DISABLED = 'TPLOT_NOGUI' in os.environ

if MPL_DISABLED:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


__all__ = ['Colors', 'TPlot', 'main', 'run']



if IS_PY_VERSION_3:
    def to_braille(m, kernel):
        # 10240 = int('2800', 16)
        return chr((kernel*m).sum()+10240)
else:
    def to_braille(m, kernel):
        return unichr((kernel*m).sum()+10240).encode('utf-8')



class Colors:

    RESET = '\033[0m'
    BOLD  = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    _COLORS = None
    ENABLED = True
    
    @staticmethod
    def load():
        def get(name, default):
            # red, green, yellow, blue, magenta, cyan
            # change (30+n) to (90+n) for light-color variants
            color = os.getenv(name, None)
            if color is None:
                color = default
            else:
                if IS_PY_VERSION_3:
                    color = bytes(color, 'utf-8').decode('unicode_escape')
                else:
                    color = color.decode('string_escape')
            return color
        
        Colors._COLORS = {
            'COLOR%d'%n: get('TPLOT_COLOR%d'%n, '\033[%dm'%(30+n)) for n in range(1,7)
        }
        
        # light gray
        Colors._COLORS['GRID'] = get('TPLOT_GRID', '\033[2m')
        
    @staticmethod
    def get(name):
        if Colors._COLORS is None:
            Colors.load()
        return Colors._COLORS.get(name, '')
    
    @staticmethod
    def as_list():
        if Colors._COLORS is None:
            Colors.load()
        colors = [Colors._COLORS[s]
                    for s in sorted(Colors._COLORS.keys())
                    if s.startswith('COLOR')]
        return colors
        
    @staticmethod
    def format(s, *prefixes):
        if Colors.ENABLED:
            return ''.join(prefixes) + s + Colors.RESET
        return s


BRAILLE_KERNEL = np.array([
    [  1,   8],
    [  2,  16],
    [  4,  32],
    [ 64, 128]
])

KERNEL41 = np.array([
    [ 1],
    [ 2],
    [ 4],
    [ 8],
])



class TPlot(object):
    
    def __init__(self, columns, lines, logx=False, logy=False, padding=10, connect_points=False):
        self.columns = columns - padding - 5
        self.lines = lines - 5
        self.padding = padding
        self.datasets = []
        
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111)

        self.logx = logx
        if logx:
            self.ax.set_xscale('log')
        
        self.logy = logy
        if logy:
            self.ax.set_yscale('log')
        
        self._xticks = None
        self._grid = False
        self.connect = connect_points
        
        self._colors  = cycle(Colors.as_list())
        self._markers = cycle('ox+.')

        self.set_xtick_format('%9.2e')
        self.set_ytick_format('%8d')

    
    def plot(self, x, y=None, color=None, marker=None, label=None, fill=False, connect=False):
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
        
        if marker=='?':
            marker = next(self._markers)
        
        self.ax.plot(x, y, (marker + '-') if (self.connect or connect) else marker, label=label)

    def hist(self, data, bins=10, range=None, label=None):
        hist, bin_edges = np.histogram(data, bins=bins, range=range)
        nonzero = hist > 0
        x = bin_edges[:-1] + bin_edges[1:]
        self.plot(0.5*x[nonzero], hist[nonzero], label=label, fill=True)
        self.set_xticks(bin_edges)

        x = np.percentile(data, [25, 50, 75])
        y = np.ones_like(x) * hist[nonzero].max()/2
        # y = np.zeros_like(x)
        self.plot(x, y,
                marker='?',
                label='percentiles',
                color=self.datasets[-1][2],
                connect=True)
        # self.plot(x, y, marker='┼', label='percentiles')


    def show_grid(self, show):
        self.ax.grid(show)
        self._grid = show
    
    def set_xlim(self, xmin, xmax):
        self.ax.set_xlim(xmin, xmax)


    def set_ylim(self, ymin, ymax):
        self.ax.set_ylim(ymin, ymax)        
    

    def transform(self, x, y, kernel=None):

        xy = np.c_[x,y]
        if self.logx:
            xy[:,0] = np.log10(xy[:,0])
        if self.logy:
            xy[:,1] = np.log10(xy[:,1])
        
        # transform to axes coordinates
        mapped = self.ax.transLimits.transform(xy)
        
        # keep only the ones within the canvas
        x_in_range,y_in_range = np.logical_and(mapped>=0.0, mapped<=1.0).T
        idx, = np.nonzero(x_in_range & y_in_range)
        mapped = mapped[idx]

        # pixelate the results
        if kernel is None:
            mapped = np.round(mapped * [self.columns-1,self.lines-1])
        else:
            L,C = kernel.shape
            mapped = np.round(mapped * [C*(self.columns-1), L*(self.lines-1)])
        
        # keep the unique pairs
        if mapped.size:
          mapped = np.unique(mapped, axis=0)

        return mapped.astype(int), idx
    

    def get_canvas(self):
        
        ylim = self.ax.get_ylim()
        self.ax.set_ylim(reversed(ylim))
        
        canvas = [(self.columns)*[' '] for _ in range(self.lines)]
        headers = []

        # add grid
        # -----------------------------
        color = Colors.get('GRID')

        xticks = self.get_xticks()
        yticks = self.get_yticks()
        if self._grid:
            # vertical lines
            s = Colors.format('│', color)
            for i,_ in xticks:
                for line in canvas:
                    line[i] = s
            
            # horizontal lines
            s = Colors.format('─', color)
            for i,_ in yticks:
                canvas[i] = (self.columns)*[s]
            
            # intersection points
            s = Colors.format('┼', color)
            for i,_ in xticks:
                for j,_ in yticks:
                    canvas[j][i] = s
        
        # add curves
        # -----------------------------
        for x,y,c,m,l,fill in self.datasets:
            
            # ADD LINES
            # ---------
            if self.connect:
                kernel = BRAILLE_KERNEL
                L,C = kernel.shape
                xi = np.linspace(0.0, 1.0, C*self.columns)

                yi = np.ones_like(xi) * 0.5
                pts = np.c_[xi,yi]
                xi = self.ax.transLimits.inverted().transform(pts)[:,0]
                xi = xi[np.logical_and(xi>=x.min(), xi<=x.max())]
                yi = np.interp(xi, x, y)

                mapped,_ = self.transform(xi, yi, kernel=kernel)

                pixels = np.zeros((L*(self.lines), C*(self.columns)), dtype=int)
                i,j = mapped.T
                pixels[j,i] = 1
                for i in range(0, C*self.columns, C):
                    for j in range(0, L*self.lines, L):
                        mat = pixels[j:j+L, i:i+C]
                        if np.any(mat):
                            canvas[j//L][i//C] = Colors.format(to_braille(mat, kernel), c)
            
            
            # ADD POINTS
            # ----------
            kernel = KERNEL41
            L,C = kernel.shape
            mapped,_ = self.transform(x, y, kernel=kernel)
            
            if m == '?':
                header = self.columns * [' ']
                ij = (mapped//[C,L])[:,0]
                s = Colors.format('━', c)
                for k in range(ij.min(), ij.max()):
                    header[k] = s
                for k in ij:
                    header[k] = Colors.format('╋', c)
                headers.append(header)
                continue
            
            marker = Colors.format('█' if fill else m, c)
            for i,j in mapped//[C,L]:
                if fill:
                    for line in canvas[j:]:
                        line[i] = marker
                else:
                    canvas[j][i] = marker

        # add legends
        # -----------------------------
        for n,dataset in enumerate(self.datasets):
            _,_,color,marker,label,_ = dataset
            if self.connect:
                label = '%s -%s-' % (label,marker)
            else:
                label = '%s %s' % (label,marker)
            k = len(label)
            legend = Colors.format(label, color, Colors.UNDERLINE)
            canvas[n+1] = canvas[n+1][:-k-4] + [' ', legend]
        
        # add frame
        # -----------------------------
        padding = self.padding*' '
        for n,line in enumerate(canvas):
            line.insert(0, padding + '┃')
        canvas.append(len(line)*['━'])

        for n,line in enumerate(headers):
            line.insert(0, padding + ' ')

        # add y-ticks
        # -----------------------------
        if yticks:
            fmts = '%%%ss ┨' % (self.padding-1)
            fmtn = self._ytick_fmt
            #fmt = '%%%d.2e ┨' % (self.padding-1)
            for i,label in yticks:
                
                canvas[i][0] = fmts % (fmtn%label)
                #canvas[i][0] = fmt%label
         
        # add x-ticks
        # -----------------------------
        if xticks:
            fmtn = self._xtick_fmt
            fmts = '%%-%ss' % (xticks[1][0]-xticks[0][0])
            #fmt = '%%-%d.2e'%(xticks[1][0]-xticks[0][0])
            labels = ''
            for i,label in xticks:
                canvas[-1][i] = '┯'
                # labels += fmt%label
                labels += fmts % (fmtn%label)

            canvas[-1].insert(0, padding + '┗')
            canvas.append([padding[:-3] + xticks[0][0]*' ' + labels.rstrip()])
        
        # reset y-limits        
        self.ax.set_ylim(ylim)
        
        return headers + canvas
        
    def set_xtick_format(self, fmt):
        self._xtick_fmt = fmt
    
    def set_ytick_format(self, fmt):
        self._ytick_fmt = fmt

    def set_xticks(self, ticks):
        self._xticks = ticks

    def get_xticks(self):
        if self._xticks is None:
            ticks = self.ax.get_xticks()
        else:
            ticks = self._xticks
        
        # find center y-coordinate
        yc = 0.5*np.sum(self.ax.get_ylim())
        pos, idx = self.transform(ticks, yc*np.ones_like(ticks))
        
        return list(zip(pos[:,0], ticks[idx]))

    def get_yticks(self):
        
        # Problems with Log10Transform in earlier versions of MPL.
        # > Works with 2.2.3 and above

        # reverse ordering due to differences in axes origin
        # between MPL and Tplot
        ticks = self.ax.get_yticks()[::-1]

        # get the center x-coordinate
        xc = 0.5*np.sum(self.ax.get_xlim())
        pos, idx = self.transform(xc*np.ones_like(ticks), ticks)
        
        return list(zip(pos[:,1], ticks[idx]))


    def __str__(self):
        canvas = self.get_canvas()
        return '\n' + '\n'.join((''.join(line) for line in canvas)) + '\n'

    def __repr__(self):
        return str(self)
    
    def show(self):
        print(self)

    def close(self):
        plt.close(self.fig)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
                logx=args.logx,
                logy=args.logy,
                padding=args.padding,
                connect_points=args.lines
    )

    # configure output
    plot.show_grid(args.grid)
    Colors.ENABLED = args.no_color

    # if no type is provided, simply plot all 
    # columns as series
    if not (args.c or args.xy or args.hist):
        for n,row in enumerate(data):
            plot.plot(row, label='col-%d'%n)

    # add series
    for col,l in args.c:
        if l is None:
            l = 'col-%d'%col
        plot.plot(data[col], label=l)
    
    # add histograms
    for col,l in args.hist:
        if l is None:
            l = 'hist-%d'%col
        limits = args.ax[-1] if args.ax else None
        plot.hist(data[col], bins=args.bins, range=limits, label=l)
        #limits = args.ax[-1] if args.ax else None
        #hist, bin_edges = np.histogram(data[col], bins=args.bins, range=limits)
        #nonzero = hist > 0
        #x = bin_edges[:-1] + bin_edges[1:]
        #plot.plot(0.5*x[nonzero], hist[nonzero], label=l, fill=True)
        #plot.set_xticks(bin_edges)
    
    # add scatter plots
    for i,j,l in args.xy:
        if l is None:
            l = '%d-vs-%d'%(j,i)
        plot.plot(data[i], data[j], label=l)
    
    # finally set axis limits
    if args.ax:
        plot.set_xlim(*args.ax[-1])
    if args.ay:
        plot.set_ylim(*args.ay[-1])
    
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

    # parser: output configuration
    # ------------------------------------------- 
    group = parser.add_argument_group('Output configuration')
    group.add_argument('--width', type=int,
        metavar='W', help='output width', default=tsize.columns)
    group.add_argument('--height', type=int,
        metavar='H', help='output height', default=tsize.lines)
    group.add_argument('--padding', type=int,
        metavar='P', help='left padding', default=10)
    group.add_argument('--mpl', action='store_true', help='show plot in matplotlib window')
    group.add_argument('--no-color', action='store_false', help='suppress colored output')


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

