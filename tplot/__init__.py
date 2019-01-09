# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

__version__ = '0.1.1'

class Colors:

    _LYELLOW = '\033[33m'
    _LGRAY = '\033[2m'
    _PURPLE = '\033[95m'
    _BLUE = '\033[94m'
    _GREEN = '\033[92m'
    _RED = '\033[91m'
    _ENDC = '\033[0m'
    _BOLD = '\033[1m'
    _UNDERLINE = '\033[4m'
    
    def __init__(self, use_colors=True):
        self.use_colors = use_colors

    def __getattr__(self, attr):
        if self.use_colors:
            return getattr(self, '_'+attr)
        return ''
    
# KERNEL42 = np.array([
#     [  1,   8, 0, 0, 0],
#     [  2,  16, 0, 0, 0],
#     [  4,  32, 0, 0, 0],
#     [ 64, 128, 0, 0, 0]
#     ,[  0,   0, 0, 0, 0]
#     ,[  0,   0, 0, 0, 0]
#     ,[  0,   0, 0, 0, 0]
#     ,[  0,   0, 0, 0, 0]
# ])

KERNEL42 = np.array([
    [  1,   8],
    [  2,  16],
    [  4,  32],
    [ 64, 128]
])

KERNEL22 = np.array([
    [  1,   2],
    [  4,   8]
])

KERNEL21 = np.array([
    [ 1],
    [ 2]
])


SQUARE_MAP = {
    0: u'',
    1: u'\u2598',
    2: u'\u259D',
    3: u'\u2580',
    4: u'\u2596',
    5: u'\u258C',
    6: u'\u259E',
    7: u'\u259B',
    8: u'\u2597',
    9: u'\u259A',
   10: u'\u2590',
   11: u'\u259C',
   12: u'\u2584',
   13: u'\u2599',
   14: u'\u259F',
   15: u'\u2588',
}

KERNEL = KERNEL42

def to_dots(m):
    # 10240 = int('2800', 16)
    # return chr((KERNEL*m).sum()+10240)
    # print (unichr((KERNEL*m).sum()+10240))

    return unichr((KERNEL*m).sum()+10240).encode('utf-8')
    # return SQUARE_MAP[(KERNEL*m).sum()].encode('utf-8')


class TPlot(object):
    
    def __init__(self, columns, lines, logx=False, logy=False, padding=10, use_colors=True):
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
        
        colors = Colors(use_colors)
        self._colors = cycle([colors.BLUE, colors.GREEN, colors.RED, colors.PURPLE])
        self._endc = colors.ENDC
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
    

    def transform(self, x, y, kernel=None):

        if self.logx:
            x = np.log10(x[x>0])
        if self.logy:
            y = np.log10(y[y>0])
        
        if (self.logx and x.size==0) or (self.logy and y.size==0):
            raise Exception('Data has no positive values, and therefore cannot be log-scaled')
            
        mapped = self.ax.transLimits.transform(list(zip(x,y)))
        x_in_range,y_in_range = np.logical_and(mapped>=0.0, mapped<=1.0).T
        idx, = np.nonzero(x_in_range & y_in_range)
        
        mapped = mapped[idx]
        if kernel is None:
            mapped = np.round(mapped * [self.columns-1,self.lines-1])
        else:
            L,C = kernel.shape
            mapped = np.round(mapped * [C*(self.columns-1), L*(self.lines-1)])
            # mapped = np.round(mapped * [C*self.columns-1, L*self.lines-1])
        
        if mapped.size:
           mapped = np.unique(mapped, axis=0)

        return mapped.astype(int), idx
    
    def get_canvas(self):
        
        ylim = self.ax.get_ylim()
        self.ax.set_ylim(reversed(ylim))
        
        canvas = [(self.columns)*[' '] for _ in range(self.lines)]
        
        # add grid
        # -----------------------------
        colors = Colors(True)
        c = colors.LGRAY

        xticks = self.get_xticks()
        yticks = self.get_yticks()
        
        for i,_ in xticks:
            for line in canvas:
                line[i] = c+'│'+self._endc
        for i,_ in yticks:
            canvas[i] = (self.columns)*[c+'─'+self._endc]
        for i,_ in xticks:
            for j,_ in yticks:
                canvas[j][i] = c+'┼'+self._endc
        
        # add curves
        # -----------------------------
        for x,y,c,m,l,fill in self.datasets:
            L,C = KERNEL42.shape
            xi = np.linspace(0.0, 1.0, C*self.columns)
            yi = np.ones_like(xi) * 0.5
            pts = np.c_[xi,yi]
            xi = self.ax.transLimits.inverted().transform(pts)[:,0]
            yi = np.interp(xi, x, y)
            mapped,_ = self.transform(xi, yi, kernel=KERNEL42)

            L,C = KERNEL42.shape
            tmp = np.zeros((L*(self.lines), C*(self.columns)), dtype=int)
            i,j = mapped.T
            tmp[j,i] = 1
            print(tmp.shape, (self.lines, self.columns))
            for i in range(0, C*self.columns, C):
                for j in range(0, L*self.lines, L):
                    # mat = tmp[j+1:j+L+1, i+1:i+C+1]
                    mat = tmp[j:j+L, i:i+C]
                    if np.any(mat):
                        canvas[j//L][i//C] = c + to_dots(mat) + self._endc
                        # print(i//C,j//L)

                        # for line in canvas[j//L+1:]:
                        #     line[i//C] = '█'
            
            
            mapped,_ = self.transform(x, y, kernel=KERNEL21)
            L,C = KERNEL21.shape
            tmp = np.zeros((L*(self.lines), C*(self.columns)), dtype=int)
            i,j = mapped.T
            tmp[j,i] = 1

            marker = c + ('█' if fill else m) + self._endc
            
            for i in range(0, C*self.columns, C):
                for j in range(0, L*self.lines, L):
                    mat = tmp[j:j+L, i:i+C]
                    if np.any(mat):
                        canvas[j//L][i//C] = marker

                        if fill:
                            for line in canvas[j//L+1:]:
                                line[i//C] = marker

            # mapped,_ = self.transform(x, y, pixelate=False)
            # marker = c + ('█' if fill else m) + self._endc
            # for i,j in mapped:
            #     canvas[j][i] = marker
            #     if fill:
            #        for line in canvas[j:]:
            #            line[i] = marker
                

            #mapped,_ = self.transform(np.array([1]), np.array([1]), pixelate=False)
            # print('no-pixel:')
            # print(mapped)

        # add legends
        # -----------------------------
        for n,dataset in enumerate(self.datasets):
            _,_,color,marker,label,_ = dataset
            k = len(label)
            legend = color + label + ' ' + marker + self._endc
            canvas[n+1] = canvas[n+1][:-k-4] + [' ', legend]
        
        # add frame
        # -----------------------------
        padding = self.padding*' '
        for n,line in enumerate(canvas):
            line.insert(0, padding + '┃')
        canvas.append(len(line)*['━'])

        # add y-ticks
        # -----------------------------
        yticks = self.get_yticks()
        fmt = '%%%d.2e ┨' % (self.padding-1)
        for i,label in yticks:
            canvas[i][0] = fmt%label
         
        # add x-ticks
        # -----------------------------
        # xticks = self.get_xticks()
        fmt = '%%-%d.2e'%(xticks[1][0]-xticks[0][0])

        labels = ''
        for i,label in xticks:
            canvas[-1][i] = '┯'
            labels += fmt%label

        canvas[-1].insert(0, padding + '┗')
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
        ymin,ymax = sorted(self.ax.get_ylim())
        yc = max(ymin, ymax)-0.95*(ymax-ymin)
        # _,yc = self.ax.transLimits.inverted().transform((0.05,0.05))
        pos, idx = self.transform(ticks, yc*np.ones_like(ticks))
        return list(zip(pos[:,0], ticks[idx]))
    
    def get_yticks(self):
        ticks = self.ax.get_yticks()
        xmin,xmax = sorted(self.ax.get_xlim())
        xc = max(xmin, xmax)-0.95*(xmax-xmin)
        # xc,_ = self.ax.transLimits.inverted().transform((0.95,0.95))
        pos, idx = self.transform(xc*np.ones_like(ticks), ticks)
        return list(zip(pos[:,1], ticks[idx]))

    def __str__(self):
        canvas = self.get_canvas()
        # for line in canvas:
        #     for n,c in enumerate(line):
        #         if isinstance(c, unicode):
        #             line[n] = c.encode('utf-8')
        return '\n' + '\n'.join((''.join(line) for line in canvas)) + '\n'

    def __repr__(self):
        return str(self)


def run(args):

    data = np.loadtxt(args.file, skiprows=args.skip, delimiter=args.delimiter)
    if len(data.shape) == 1:
        data = data.reshape(1,-1)
    else:
        data = data.T
    
    plot = TPlot(args.width, args.height,
                logx=args.logx,
                logy=args.logy,
                padding=args.padding,
                use_colors=not args.no_color
    )
    
    if not (args.c or args.xy or args.hist):
        for n,row in enumerate(data):
            plot.plot(row, label='col-%d'%n)

    for col,l in args.c:
        if l is None:
            l = 'col-%d'%col
        plot.plot(data[col], label=l)
    
    for col,l in args.hist:
        if l is None:
            l = 'hist-%d'%col
        limits = args.ax[-1] if args.ax else None
        hist, bin_edges = np.histogram(data[col], bins=args.bins, range=limits)
        x = bin_edges[:-1] + bin_edges[1:]
        plot.plot(0.5*x, hist, label=l, fill=True)
        plot.set_xticks(bin_edges)
    
    for i,j,l in args.xy:
        if l is None:
            l = '%d-vs-%d'%(j,i)
        plot.plot(data[i], data[j], label=l)
    
    if args.ax:
        plot.set_xlim(*args.ax[-1])
    if args.ay:
        plot.set_ylim(*args.ay[-1])
    
    if args.mpl:
        plt.show()
    else:
        print(plot)


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
    print('TERMINAL SIZE =', tsize.lines, tsize.columns)

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

    # parser> Input source
    # ------------------------------------------- 
    parser.add_argument('-f', '--file', default=SRC, help='source file. Use "-" to read from stdin')

    # parser> plot arguments
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

    # parser> data parsing
    # ------------------------------------------- 
    group = parser.add_argument_group('Data parsing')
    group.add_argument('-d', '--delimiter', type=str,
        metavar='D', help='delimiter')
    group.add_argument('-s', '--skip', type=int, default=0,
        metavar='N', help='skip first N rows')

    # parser> axes configuration
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

    # parser> output configuration
    # ------------------------------------------- 
    group = parser.add_argument_group('Output configuration')
    group.add_argument('--width', type=int,
        metavar='W', help='output width', default=tsize.columns)
    group.add_argument('--height', type=int,
        metavar='H', help='output height', default=tsize.lines)
    group.add_argument('--padding', type=int,
        metavar='P', help='left padding', default=10)
    group.add_argument('--mpl', action='store_true', help='show plot in matplotlib window')
    group.add_argument('--no-color', action='store_true', help='suppress colored output')


    # parser> run parser
    # ------------------------------------------- 
    args = parser.parse_args()

    if args.file is None:
        print('Error: Missing "file" (-f) argument.')
        exit(1)
    elif args.file == '-':
        args.file = sys.stdin

    run(args)

