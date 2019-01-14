# -*- coding: utf-8 -*-
from __future__ import print_function

"""
TPlot.py

A Python package for creating and displaying matplotlib plots in the console/terminal
"""

__license__ = "MIT"
__version__ = '0.4.0'
__author__ = 'Sérgio Miguel Santos'
__copyright__ = "Copyright 2019, Sérgio Miguel Santos, Univ. Aveiro - Portugal"

from itertools import cycle
import sys
import os

IS_PY_VERSION_3 = sys.version_info[0] == 3
MPL_DISABLED = 'TPLOT_NOGUI' in os.environ

if MPL_DISABLED or True:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


__all__ = ['Colors', 'TPlot', 'main', 'run']


if IS_PY_VERSION_3:
    # def to_braille(m, kernel):
    #     # 10240 = int('2800', 16)
    #     return chr((kernel*m).sum()+10240)
    def to_braille(m):
        # 10240 = int('2800', 16)
        return chr(m+10240)
else:
    # def to_braille(m, kernel):
    #     return unichr((kernel*m).sum()+10240).encode('utf-8')
    def to_braille(m):
        return unichr(m+10240).encode('utf-8')

try:
    from scipy.signal import convolve2d
except ImportError as ex:
    def convolve2d(mat, kernel, **kwargs):
        s = kernel.shape + tuple(np.subtract(mat.shape, kernel.shape) + 1)
        strd = np.lib.stride_tricks.as_strided
        subM = strd(mat, shape = s, strides = mat.strides * 2)
        return np.einsum('ij,ijkl->kl', kernel, subM)


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
                    # color = color.decode('unicode_escape')
                else:
                    # color = color.decode('string_escape')
                    color = color.decode('unicode_escape')
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


class TPlotType(object):
    LINE = 1
    BAR = 2

class Format:
    NONE   = 0
    LEFT   = 1
    TOP    = 2
    RIGHT  = 4
    BOTTOM = 8
    ALL    = 15
    TOP_LEFT = 3
    TOP_RIGHT = 6
    BOTTOM_LEFT = 9
    BOTTOM_RIGHT = 12


def get_unicode_array(size, fill=u''):
    ar = np.empty(size, dtype='U32')
    ar[:] = fill
    return ar


class TPlot(object):
    
    def __init__(self, columns, lines,
                logx=False, logy=False,
                borders=Format.BOTTOM_LEFT,
                tick_position=Format.BOTTOM_LEFT,
                xtick_format='%r',
                ytick_format='%r',
                padding=0):

        self.size = (lines, columns)
        # self._columns = columns# - padding - 5
        # self._lines = lines# - 5
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
        
        self._colors  = cycle(Colors.as_list())
        self._markers = cycle('ox+.')

        self.set_tick_position(tick_position)
        self.set_xtick_format(xtick_format)
        self.set_ytick_format(ytick_format)
        self.set_padding(padding)
        self.set_border(borders)

    def set_tick_position(self, position):
        self._tick_position = position


    def set_padding(self, *padding):
        count = len(padding)
        if count == 1:
            self.padding = 4*padding
        elif count == 2:
            self.padding = 2*padding
        elif count == 4:
            self.padding = padding
        else:
            raise ValueError('invalid number of arguments: expected 1, 2 or 4 and found %d'%count)
        
    def _get_dataset(self, x, y, **kwargs):
        if y is None:
            y = x
            x = np.arange(len(y))
        
        dataset = dict(
            x=x,
            y=y,
        )
        
        if ('color' in kwargs) and (kwargs['color'] is None):
            kwargs['color'] = next(self._colors)
        
        if ('marker' in kwargs) and (kwargs['marker'] is None):
            kwargs['marker'] = next(self._markers)
        
        dataset.update(kwargs)
        return dataset

    def line(self, x, y=None, color=None, marker=None, label=None, connect=False):
        dataset = self._get_dataset(
            x, y,
            fill=False,
            label=label,
            color=color,
            marker=marker,
            connect=connect,
            #type=TPlotType.LINE
        )
        self.datasets.append(dataset)
        
        marker = dataset['marker']
        if connect:
            marker += '-'
        
        self.ax.plot(
            dataset['x'],
            dataset['y'],
            dataset['marker'],
            label=label)
    
    def bar(self, x, y=None, label=None, color=None, fill=True, marker = u'█'):
        dataset = self._get_dataset(
            x, y,
            color=color,
            label=label,
            fill = fill,
            marker = u'█',
            # percentile = None,
            # type=TPlotType.BAR
        )
        self.datasets.append(dataset)

        self.ax.bar(dataset['x'], dataset['y'], label=label)

        return dataset

    def hist(self, y, bins=10,
            range=None, label=None, add_percentile=True,
            marker = u'█', color=None):
        hist, bin_edges = np.histogram(y, bins=bins, range=range)
        x = 0.5*(bin_edges[1:] + bin_edges[:-1])
        #nonzero = hist > 0
        
        dataset = self.bar(x, hist, label=label, color=color, fill=True, marker=marker)
        # dataset = self._get_dataset(
        #     x, hist,
        #     color=None,
        #     label=label,
        #     fill = True,
        #     marker = u'█',
        #     percentile = None,
        #     #type=TPlotType.BAR
        # )

        if add_percentile:
            dataset['percentile'] = np.percentile(y, [25, 50, 75])
        
        # self.datasets.append(dataset)
        
        # self.ax.bar(x, hist, label=label)
        # self.set_xticks(bin_edges)
        
    def show_grid(self, show=True):
        self.ax.grid(show)
        self._grid = show
    
    def _set_xlim(self, xlim):
        self.ax.set_xlim(xlim)
    def _get_xlim(self):
        return self.ax.get_xlim()
    xlim = property(_get_xlim, _set_xlim)

    def _set_ylim(self, ylim):
        self.ax.set_ylim(ylim)        
    def _get_ylim(self):
        return self.ax.get_ylim()
    ylim = property(_get_ylim, _set_ylim)


    def transform(self, x, y, interpolation_ratio=None, unique=True):

        if self.logx:
            x = np.log10(x)
        if self.logy:
            y = np.log10(y)
        xy = np.c_[x,y]
        
        # transform to axes coordinates
        mapped = self.ax.transLimits.transform(xy)
        
        # keep only the ones within the canvas
        x_in_range,y_in_range = np.logical_and(mapped>=0.0, mapped<=1.0).T
        idx, = np.nonzero(x_in_range & y_in_range)
        mapped = mapped[idx]

        # pixelate the results
        #if kernel is None:
        #    mapped = np.round(mapped * [self._columns-1,self._lines-1])
        #else:
        #    L,C = kernel.shape
        if interpolation_ratio is None:
            interpolation_ratio = (1,1)

        L,C = interpolation_ratio
        mapped = np.round(mapped * [C*(self._columns-1), L*(self._lines-1)])
        
        mapped = mapped.astype(int)
        
        # keep the unique pairs
        if unique and mapped.size:
         mapped = np.unique(mapped, axis=0)

        return mapped, idx
    

    def _get_figure(self):
        
        pl,pt,pr,pb = self.padding

        ylim_min, ylim_max = sorted(self.ylim)
        
        # get candidate tick labels
        y_labels = self.ax.get_yticks()
        # extract the ones that fit inside the y-limits
        y_labels = y_labels[(y_labels>=ylim_min) & (y_labels<=ylim_max)]
        # stringify the labels
        y_labels =  [(self._ytick_fmt%l).strip() for l in y_labels]
        
        # -1 line for spacing between label and tick
        #columns = self._columns - max(map(len, y_labels)) - pl - pr - 1
        
        # -1 line for x-labels
        #lines = self._lines - pt - pb - 1
        l,c = self.size
        lines = l - pt - pb# - 1 # minus one for prompt
        columns = c - pl - pr
        if (self._tick_position & Format.TOP) or (self._tick_position & Format.BOTTOM):
            lines -= 1
        
        if (self._tick_position & Format.LEFT) or (self._tick_position & Format.RIGHT):
            # get candidate tick labels
            y_labels = self.ax.get_yticks()
            # extract the ones that fit inside the y-limits
            y_labels = y_labels[(y_labels>=ylim_min) & (y_labels<=ylim_max)]
            # stringify the labels
            y_labels =  [(self._ytick_fmt%l).strip() for l in y_labels]
            columns -= max(map(len, y_labels)) + 1
        

        figure = get_unicode_array((lines,columns), u' ')
        
        lf = cf = 0
        lt = ct = None
        
        if self._borders & Format.TOP:
            figure[0,:] = u'━'
            lf = 1
        if self._borders & Format.BOTTOM:
            figure[-1,:] = u'━'
            lt = -1
        if self._borders & Format.LEFT:
            figure[:,0] = u'┃'
            cf = 1
        if self._borders & Format.RIGHT:
            figure[:,-1] = u'┃'
            ct = -1
        
        if (self._borders & Format.TOP) and (self._borders & Format.LEFT):
            figure[0,0] = u'┏'
        if (self._borders & Format.TOP) and (self._borders & Format.RIGHT):
            figure[0,-1] = u'┓'
        if (self._borders & Format.BOTTOM) and (self._borders & Format.LEFT):
            figure[-1,0] = u'┗'
        if (self._borders & Format.BOTTOM) and (self._borders & Format.RIGHT):
            figure[-1,-1] = u'┛'

        canvas = figure[lf:lt, cf:ct]
        print('FIGURE =', figure.shape)
        print('CANVAS =', canvas.shape)
        return figure, canvas


    def set_border(self, borders=Format.ALL):
        self._borders = borders

    def __str__(self):
        
        '''
              padding-top
        p   xxxxxxxxxxxxxxx p
        a y x             x a
        d l x             x d
          a x             x
        l b x             x r
        e e x             x i
        f l x             x g
        t   xxxxxxxxxxxxxxx h
                x-labels    t
             padding-bottom
        '''
        #self.set_padding(0)
        # self.set_border(Format.NONE|Format.BOTTOM|Format.LEFT)
        # self.set_border(Format.ALL)#BOTTOM_LEFT)
        # self.set_tick_position(Format.BOTTOM_LEFT)

        ylim = self.ylim
        self.ylim = reversed(ylim)
        
        # GET FIGURE AND CANVAS
        # -----------------------------
        figure, canvas = self._get_figure()
        canvas_lines, canvas_columns = canvas.shape
        lines, columns = figure.shape

        self._columns = canvas_columns
        self._lines = canvas_lines

        # canvas = figure[1:-1,1:-1]

        # add grid
        # -----------------------------
        color = Colors.get('GRID')

        xpos,xlabels = self.get_xticks()
        ypos,ylabels = self.get_yticks()
        
        if self._grid:
            
            # horizontal thin lines
            canvas[ypos,:] = Colors.format(u'─', color)
            
            # vertical thin lines
            canvas[:,xpos] = Colors.format(u'│', color)
            
            # intersection this crosses
            xmarker = Colors.format(u'┼', color)
            for i in ypos:
                canvas[i,xpos] = xmarker
        
        # add tick labels
        # -----------------------------

        pl,pt,pr,pb = self.padding

        yticks_left = 1
        xticks_top = 0

        labels = [(self._ytick_fmt%l).strip() for l in ylabels]
        if self._borders & Format.TOP:
           ypos += 1 
        
        w = 0
        if self._tick_position & Format.LEFT:
            if self._borders & Format.LEFT:
                figure[ypos, 0] = u'┨'
            w = max(map(len, labels)) +  pl
            fmt = '%%%ds '%w
            lmargin = get_unicode_array(lines, u' '*(w+1))
            lmargin[ypos] = [fmt%l for l in labels]

            rmargin = get_unicode_array(lines)
        elif self._tick_position & Format.RIGHT:
            if self._borders & Format.RIGHT:
                figure[ypos, -1] = u'┠'
            w = pl
            lmargin = get_unicode_array(lines, u' '*pl)
            rmargin = get_unicode_array(lines)
            rmargin[ypos] = [' %s'%l for l in labels]
        else:
            lmargin = rmargin = get_unicode_array(lines)
            
            
        headers = []
        footers = []

        

        
        
        if self._borders & Format.LEFT:
           xpos += 1
        # fmts = ['%%-%d.2f'%n for n in np.diff(xpos)] + ['%-.2f']
        labels = [(self._xtick_fmt%l).strip() for l in xlabels]
        fmts = ['%%-%ds'%n for n in np.diff(xpos)] + ['%s']
        footer = [fmt%l for fmt,l in zip(fmts, labels)]
        footer.insert(0, (w+xpos[0]-2)* ' ')


        if self._tick_position & Format.BOTTOM:
            if self._borders & Format.BOTTOM:
                figure[-1,xpos] = u'┯'
            footers.append(footer)
        elif self._tick_position & Format.TOP:
            if self._borders & Format.TOP:
                figure[ 0,xpos] = u'┷'
            headers.append(footer)
        
        self._add_curves(canvas)

        
        figure = figure.tolist()
        lmargin = lmargin.tolist()
        rmargin = rmargin.tolist()

        self._add_legends(figure)

        headers.extend((pt-len(headers))*[[]])
        footers.extend((pb-len(footers))*[[]])

        output = headers + [[r]+line+[l] for r,line,l in zip(lmargin,figure,rmargin)] + footers
        # output = headers + np.c_[lmargin, figure, rmargin].tolist() + footers
        s = '\n'.join((''.join(line) for line in output))
        
        # reset y-limits        
        self.ax.set_ylim(ylim)


        if not IS_PY_VERSION_3:
            s = s.encode('utf-8')
        return s
    

    def _add_legends(self, figure):
        datasets = [d for d in self.datasets if d.get('label',None)]
        for n,dataset in enumerate(datasets):
            label = '{label} {marker}'.format(**dataset)
            k = len(label)
            label = Colors.format(label, dataset['color'], Colors.UNDERLINE)
            line = figure[n+2]
            figure[n+2] = line[:-k-4] + [label] + line[-4:]
    
    def _add_curves(self, canvas):
        for ds in self.datasets:
            
            x = ds['x']
            y = ds['y']
            color = ds.get('color', '')

            if ds.get('connect', False):
                
                L,C = BRAILLE_KERNEL.shape
                xi = np.linspace(0.0, 1.0, 10*C*self._columns)
                yi = np.ones_like(xi) * 0.5
                pts = np.c_[xi,yi]
                xi = self.ax.transLimits.inverted().transform(pts)[:,0]
                xi = xi[np.logical_and(xi>=x.min(), xi<=x.max())]
                yi = np.interp(xi, x, y)

                mapped,_ = self.transform(xi, yi, BRAILLE_KERNEL.shape)
                
                pixels = np.zeros((L*(self._lines), C*(self._columns)), dtype=int)
                i,j = mapped.T
                pixels[j,i] = 1
                tmp = convolve2d(pixels, BRAILLE_KERNEL, mode='valid')[::L,::C]
                
                #f = np.vectorize(lambda n,c: Colors.format(to_braille(n), c))
                #i,j=np.nonzero(tmp)
                #canvas[i,j] = f(tmp[i,j], c)
                
                for i,j in zip(*np.nonzero(tmp)):
                   canvas[i,j] = Colors.format(to_braille(tmp[i,j]), color)

            L,C = KERNEL41.shape
            mapped,_ = self.transform(x, y, KERNEL41.shape)
            mapped = mapped//[C,L]
            
            if ds.get('fill', False):
                zero,_ = self.transform(x[0], [0])
                _,k = zero[0]
                for i,j in mapped:
                    k = zero[0,1]
                    if zero[0,1] > j:
                     k += 1
                    j,k = sorted((j,k))
                    canvas[j:k,i] = Colors.format(ds['marker'], color)
            
            if ds.get('marker', None) is not None:
                i,j = mapped.T
                canvas[j,i] = Colors.format(ds['marker'], color)
            
            if 'percentile' in ds:
                xp = ds['percentile']
                yp = min(self.ylim) * np.ones_like(xp)
                mapped,_ = self.transform(xp, yp)
                i = mapped[:,0]
                canvas[0,i.min():i.max()] = Colors.format(u'━', color)
                canvas[0,mapped[:,0]] = Colors.format(u'╋', color)



        
        
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
        pos, idx = self.transform(ticks, yc*np.ones_like(ticks), unique=False)
        
        # return list(zip(pos[:,0], ticks[idx]))
        return pos[:,0], ticks[idx]

    def get_yticks(self):
        
        # Problems with Log10Transform in earlier versions of MPL.
        # > Works with 2.2.3 and above

        # reverse ordering due to differences in axes origin
        # between MPL and Tplot
        ticks = self.ax.get_yticks()[::-1]

        # get the center x-coordinate
        xc = 0.5*np.sum(self.ax.get_xlim())
        pos, idx = self.transform(xc*np.ones_like(ticks), ticks, unique=False)
        
        # return list(zip(pos[:,1], ticks[idx]))
        return pos[:,1], ticks[idx]


    # def __str__(self):
    #     s = self.as_string()
    #     if not IS_PY_VERSION_3:
    #         s = s.encode('utf-8')
    #     return s

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
                # connect_points=args.lines
    )

    # configure output
    
    plot.show_grid(args.grid)
    plot.set_padding(*args.padding)
    
    plot.set_tick_position(getattr(Format, args.labels))
    plot.set_border(getattr(Format, args.border))
    plot.set_xtick_format(args.x_fmt)
    plot.set_ytick_format(args.y_fmt)

    Colors.ENABLED = args.no_color

    # if no type is provided, simply plot all 
    # columns as series
    if not (args.c or args.xy or args.hist):
        for n,row in enumerate(data):
            plot.bar(row, label='col-%d'%n)

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

