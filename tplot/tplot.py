# -*- coding: utf-8 -*-

from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

from .utils import get_output_size
from . import IS_PYTHON3
from .ansi import Ansi

__all__ = ['Format', 'TPlot']


def block_multiply(mat, kernel):
    # adapted from https://stackoverflow.com/a/43087771
    shape = kernel.shape + tuple(np.subtract(mat.shape, kernel.shape) + 1)
    sub_mat = as_strided(mat, shape=shape, strides=mat.strides*2)
    out = np.einsum('ij,ijkl->kl', kernel, sub_mat)
    kr,kc = kernel.shape
    return out[::kr,::kc]


if not IS_PYTHON3:
    chr = unichr


def to_braille(m):
    # 10240 = int('2800', 16)
    return chr(m+10240)#.encode('utf-8')



BRAILLE_KERNEL = np.array([
    [  1,   8],
    [  2,  16],
    [  4,  32],
    [ 64, 128]
])


KERNEL41 = np.array([
    [ 1],
    [ 1],
    [ 1],
    [ 1],
])


def get_unicode_array(size, fill=u''):
    ar = np.empty(size, dtype='U32')
    ar[:] = fill
    return ar


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


class TPlot(object):
    
    def __init__(self, size=None,
                tick_position=None,
                xtick_format=None,
                ytick_format=None,
                borders=None,
                padding=(0,)):

        self._datasets = []
        
        self._fig = plt.figure()
        self._ax  = self._fig.add_subplot(111)

        self._is_logx = False
        self._is_logy = False
        
        self._xticks = None
        self._grid = False
        self._title = None

        if size is None:
            size = get_output_size()
        self.set_size(*size)

        self.set_tick_position(tick_position)
        self.set_xtick_format(xtick_format)
        self.set_ytick_format(ytick_format)
        self.set_padding(*padding)
        self.set_border(borders)
        
        self.reset()

    def set_title(self, title):
        self._title = title
    
    def set_size(self, lines, columns):
        self._size = (lines, columns)
    
    def reset(self):
        self._colors  = cycle(Ansi.available_colors())
        self._markers = cycle(u'ox+.')
        return self
        
    def clear(self):
        self._datasets = []
        self._ax.clear()
        return self
        

    def set_yscale(self, scale):
        if not scale in ('linear', 'log'):
            raise ValueError('Only "linear" and "log" scales are supported')
        self._is_logy = scale=='log'
        self._ax.set_yscale(scale)
    
    def set_xscale(self, scale):
        if not scale in ('linear', 'log'):
            raise ValueError('Only "linear" and "log" scales are supported')
        self._is_logx = scale=='log'
        self._ax.set_xscale(scale)
    

    def set_tick_position(self, position):
        if position is None:
            position = Format.BOTTOM_LEFT
        self._tick_position = position


    def set_padding(self, *padding):
        '''
        Set spacing (number of spaces and lines) around the plot canvas.

        ## **inputs**

        The actual spacing around the canvas depends on the number
        of values passed to the method:
        
        * single value - `top = bottom = left = right = padding`
        * two values - `left = right = padding[0]`, `top = bottom = padding[1]`
        * fours values - `left,top,right,bottom = padding`
        
        '''
        
        count = len(padding)
        if count == 1:
            self._padding = 4*padding
        elif count == 2:
            self._padding = 2*padding
        elif count == 4:
            self._padding = padding
        else:
            raise ValueError('invalid number of arguments: expected 1, 2 or 4 but found %d'%count)
    
    def get_padding(self):
        return self._padding
    
    def _build_dataset(self, x, y, **kwargs):
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
        dataset = self._build_dataset(
            x, y,
            fill=False,
            label=label,
            color=color,
            marker=marker,
            connect=connect,
        )
        self._datasets.append(dataset)
        
        marker = dataset['marker']
        if connect:
            marker += '-'
        
        self._ax.plot(
            dataset['x'],
            dataset['y'],
            dataset['marker'],
            label=label)

        return dataset
    

    def bar(self, x, y=None, label=None, color=None, fill=True, marker = u'█'):
        
        dataset = self._build_dataset(
            x, y,
            fill=fill,
            color=color,
            label=label,
            marker = marker,
        )
        self._datasets.append(dataset)

        self._ax.bar(dataset['x'], dataset['y'], label=label)

        return dataset


    def hist(self, y, bins=10,
            range=None, label=None, add_percentile=True,
            marker = u'█', color=None):
        
        hist, edges = np.histogram(y, bins=bins, range=range)
        x = 0.5*(edges[1:] + edges[:-1])
        
        dataset = self.bar(x, hist, label=label, color=color, fill=True, marker=marker)

        if add_percentile:
            dataset['percentile'] = np.percentile(y, [25, 50, 75])
        
        return dataset

    def show_grid(self, show=True):
        self._ax.grid(show)
        self._grid = show
    
    def set_xlim(self, xlim):
        self._ax.set_xlim(xlim)
    
    def get_xlim(self):
        return self._ax.get_xlim()
    
    def set_ylim(self, ylim):
        self._ax.set_ylim(ylim)        
    
    def get_ylim(self):
        return self._ax.get_ylim()
    

    def transform(self, x, y, sub_sampling=None, unique=True):

        if self._is_logx:
            x = np.log10(x)
        if self._is_logy:
            y = np.log10(y)
        xy = np.c_[x,y]
        
        # transform to axes coordinates
        mapped = self._ax.transLimits.transform(xy)
        
        # keep only the ones inside the canvas
        x_in_range,y_in_range = np.logical_and(mapped>=0.0, mapped<=1.0).T
        idx, = np.nonzero(x_in_range & y_in_range)
        mapped = mapped[idx]

        if sub_sampling is None:
            sub_sampling = (1,1)

        L,C = sub_sampling
        mapped = np.round(mapped * [C*(self._columns-1), L*(self._lines-1)])
        
        mapped = mapped.astype(int)
        
        # keep the unique pairs
        if unique and mapped.size:
            mapped = np.unique(mapped, axis=0)
        
        return mapped, idx
    

    def _get_figure(self):
        
        pl,pt,pr,pb = self._padding

        ylim_min, ylim_max = sorted(self.get_ylim())
        
        # get candidate tick labels
        y_labels = self._ax.get_yticks()
        # extract the ones that fit inside the y-limits
        y_labels = y_labels[(y_labels>=ylim_min) & (y_labels<=ylim_max)]
        # stringify the labels
        y_labels =  [(self._ytick_fmt%l).strip() for l in y_labels]
        
        # -1 line for spacing between label and tick
        #columns = self._columns - max(map(len, y_labels)) - pl - pr - 1
        
        # -1 line for x-labels
        #lines = self._lines - pt - pb - 1
        l,c = self._size
        lines = l - pt - pb# - 1 # minus one for prompt
        columns = c - pl - pr
        if (self._tick_position & Format.TOP) or (self._tick_position & Format.BOTTOM):
            lines -= 1
        
        if (self._tick_position & Format.LEFT) or (self._tick_position & Format.RIGHT):
            # get candidate tick labels
            y_labels = self._ax.get_yticks()
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
        #print('FIGURE =', figure.shape)
        #print('CANVAS =', canvas.shape)
        return figure, canvas


    def set_border(self, borders=None):
        if borders is None:
            borders = Format.ALL
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
        
        ylim = self.get_ylim()
        self.set_ylim(reversed(ylim))
        
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
        color = Ansi.get('GRID')

        xpos,xlabels = self.get_xticks()
        ypos,ylabels = self.get_yticks()
        
        if self._grid:
            
            # horizontal thin lines
            canvas[ypos,:] = Ansi.format(u'─', color)
            
            # vertical thin lines
            canvas[:,xpos] = Ansi.format(u'│', color)
            
            # intersection this crosses
            xmarker = Ansi.format(u'┼', color)
            for i in ypos:
                canvas[i,xpos] = xmarker
        
        # add tick labels
        # -----------------------------

        pl,pt,pr,pb = self._padding

        yticks_left = 1
        xticks_top = 0

        labels = [(self._ytick_fmt%l).strip() for l in ylabels]
        if self._borders & Format.TOP:
           ypos += 1 
        
        w = pl
        if self._tick_position & Format.LEFT:
            if self._borders & Format.LEFT:
                figure[ypos, 0] = u'┨'
            # w = max(map(len, labels)) +  pl
            w += max(map(len, labels))
            fmt = '%%%ds '%w
            lmargin = get_unicode_array(lines, u' '*(w+1))
            lmargin[ypos] = [fmt%l for l in labels]

            rmargin = get_unicode_array(lines, u' '*pr)
        elif self._tick_position & Format.RIGHT:
            if self._borders & Format.RIGHT:
                figure[ypos, -1] = u'┠'
            # w = pl
            lmargin = get_unicode_array(lines, u' '*pl)
            rmargin = get_unicode_array(lines)
            rmargin[ypos] = [' %s'%l for l in labels]
        else:
            # w = pl
            lmargin = get_unicode_array(lines, u' '*pl)
            rmargin = get_unicode_array(lines, u' '*pr)
            
            
        headers = []
        footers = []

        if self._title is not None:
            headers.append([self._title.center(columns-1)])
        
        
        
        if self._borders & Format.LEFT:
            xpos += 1
            w += 1
        if self._tick_position & Format.LEFT:
            w += 1
        
        # fmts = ['%%-%d.2f'%n for n in np.diff(xpos)] + ['%-.2f']
        labels = [(self._xtick_fmt%l).strip() for l in xlabels]
        fmts = ['%%-%ds'%n for n in np.diff(xpos)] + ['%s']
        footer = [fmt%l for fmt,l in zip(fmts, labels)]
        # footer.insert(0, (w+xpos[0]-2)* ' ')
        footer.insert(0, (xpos[0]-2)* ' ')

        
        if self._tick_position & Format.BOTTOM:
            if self._borders & Format.BOTTOM:
                figure[-1,xpos] = u'┯'
            footers.append(footer)
        elif self._tick_position & Format.TOP:
            if self._borders & Format.TOP:
                figure[ 0,xpos] = u'┷'
            headers.append(footer)
        
        self._add_curves(canvas, headers, footers)

        
        
        for item in headers+footers:
           item.insert(0, w*' ')
        
        figure = figure.tolist()
        lmargin = lmargin.tolist()
        rmargin = rmargin.tolist()

        self._add_legends(figure)

        # headers.extend((pt-len(headers))*[[]])
        headers = (pt-len(headers))*[[]] + headers
        footers.extend((pb-len(footers))*[[]])

        output = headers + [[r]+line+[l] for r,line,l in zip(lmargin,figure,rmargin)] + footers
        # output = headers + np.c_[lmargin, figure, rmargin].tolist() + footers
        s = '\n'.join((''.join(line) for line in output))
        
        # reset y-limits        
        self._ax.set_ylim(ylim)


        if not IS_PYTHON3:
            s = s.encode('utf-8')
        return s + '\n'
    

    def _add_legends(self, figure):
        datasets = [d for d in self._datasets if d.get('label',None)]
        for n,dataset in enumerate(datasets):
            label = '{label} {marker}'.format(**dataset)
            k = len(label)
            label = Ansi.format(label, dataset['color'], Ansi.UNDERLINE)
            line = figure[n+2]
            figure[n+2] = line[:-k-4] + [label] + line[-4:]
    
    def _add_curves(self, canvas, headers, footers):
        for ds in self._datasets:
            
            x = ds['x']
            y = ds['y']
            color = ds.get('color', '')

            if ds.get('connect', False):
                
                L,C = BRAILLE_KERNEL.shape
                xi = np.linspace(0.0, 1.0, 10*C*self._columns)
                yi = np.ones_like(xi) * 0.5
                pts = np.c_[xi,yi]
                xi = self._ax.transLimits.inverted().transform(pts)[:,0]
                xi = xi[np.logical_and(xi>=x.min(), xi<=x.max())]
                yi = np.interp(xi, x, y)

                mapped,_ = self.transform(xi, yi, BRAILLE_KERNEL.shape)
                
                pixels = np.zeros((L*(self._lines), C*(self._columns)), dtype=int)
                i,j = mapped.T
                pixels[j,i] = 1
                
                tmp = block_multiply(pixels, BRAILLE_KERNEL)
                
                for i,j in zip(*np.nonzero(tmp)):
                   canvas[i,j] = Ansi.format(to_braille(tmp[i,j]), color)

            L,C = KERNEL41.shape
            mapped,_ = self.transform(x, y, KERNEL41.shape)
            mapped = mapped//[C,L]
            
            if ds.get('fill', False):
                #zero,_ = self.transform(x[0], [0])
                #_,k = zero[0]
                for i,j in mapped:
                    # k = zero[0,1]
                    # if zero[0,1] > j:
                    #     k += 1
                    # j,k = sorted((j,k))
                    # canvas[j:k,i] = Ansi.format(ds['marker'], color)
                    canvas[j:,i] = Ansi.format(ds['marker'], color)
            
            if ds.get('marker', None) is not None:
                i,j = mapped.T
                canvas[j,i] = Ansi.format(ds['marker'], color)
            
            for percentile in ds.get('percentile', []):
                xp = percentile
                llabel = (self._xtick_fmt%xp[0]).strip()
                rlabel = (self._xtick_fmt%xp[-1]).strip()
                
                yp = min(self.get_ylim()) * np.ones_like(xp)
                mapped,_ = self.transform(xp, yp)
                
                i = mapped[:,0]
                # canvas[0,i.min():i.max()] = Ansi.format(u'━', color)
                # canvas[0,mapped[:,0]] = Ansi.format(u'╋', color)
                
                s = u'╋'.join([''] + [u'━'*d for d in np.diff(mapped[:,0])-1] + [''])
                s2 = '%s %s %s'%(llabel,s,rlabel)
                item = Ansi.format((i.min()-len(llabel)-1)*' '+ s2, color)
                footers.append([item])
                #item = Ansi.format(i.min()*' '+ s, color)
                #footers.append([item])
                # headers.append([item])
            
            # if 'percentile' in ds:
            #     xp = ds['percentile']
            #     yp = min(self.get_ylim()) * np.ones_like(xp)
            #     mapped,_ = self.transform(xp, yp)
                
            #     i = mapped[:,0]
            #     # canvas[0,i.min():i.max()] = Ansi.format(u'━', color)
            #     # canvas[0,mapped[:,0]] = Ansi.format(u'╋', color)
                
            #     s = u'╋'.join([''] + [u'━'*d for d in np.diff(mapped[:,0])-1] + [''])
            #     item = Ansi.format(i.min()*' '+ s, color)
            #     footers.append([item])
            #     # headers.append([item])




        
        
    def set_xtick_format(self, fmt):
        if fmt is None:
            fmt = '%r'
        self._xtick_fmt = fmt
    
    def set_ytick_format(self, fmt):
        if fmt is None:
            fmt = '%r'
        self._ytick_fmt = fmt

    def set_xticks(self, ticks):
        self._xticks = ticks

    def get_xticks(self):
        if self._xticks is None:
            ticks = self._ax.get_xticks()
        else:
            ticks = self._xticks
        
        # find center y-coordinate
        yc = 0.5*np.sum(self._ax.get_ylim())
        pos, idx = self.transform(ticks, yc*np.ones_like(ticks), unique=False)
        
        # return list(zip(pos[:,0], ticks[idx]))
        return pos[:,0], ticks[idx]

    def get_yticks(self):
        
        # Problems with Log10Transform in earlier versions of MPL.
        # > Works with 2.2.3 and above

        # reverse ordering due to differences in axes origin
        # between MPL and Tplot
        ticks = self._ax.get_yticks()[::-1]

        # get the center x-coordinate
        xc = 0.5*np.sum(self._ax.get_xlim())
        pos, idx = self.transform(xc*np.ones_like(ticks), ticks, unique=False)
        
        # return list(zip(pos[:,1], ticks[idx]))
        return pos[:,1], ticks[idx]


    def __repr__(self):
        return str(self)
    
    # def show(self):
    #     print(self)

    def close(self):
        plt.close(self._fig)

    # def __enter__(self):
    #     return self

    # def __exit__(self, *args):
    #     self.close()
    
    