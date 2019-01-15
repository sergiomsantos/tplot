import numpy as np
import sys
import os


IS_PYTHON3 = sys.version_info[0] == 3
MPL_DISABLED = 'TPLOT_NOGUI' in os.environ


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



class Colors:

    RESET = '\033[0m'
    BOLD  = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    _COLORS = None
    _ENABLED = True
    
    @staticmethod
    def load():
        def get(name, default):
            # red, green, yellow, blue, magenta, cyan
            # change (30+n) to (90+n) for light-color variants
            color = os.getenv(name, None)
            if color is None:
                color = default
            else:
                if IS_PYTHON3:
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
    def disable():
        Colors._ENABLED = False
    
    @staticmethod
    def enable():
        Colors._ENABLED = True
    
    @staticmethod
    def format(s, *prefixes):
        if Colors._ENABLED:
            return ''.join(prefixes) + s + Colors.RESET
        return s
