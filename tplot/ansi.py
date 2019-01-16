import os

__all__ = ['Ansi']

class Ansi:

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
        
        Ansi._COLORS = {
            'COLOR%d'%n: get('TPLOT_COLOR%d'%n, '\033[%dm'%(30+n)) for n in range(1,7)
        }
        
        # light gray
        Ansi._COLORS['GRID'] = get('TPLOT_GRID', '\033[2m')
        
    @staticmethod
    def get(name):
        if Ansi._COLORS is None:
            Ansi.load()
        return Ansi._COLORS.get(name, '')
    
    @staticmethod
    def available_colors():
        if Ansi._COLORS is None:
            Ansi.load()
        colors = [Ansi._COLORS[s]
                    for s in sorted(Ansi._COLORS.keys())
                    if s.startswith('COLOR')]
        return colors
    
    @staticmethod
    def disable():
        Ansi._ENABLED = False
    
    @staticmethod
    def enable():
        Ansi._ENABLED = True
    
    @staticmethod
    def format(s, *prefixes):
        if Ansi._ENABLED:
            return ''.join(prefixes) + s + Ansi.RESET
        return s
