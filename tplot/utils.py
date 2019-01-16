__all__ = ['get_output_size']

def get_output_size():
    from collections import namedtuple
    import os

    TSize = namedtuple('TSize', ['columns', 'lines'])
    fallback=(80,24)

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
        return shutil.get_terminal_size(fallback=fallback)
    except:
        pass
    
    # try stty if py2
    try:
        r,c = os.popen('stty size', 'r').read().split()
        return TSize(int(c), int(r))
    except:
        pass

    # final fallback option (80 columns, 24 lines)
    return TSize(*fallback)

