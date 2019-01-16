__all__ = ['get_output_size']

def get_output_size():
    import os

    # try to get default size from env variables
    try:
        size = os.getenv('TPLOT_SIZE', None)
        lines,columns = size.split(',')
        return int(lines), int(columns)
    except:
        pass
    
    # try shutil if py3
    try:
        import shutil
        tsize = shutil.get_terminal_size()
        return tsize.lines, tsize.columns
    except:
        pass
    
    # try stty if py2
    try:
        lines,columns = os.popen('stty size', 'r').read().split()
        return int(lines), int(columns)
    except:
        pass

    # final fallback option (80 columns, 24 lines)
    return (24, 80)

