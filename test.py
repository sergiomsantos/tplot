import numpy as np
import tplot

size = tplot.get_output_size()
plot = tplot.TPlot(size.columns, size.lines)
plot.bar(np.arange(20), np.random.uniform(-10,10,20))
plot.show_grid(True)
plot.set_border(tplot.Format.ALL)
plot.show()