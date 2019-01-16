import numpy as np
import tplot.utils
import tplot

# tplot.Ansi.disable()

# size = tplot.utils.get_output_size()
# plot = tplot.TPlot(size.columns, size.lines)
# plot = tplot.TPlot(80,40)
plot = tplot.TPlot((20,50))
# plot = tplot.TPlot()

plot.show_grid(True)
plot.set_padding(0)
plot.set_border(tplot.Format.ALL)
plot.set_tick_position(tplot.Format.BOTTOM_LEFT)

plot.set_title('o pai natal morreu')
N = 1000000
bins = 20

a = np.random.normal(0, 5, N)
h,e = np.histogram(a, bins=bins, density=False)

x = 0.5*(e[1:] + e[:-1])

# approximate percentiles
# dataset = plot.bar(x, h)
# p = np.interp([0.25, 0.50, 0.75], np.cumsum(h)/N, e[1:])
# dataset['percentile'] = p

#plot.hist(a, bins=bins, add_percentile=True)
plot.line(x, h, connect=True)

# plot.xlim = (-100, 100)
# plot.show()
print(plot)

# clear plot and draw as line
# plot.clear().line(x, h, connect=True)
# print(plot)
# plot.clear().line(x, h, connect=True)
# print(plot)
# plot.clear().line(x, h, connect=True)
# print(plot)
# plot.clear().line(x, h, connect=True)
# print(plot)
# plot.clear().line(x, h, connect=True)
# print(plot)
# plot.clear().line(x, h, connect=True)
# print(plot)


