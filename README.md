# TPlot

A Python package for creating and displaying matplotlib plots in the console/terminal.

## Installation

```console
# clone repository
$ git clone https://github.com/sergiomsantos/tplot.git

# install package
$ python setup.py install

# test installation
#  as an executable on the user's PATH
$ tploh -h
usage: tplot [-h] [--version] [-f FILE] [-c C L? [C L? ...]]
  [...]
  
#  as a module
$ python -m tplot
usage: tplot [-h] [--version] [-f FILE] [-c C L? [C L? ...]]
  [...]

```


# Usage

### Help

Help is available through the `-h` or `--help` flags:

```console
$ tplot -h
usage: tplot [-h] [--version] [-f FILE] [-c C L? [C L? ...]]
             [-xy X Y L? [X Y L? ...]] [--hist H L? [H L? ...]] [--bins N]
             [-d D] [-s N] [-ax xmin xmax] [-ay ymin ymax] [--logx] [--logy]
             [--width W] [--height H] [--padding P] [--mpl] [--no-color]

A Python package for creating and displaying matplotlib plots in the
console/terminal

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -f FILE, --file FILE  source file. Use "-" to read from stdin

Plot arguments:
  -c C L? [C L? ...]    series plot of column(s) C with optional label L
  -xy X Y L? [X Y L? ...]
                        scatter plot of column X vs Y with optional label L
  --hist H L? [H L? ...]
                        histogram of column(s) H with optional label L
  --bins N              number of bins

Data parsing:
  -d D, --delimiter D   delimiter
  -s N, --skip N        skip first N rows

Axis configuration:
  -ax xmin xmax         x-axis limits
  -ay ymin ymax         y-axis limits
  --logx                set log-scale on the x-axis
  --logy                set log-scale on the y-axis

Output configuration:
  --width W             output width
  --height H            output height
  --padding P           left padding
  --mpl                 show plot in matplotlib window
  --no-color            suppress colored output
```

### Simple series plot

Request a series plot (`-c`) of column 3

```console
$ tplot -f resources/data.txt -c 3
```

![simple series image](resources/images/example6.png)

### Simple scatter plot

Request a scatter plot (`-xy`) of columns 1 vs 0 of file `data.txt`:

```console
$ tplot -f resources/data.txt -xy 0 1
```

![simple scatter plot image](resources/images/example1.png)

### Multiple scatter plots

Request multiple scatter plots (`-xy`) of columns 1 vs 0 and 2 vs 0,
with the second set labelled as `5*cos(x)`

```console
$ tplot -f resources/data.txt -xy 0 1 -xy 0 2 '5*cos(x)'
```

![multiple scatter plots image](resources/images/example2.png)

### Pipes as data source

Feed data into `tplot` directly using a pipe:

```console
$ seq 1 100 | tplot -c 0 'simple sequence'
```

![piping data image](resources/images/example3.png)

### STDIN as data source

Feed data into `tplot` directly using a pipe:

```console
$ tplot -f - -xy 0 1 'first fibonacci numbers'
1
1
2
3
4
8
^D
```

![stdin data image](resources/images/example4.png)

### Histogram

Request an histogram (`--hist`) of column 1 and specify the
number of bins (`--bins`) and data range (`-ax`):

```console
$ tplot -f resources/data.txt --hist 1 'an histogram' --bins=5 -ax -5 5
```

![histogram image](resources/images/example5.png)

### Series plot with log-scale on the y-axis

Request a series plot (`-c`) of column 3 ith a log-scaled y-axis (`--logy`)

```console
$ tplot -f resources/data.txt -c 3 --logy
```

![logy series image](resources/images/example7.png)

### Supress colored output

Supress colored output (`--no-color`)

```console
$ tplot -f resources/data.txt -c 2 --no-color
```

![dull image](resources/images/example8.png)

### Output as a Matplotlib plot

```console
$ tplot -f resources/data.txt -xy 0 1 --mpl
```

![dull image](resources/images/example9.png)