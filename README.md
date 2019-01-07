# TPlot

Output Matplotlib plots to the console

### Example:

```bash
$ ./tplot.x -h
usage: tplot [-h] [-xy X [Y ...]] [-c [C [C ...]]] [--hist [H [H ...]]]
             [--bins N] [-d D] [-s N] [-ax xmin [xmax ...]]
             [-ay ymin [ymax ...]] [--logx] [--logy] [--width W] [--height H]
             [--mpl]
             [file]

positional arguments:
  file

optional arguments:
  -h, --help           show this help message and exit

Plot arguments:
  -xy X [Y ...]        scatter plot of column X vs Y
  -c [C [C ...]]       series plot of column(s) C
  --hist [H [H ...]]   histogram of column(s) H
  --bins N             number of bins

Data parsing:
  -d D, --delimiter D  delimiter
  -s N, --skip N       skip first N rows

Axis configuration:
  -ax xmin [xmax ...]  x-axis limits
  -ay ymin [ymax ...]  y-axis limits
  --logx               set log-scale on the x-axis
  --logy               set log-scale on the y-axis

Output configuration:
  --width W            output width
  --height H           output height
  --mpl                show plot in matplotlib window
```

```bash
$ ./tplot.x data.txt -c 1 2 --width=80 --height=24


          │                                                                 
          │             [92mx[0m[92mx[0m[92mx[0m               [92mx[0m[92mx[0m[92mx[0m               [92mx[0m[92mx[0m[92mx[0m     [1m[94mcol-1 o[0m
 4.00e+00 ┤   [94mo[0m        [92mx[0m  [92mx[0m[92mx[0m             [92mx[0m[92mx[0m [92mx[0m[92mx[0m             [92mx[0m[92mx[0m  [92mx[0m    [1m[92mcol-2 x[0m
          │   [94mo[0m       [92mx[0m[92mx[0m   [92mx[0m             [92mx[0m   [92mx[0m             [92mx[0m   [92mx[0m[92mx[0m       [94mo[0m   
          │   [94mo[0m[94mo[0m     [94mo[0m[92mx[0m[94mo[0m   [92mx[0m[92mx[0m           [92mx[0m     [92mx[0m           [92mx[0m[92mx[0m   [94mo[0m[92mx[0m[94mo[0m     [94mo[0m[94mo[0m   
          │    [94mo[0m     [92mx[0m[92mx[0m[94mo[0m[94mo[0m   [92mx[0m           [92mx[0m     [92mx[0m           [92mx[0m   [94mo[0m[94mo[0m[92mx[0m[92mx[0m     [94mo[0m    
 2.00e+00 ┤    [94mo[0m     [92mx[0m  [94mo[0m   [92mx[0m [94mo[0m[94mo[0m[94mo[0m      [92mx[0m[92mx[0m     [92mx[0m[92mx[0m      [94mo[0m[94mo[0m[94mo[0m [92mx[0m   [94mo[0m  [92mx[0m     [94mo[0m    
          │    [94mo[0m     [92mx[0m  [94mo[0m   [92mx[0m[92mx[0m[94mo[0m [94mo[0m[94mo[0m     [92mx[0m       [92mx[0m     [94mo[0m[94mo[0m [94mo[0m[92mx[0m[92mx[0m   [94mo[0m  [92mx[0m     [94mo[0m    
          │    [94mo[0m    [92mx[0m[92mx[0m  [94mo[0m[94mo[0m   [92mx[0m[94mo[0m  [94mo[0m     [92mx[0m[94mo[0m[94mo[0m   [94mo[0m[94mo[0m[92mx[0m     [94mo[0m  [94mo[0m[92mx[0m   [94mo[0m[94mo[0m  [92mx[0m[92mx[0m    [94mo[0m    
 0.00e+00 ┤     [94mo[0m   [92mx[0m    [94mo[0m   [92mx[0m    [94mo[0m   [92mx[0m[92mx[0m [94mo[0m[94mo[0m[94mo[0m[94mo[0m[94mo[0m [92mx[0m[92mx[0m   [94mo[0m    [92mx[0m   [94mo[0m    [92mx[0m   [94mo[0m     
          │     [94mo[0m   [92mx[0m    [94mo[0m   [94mo[0m[92mx[0m   [94mo[0m[94mo[0m [94mo[0m[92mx[0m         [92mx[0m[94mo[0m [94mo[0m[94mo[0m   [92mx[0m[94mo[0m   [94mo[0m    [92mx[0m   [94mo[0m     
          │     [94mo[0m  [92mx[0m[92mx[0m    [94mo[0m  [94mo[0m[94mo[0m[92mx[0m    [94mo[0m[94mo[0m[94mo[0m[92mx[0m         [92mx[0m[94mo[0m[94mo[0m[94mo[0m    [92mx[0m[94mo[0m[94mo[0m  [94mo[0m    [92mx[0m[92mx[0m  [94mo[0m     
-2.00e+00 ┤     [94mo[0m  [92mx[0m[94mo[0m     [94mo[0m [94mo[0m [92mx[0m      [92mx[0m           [92mx[0m      [92mx[0m [94mo[0m [94mo[0m     [94mo[0m[92mx[0m  [94mo[0m     
          │     [94mo[0m  [92mx[0m      [94mo[0m[94mo[0m[94mo[0m  [92mx[0m     [92mx[0m           [92mx[0m     [92mx[0m  [94mo[0m[94mo[0m[94mo[0m      [92mx[0m  [94mo[0m     
          │      [94mo[0m[92mx[0m[94mo[0m       [94mo[0m   [92mx[0m    [92mx[0m[92mx[0m           [92mx[0m[92mx[0m    [92mx[0m   [94mo[0m       [94mo[0m[92mx[0m[94mo[0m      
          │      [94mo[0m[92mx[0m[94mo[0m            [92mx[0m   [92mx[0m             [92mx[0m   [92mx[0m            [94mo[0m[92mx[0m[94mo[0m      
-4.00e+00 ┤   [92mx[0m  [92mx[0m[94mo[0m             [92mx[0m[92mx[0m [92mx[0m[92mx[0m             [92mx[0m[92mx[0m [92mx[0m[92mx[0m             [94mo[0m[92mx[0m  [92mx[0m   
          │   [92mx[0m[92mx[0m[92mx[0m[92mx[0m               [92mx[0m[92mx[0m[92mx[0m               [92mx[0m[92mx[0m[92mx[0m               [92mx[0m[92mx[0m[92mx[0m[92mx[0m   
          │                                                                 
          └───┬───────────┬──────────┬───────────┬───────────┬──────────┬────
          0.00e+00    1.00e+02    2.00e+02    3.00e+02    4.00e+02    5.00e+02
```

