#!/usr/bin/env sh

# see: http://blog.ablepear.com/2012/10/bundling-python-files-into-stand-alone.html
rm -f app.zip

cd tplot
zip -r ../app.zip *
cd ..

echo '#!/usr/bin/env python' | cat - app.zip > tplot.x
chmod +x tplot.x

rm -f app.zip

