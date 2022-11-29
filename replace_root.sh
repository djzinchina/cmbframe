#!/bin/bash/

org="/home/doujzh/Healpix_3.70/data"
des="/disk1/home/doujz/Healpix_3.80/data"
sed -i "s#$org#$des#g" ./cmbframe/*.py
sed -i "s#$org#$des#g" ~/codes/*.py

org="/home/doujzh/DATA/HPX_pix_wgts"
des="/disk1/home/doujz/Healpix_3.80/full_weights"
sed -i "s#$org#$des#g" ./cmbframe/*.py
sed -i "s#$org#$des#g" ~/codes/*.py


