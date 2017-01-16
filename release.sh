#!/bin/bash

if [ $# -lt 1 ]; then
	echo "usage: $0 <version>"
	exit 1
fi

ver=$1
#PKG=hsh_signal

wd=$(basename $(dirname $(readlink -f $0)))

cd ..
mkdir $wd/release/$ver
zip -r - $wd/hsh_signal $wd/gr_pll $wd/gr_firdes $wd/README.md $wd/setup.py -x '*.pyc' '*.so' > $wd/release/$ver/$wd.zip


scp $wd/release/$ver/$wd.zip well:/var/www/$wd-$ver.zip
scp $wd/release/$ver/$wd.zip well:/var/www/$wd.zip
