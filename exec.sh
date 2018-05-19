#!/bin/bash
mkdir /tmp/3302938
make

./bin/shalw --export --export-path /tmp/3302938 -x 256 -y 256 -t 100

./visu.py /tmp/3302938/shalw_256x256_T100.sav 

