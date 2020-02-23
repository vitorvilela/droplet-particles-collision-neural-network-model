#!/bin/bash
rm droplet
qcc -Wall -O2 droplet.c -o droplet -lm
./droplet &

