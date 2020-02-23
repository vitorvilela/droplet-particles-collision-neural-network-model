#!/bin/bash
rm droplet
#qcc -Wall -O2 droplet.c -o droplet -lm
#./droplet
CC99='mpicc -std=c99' qcc -Wall -O2 -grid=octree -D_MPI=1 droplet.c -o droplet -lm
#CC99='mpicc -std=c99' qcc -Wall -O2 -D_MPI=1 droplet.c -o droplet -lm
mpirun -np 8 ./droplet &

