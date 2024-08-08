#!/bin/bash

for i in $(seq 2 612);
do
    cd "episode_$i"
    ffmpeg -framerate 35 -i %d.png -c:v libx264 -r 35 -pix_fmt yuv420p output.mp4
    rm -rf *.png
    cd ..
done
