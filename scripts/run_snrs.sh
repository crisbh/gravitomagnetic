#!/bin/bash -l

for z in $(seq 0.5 0.1 3); do
    printf "\rProcessing redshift: %s..." "$z"
    python python/SNR.py --in-dir output_cosma/ --out-dir output_cosma/ --z_source "$z"
done

echo ""
echo "Done."