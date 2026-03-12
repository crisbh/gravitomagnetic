#!/bin/bash -l

for z in $(seq 0.5 0.1 3); do
    python3 python/angular_powerspec_z.py --in-dir output_cosma/lcdm/ --out-dir output_cosma/lcdm --z_source "$z"
    python3 python/angular_powerspec_z.py --in-dir output_cosma/frhs/ --out-dir output_cosma/frhs --z_source "$z"
    python3 python/angular_powerspec_z.py --in-dir output_cosma/ndgp/ --out-dir output_cosma/ndgp --z_source "$z"
done