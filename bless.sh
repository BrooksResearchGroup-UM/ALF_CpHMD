#!/usr/bin/env bash

# Ensure we have one argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <j>"
    exit 1
fi

j="$1"

SCRIPT="minimize_setup.slurm"

# Check if j is a valid integer
if ! [[ "$j" =~ ^-?[0-9]+$ ]]; then
    echo "Error: j must be an integer."
    exit 1
fi

# Convert to absolute value in case a negative number is given
j_abs=$(( j < 0 ? -j : j ))

# We will do a total of (2 * j_abs) + 1 iterations:
# From i = 0 to j (inclusive) gives (j_abs + 1) values
# Then from i = j+1 to 2*j (inclusive) gives j_abs more values
#
# For the second phase:
#   At i = j+1, we want to print -j_abs
#   At i = j+2, we want to print (-j_abs + 1)
#   ...
#   At i = 2*j_abs, we end up at -1.
#
# Thus total steps go from i=0 to i=2*j_abs.

for (( i=0; i <= 2*j_abs; i++ )); do
    if (( i <= j_abs )); then
        # First half: print increasing from 0 to j_abs
        val=$i
    else
        # Second half: print increasing from -j_abs up to -1
        # When i = j_abs+1, val = -j_abs
        # When i = j_abs+2, val = -j_abs+1
        # ...
        val=$(( -j_abs + (i - (j_abs + 1)) ))
    fi

    sed -i "s|^#SBATCH --output=.*|#SBATCH --output=setup_$i.out|" "$SCRIPT"
    sed -i "s|^#SBATCH --error=.*|#SBATCH --error=setup_$i.err|" "$SCRIPT"
    sed -i "s|^#SBATCH --job-name=.*|#SBATCH --job-name=setup_$i|" "$SCRIPT"
    sed -i "s|^drxn=.*|drxn=$val|" "$SCRIPT"
    sed -i "s|^setup=.*|setup=$i|" "$SCRIPT"
    

    echo "i = $i, val = $val"
    head -n 17 "$SCRIPT"
done