#!/bin/bash
#$ -N Mandelbrot
#$ -q pub8i
#$ -pe mpi 64
#$ -R y


# Grid Engine Notes:
# -----------------
# 1) Use "-R y" to request job reservation otherwise single 1-core jobs
#    may prevent this multicore MPI job from running.   This is called
#    job starvation.

# Module load boost
module load boost/1.57.0

# Module load OpenMPI
module load openmpi-1.8.3/gcc-4.9.2

# Run the program 
# echo "proc = 16"
# mpirun -np 16  ./mandelbrot_joe 1000 1000
# mpirun -np 16  ./mandelbrot_susie 1000 1000
# mpirun -np 16  ./mandelbrot_ms 1000 1000


echo "proc = 48"
mpirun -np 48  ./mandelbrot_joe 1000 1000
mpirun -np 48  ./mandelbrot_susie 1000 1000
mpirun -np 48  ./mandelbrot_ms 1000 1000

echo "proc = 64"
mpirun -np 64  ./mandelbrot_joe 4000 4000
mpirun -np 64  ./mandelbrot_susie 4000 4000
mpirun -np 64  ./mandelbrot_ms 4000 4000