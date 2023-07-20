# OpenMM Benchmark
Benchmark on a small peptide with varying size of waterbox.
Tests tip3p and tip4pew waters and single/mixed/double precision on CUDA or HIP platforms.

Simply run `python benchmark.py --platform cuda` or `python benchmark.py --platform hip` respectively.
The conda environment used for either platforms is defined in the environment yaml files.
Results on Nvidia A40 and AMD MI210 using 16 AMD EPYC 7763 cores are located in the [results](./results) folder.
