# openmm-dna
Code for simulating 3D chromatin interactions under the loop extrusion hypothesis [Fudenberg et al. (2016)](http://www.sciencedirect.com/science/article/pii/S2211124716305307). Relies heavily on the Mirnylab openmm-polymer implementation. This is mostly a wrapper making their implementation easier to use to simulate custom genomic regions. 

## Quick start
Given a file defining CTCF sites and a specific genomeic region, you can generate 3D stuctures with the default parameters. 
```python src/directional_model/loop_extrusion_simulation.py -i data/example_ctcf_sites.txt -o ~/loop_extrusion_test```
If you are using a computer without a cpu, specify the ```--cpu_simulation``` flag. Setting the ```--skip_start``` argument to 1 will output 3D structures quickly for testing (won't skip 100 blocks of simulation to get conformations far away from the starting conditions). 

## Prerequisites
[openmm-polymer](https://bitbucket.org/mirnylab/openmm-polymer/overview) and prereqs for that repository. _This is probably the hardest part. Getting the package and GPU support configured can take a while._


## Simulation of loop extrusion 
After configuring the proper tools, it's pretty straightforward to simulate an ensemble of 3D structures representing chromatin folding via loop extrusion. The model takes in one main data type: a list of CTCF sites defined at positions in the genome. This is a 8 column file that is defined as the overlap of CTCF binding peaks and another feature, typically RAD21 binding or CTCF motifs in the genome. The example data for the region I've been working with in K562 can be found in data/example_ctcf_sites.txt.
```
chrom	start	end	fc	f2_start	f2_end	f2_fc	summitDist
chr21	28937062	28937302	23.9148320018102	28937186	28937205	0.109	-1
chr21	29073396	29073636	155.192916651482	29073497	29073516	0.29	-1
```
Other paramaters are as definied in the script loop_extrusion_simulation.py (See the help text). See the Fudenberg et al. paper above and the supplementary material for a detailed explanation of the extusion methodology and the separation, lifetime, mu, divide_logisitic parameters that define the properties of the simulation. 

## Processing simulated polymers
Several steps are taken to process simulated polymers. The output trajectory is loaded into memory, the ends of the structure are trimmed off (the same amount that was extended during the simulation process), and the structure is binned in 3D by taking the centroid of a number of points. Structures that are from nearby timesteps are correlated - the default pipeline doesn't produce independent draws from the distribution of possible structures. I therefore only load every 100th or 200th structure from the simulation output if I'm looking for independent samples.

A simple processing pipeline can be found in src/process_simulated_structures.py. 

## Analysis of simulated ensemble
The file data/example_simulation.npy contains an example ensemble of structures. There are some handy analysis and plotting functions in src/useful_3D.py and src/useful_plotting.py. 

An example script to load the sample data and generate a figure can be found at src/example_analysis.py
!(example_maps.png)