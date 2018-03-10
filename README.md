# openmm-dna
Code for simulating 3D chromatin interactions under the loop extrusion hypothesis ([Fudenberg et al. (2016)])(http://www.sciencedirect.com/science/article/pii/S2211124716305307). Relies heavily on the Mirnylab openmm-polymer implementation. This is mostly a wrapper making their implementation easier to use to simulate custom genomic regions. 


## prerequisites
[openmm-polymer](https://bitbucket.org/mirnylab/openmm-polymer/overview) and prereqs for that repository. _this is probably the hardest part_


## Simulation of loop extrusion 
After configuring the proper tools, it's pretty straightforward to simulate an ensemble of 3D structures representing chromatin folding via loop extrusion. The model takes in one main data type: a list of CTCF sites defined at positions in the genome. This is a 5 column file in the following format 
```
chrom	start	end	fc	summitDist
chr21	29402390	29432389	0.031439872213307296	-1
chr21	29432390	29462389	0.42000392777653356	1
```
Other paramaters are as definied in the script flagshipNormLifetime_BS_chr21.py. See the Fudenberg et al. paper above and the supplementary material for a detailed explanation of the extusion methodology and the separation, lifetime, mu, divide_logisitic parameters that define the properties of the simulation. 

```
usage: flagshipNormLifetime_BS_chr21.py [-h] -i INPUT_CTCF -o
                                        OUTFOLDER_BASENAME [-s SEPARATION]
                                        [-l LIFETIME] [-m MU]
                                        [-d DIVIDE_LOGISTIC] [-e EXTEND]
                                        [-sb SAVE_BLOCKS] [-nl] [-imp]
                                        [-smc_steps SMC_STEPS]
                                        [-gpuN GPU_NUMBER] [-cpu]
                                        [-monomer_size MONOMER_SIZE] [-no_SMC]
                                        [-randomize_SMC] [-timeStep TIMESTEP]
                                        [-skip_start SKIP_START]

Simulate chromatin foling under the mirnylab loop extrusion directional model.
Currently simulates region Chr21:29372390-31322258 by default.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_CTCF, --input_ctcf INPUT_CTCF
                        input CTCF peak file. 5 column, tab separated. Ex:
                        chrom start end fc summitDist chr21 29402390 29432389
                        0.03143 -1 chr21 29432390 29462389 0.42000 1
  -o OUTFOLDER_BASENAME, --outfolder_basename OUTFOLDER_BASENAME
                        Basename of folder to save results. Parameters will be
                        appended to the end. Will be created if does not
                        exist.
  -s SEPARATION, --separation SEPARATION
                        Extruder separation parameter. Default=200
  -l LIFETIME, --lifetime LIFETIME
                        Extruder lifetime parameter. Default=300
  -m MU, --mu MU        Logistic function Mu parameter. Default=3
  -d DIVIDE_LOGISTIC, --divide_logistic DIVIDE_LOGISTIC
                        Logistic function divide parameter. Default=20
  -e EXTEND, --extend EXTEND
                        Extend simulation by this fraction past the start and
                        end to reduce edge effects. Default=0.10
  -sb SAVE_BLOCKS, --save_blocks SAVE_BLOCKS
                        Save this many simulation blocks. Default=2000
  -nl, --no_logistic    No logistic scaling of CTCF boundary strengths
  -imp, --impermiable_boundaries
                        Specify this flag to make CTCF boundaries impermiable,
                        theres no probability a SMC can slide past. Still
                        directional.
  -smc_steps SMC_STEPS, --smc_steps SMC_STEPS
                        Number of SMC steps per 3D polymer simulation steps.
                        Default of 4. Must be integer.
  -gpuN GPU_NUMBER, --gpu_number GPU_NUMBER
                        Which GPU to run the simulation on, for systems with
                        more than one GPU. Default takes the first available
                        GPU.
  -cpu, --cpu_simulation
                        Do simulations using the CPU only, not a GPU
  -monomer_size MONOMER_SIZE, --monomer_size MONOMER_SIZE
                        Change monomer representation in the model. Default of
                        600bp is from Jeffs paper. Changing requires adapting
                        lifetime, separation parameter also. EXPERIMENTAL!
  -no_SMC, --no_SMC     Remves the action of SMCs in the model. Just a polymer
                        floating about randomly now.
  -randomize_SMC, --randomize_SMC
                        Fully randomize the positions of SMCs each simulation
                        block. Should probably up timeStep along with this
                        option.
  -timeStep TIMESTEP, --timeStep TIMESTEP
                        Number of simulation timestep per block
  -skip_start SKIP_START, --skip_start SKIP_START
                        Skip this many simulation blocks at the start.

```
