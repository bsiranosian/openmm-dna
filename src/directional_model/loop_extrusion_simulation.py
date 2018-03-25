import sys
import argparse
from os import makedirs, getenv
from os.path import exists
import socket
import numpy as np
import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()})
import matplotlib
matplotlib.use("Agg")

# OLD IMPORT CODE
'''
# import modules and get the location of my github repo
# depending on the hostname
hn = socket.gethostname()
if hn == 'aspire':
    supercontactSrc = '/home/ben/projects/supercontact/src/'
elif hn == 'BoettigerServer':
    supercontactSrc = 'C:\\Users\\Ben\\projects\\supercontact\\src\\'
elif (hn.startswith('sherlock') or
      getenv('SHERLOCK') == '1' or getenv('SHERLOCK') == '2'):
    supercontactSrc = '~/projects/supercontact/src/'
else:
    print('Warning: set location of supercontact \
        repository src file in this script')
    supercontactSrc = '~/projects/supercontact/src/'
sys.path.append(supercontactSrc)
sys.path.append(supercontactSrc + 'directional_model')
'''
import openmmlib
import polymerutils
from loop_extrusion_functions import get_forw_rev, do_polymer_simulation


def main():
    parser = argparse.ArgumentParser(description="Simulate chromatin foling under the mirnylab \
    loop extrusion directional model. Currently simulates\
    region Chr21:29372390-31322258 by default.")

    parser.add_argument('-i',
                        '--input_ctcf',
                        required=True,
                        action='store',
                        dest='input_ctcf',
                        help='input CTCF peak file. 5 column, tab separated. Ex:\
        chrom start   end fc  summitDist\
        chr21   29402390    29432389    0.03143   -1\
        chr21   29432390    29462389    0.42000    1')
    parser.add_argument('-o',
                        '--outfolder_basename',
                        required=True,
                        action='store',
                        dest='outfolder_basename',
                        help='Basename of folder to save results. \
                        Parameters will be appended to the end. Will be \
        created if does not exist.')
    parser.add_argument('-s',
                        '--separation',
                        required=False,
                        action='store',
                        dest='separation',
                        default=200,
                        help='Extruder separation parameter. Default=200')
    parser.add_argument('-l',
                        '--lifetime',
                        required=False,
                        action='store',
                        dest='lifetime',
                        default=300,
                        help='Extruder lifetime parameter. Default=300')
    parser.add_argument('-m',
                        '--mu',
                        required=False,
                        action='store',
                        dest='mu',
                        default=3,
                        help='Logistic function Mu parameter. Default=3')
    parser.add_argument('-d',
                        '--divide_logistic',
                        required=False,
                        action='store',
                        dest='divide_logistic',
                        default=20,
                        help='Logistic function divide parameter. Default=20')
    parser.add_argument('-e',
                        '--extend',
                        required=False,
                        action='store',
                        dest='extend',
                        default=0.10,
                        help='Extend simulation by this fraction past the start\
                        and end to reduce edge effects. Default=0.10')
    parser.add_argument('-sb',
                        '--save_blocks',
                        required=False,
                        action='store',
                        dest='save_blocks',
                        default=2000,
                        help='Save this many simulation blocks. Default=2000')
    parser.add_argument('-nl',
                        '--no_logistic',
                        required=False,
                        action='store_true',
                        dest='no_logistic',
                        default=False,
                        help='No logistic scaling of CTCF boundary strengths')
    parser.add_argument('-imp',
                        '--impermiable_boundaries',
                        required=False,
                        action='store_true',
                        dest='impermiable_boundaries',
                        default=False,
                        help='Specify this flag to make CTCF boundaries impermiable, \
        theres no probability a SMC can slide past. Still directional.')
    parser.add_argument('-smc_steps',
                        '--smc_steps',
                        required=False,
                        action='store',
                        dest='smc_steps',
                        default=4,
                        help='Number of SMC steps per 3D polymer\
                        simulation steps. Default of 4. Must be integer.')
    parser.add_argument('-gpuN',
                        '--gpu_number',
                        required=False,
                        action='store',
                        dest='gpu_number',
                        default='default',
                        help='Which GPU to run the simulation on, for\
                        systems with more than one GPU. Default takes \
                        the first available GPU.')
    parser.add_argument('-cpu',
                        '--cpu_simulation',
                        required=False,
                        action='store_true',
                        dest='cpu_simulation',
                        default=False,
                        help='Do simulations using the CPU only, not a GPU')
    parser.add_argument('-monomer_size',
                        '--monomer_size',
                        required=False,
                        action='store',
                        dest='monomer_size',
                        default=600,
                        help='Change monomer representation in the model. Default of 600bp is \
        from Jeffs paper. Changing requires adapting lifetime, separation \
        parameter also. EXPERIMENTAL!')
    parser.add_argument('-no_SMC',
                        '--no_SMC',
                        required=False,
                        action='store_true',
                        dest='no_SMC',
                        default=False,
                        help='Remves the action of SMCs in the model. Just a polymer \
        floating about randomly now.')
    parser.add_argument('-randomize_SMC',
                        '--randomize_SMC',
                        required=False,
                        action='store_true',
                        dest='randomize_SMC',
                        default=False,
                        help='Fully randomize the positions of SMCs each simulation block.\
        Should probably up time_step along with this option.')
    parser.add_argument('-time_step',
                        '--time_step',
                        required=False,
                        action='store',
                        dest='time_step',
                        default=5000,
                        help='Number of simulation time_step per block')
    parser.add_argument('-skip_start',
                        '--skip_start',
                        required=False,
                        action='store',
                        dest='skip_start',
                        default=100,
                        help='Skip this many simulation blocks at the start.')
    # should add args for chr start, end
    # but that can come later!!

    # get args
    args = parser.parse_args()
    input_ctcf = args.input_ctcf
    outfolder_basename = args.outfolder_basename
    SEPARATION = int(args.separation)
    LIFETIME = int(args.lifetime)
    mu = float(args.mu)
    divide_logistic = float(args.divide_logistic)
    extend_factor = float(args.extend)
    save_blocks = int(args.save_blocks)
    no_logistic = args.no_logistic
    # impermiable_boundaries = args.impermiable_boundaries
    smc_steps = int(args.smc_steps) - 1
    gpu_number = args.gpu_number
    cpu_simulation = args.cpu_simulation
    monomer_size = int(args.monomer_size)
    no_SMC = args.no_SMC
    randomize_SMC = args.randomize_SMC
    time_step = int(args.time_step)
    skip_start = int(args.skip_start)
    # done

    # make output folder
    if no_logistic:
        outFolder = "{0}_S{1}_L{2}_smcX{3}_noLog".format(
            outfolder_basename, SEPARATION, LIFETIME, str(smc_steps + 1))

    else:
        outFolder = "{0}_S{1}_L{2}_Mu{3}_d{4}_smcX{5}".format(
            outfolder_basename, SEPARATION, LIFETIME,
            str(int(mu)), str(int(divide_logistic)), str(smc_steps + 1))

    # make the outFolder
    if not (exists(outFolder)):
        print('making output directory: ' + outFolder)
        makedirs(outFolder)

    # ensure divide_logistic is >0
    if divide_logistic < 0:
        print('divide_logistic error. setting to default=20')
        divide_logistic = 20
    # if no_logistic, don't divide
    if no_logistic:
        print('Not doing logistic scaling')
        divide_logistic = 1

    # change to simulate a different chr region
    mychr = 21
    mystart = 29372390
    myend = 31322258
    forw, rev = get_forw_rev(
        input_ctcf, mu=mu, divide_logistic=divide_logistic,
        extend_factor=extend_factor, do_logistic=not no_logistic,
        monomer_size=monomer_size, mychr=mychr, mystart=mystart, myend=myend)
    print('CTCF sites on forward: ' + str(len(forw[forw > 0])))
    print('CTCF sites on reverse: ' + str(len(rev[rev > 0])))
    # number of monomers to simulate
    N = len(forw)

    # This actually does a polymer simulation
    do_polymer_simulation(
        steps=time_step, dens=0.2, stiff=2,
        folder=outFolder, N=N, SEPARATION=SEPARATION, LIFETIME=LIFETIME,
        forw=forw, rev=rev, save_blocks=save_blocks, smc_steps=smc_steps,
        no_SMC=no_SMC, randomize_SMC=randomize_SMC,
        gpu_number=gpu_number, cpu_simulation=cpu_simulation,
        skip_start=skip_start)


if __name__ == '__main__':
    main()
