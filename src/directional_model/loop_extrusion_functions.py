# function definitions for the flagshipNormLifetime script
# cleaning up the mirny lab code again
from os.path import join, exists
import numpy as np
import pandas as pd
import ctypes
import multiprocessing as mp
import pyximport; pyximport.install()
import polymerutils
from polymerutils import scanBlocks
from mirnylib.numutils import coarsegrain
from smcTranslocator import smcTranslocatorDirectional


# logistic function to transform fc to boundary score
def logistic(x, mu=3):
    x[x == 0] = -99999999
    return 1 / (1 + np.exp(-(x - mu)))


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())  # .reshape((N,N))


def initModel(i, N, SEPARATION, LIFETIME, forw, rev):
    birthArray = np.zeros(N, dtype=np.double) + 0.1
    deathArray = np.zeros(N, dtype=np.double) + 1. / LIFETIME
    stallArrayLeft = forw
    stallArrayRight = rev
    stallDeathArray = np.zeros(N, dtype=np.double) + 1 / LIFETIME
    pauseArray = np.zeros(N, dtype=np.double)
    smcNum = N // SEPARATION
    myDeathArray = deathArray + (1. / (LIFETIME * LIFETIME)) * i
    SMCTran = smcTranslocatorDirectional(
        birthArray, myDeathArray, stallArrayLeft,
        stallArrayRight, pauseArray, stallDeathArray, smcNum)
    return SMCTran


def doSim(i, N, SEPARATION, LIFETIME, shared_arr, forw, rev):
    nparr = tonumpyarray(shared_arr)
    SMCTran = initModel(i, N, SEPARATION, LIFETIME, forw, rev)

    for j in range(1):
        SMC = []
        N1 = 10000
        for k in range(np.random.randint(N1 // 2, N1)):
            SMCTran.steps(150)
            SMC.append(SMCTran.getSMCs())
        SMC = np.concatenate(SMC, axis=1)
        SMC1D = SMC[0] * N + SMC[1]
        position, counts = np.unique(SMC1D, return_counts=True)

        with shared_arr.get_lock():
            nparr[position] += counts

    print("Finished!")

    return None


def get_forw_rev(ctcf_file, mu=3, divide_logistic=20,
                 extend_factor=0.10, do_logistic=True,
                 monomer_size=600, mychr=21, mystart=29372390, myend=31322258):

    # using a CTCF file and other inputs,
    # get scaled forw and rev array of stall sites
    # for LEFs

    # set this to be sure no division takes place
    # when not doing logistic scaling
    if not do_logistic:
        divide_logistic = 1
    # extend by extend_factor in either direction to remove edge effects
    nmon = int(np.ceil((myend - mystart) / monomer_size))
    low = int(mystart - (nmon * extend_factor * monomer_size))
    high = int(myend + (nmon * extend_factor * monomer_size))

    # monomer locations for this chromosome regiion
    lowMon = low // monomer_size
    highMon = high // monomer_size

    # read in information on the CTCF sites
    df = pd.read_csv(ctcf_file, sep="\t")
    # subset to rows with a summitDist value
    df = df.loc[(~pd.isnull(df["summitDist"]))]
    # subset to the right chr
    df = df.loc[df["chrom"] == "chr{0}".format(mychr)]

    # get information on the CTCF sites
    start = df["start"].values
    end = df["end"].values
    strand = df["summitDist"].values > 0
    strength = df["fc"]

    # midpoints in monomers
    mid_1k = (start + end) // (monomer_size * 2)
    # M = mid_1k.max() + 1

    # counts the number of ctcf sites that fall in each monomer.
    # here theyre weighted by the strength variable
    forw = np.bincount(mid_1k[strand], weights=(
        strength[strand] / divide_logistic), minlength=highMon)
    rev = np.bincount(mid_1k[~strand], weights=(
        strength[~strand] / divide_logistic), minlength=highMon)

    # pick out the sections in forw and reverse that we're actually simulating
    forw = forw[lowMon:highMon]
    rev = rev[lowMon:highMon]
    # print('CTCF sites on forward: ' + str(len(forw[forw>0])))
    # print('CTCF sites on reverse: ' + str(len(rev[rev>0])))

    # transformed arrays of stall sites
    if do_logistic:
        forw = logistic(forw, mu)
        rev = logistic(rev, mu)

    # scale by the max value
    doScale = False
    if doScale:
        forw = forw / np.max(forw)
        rev = rev / np.max(rev)

    return(forw, rev)

# takes in a file wth CTCF sites, returns extruder positioning arr and logarr


def do_extruder_position(forw, rev, SEPARATION=200,
                         LIFETIME=300, nSim=10, trim=0, binSize=0):
    # number of monomers to simulate
    N = len(forw)
    shared_arr = mp.Array(ctypes.c_double, N**2)
    arr = tonumpyarray(shared_arr)
    arr.shape = (N, N)

    # can do parallel with fmap, not doing that here
    # setExceptionHook()
    # fmap(doSim, range(30), n = 1 )  # number of threads to use.
    # On a 20-core machine I use 20.
    [doSim(i, N, SEPARATION, LIFETIME, shared_arr, forw, rev)
     for i in range(nSim)]

    # trim before coarsegraining, if desired
    if trim > 0:
        print('trimming ' + str(arr.shape))
        npoints = arr.shape[0]
        origSize = int(npoints / (1 + (trim * 2)))
        removeTotal = npoints - origSize
        if removeTotal % 2 != 0:
            removeLeft = int(np.floor(removeTotal / 2))
            removeRight = int(np.ceil(removeTotal / 2))
        else:
            removeLeft = removeRight = removeTotal // 2

        arr = arr[removeLeft:npoints - removeRight,
                  removeLeft:npoints - removeRight]
        print('done trimming ' + str(arr.shape))

    # bin to a lower resolution if desired
    if binSize > 0:
        arr = coarsegrain(arr, binSize)

    arr = np.clip(arr, 0, np.percentile(arr, 99.9))
    arr /= np.mean(np.sum(arr, axis=1))
    logarr = np.log(arr + 0.0001)
    return(arr, logarr)

# given a CTCF file and all the other parameters, init a
# smcTranslocatorDirectional object
# basically a wrapper for the above functions


def init_SMCTran(ctcf_file, SEPARATION=200, LIFETIME=300,
                 mu=3, divide_logistic=20, extend_factor=0.10,
                 do_logistic=True, monomer_size=600, mychr=21,
                 mystart=29372390, myend=31322258):
    # get stall sites
    forw, rev = get_forw_rev(
        ctcf_file, mu=mu, divide_logistic=divide_logistic,
        extend_factor=extend_factor, do_logistic=do_logistic,
        monomer_size=monomer_size, mychr=mychr, mystart=mystart,
        myend=myend)
    N = len(forw)
    SMCTran = initModel(0, N, SEPARATION, LIFETIME, forw, rev)
    return(SMCTran)


def calculateAverageLoop():
    SMCTran = initModel()
    SMCTran.steps(1000000)
    dists = []
    for i in range(10000):
        SMCTran.steps(1000)
        left, right = SMCTran.getSMCs()
        dist = np.mean(right - left)
        # print(dist)
        dists.append(dist)
    print("final dist", np.mean(dists))
    exit()


def do_polymer_simulation(steps, dens, stiff, folder, N, SEPARATION,
                          LIFETIME, forw, rev, save_blocks=2000, smc_steps=3,
                          no_SMC=False, randomize_SMC=False,
                          gpu_number='default', cpu_simulation=False,
                          skip_start=100):
    from openmmlib import Simulation
    from polymerutils import grow_rw
    import time
    i = 0
    SMCTran = initModel(i, N, SEPARATION, LIFETIME, forw, rev)

    box = (N / dens) ** 0.33  # density = 0.1
    if exists(join(folder, "block10.dat")):
        block = scanBlocks(folder)["keys"].max() - 1
        data = polymerutils.load(join(folder, "block{0}.dat".format(block)))
    else:
        data = grow_rw(N, int(box) - 2)
        block = 0
    assert len(data) == N
    skip = 0
    time.sleep(0.1)

    while True:
        SMCTran.steps(smc_steps)
        if randomize_SMC:
            SMCTran.steps(500000)
        if (block % 2000 == 0) and (skip == 0):
            print ("doing dummy steps")
            SMCTran.steps(500000)
            skip = skip_start
            print (skip, "blocks to skip")

        # initialize a polymer for the 3D simualton step.
        a = Simulation(timestep=80, thermostat=0.01)
        if cpu_simulation:
            a.setup(platform="cpu", PBC=True, PBCbox=[
                    box, box, box], precision="mixed")
        else:
            a.setup(platform="OpenCL", PBC=True, PBCbox=[
                    box, box, box], precision="mixed", GPU=gpu_number)
            # a.setup(platform="CUDA", PBC=True, PBCbox=[box, box, box],
            # GPU=sys.argv[4], precision="mixed")
        a.saveFolder(folder)
        a.load(data)
        a.addHarmonicPolymerBonds(wiggleDist=0.1)
        if stiff > 0:
            a.addGrosbergStiffness(stiff)
        a.addPolynomialRepulsiveForce(trunc=1.5, radiusMult=1.05)

        # get positions of SMCs, add bonds to polymer
        left, right = SMCTran.getSMCs()
        for l, r in zip(left, right):
            a.addBond(l, r, bondWiggleDistance=0.5,
                      distance=0.3, bondType="harmonic")
        a.step = block

        if skip > 0:
            print ("skipping block")
            a.doBlock(steps, increment=False)
            skip -= 1

        if skip == 0:
            a.doBlock(steps)
            block += 1
            a.save(mode='txt')
        if block == save_blocks:
            break

        data = a.getData()

        del a

        time.sleep(0.1)


"""
Mirnylib custom functions I dont use3

def indexing(smaller, larger, M):
    return larger + smaller * (M - 1) - smaller * (smaller - 1) / 2



def init(*args):
    global sharedArrays
    sharedArrays = args

def worker(filenames):
    N = int(np.sqrt(sum(map(len, sharedArrays))))
    chunks = range()
def chunk(mylist, chunksize):
    N = len(mylist)
    chunks = list(range(0, N, chunksize)) + [N]
    return [mylist[i:j] for i, j in zip(chunks[:-1], chunks[1:])]
def averageContacts(contactFunction, inValues, N, **kwargs):
    arrayDtype = kwargs.get("arrayDtype", ctypes.c_int32)

    nproc = min(kwargs.get("nproc", 4), len(filenames))
    blockSize = max(min(kwargs.get("blockSize", 50),
                        len(filenames) // (3 * nproc)), 1)

    finalSize = N * (N + 1) // 2
    boundaries = np.linspace(0, finalSize, bucketNum + 1)
    chunks = zip(boundaries[:-1], boundaries[1:])
    sharedArrays_ = [mp.Array(arrayDtype, j - i) for j, i in chunks]

    filenameChunks = [filenames[i::nproc] for i in range(nproc)]

    with closing(mp.Pool(processes=nproc,
    initializer=init, initargs=sharedARrays_)) as p:
        p.map(worker, filenameChunks)
"""
