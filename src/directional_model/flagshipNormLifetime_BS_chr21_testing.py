import sys
import argparse
import matplotlib
# matplotlib.use("Agg")

import os
from os import listdir, makedirs     
from os.path import isfile, join, basename, normpath, exists, dirname
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mirnylib.plotting import nicePlot
import multiprocessing as mp
from scipy.stats import spearmanr
from mirnylib.numutils import coarsegrain
# import pyximport;
# pyximport.install()

# import definitions from the original script
sys.path.append('/home/ben/projects/supercontact/src/')
sys.path.append('/home/ben/projects/supercontact/src/directional_model')
from useful_3D import *
from flagshipNormLifetime_functions import logistic, indexing, \
tonumpyarray, init, worker, chunk, averageContacts, initModel, \
doSim, displayHeatmap, calculateAverageLoop, doPolymerSimulation
from logarr_param_explore import do_extruder_position

# def main() 

# TESTING ARGS
testing=True
if testing: 
    # ctcfFile =  '/home/ben/ab_local/directional_model/k562_CTCF_M1Motif_peaks.txt'
    # ctcfFile =  '/home/ben/ab_local/directional_model/k562_CTCF_RenMotif_peaks.txt'
    ctcfFile =  '/home/ben/ab_local/k562/chipseq_data/unipk/k562_ctcf_encode_rad21_forsimulation.txt'
    # ctcfFile =  '/home/ben/ab_local/k562/chipseq_data/unipk/k562_ctcf_encode_rad21_customDir_forsimulation.txt'
    # ctcfFile =  '/home/ben/ab_local/directional_model/k562_CTCF_heuristic_peaks.txt'
    # ctcfFile =  '/home/ben/ab_local/directional_model/k562_CTCF_encode_RAD21_simulation.txt'
    # ctcfFile =  '/home/ben/ab_local/directional_model/k562_CTCF_forsimulation.txt'
    outfolderBasename = '/home/ben/ab_local/directional_model/testing_local'
    SEPARATION = 200
    LIFETIME = 300
    mu = 3
    divideLogistic = 20
    extendFactor = 0.10
    saveBlocks = 100

# make output folder
outfolder="{0}_S{1}_L{2}_Mu{3}_d{4}".format(outfolderBasename, SEPARATION, LIFETIME, str(int(mu)), str(int(divideLogistic)))

# make the outfolder
if not (exists(outfolder)):
    print('making output directory: ' + outfolder)
    makedirs(outfolder)

# chromosome, start and end to simulate
mychr = 21
mystart = 29372390 
myend = 31322258
# extend by extendFactor in either direction to remove edge effects hopefully
nmon = int(np.ceil((myend - mystart) / 600.0))
low = int(mystart - (nmon * extendFactor * 600))
high = int(myend + (nmon * extendFactor * 600))

# monomer locations for this chromosome regiion
lowMon = low // 600     
highMon = high // 600 

# read in information on the CTCF sites 
df = pd.read_csv(ctcfFile, sep = "\t")
# subset to rows with a summitDist value
df = df.loc[(~pd.isnull(df["summitDist"]) ) ]
# subset to the right chr
df = df.loc[df["chrom"] == "chr{0}".format(mychr)]

# get information on the CTCF sites
start = df["start"].values
end = df["end"].values
strand = df["summitDist"].values > 0
strength = df["fc"]

# midpoints in monomers
mid_1k = (start + end) // 1200
M = mid_1k.max() + 1

# print(len(mid_1k[strand]))
# print(len(mid_1k[~strand]))
# counts the number of ctcf sites that fall in each monomer.
# here theyre weighted by the strength variable
forw = np.bincount(mid_1k[strand], weights = (strength[strand] / divideLogistic), minlength=highMon)
rev = np.bincount(mid_1k[~strand], weights = (strength[~strand] / divideLogistic), minlength=highMon)

# pick out the sections in forw and reverse that we're actually simulating
forw = forw[lowMon:highMon]
rev = rev[lowMon:highMon]
print('CTCF sites on forward: ' + str(len(forw[forw>0])))
print('CTCF sites on reverse: ' + str(len(rev[rev>0])))

# transformed arrays of stall sites
forw = logistic(forw,mu)
rev = logistic(rev, mu)

# scale by the max value
doScale = False
if doScale:
    forw = forw / np.max(forw)
    rev = rev / np.max(rev)

# number of monomers to simulate
# N = highMon-lowMon
N = len(forw)
print('N: ' + str(N))


def genome_to_bin(coord, roundMethod='floor'):
    mystart = 29372390 
    myend = 31322258
    binSize = 30000
    if roundMethod=='floor':
       return np.floor((coord - mystart)/binSize)
    elif roundMethod=='round':
       return np.round((coord - mystart)/binSize)
    elif roundMethod=='ceil':
       return np.ceil((coord - mystart)/binSize)
    else:
       return ((coord - mystart)/binSize)

# plot some additional domain inforation on here
# contact domains from HiC 
# peaks from Hic
import matplotlib.patches as patches
contactDomainFile = '/home/ben/ab_local/k562/hic_data/Contact_domains_chr21_myregion.txt'
peakFile = '/home/ben/ab_local/k562/hic_data/Peaks_chr21_myregion.txt'
contactDomains = pd.read_csv(contactDomainFile, sep='\t', header=0)
peaks = pd.read_csv(peakFile, sep='\t', skiprows=0)
peaksFloat = genome_to_bin(peaks[['x1','x2','y1','y2']], roundMethod='none')
peaks = genome_to_bin(peaks[['x1','x2','y1','y2']], roundMethod='floor')
# set first domain to the start to simplify plotting 
contactDomains['x1'][0] = 29372390
contactDomains = genome_to_bin(contactDomains[['x1','x2','y1','y2']], roundMethod=False)


#This does the heatmap  from positions of loop extrudors only. 
nsim=10
# arr, logarr = displayHeatmap(N, low, high, SEPARATION, LIFETIME, forw, rev, nsim)
manyContactAv = load_K562_contact()
hicSum1 = load_HiC_contact()


# plot with CTCF sites and genomic or HiC annotations as defined above
def plot_with_annotations(arr, title, cmap, peaks, contactDomains, forw, rev, showPlot=True, maxPercentile=99.9, extendFactor=0.10):
    fig, ax0 = plt.subplots()
    im0 = ax0.matshow(arr, cmap=cmap, vmax = np.nanpercentile(arr, maxPercentile))
    ax0.set_title(title, y=1.08)
    ax0.set_ylim(64,-20)
    fig.colorbar(im0, ax=ax0)

    # CTCF site information
    trim = extendFactor
    npoints = len(forw)
    orig_size = int(npoints / (1 + (trim * 2)))
    remove_each = int((npoints - orig_size) / 2)
    forwTrim = forw[remove_each:npoints - remove_each]
    forwTrimC = coarsegrain(forwTrim, 50)
    revTrim = rev[remove_each:npoints - remove_each]
    revTrimC = coarsegrain(revTrim, 50)
    forwTrimC[forwTrimC >1] =1
    revTrimC[revTrimC >1] =1
    # where on the chart we're plotting
    siteStart = -10 
    siteExpand = 9
    # add lines for CTCF sites
    ax0.vlines([np.where(forwTrimC != 0)], ymin=siteStart, ymax=siteStart - (forwTrimC[np.where(forwTrimC != 0)] * siteExpand), color='red')
    ax0.vlines([np.where(revTrimC != 0)], ymin=siteStart, ymax=siteStart + (revTrimC[np.where(revTrimC != 0)] * siteExpand), color='blue')

    # add lines all the way down the plot
    ax0.vlines([np.where(forwTrimC != 0)], ymin=-20, ymax=[np.where(forwTrimC != 0)], color='magenta', lw=0.5, linestyles='dashed')
    ax0.vlines([np.where(revTrimC != 0)], ymin=-20, ymax=[np.where(revTrimC != 0)], color='aqua', lw=0.5, linestyles='dashed')
    ax0.hlines([np.where(forwTrimC != 0)], xmax=arr.shape[0]-1, xmin=[np.where(forwTrimC != 0)], color='magenta', lw=0.5, linestyles='dashed')
    ax0.hlines([np.where(revTrimC != 0)], xmax=arr.shape[0]-1, xmin=[np.where(revTrimC != 0)], color='aqua', lw=0.5, linestyles='dashed')

    # add information on domains and peak calls from HiC
    for i in range(contactDomains.shape[0]):
        side = (contactDomains['x2'][i] - contactDomains['x1'][i])
        xy=[contactDomains['x1'][i], contactDomains['x1'][i] + side]
        p = patches.Rectangle([x - 0.5 for x in xy], width = side, height= -1*side,
            fill=False, edgecolor='palegreen')
        ax0.add_patch(p)
    for i in range(peaks.shape[0]):
        side = 1
        xy=[peaks['x1'][i] - 0.5, peaks['y1'][i] + 0.5]
        p = patches.Rectangle([x for x in xy], width = side, height= -1*side,
            fill=False, edgecolor='palegreen')
        ax0.add_patch(p)

    for i in range(peaks.shape[0]):
        side = 1
        xy=[peaks['y1'][i] - 0.5,peaks['x1'][i] + 0.5]
        p = patches.Rectangle([x for x in xy], width = side, height= -1*side,
            fill=False, edgecolor='palegreen')
        ax0.add_patch(p)

    if showPlot:
        plt.show(block=False)

# arr, logarr, forw, rev = do_extruder_position('/home/ben/ab_local/k562/chipseq_data/unipk/k562_ctcf_encode_rad21_forsimulation.txt')
plot_with_annotations(logarr,'log extrusion occupancy', 'viridis', peaks, contactDomains, forw, rev, maxPercentile=99.9)
plot_with_annotations(logarr,'log extrusion occupancy', 'viridis', peaksFloat, contactDomains, forw, rev, maxPercentile=99.9)

# plot_with_annotations(manyContactAv,'K562 average contact', 'magma', peaks, contactDomains,   maxPercentile=95)
plot_with_annotations(hicSum1, 'HiC contact frequency', 'magma', peaks, contactDomains, forw, rev, maxPercentile=95)
plot_with_annotations(hicSum1, 'HiC contact frequency', 'magma', peaksFloat, contactDomains, forw, rev, maxPercentile=95)

# can we directly compare this logarr and some
arrUpper = arr[np.triu_indices(arr.shape[0], 1)]
hicUpper = hicSum1[np.triu_indices(hicSum1.shape[0], 1)] 
lhCor = spearmanr(arrUpper, hicUpper)[0]

plt.scatter(arrUpper, hicUpper, alpha=0.5)
plt.xlabel('extrusion occupancy of bin')
plt.ylabel('HiC contact of bin')
plt.show(block=False)
# cor of this map is 0.86

# whats the correlation with no sites though?
forwNoSites =forw
forwNoSites[forw !=0] = 0
revNoSites =rev
revNoSites[rev !=0] = 0
arrNoSites, logarrNoSites = displayHeatmap(N, low, high, SEPARATION, LIFETIME, forwNoSites, revNoSites, nsim)

arrUpper = arrNoSites[np.triu_indices(arrNoSites.shape[0], 1)]
hicUpper = hicSum1[np.triu_indices(hicSum1.shape[0], 1)] 
lhCor = spearmanr(arrUpper, hicUpper)[0]

plt.scatter(arrUpper, hicUpper, alpha=0.5)
plt.xlabel('extrusion occupancy of bin')
plt.ylabel('HiC contact of bin')
plt.show(block=False)
# cor of random is 0.8899
# that's better than the map with sites on it! hahaha...



def cor_with_hic(arr, hicSum1=hicSum1):
    arrUpper = arr[np.triu_indices(arr.shape[0], 1)]
    hicUpper = hicSum1[np.triu_indices(hicSum1.shape[0], 1)] 
    return(spearmanr(arrUpper, hicUpper)[0])


# custom dir sites
a1, la1, fw1, rv1 = do_extruder_position('/home/ben/ab_local/k562/chipseq_data/unipk/k562_ctcf_encode_rad21_forsimulation.txt') 
plot_with_annotations(la1, 'Rad21 overlaps', 'viridis', peaksFloat, contactDomains, fw1, rv1)
plot_with_annotations(hicSum1, 'HiC with Rad21 annotations', 'magma', peaksFloat, contactDomains, fw1, rv1, maxPercentile=95)
a2, la2, fw2, rv2 = do_extruder_position('/home/ben/ab_local/k562/chipseq_data/unipk/k562_ctcf_encode_rad21_customDir_forsimulation.txt') 
plot_with_annotations(la2, 'Rad21 overlaps, custom Dir', 'viridis', peaksFloat, contactDomains, fw2, rv2)
plot_with_annotations(hicSum1, 'HiC with Rad21 custom annotations', 'magma', peaksFloat, contactDomains, fw2, rv2, maxPercentile=95)
# ren MOtif sites
a3, la3, fw3, rv3 = do_extruder_position('/home/ben/ab_local/k562/chipseq_data/unipk/k562_ctcf_encode_ren_max1_forsimulation.txt', mu=4) 
plot_with_annotations(la3, 'Ren Motifs', 'viridis', peaksFloat, contactDomains, fw3, rv3)
plot_with_annotations(hicSum1, 'HiC with Ren motif annotation', 'magma', peaksFloat, contactDomains, fw3, rv3, maxPercentile=95)


d=4
arrDI =directionalityIndex(arr, d)
diExpand1 = -10
diExpand2 = -4
fig, ax0 = plt.subplots()
# fig, [ax0, ax1] = plt.subplots(ncols=2, gridspec_kw = {'height_ratios':[3, 1], })
im0 = ax0.matshow(logarr, vmax = np.percentile(logarr, 99.9))
ax0.plot([i for i in range(len(arrDI))], [(i*diExpand1)-10 for i in arrDI], lw=0.75)
ax0.plot([i for i in range(len(arrDI))], [-10 for i in arrDI], ls='--', color='black', lw=0.5)
# fig.colorbar(im0, ax=ax0)
ax0.set_title('log extrusion occupancy', y=1.08)
ax0.vlines([i+0.5 for i in range(65)], ymin=-20, ymax=64.5, linestyles='dashed', lw=0.5, color='grey')
# fig.colorbar(im0, ax=ax0)


# plt.imshow(arr, vmax = np.percentile(arr, 99.9), extent = [low, high, high, low], interpolation = "none")
# nicePlot(show=False)
# plt.show(block=False)
# np.save('/home/ben/ab_local/directional_model/extrusion_fitting/\
#     arr_CTCF_heuristic.npy', arr)
# np.save('/home/ben/ab_local/directional_model/extrusion_fitting/\
#     logarr_CTCF_heuristic.npy', logarr)




'''
# directionality index of logarr
arrDI = directionalityIndex(arr, 5)

plt.plot([i for i in range(arr.shape[0])], arrDI)
plt.show(block=False)

# let's plot on on top of the other
fig, (ax0, ax1) = plt.subplots(nrows=2, gridspec_kw = {'height_ratios':[3, 1], })
im0 = ax0.imshow(logarr, vmax = np.percentile(logarr, 99.9),\
 extent = [low, high, high, low], interpolation = "none")
# nicePlot(show=False)
# fig.colorbar(im0, ax=ax0)
ax0.set_title('log extrusion occupancy')

im1 = ax1.plot([i for i in range(arr.shape[0])], arrDI)
ax1.set_title('Directionality Index')
plt.show(block=False)
'''

# get K562 distance map

# plt.matshow(manyDistAv, cmap='magma_r')
# plt.show(block=False)

# calculate DI for these 
# contactDI = directionalityIndex(manyContactAv, 5)
# plt.matshow(manyContactAv, cmap='magma')
# plt.plot([i for i in range(len(contactDI))], contactDI)
# plt.show(block=False)


def diCompare(logarr, arrDI,  manyContactAv, contactDI, savename):
    r=250
    # plot all 4
    # normalize so that the two are on the same scale
    arrDI = arrDI / np.max(abs(arrDI)) * 9
    contactDI = contactDI / np.max(abs(contactDI)) * 9

    fig, [ax0, ax1] = plt.subplots(ncols=2)
    # fig, [ax0, ax1] = plt.subplots(ncols=2, gridspec_kw = {'height_ratios':[3, 1], })
    im0 = ax0.matshow(logarr, vmax = np.percentile(logarr, 99.9))
    ax0.set_title('log extrusion occupancy', y=1.08)
    ax0.plot([i for i in range(len(arrDI))], [(i*-1)-10 for i in arrDI], lw=0.75)
    ax0.plot([i for i in range(len(arrDI))], [-10 for i in arrDI], ls='--', color='black', lw=0.5)
    # fig.colorbar(im0, ax=ax0)
    ax0.set_ylim(64,-20)

    im1 = ax1.matshow(manyContactAv, cmap='magma')
    # fig.colorbar(im1, ax=ax1)
    ax1.set_title('microscopy contact at r=' + str(r), y=1.08)
    ax1.plot([i for i in range(len(contactDI))], [(i*-1)- (10) for i in contactDI], lw=0.75)
    ax1.plot([i for i in range(len(contactDI))], [(-10) for i in contactDI], ls='--', color='black', lw=0.5)
    ax1.set_ylim(64,-20)

    # plt.show(block=False)
    plt.savefig(savename, bbox_inches='tight', dpi=250)
    plt.close()

def doDiCompare(logarr, arr, manyContactAv, saveBasename):
    for d in range(4,10):
        diCompare(logarr, directionalityIndex(arr, d),\
              manyContactAv, directionalityIndex(manyContactAv * 10, d), \
              saveBasename+str(d)+'.png')

manyContactAv = load_K562_contact()
doDiCompare(logarr, arr, manyContactAv,'/home/ben/ab_local/directional_model/extrusion_fitting/diCompare/heuristic_di_d')




def diCor(logarr, arr, manyContactAv, d):
    arrDI = directionalityIndex(arr, d)
    contactDI = directionalityIndex(manyContactAv * 10, d)

    plt.scatter(arrDI, contactDI, )
    plt.show(block=False)


import peakutils
def doPlots():
    # calculation of the inflection points 
    # this is where the DI changes signs (including going to zero)
    for d in range(4,10):

        diExpand2 = 4
        contactDI  = directionalityIndex(manyContactAv *10, d)
        diChange = np.append(np.array(0), ((np.diff(np.sign(contactDI)) != 0)*1))
        diChangeInd = np.where(diChange ==1)[0] - 0.5
        peaksOld = np.array([2,8,15,17,22,26,31,36,40,44,51,57,61]) - 0.5

        contactDIPos = np.array([i if i >0 else 0 for i in contactDI])
        contactDINeg = np.array([i if i <0 else 0 for i in contactDI])
        min_dist = 3
        thres = 0.25
        posPeaks = peakutils.indexes(contactDIPos, min_dist=min_dist, thres=thres)
        negPeaks = peakutils.indexes(contactDINeg * -1, min_dist=min_dist, thres=thres)
        # peaks = posPeaks
        peaks = np.append(posPeaks +0.5, negPeaks - 0.5)

        fig, ax1 = plt.subplots()
        im1 = ax1.matshow(manyContactAv, cmap='magma')
        # fig.colorbar(im1, ax=ax1)
        ax1.set_title('microscopy contact at r=' + str(r), y=1.08)
        ax1.plot([i for i in range(len(contactDI))], [(i*diExpand2 * 1)- (10) for i in contactDI], lw=0.75)
        ax1.plot([i for i in range(len(contactDI))], [(-10) for i in arrDI], ls='--', color='black', lw=0.5)
        ax1.vlines(peaks, ymin=-20, ymax=65, linestyles='dashed', lw=0.5, color='orange')
        ax1.hlines(peaks, xmin=0, xmax=64.5, linestyles='dashed', lw=0.5, color='orange')
        # ax1.vlines(diChangeInd, ymin=-20, ymax=65, linestyles='dashed', lw=0.5, color='yellow')
        # ax1.hlines(diChangeInd, xmin=0, xmax=64.5, linestyles='dashed', lw=0.5, color='yellow')
        # plt.show(block=False)
        plt.savefig('/home/ben/ab_local/directional_model/extrusion_fitting/diCompare/trypeak_'+str(d)+'.png', bbox_inches='tight', dpi=250)
    plt.close('all')

    for d in range(4,10):
        diExpand2 = 1
        contactDI  = directionalityIndex(arr * 10, d)
        diChange = np.append(np.array(0), ((np.diff(np.sign(contactDI)) != 0)*1))
        diChangeInd = np.where(diChange ==1)[0] - 0.5
        peaksOld = np.array([2,8,15,17,22,26,31,36,40,44,51,57,61]) - 0.5

        contactDIPos = np.array([i if i >0 else 0 for i in contactDI])
        contactDINeg = np.array([i if i <0 else 0 for i in contactDI])
        min_dist = 3
        thres = 0.25
        posPeaks = peakutils.indexes(contactDIPos, min_dist=min_dist, thres=thres)
        negPeaks = peakutils.indexes(contactDINeg * -1, min_dist=min_dist, thres=thres)
        # peaks = posPeaks
        peaks = np.append(posPeaks +0.5, negPeaks - 0.5)

        fig, ax1 = plt.subplots()
        im1 = ax1.matshow(logarr)
        # fig.colorbar(im1, ax=ax1)
        ax1.set_title('simulation LEF occupancy', y=1.08)
        ax1.plot([i for i in range(len(contactDI))], [(i*diExpand2 * 1)- (10) for i in contactDI], lw=0.75)
        ax1.plot([i for i in range(len(contactDI))], [(-10) for i in contactDI], ls='--', color='black', lw=0.5)
        ax1.vlines(peaks, ymin=-20, ymax=65, linestyles='dashed', lw=0.5, color='orange')
        ax1.hlines(peaks, xmin=0, xmax=64.5, linestyles='dashed', lw=0.5, color='orange')
        # ax1.vlines(diChangeInd, ymin=-20, ymax=65, linestyles='dashed', lw=0.5, color='yellow')
        # ax1.hlines(diChangeInd, xmin=0, xmax=64.5, linestyles='dashed', lw=0.5, color='yellow')
        # plt.show(block=False)
        plt.savefig('/home/ben/ab_local/directional_model/extrusion_fitting/diCompare/heuristic_trypeak_simulation_'+str(d)+'.png', bbox_inches='tight', dpi=250)
    plt.close('all')


# what if... instead of CTCF calls we just use the DI from the microscopy data. This should simulate
# structures that are comparable, even if not motiviated entirely by ChipSeq data
manyContactAv = load_K562_contact()
d = 5
contactDI =directionalityIndex(hicSum1, d)
contactDIPos = np.array([i if i >0 else 0 for i in contactDI])
contactDINeg = np.array([i if i <0 else 0 for i in contactDI])
min_dist = 3
thres = 0.25
# posPeaks = peakutils.indexes(contactDIPos, min_dist=min_dist, thres=thres)
# negPeaks = peakutils.indexes(contactDINeg * -1, min_dist=min_dist, thres=thres)

# attempt 1: each bin can have at most 1 ctcf site
# this will be located at the center monomer for that bin.
# ctcf directionality and strength will be from the DI of the microsocpy data
# normalized so the strongest site has a value of 1
# and then redo the simulation 
# contactDI of 0 means no site

# chromosome, start and end to simulate
mychr = 21
mystart = 29372390 
myend = 31322258

# scale contactDI by max val
contactDIScaled = contactDI/np.max(abs(contactDI))
# mid base pair of our bins
binSize = 30000
nBins = 65
binStarts = np.array([int(mystart + (binSize * i)) for i in range(nBins)])
binEnds = np.array([int(i + binSize -1) for i in binStarts])
binMids = np.array([int(i) for i in binStarts + (binSize/2)])


strength =  abs(contactDIScaled[contactDIScaled != 0])
strand = contactDIScaled[contactDIScaled != 0] > 0
mid_1k = binMids[contactDIScaled != 0] // 600

# extend by extendFactor in either direction to remove edge effects hopefully
nmon = int(np.ceil((myend - mystart) / 600.0))
low = int(mystart - (nmon * extendFactor * 600))
high = int(myend + (nmon * extendFactor * 600))

# monomer locations for this chromosome regiion
lowMon = low // 600     
highMon = high // 600 

# midpoints in monomers
M = mid_1k.max() + 1

# print(len(mid_1k[strand]))
# print(len(mid_1k[~strand]))
# counts the number of ctcf sites that fall in each monomer.
# here theyre weighted by the strength variable
forw = np.bincount(mid_1k[strand], weights = (strength[strand]), minlength=highMon)
rev = np.bincount(mid_1k[~strand], weights = (strength[~strand]), minlength=highMon)

# pick out the sections in forw and reverse that we're actually simulating
forw = forw[lowMon:highMon]
rev = rev[lowMon:highMon]
print('CTCF sites on forward: ' + str(len(forw[forw>0])))
print('CTCF sites on reverse: ' + str(len(rev[rev>0])))

# transformed arrays of stall sites
doLogistic = False
if doLogistic:
    forw = logistic(forw,mu)
    rev = logistic(rev, mu)

# scale by the max value
doScale = False
if doScale:
    forw = forw / np.max(forw)
    rev = rev / np.max(rev)

# number of monomers to simulate
# N = highMon-lowMon
N = len(forw)
print('N: ' + str(N))
nsim=10
arr, logarr = displayHeatmap(N, low, high, SEPARATION, LIFETIME, forw, rev, nsim)


arrDI =directionalityIndex(arr, d)
arrDI = (arrDI / np.max(abs(arrDI))) * 9
fig, ax0 = plt.subplots()
# fig, [ax0, ax1] = plt.subplots(ncols=2, gridspec_kw = {'height_ratios':[3, 1], })
im0 = ax0.matshow(logarr, vmax = np.percentile(logarr, 99.9))
ax0.plot([i for i in range(len(arrDI))], [(i*-1)-10 for i in arrDI], lw=0.75)
ax0.plot([i for i in range(len(arrDI))], [-10 for i in arrDI], ls='--', color='black', lw=0.5)
# fig.colorbar(im0, ax=ax0)
ax0.set_title('log extrusion occupancy', y=1.08)
# ax0.vlines([i+0.5 for i in range(65)], ymin=-20, ymax=64.5, linestyles='dashed', lw=0.5, color='grey')
# fig.colorbar(im0, ax=ax0)
ax0.set_ylim(64,-20)

# can also plot some ctcf stuff
trim = extendFactor
npoints = len(forw)
orig_size = int(npoints / (1 + (trim * 2)))
remove_each = int((npoints - orig_size) / 2)
forwTrim = forw[remove_each:npoints - remove_each]
forwTrimC = coarsegrain(forwTrim, 50)
revTrim = rev[remove_each:npoints - remove_each]
revTrimC = coarsegrain(revTrim, 50)

siteStart = -10 
siteExpand = 9
ax0.vlines([np.where(forwTrimC != 0)], ymin=siteStart, ymax=siteStart - (forwTrimC[np.where(forwTrimC != 0)] * siteExpand), color='red')
ax0.vlines([np.where(revTrimC != 0)], ymin=siteStart, ymax=siteStart + (revTrimC[np.where(revTrimC != 0)] * siteExpand), color='blue')
plt.show(block=False)

doDiCompare(logarr, arr, manyContactAv,'/home/ben/ab_local/directional_model/extrusion_fitting/diCompare/diSites_d')

# we can just export this as a file to use with the normal simulation code on Windows
saveDf = pd.DataFrame({'chrom'      : 'chr21', 
                       'start'      : binStarts[contactDIScaled != 0],
                       'end'        : binEnds[contactDIScaled != 0],
                       'fc'         : strength,
                       'summitDist' : strand  * 2 -1})
saveDf = saveDf[['chrom', 'start', 'end', 'fc', 'summitDist']]
saveDf.to_csv('/home/ben/ab_local/directional_model/k562_DIsites_forsimulation.txt', sep='\t', index=False)
saveDf.to_csv('/home/ben/ab/cdrive/Software/MirnyLab_OpenMM_Polymer/examples/loopExtrusion/directionalModel/k562_DIsites_forsimulation.txt', sep='\t', index=False)

##############################################
### new 2017-11-09   #########################
##############################################

# what if we modify DI calculations
# to be more zscore like and can have multiple sites per bin
manyContactAv = load_K562_contact()

# new directionality index
# for each bin, calculate simply mean up and down 
def new_DI(arr, d=5, distMat=False):
    nbin = arr.shape[0]
    DIList = []
    upMeanList = []
    downMeanList = []
    for i in range(nbin):
        if (i == 0) or (i == nbin-1):
            DIList.append(0)
            upMeanList.append(0)
            downMeanList.append(0)
        else: 
            newD = np.min([d, i, (nbin-i-1)])
            # print(str(i), ' ', str(newD))
            upInd = i + newD
            downInd =  i - newD 
            upVal = arr[i , i+1:upInd+1]
            downVal = arr[downInd:i , i]
            if(all(np.isnan(upVal)) or all(np.isnan(downVal))):
                DI=0
                UM=0
                DM=0
            else:
                B = np.nanmean(upVal)
                A = np.nanmean(downVal)
            upMeanList.append(B)
            downMeanList.append(A)
    
    return(np.array([upMeanList, downMeanList]))


a,b=new_DI(manyContactAv)
arrDI =directionalityIndex(manyContactAv)
arrDI = (arrDI / np.max(abs(arrDI))) * 9

plt.plot([i for i in range(len(a))], a, label='upMean')
plt.plot([i for i in range(len(b))], b, label='downMean')
plt.legend()
plt.show(block=False)

# plot this ontop of a heatmap
# plot as zscore instead
azs = np.array([(i - np.mean(a))/np.std(a) for i in a])
bzs = np.array([(i - np.mean(b))/np.std(b) for i in b])

plt.plot([i for i in range(len(azs))], azs, label='upMean')
plt.plot([i for i in range(len(bzs))], bzs, label='downMean')
plt.legend()
plt.show(block=False)


a,b=new_DI(manyContactAv, d =7)
arrDI =directionalityIndex(manyContactAv, d =7)
arrDI = (arrDI / np.max(abs(arrDI))) * 9

fig, ax0 = plt.subplots()
# fig, [ax0, ax1] = plt.subplots(ncols=2, gridspec_kw = {'height_ratios':[3, 1], })
im0 = ax0.matshow(manyContactAv, cmap='magma')
ax0.plot([i for i in range(len(azs))], [(i*-5)-10 for i in azs], lw=0.75, label='upMean')
ax0.plot([i for i in range(len(bzs))], [(i*-5)-10 for i in bzs], lw=0.75, label='downMean')
ax0.plot([i for i in range(len(arrDI))], [(i*-1)-10 for i in arrDI], lw=0.75, color='black', label='DI')
ax0.plot([i for i in range(len(azs))], [-10 for i in azs], ls='--', color='black', lw=0.5)
# fig.colorbar(im0, ax=ax0)
ax0.set_title('Z-score ', y=1.08)
# ax0.vlines([i+0.5 for i in range(65)], ymin=-20, ymax=64.5, linestyles='dashed', lw=0.5, color='grey')
# fig.colorbar(im0, ax=ax0)
ax0.set_ylim(64,-20)
ax0.legend(loc='lower right')
plt.show(block=False)


# now there can be at most 65 CTCF sites distributed
# but each bin can have a forward and reverse. 
# first/last bin or two shouldn't have a site... no information to make a call?
azs[0] = 0
azs[len(azs)-1] = 0
bzs[0] = 0
bzs[len(bzs)-1] = 0

azsScaled = azs
azsScaled[azsScaled <0 ] =0

bzsScaled = bzs
bzsScaled[bzsScaled <0 ] =0
# normalize so that the max of each is 1
azsScaled = azsScaled/np.max(azsScaled)
bzsScaled = bzsScaled/np.max(bzsScaled)

# Put this in the context of chr21
# chromosome, start and end to simulate
mychr = 21
mystart = 29372390 
myend = 31322258

# mid base pair of our bins
binSize = 30000
nBins = 65
binStarts = np.array([int(mystart + (binSize * i)) for i in range(nBins)])
binEnds = np.array([int(i + binSize -1) for i in binStarts])
binMids = np.array([int(i) for i in binStarts + (binSize/2)])

# split up because we can have more than one site in a bin
strengthA =  abs(azsScaled[azsScaled != 0])
strandA = azsScaled[azsScaled != 0] > 0
mid_1kA = binMids[azsScaled != 0] // 600
strengthB =  abs(bzsScaled[bzsScaled != 0])
strandB = bzsScaled[bzsScaled != 0] > 0
mid_1kB = binMids[bzsScaled != 0] // 600

# extend by extendFactor in either direction to remove edge effects hopefully
nmon = int(np.ceil((myend - mystart) / 600.0))
low = int(mystart - (nmon * extendFactor * 600))
high = int(myend + (nmon * extendFactor * 600))

# monomer locations for this chromosome regiion
lowMon = low // 600     
highMon = high // 600 

# midpoints in monomers
M = mid_1k.max() + 1

# print(len(mid_1k[strand]))
# print(len(mid_1k[~strand]))
# counts the number of ctcf sites that fall in each monomer.
# here theyre weighted by the strength variable
forw = np.bincount(mid_1kA, weights = (strengthA), minlength=highMon)
rev = np.bincount(mid_1kB, weights = (strengthB), minlength=highMon)

# pick out the sections in forw and reverse that we're actually simulating
forw = forw[lowMon:highMon]
rev = rev[lowMon:highMon]
print('CTCF sites on forward: ' + str(len(forw[forw>0])))
print('CTCF sites on reverse: ' + str(len(rev[rev>0])))


# number of monomers to simulate
# N = highMon-lowMon
N = len(forw)
print('N: ' + str(N))
nsim=10
arr, logarr = displayHeatmap(N, low, high, SEPARATION, LIFETIME, forw, rev, nsim)

fig, ax0 = plt.subplots()
im0 = ax0.matshow(logarr, vmax = np.percentile(logarr, 99.9))
ax0.plot([i for i in range(len(azs))], [(i*-5)-10 for i in azs], lw=0.75, label='upMean')
ax0.plot([i for i in range(len(bzs))], [(i*-5)-10 for i in bzs], lw=0.75, label='downMean')
ax0.plot([i for i in range(len(arrDI))], [(i*-1)-10 for i in arrDI], lw=0.75, color='black', label='DI')
ax0.plot([i for i in range(len(azs))], [-10 for i in azs], ls='--', color='black', lw=0.5)

# fig.colorbar(im0, ax=ax0)
ax0.set_title('log extrusion occupancy', y=1.08)
# ax0.vlines([i+0.5 for i in range(65)], ymin=-20, ymax=64.5, linestyles='dashed', lw=0.5, color='grey')
# fig.colorbar(im0, ax=ax0)
ax0.set_ylim(64,-20)

# can also plot some ctcf stuff
trim = extendFactor
npoints = len(forw)
orig_size = int(npoints / (1 + (trim * 2)))
remove_each = int((npoints - orig_size) / 2)
forwTrim = forw[remove_each:npoints - remove_each]
forwTrimC = coarsegrain(forwTrim, 50)
revTrim = rev[remove_each:npoints - remove_each]
revTrimC = coarsegrain(revTrim, 50)

siteStart = -10 
siteExpand = 9
ax0.vlines([np.where(forwTrimC != 0)], ymin=siteStart, ymax=siteStart - (forwTrimC[np.where(forwTrimC != 0)] * siteExpand), color='red')
ax0.vlines([np.where(revTrimC != 0)], ymin=siteStart, ymax=siteStart + (revTrimC[np.where(revTrimC != 0)] * siteExpand), color='blue')
ax0.legend(loc='lower left')
plt.show(block=False)

# we can just export this as a file to use with the normal simulation code on Windows
# define start and end 
# +1 to have strength on the positive and negative strand
startA = binStarts[azs >0]
startB = binStarts[bzs >0] +1
startA = binStarts[azs >0]
startB = binStarts[bzs >0] +1

saveDf = pd.DataFrame({'chrom'      : 'chr21', 
                       'start'      : binStarts[contactDIScaled != 0],
                       'end'        : binEnds[contactDIScaled != 0],
                       'fc'         : strength,
                       'summitDist' : strand  * 2 -1})
saveDf = saveDf[['chrom', 'start', 'end', 'fc', 'summitDist']]
saveDf.to_csv('/home/ben/ab_local/directional_model/k562_DIsites_forsimulation.txt', sep='\t', index=False)
saveDf.to_csv('/home/ben/ab/cdrive/Software/MirnyLab_OpenMM_Polymer/examples/loopExtrusion/directionalModel/k562_DIsites_forsimulation.txt', sep='\t', index=False)
