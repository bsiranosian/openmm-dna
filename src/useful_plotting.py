# useful plotting functions for supercontact
import socket
import sys
import os
import matplotlib.pyplot as plt
from mirnylib.numutils import coarsegrain

hn = socket.gethostname()
if hn == 'aspire':
    supercontactSrc = '/home/ben/projects/supercontact/src/'
elif hn == 'BoettigerServer':
    supercontactSrc = 'C:\\Users\\Ben\\projects\\supercontact\\src\\'
elif hn.startswith('sherlock') or\
    os.getenv('SHERLOCK') == '1' or\
        os.getenv('SHERLOCK') == '2':
    supercontactSrc = '~/projects/supercontact/src/'
else:
    print('Warning: set location of supercontact\
     repository src file in this script')
    supercontactSrc = '~/projects/supercontact/src/'

sys.path.append(supercontactSrc)
sys.path.append(supercontactSrc + 'directional_model')
from useful_3D import *


# from a simulation run trajectory, load up all 3D structures
# and plot an average distance and contact map for them
# options control which structures to load
def plot_avDist_contact(in_folder, out_folder, load_format='joblib', r=10,
                        start_ind=1, load_n=1, save_fig=True,
                        contact_vmax_percentile=97, ):
    """from a simulation run trajectory, load up all 3D structures
    and plot an average distance and contact map for them
    options control which structures to load
    """
    many_struc = load_run_trajectory(
        in_folder, load_format, start_ind=start_ind, load_n=load_n)
    many_dist = xyz_to_distance_many(many_struc)
    av_dist = np.mean(many_dist, axis=0)
    av_contact = average_contact_from_distance_k562(
        many_dist, contact_threshold=r)

    # save average distance map
    plt.matshow(av_dist, cmap='magma_r')
    plt.colorbar()
    plt.title('AvDist skip=' + str(load_n) + ' n=' +
              str(many_struc.shape[0]), y=1.08)
    if save_fig:
        plt.savefig(join(out_folder, 'avDist.png'), bbox_inches='tight')
    else:
        plt.show(block=False)
    # save average contact map
    plt.matshow(av_contact, cmap='magma', vmax=np.nanpercentile(
        av_contact, contact_vmax_percentile))
    plt.colorbar()
    plt.title('avContact skip=' + str(load_n) +
              ' n=' + str(many_struc.shape[0]) +
              ' r=' + str(r) + 'nm', y=1.08)
    if save_fig:
        plt.savefig(join(out_folder, 'avContact.png'), bbox_inches='tight')
        plt.close('all')
    else:
        plt.show(block=False)


def save_3Dimage_2domain(data, compartments_mon, save_name=None):
    """save a 3D rendering of the structure
    ideally using something better that matplotlib3d
    currently only works for 2 domains
    """

    fig = plt.figure(figsize=(5, 5))

    d1Bins = np.where(compartments_mon == 1)[0]
    d2Bins = np.where(compartments_mon == 2)[0]
    otherBins = np.where(
        np.isin(compartments_mon, [1, 2], invert=True))[0]

    ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    d1Points = data[d1Bins, :]
    d2Points = data[d2Bins, :]
    otherPoints = data[otherBins, :]
    domainColors = ['blue', 'green', 'dimgray']
    domainNames = ['d1', 'd2', 'other']
    # plot trace line
    ax2.plot(data[:, 0], data[:, 1],
             data[:, 2], lw=0.5, label='trace')
    # ax.scatter(data[:,0],data[:,1],data[:,2],
    # label=n, marker='o')
    # then plot points with colors
    if len(otherBins) > 0:
        [ax2.scatter(d[:, 0], d[:, 1], d[:, 2],
                     color=c, label=n, marker='o',
                     depthshade=False)
         for d, c, n in zip([d1Points, d2Points, otherPoints],
                            domainColors, domainNames)]
    else:
        [ax2.scatter(d[:, 0], d[:, 1], d[:, 2],
                     color=c, label=n, marker='o',
                     depthshade=False)
         for d, c, n in zip([d1Points, d2Points],
                            domainColors, domainNames)]
    ax2.legend()
    if save_name is not None:
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show(block=False)


def save_basic_heatmap(struc_dist, save_name, cmap='magma_r'):
    """Save a basic heatmap for a distance matrix."""
    plt.matshow(struc_dist, cmap='magma_r')
    plt.colorbar()
    if save_name is not None:
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show(block=False)


# plot the log extrusion occupancy and HiC or microscopy on the same map
# as well as the sites from simulation


def plot_logarr_sites(logarr, forw, rev, title='',
                      cmap='viridis', max_percentile=99.9,
                      extend_factor=0.10,
                      coarsegrain_factor=50,
                      plot_CTCF_lines=True, save_plot=None):
    '''Plot the log transformed array of extrusion occupancy with a
    track of the stall sites above the plot.

    Arguments:
    logarr: log extrusion occupancy matrix
    forw: forward array of stall sites
    rev: reverse array of stall sites

    Kwargs:
    title: plot title
    cmap: plot colormap
    max_percentile: threshold colormap at this percentile
    extenc_fator: for simulaion, how much did the region get extended
    coarsegrain_factor: Binning size for final plot
    plot_CTCF_lines: plot lines ontop of the matrix for each CTCF site
    save_plot: if a string, saves the plot at this file
    '''

    # CTCF site information
    # trim and coarsegrain
    trim = extend_factor
    npoints = len(forw)
    orig_size = int(npoints / (1 + (trim * 2)))
    remove_each = int((npoints - orig_size) / 2)
    forwTrim = forw[remove_each:npoints - remove_each]
    forwTrimC = coarsegrain(forwTrim, coarsegrain_factor)
    revTrim = rev[remove_each:npoints - remove_each]
    revTrimC = coarsegrain(revTrim, coarsegrain_factor)
    forwTrimC[forwTrimC > 1] = 1
    revTrimC[revTrimC > 1] = 1

    # init plot
    fig, ax0 = plt.subplots(ncols=1)

    # set vmax lims
    use_vmax = np.nanpercentile(logarr, max_percentile)
    use_vmin = np.nanmin(logarr)

    # symmetrize logarr
    keep = logarr[np.triu_indices(logarr.shape[0])]
    logarr = logarr.transpose()
    logarr[np.triu_indices(logarr.shape[0])] = keep

    # set up loarr plot
    im0 = ax0.matshow(logarr, cmap=cmap, vmax=use_vmax, vmin=use_vmin)
    ax0.set_title(title, y=1.08)
    # ctcf site plot should extend 1/4 beyond
    ctcf_plot_size = int(logarr.shape[0] * 0.25)
    ax0.set_ylim(logarr.shape[0] - 1, -1 * ctcf_plot_size)
    fig.colorbar(im0, ax=ax0)
    # where on the chart we're plotting
    siteStart = -1 * (ctcf_plot_size / 2)
    siteExpand = (ctcf_plot_size / 2) * 0.9
    # add lines for CTCF sites
    ax0.vlines([np.where(forwTrimC != 0)], ymin=siteStart, ymax=siteStart -
               (forwTrimC[np.where(forwTrimC != 0)] * siteExpand), color='red')
    ax0.vlines([np.where(revTrimC != 0)], ymin=siteStart, ymax=siteStart +
               (revTrimC[np.where(revTrimC != 0)] * siteExpand), color='blue')
    if plot_CTCF_lines:
        # add lines all the way down the plot
        ax0.vlines([np.where(forwTrimC != 0)], ymin=-20,
                   ymax=[np.where(forwTrimC != 0)],
                   color='magenta', lw=0.5, linestyles='dashed')
        ax0.vlines([np.where(revTrimC != 0)], ymin=-20,
                   ymax=[np.where(revTrimC != 0)],
                   color='aqua', lw=0.5, linestyles='dashed')
        ax0.hlines([np.where(forwTrimC != 0)], xmax=logarr.shape[0] - 1,
                   xmin=[np.where(forwTrimC != 0)],
                   color='magenta', lw=0.5, linestyles='dashed')
        ax0.hlines([np.where(revTrimC != 0)], xmax=logarr.shape[0] - 1,
                   xmin=[np.where(revTrimC != 0)],
                   color='aqua', lw=0.5, linestyles='dashed')

    plt.tight_layout()
    if save_plot is not None:
        plt.savefig(save_plot)
    else:
        plt.show(block=False)
