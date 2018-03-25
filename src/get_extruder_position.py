import argparse
from os import makedirs
from os.path import exists
import numpy as np
import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()})
import matplotlib
matplotlib.use("Agg")

# import openmmlib
# import polymerutils
from loop_extrusion_functions import get_forw_rev, do_extruder_position
from useful_plotting import plot_logarr_sites

def main():
    parser = argparse.ArgumentParser(description="Test where loop extruding factors\
   	will be on average, given a file with CTCF sites and parameters.\
   	Currently simulates region Chr21:29372390-31322258 by default.")

    parser.add_argument('-i',
                        '--input_ctcf',
                        required=True,
                        action='store',
                        dest='input_ctcf',
                        help='input CTCF peak file. 5 column, tab separated. Ex:\
        chrom start   end fc  summitDist\
        chr21   29402390    29432389    0.03143   -1\
        chr21   29432390    29462389    0.42000    1')
    parser.add_argument('-save_plot',
                        '--save_plot',
                        required=False,
                        action='store',
                        dest='save_plot',
                        default=None,
                        help='Specify a name to save an image of the plot.\
                        Otherwise will just display.')
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
    parser.add_argument('-chr',
                        '--chromosome',
                        required=False,
                        action='store',
                        dest='chromosome',
                        default=21,
                        help='Chromosome to simulate')
    parser.add_argument('-start',
                        '--simulation_start',
                        required=False,
                        action='store',
                        dest='simulation_start',
                        default=29372390,
                        help='Start of region to simulate.')
    parser.add_argument('-end',
                        '--simulation_end',
                        required=False,
                        action='store',
                        dest='simulation_end',
                        default=31322258,
                        help='End of region to simulate.')
    parser.add_argument('-nsim',
                        '--nsim',
                        required=False,
                        action='store',
                        dest='nsim',
                        default=10,
                        help='Rounds of extruder positioning simulation\
                        default should be fine.')
    parser.add_argument('-bin_size',
                        '--bin_size',
                        required=False,
                        action='store',
                        dest='bin_size',
                        default=50,
                        help='Bin the resulting matrix by taking the average\
                        of this many points.')
    parser.add_argument('-monomer_size',
	                    '--monomer_size',
	                    required=False,
	                    action='store',
	                    dest='monomer_size',
	                    default=600,
	                    help='Change monomer representation in the model. Default of 600bp is \
	    from Jeffs paper. Changing requires adapting lifetime, separation \
	    parameter also. EXPERIMENTAL!')
    parser.add_argument('-plot_CTCF_lines',
                        '--plot_CTCF_lines',
                        required=False,
                        action='store_true',
                        dest='plot_CTCF_lines',
                        default=False,
                        help='Draw lines down through the plot for each CTCF site.')
    parser.add_argument('-plot_title',
                        '--plot_title',
                        required=False,
                        action='store',
                        dest='plot_title',
                        default='log extrusion occupancy',
                        help='Title for the plot.')

    # get args
    args = parser.parse_args()
    input_ctcf = args.input_ctcf
    save_plot = args.save_plot
    SEPARATION = int(args.separation)
    LIFETIME = int(args.lifetime)
    mu = float(args.mu)
    divide_logistic = float(args.divide_logistic)
    extend_factor = float(args.extend)
    monomer_size = int(args.monomer_size)
    mychr = int(args.chromosome)
    mystart = int(args.simulation_start)
    myend = int(args.simulation_end)
    nsim = int(args.nsim)
    bin_size = int(args.bin_size)
    plot_CTCF_lines = args.plot_CTCF_lines
    plot_title = args.plot_title

    # ensure divide_logistic is >0
    if divide_logistic < 0:
        print('divide_logistic error. setting to default=20')
        divide_logistic = 20

    forw, rev = get_forw_rev(input_ctcf, mu=mu,
    divide_logistic=divide_logistic, extend_factor=extend_factor,
    monomer_size=monomer_size, mychr=mychr, mystart=mystart, myend=myend)
    print('CTCF sites on forward: ' + str(len(forw[forw > 0])))
    print('CTCF sites on reverse: ' + str(len(rev[rev > 0])))
    
    # get logarr
    logarr = do_extruder_position(forw, rev, SEPARATION=SEPARATION,
                         LIFETIME=LIFETIME, trim=extend_factor, nsim=nsim,
                         bin_size=bin_size)

    # plot and save
    plot_logarr_sites(logarr, forw, rev, title=plot_title,
                      cmap='viridis', max_percentile=99.9,
                      extend_factor=extend_factor,
                      coarsegrain_factor=bin_size,
                      plot_CTCF_lines=plot_CTCF_lines, save_plot=save_plot)

if __name__ == '__main__':
	main()
