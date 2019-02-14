import argparse
import numpy as np
from os import makedirs
from os.path import exists, dirname
from useful_3D import *


def main():
    parser = argparse.ArgumentParser(description="Take simulated chromatin structures \
        from the loop extrusion pipleine and process them by trimming\
        the ends of the structure and binning in 3D. Can save \
        the ensemble of structures as individual text files or \
        a binary np array. Options control trim size and bin size \
        but are set as default for simulating the K562 region.")
    parser.add_argument('-i',
                        '--in_folder',
                        required=True,
                        action='store',
                        dest='in_folder',
                        help='Folder with simulated trajectory. Expects a number of filed named \
        block#.dat, where # ranges consistently in integers from \
        1-nsim. If save_format==txt, file is 3 columns sep by whitespace, \
        with a header line marking the number of points. If save_format\
        ==joblib, file is binary saved in joblib by the pipeline.')
    parser.add_argument('-o',
                        '--out_folder',
                        required=True,
                        action='store',
                        dest='out_folder',
                        help='Folder to save complete trajectories. Will be created if does not\
        exist. In the case of saving a binary result \
        file, this will be the filename.')
    parser.add_argument('-t',
                        '--trim',
                        required=False,
                        action='store',
                        default=0.10,
                        dest='trim',
                        help='Fraction of points to trim from the start and end of the structure. \
        Set to 0 to not perform trimming.')
    parser.add_argument('-b',
                        '--bin_size',
                        required=False,
                        action='store',
                        default=50,
                        dest='bin_size',
                        help='Centroid binning will use this many original points to a single\
        bin in the final structure. Set to 0 to not perform any binning.')
    parser.add_argument('-save_text',
                        '--save_text',
                        required=False,
                        action='store_true',
                        dest='save_text',
                        default=False,
                        help='Instead of a binary numpy array,\
                        save processed structures as text files.')
    parser.add_argument('-load_n',
                        '--load_n',
                        required=False,
                        action='store',
                        dest='load_n',
                        default=1,
                        help='Only load every nth structure, due to correlation between \
        structures simulated one after the other. Prelim results \
        show n=200 gives statistically independent structures.')
    parser.add_argument('-load_format',
                        '--load_format',
                        required=False,
                        action='store',
                        dest='load_format',
                        default='joblib',
                        help='Load structures from txt or joblib format.')

    # get args
    args = parser.parse_args()
    in_folder = args.in_folder
    out_folder = args.out_folder
    trim = float(args.trim)
    bin_size = int(args.bin_size)
    save_binary = not args.save_text
    load_n = int(args.load_n)
    load_format = args.load_format
    # done

    assert(args.load_format == 'txt' or args.load_format == 'joblib')
    # make the out_folder
    if not (save_binary) and not (exists(out_folder)):
        print('making output directory: ' + out_folder)
        makedirs(out_folder)
    if save_binary and not (exists(dirname(out_folder))):
        print('making output directory: ' + dirname(out_folder))
        makedirs(dirname(out_folder))

    # load up the strucs
    print('Loading structures...')
    many_struc = load_run_trajectory(
        in_folder, load_format, print_prog=True, load_n=load_n)
    print(many_struc.shape)
    # trim the ends
    print('Trimming...')
    trim_struc = trim_structure_many(many_struc, trim)

    # bin in 3D
    print('Binning in 3D...')
    bin_struc = bin_structure_3D_many(trim_struc, bin_size)

    # save each structure
    if save_binary:
        # save in binary format
        print('Saving in binary format')
        np.save(out_folder, bin_struc)
    else:
        print('Saving individual .dat files')
        save_many_structures(out_folder, bin_struc)

    print('Done :)')


if __name__ == '__main__':
    main()
