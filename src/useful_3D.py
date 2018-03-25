# useful functions for working with xyz structures and distance matrices
from scipy.spatial.distance import pdist, squareform, euclidean
import numpy as np
from os.path import isfile, join
from os import listdir
import pandas as pd
import joblib


def xyz_to_distance(xyz, diagNan=False):
    """Return pairwise distance matrix from nx3 array"""
    assert(xyz.shape[1] == 3 and len(xyz.shape) == 2)
    to_return = squareform(pdist(xyz, metric='euclidean'))
    if diagNan:
        to_return[np.diag_indices(to_return.shape[0])] = np.nan

    return(to_return)


def xyz_to_distance_many(many_struc, diagNan=False):
    """Return distance matrix for many structures """
    assert(len(many_struc.shape) == 3 and many_struc.shape[2] == 3)
    return(np.array([xyz_to_distance(xyz, diagNan) for xyz in many_struc]))


def xyz_to_contact(xyz, threshold):
    """Return a pairwise contact matrix at a given threshold"""
    this_dist = xyz_to_distance(xyz)
    this_contact = (this_dist <= threshold) * 1.0
    this_contact[np.isnan(this_dist)] = np.nan
    return(this_contact)


def xyz_to_contact_many(many_struc, threshold):
    """Return contact matrix for many structures"""
    assert(len(many_struc.shape) == 3 and many_struc.shape[2] == 3)
    return(np.array([xyz_to_contact(xyz, threshold) for xyz in many_struc]))


def distance_to_contact(distance_mat, threshold):
    """Return contact matrix at threshold from a distance matrix"""
    this_contact = (distance_mat <= threshold) * 1.0
    this_contact[np.isnan(distance_mat)] = np.nan
    return(this_contact)


def distance_to_contact_many(many_dist, threshold):
    """Return contact matrix at threshold from many distance matrix"""
    assert(len(many_dist.shape) == 3)
    return(np.array([distance_to_contact(dist, threshold)
                     for dist in many_dist]))


def load_run_trajectory(in_folder, load_format,
                        print_prog=False, load_n=1, start_ind=1):
    """Return np arrayh from load all the dat files in a simulation folder"""

    # assume all files ending in dat are from the trajectory
    num_df = len([f for f in listdir(in_folder) if (
        isfile(join(in_folder, f)) and f[-3:] == 'dat' and f[0:5] == 'block')])
    # order matters so we call them by name
    df_names = ['block' + str(i) + '.dat'
                for i in range(start_ind, (num_df + start_ind))]
    df_names_full = [join(in_folder, x) for x in df_names]

    if load_format == 'txt':
        # number of points: number of lines - 1
        num_points = sum(1 for line in open(df_names_full[0])) - 1
    elif load_format == 'joblib':
        num_points = joblib.load(df_names_full[0])['data'].shape[0]
    # load every nth structure if desired
    if load_n > 1:
        load_idx = [i * load_n
                    for i in range(int(np.ceil(len(df_names) / load_n)))]
        df_names = [df_names[i] for i in load_idx]
        df_names_full = [df_names_full[i] for i in load_idx]
        num_df = len(df_names)
        print('Only loading every ' + str(load_n) + 'th structure.')
        print('Loading ' + str(len(df_names_full)) + ' total.')

    # matrix to store all the structures
    many_struc = np.zeros([num_df, num_points, 3])
    # loop over file names, load them
    if print_prog:
        for i in range(num_df):
            if ((i + 1) % 10) == 0 and print_prog:
                print('loading ' + str(i + 1) +
                      ' of ' + str(len(df_names_full)))
            if load_format == 'txt':
                many_struc[i] = np.loadtxt(df_names_full[i], skiprows=1)
            elif load_format == 'joblib':
                many_struc[i] = joblib.load(df_names_full[i])['data']
    return(many_struc)


def average_distance_k562(many_dist, diag_nan=True):
    """Return average distance with some potentially NaN bins
    many_dist: np array of shape [nstruc, nbins, nbins]
    diag_nan: return NaN along diagonal
    """
    assert(len(many_dist.shape) == 3)
    nstruc = np.shape(many_dist)[0]
    nbins = np.shape(many_dist)[1]

    count_interactions = nstruc - np.sum(np.isnan(many_dist), axis=0)
    many_dist_nonan = many_dist.copy()
    many_dist_nonan[np.isnan(many_dist)] = 0
    sum_distance = np.sum(many_dist_nonan, axis=0)
    av_distance = sum_distance / count_interactions

    if diag_nan:
        av_distance[np.diag_indices(nbins)] = np.nan
    else:
        av_distance[np.diag_indices(nbins)] = 0
    return(av_distance)


def average_contact_from_distance_k562(many_dist, contact_threshold=500,
                                       diag_nan=True):
    """Return average distance with some potentially NaN bins
    many_dist: np array of shape [nstruc, nbins, nbins]
    diag_nan: return NaN along diagonal
    """
    assert(len(many_dist.shape) == 3)
    nstruc = np.shape(many_dist)[0]
    nbins = np.shape(many_dist)[1]

    count_interactions = nstruc - np.sum(np.isnan(many_dist), axis=0)
    many_dist_nonan = many_dist
    many_dist_nonan[np.isnan(many_dist)] = np.inf
    sum_contact = np.sum(many_dist_nonan < contact_threshold, axis=0)
    av_contact = sum_contact / count_interactions

    if diag_nan:
        av_contact[np.diag_indices(nbins)] = np.nan
    else:
        av_contact[np.diag_indices(nbins)] = 0
    return(av_contact)


def binary_distance_nan(mat):
    """binary distance function that's robust to nans
    compute distance between columns on a number of features (rows)
    NOT DONE
    """

    assert(False)
    # ensure all values are 0 or 1 (or nan)
    nanmask = np.isnan(mat)
    allvals = mat[~nanmask]
    return(allvals)


def centeroid_3D(xyz):
    """Return the centroid of a matrix of 3D points"""
    assert(len(xyz.shape) == 3)
    length = xyz.shape[0]
    sum_x = np.sum(xyz[:, 0])
    sum_y = np.sum(xyz[:, 1])
    sum_z = np.sum(xyz[:, 2])
    return sum_x / length, sum_y / length, sum_z / length

# given a single structure, bin it in 3D using the centroid function
# at the specified resolution


def bin_structure_3D(xyz, bin_size):
    """Return structure binned in 3D using centroid binning
    xyz: np array of shape [nbins, 3]
    bin_size: points to bin via centroid
    """
    assert(xyz.shape[1] == 3 and len(xyz.shape) == 2)
    num_points = xyz.shape[0]
    if num_points % bin_size != 0:
        print('WARNING: number of points is not a multiple\
         of ' + str(bin_size))
        print('Final point will be the average\
         of ' + str(num_points % bin_size))

    # bins to use
    num_bins = int(np.ceil(num_points / bin_size))
    bin_starts = [b * bin_size for b in range(num_bins)]
    bin_ends = [(b * bin_size) - 1 for b in range(1, num_bins + 1)]
    # ensure we dont go past num_points
    if bin_ends[-1] > num_points:
        bin_ends[-1] = num_points

    # store structures in this matrix
    binstruc = np.zeros([num_bins, 3])
    for i in range(num_bins):
        binstruc[i] = centeroid_3D(xyz[bin_starts[i]:bin_ends[i]])
    return(binstruc)

# bin many structures, easy loop


def bin_structure_3D_many(many_struc, bin_size):
    """Return structures binned in 3D using centroid binning
    many_struc: np array of shape [nstruc, nbins, 3]
    bin_size: points to bin via centroid
    """
    assert(many_struc.shape[2] == 3 and len(many_struc.shape) == 2)
    return(np.array([bin_structure_3D(xyz, bin_size) for xyz in many_struc]))

# save many strucs in the .dat format used by the pipeline
# defaut uses block#.dat
# strucs is a n,points,3 np array


def save_many_structures(out_folder, many_struc, basename='block', ext='.dat'):
    """Save many_struc in .dat format used by mirnylab pipeline
    out_folder: where to save structures
    many_struc: np array of shape [nstruc, nbin, 3]

    kwargs:
    basename: starting name of files, which an integer will be appended
    ext: file extension for text file
    """
    assert(many_struc.shape[2] == 3 and len(many_struc.shape) == 2)

    # compose names of bins
    ofNames = [join(out_folder, basename + str(i + 1) + ext)
               for i in range(many_struc.shape[0])]

    # write out using np
    for i in range(many_struc.shape[0]):
        np.savetxt(ofNames[i], many_struc[i], fmt='%.5f',
                   delimiter=' ', header=str(many_struc.shape[1]), comments='')


# trim structures: remove the first and last percentage of points
# xyz is a n*3 np array
# trim is the percentage the structures were extended by in generation
def trim_structure(xyz, trim):
    """Return structure trimmed by removing first and last fraction
    of points
    xyz: np array of shape [nbins, 3]
    trim: fraction of points to remove
    """
    assert(xyz.shape[1] == 3 and len(xyz.shape) == 2)
    npoints = xyz.shape[0]
    origSize = int(npoints / (1 + (trim * 2)))
    removeTotal = npoints - origSize
    if removeTotal % 2 != 0:
        removeLeft = int(np.floor(removeTotal / 2))
        removeRight = int(np.ceil(removeTotal / 2))
    else:
        removeLeft = removeRight = removeTotal // 2

    return(xyz[removeLeft:npoints - removeRight])

# apply trim structure across an ensemble


def trim_structure_many(many_struc, trim):
    """Return trim_structure across many structures"""
    assert(many_struc.shape[2] == 3 and len(many_struc.shape) == 2)
    return(np.array([trim_structure(xyz, trim) for xyz in many_struc]))


def diag_to_exp(norm_vec):
    """Return expected contact matrix from a normalization vector
    Probably from Juicebox or something similar. Contains normalization
    factor at each genomic distance bin.
    """
    # dim of exp is len(norm_vec) + 1
    # 1 on the diagonal
    exp = np.diag([1.0 for i in range(len(norm_vec) + 1)])
    # build array to fill it in
    expArr = [np.repeat(a, b) for
              a, b in zip(norm_vec,
              [i for i in reversed(range(1, len(norm_vec) + 1))])]
    # iteratively fill it in
    for d in range(len(norm_vec)):
        thisInd = [[i for i in range(len(norm_vec) - d)],
                   [i + d + 1 for i in range(len(norm_vec) - d)]]
        exp[thisInd] = expArr[d]
        exp[thisInd[::-1]] = expArr[d]
    return(exp)


def radius_gyration(xyz):
    """Return radius of gyration for a 3D structure"""
    assert(xyz.shape[1] == 3 and len(xyz.shape) == 2)
    # first calculate the centroid
    centroid = centeroid_3D(xyz)
    # root mean squared distance between points and centroid
    dists = [np.square(euclidean(x, centroid)) for x in xyz]
    Rg = np.sqrt(np.sum(dists) / np.shape(xyz)[0])
    return(Rg)


def directionalityIndex(arr, d=5, dist_mat=False):
    """Return the directionality index defined across a matix. DI is defined
    from dixon 2012:
     - A is the number of upstream interactions,
     - B is the number of downstream interactions
     - E expected number of reads under null (A+B)/2
     - calculate A an B for each bin
     - DI of bin 1 is 0
     - triangle of size d along the diagonal, up and down

     args:
     arr: square np array of dim 2

     kwargs:
     d: distance to look up and down for each bin
     dist_mat: are you using this on a distance map?
    """
    nbin = arr.shape[0]
    DIList = []
    for i in range(nbin):
        if (i == 0) or (i == nbin - 1):
            DIList.append(0)
        else:
            newD = np.min([d, i, (nbin - i - 1)])
            # print(str(i), ' ', str(newD))
            upInd = i + newD
            downInd = i - newD
            upVal = arr[i, i + 1:upInd + 1]
            downVal = arr[downInd:i, i]
            if(all(np.isnan(upVal)) or all(np.isnan(downVal))):
                DI = 0
            else:
                if not dist_mat:
                    nanFracA = 1 - (np.sum(np.isnan(upVal)) / len(upVal))
                    nanFracB = 1 - (np.sum(np.isnan(downVal)) / len(downVal))
                    B = np.nansum(upVal) * nanFracB
                    A = np.nansum(downVal) * nanFracA
                    E = (A + B) / 2.
                    DI = ((B - A) / abs(B - A)) * ((np.square(A - E) / E) +
                                                   (np.square(B - E) / E))
                    # print('A: ' + str(A)+ ' B: '+ str(B) +\
                    # ' E: ' + str(E)+ ' DI: ' + str(DI))
                else:
                    # doing on distance map,
                    # probably a single structure
                    B = np.nanmean(upVal)
                    A = np.nanmean(downVal)
                    E = np.nanmean(np.concatenate((upVal, downVal)))
                    DI = ((A - B) / abs(A - B)) * ((np.square(A - E) / E) +
                                                   (np.square(B - E) / E))
            DIList.append(DI)
    return(np.array(DIList))


def load_K562_contact(many_dist_file='/home/ben/ab_local/\
    k562/microscopy_matrices/k562_microscopy_distances.pickle',
                      return_average=False, r=250):
    """Return k562 contact matrix from default file
    kwargs:
    return_average: return average contact matrix instead of many
    r: contact radius
    """
    many_dist = np.load(many_dist_file)
    many_distAv = np.nanmean(many_dist, axis=0)
    many_distAv[np.diag_indices(many_distAv.shape[0])] = np.nan
    manyContact = distance_to_contact_many(many_dist, r)

    if not return_average:
        return(manyContact)
    else:
        manyContactAv = np.nanmean(manyContact, axis=0)
        manyContactAv[np.diag_indices(manyContactAv.shape[0])] = np.nan
        return(manyContactAv)


def load_K562_distance(many_dist_file='/home/ben/ab_local/\
    k562/microscopy_matrices/k562_microscopy_distances.pickle',
                       return_average=False):
    """Return k562 distance matrix from default file
    kwargs:
    return_average: return average contact matrix instead of many
    """
    many_dist = np.load(many_dist_file)
    if not return_average:
        return(many_dist)
    else:
        many_distAv = np.nanmean(many_dist, axis=0)
        many_distAv[np.diag_indices(many_distAv.shape[0])] = np.nan
        return(many_distAv)


# loads HiC contact matrices for our region in K562 from
# a specified set of files
def load_HiC_contact(norm_type='none', obs_exp=False, sum1=True,
                     data_dir='/home/ben/ab_local/k562/hic_data/'):
    """Return HiC contact matrices stored in a default location. Options
    control which files to load.

    kwargs:
    norm_type: one of {none, coverage, coveragesqrt, balanced}
    obs_exp: binary, load observed/expected matrix
    sum1: normalize data so rows sum to 1
    data_dir: directory to load files from
    """
    norm_type = norm_type.lower()
    assert(norm_type in ['none', 'coverage', 'coveragesqrt', 'balanced'])
    # if obs_exp, cant do sum1
    if obs_exp and sum1:
        print('Cant do sum1 noraliztion on the\
               obs_exp matrix, setting it to False')
        sum1 = False
    if norm_type == 'none':
        if obs_exp:
            hicFile = join(data_dir,
                           'k562_myregion_HiC_10kb_bin30kb_Obs_exp_NoNorm.txt')
        else:
            hicFile = join(data_dir, 'k562_myregion\
                _HiC_10kb_bin30kb_NoNorm.txt')
    elif norm_type == 'coverage':
        if obs_exp:
            hicFile = join(data_dir, 'k562_myregion\
                _HiC_10kb_bin30kb_Obs_exp_CoverageNorm.txt')
        else:
            hicFile = join(data_dir, 'k562_myregion\
                _HiC_10kb_bin30kb_CoverageNorm.txt')
    elif norm_type == 'coveragesqrt':
        if obs_exp:
            hicFile = join(data_dir, 'k562_myregion\
                _HiC_10kb_bin30kb_Obs_exp_CoverageSqrtNorm.txt')
        else:
            hicFile = join(data_dir, 'k562_myregion\
                _HiC_10kb_bin30kb_CoverageSqrtNorm.txt')
    elif norm_type == 'balanced':
        if obs_exp:
            hicFile = join(data_dir, 'k562_myregion\
                _HiC_10kb_bin30kb_Obs_exp_BalancedNorm.txt')
        else:
            hicFile = join(data_dir, 'k562_myregion\
                _HiC_10kb_bin30kb_BalancedNorm.txt')

    # load matix
    hicData = pd.DataFrame.as_matrix(pd.read_csv(hicFile, sep='\t', header=0))
    if sum1:
        # normalize HiC data so that off diagonal sums to 1
        # each row here
        hicSum1 = hicData.copy()
        hicSum1[np.diag_indices(np.shape(hicData)[0])] = 0
        hicSum1 = hicSum1 / np.sum(hicSum1, axis=0)
        # diag 0
        hicSum1[np.diag_indices(np.shape(hicData)[0])] = 0
        return(hicSum1)
    else:
        # just set diagonal to zero, or 1 if obs_exp
        if obs_exp:
            hicData[np.diag_indices(np.shape(hicData)[0])] = 1
        else:
            hicData[np.diag_indices(np.shape(hicData)[0])] = 0
        return(hicData)


# returns the bin interval for a genomic coordinate
# specific to work with k562 roi right now, but can be adapted
# to anything.
def genome_to_bin(coord, roundMethod='floor', start=29372390,
                  end=31322258, bin_size=30000):
    """Return the bin interval for a genomic coordinate.
    Defaults to work with my data in k562 roi right now, but can be adapted
    to anything.
    """
    if coord > end:
        raise ValueError('coordinate is past end of region specified')
    if roundMethod == 'floor':
        return np.floor((coord - start) / bin_size)
    elif roundMethod == 'round':
        return np.round((coord - start) / bin_size)
    elif roundMethod == 'ceil':
        return np.ceil((coord - start) / bin_size)
    else:
        return ((coord - start) / bin_size)


def position_file_to_xyz(position_file, nbin=65):
    """Return xyz coordinates from postion_file in a defined format
    file has 5 tab separated columns: uniqueID, read, x, y, z
    and each uuid comes with nbin points.
    """
    positions = pd.read_csv(position_file, delimiter='\t', header=0)
    # should have the same number of positions for each uuid
    uuids = np.unique(positions['uniqueID'])
    if (positions.shape[0] / len(uuids) != nbin):
        raise ValueError('not consistent number of points')

    xyz = np.array([np.array(positions.iloc[i * nbin:(i * nbin) + nbin:, 2:5])
                    for i in range(len(uuids))])
    # xyz2 = np.array([np.array(positions.iloc
    # [i*nbin:(i*nbin)+nbin:,]) for i in range(len(uuids))])
    return(xyz)


def interpolate_NA(xyz):
    """Return xyz with NA bins interpolated linearly.
    first and last bin strategy: continue trajectory from previous
    filled in point.
    """
    assert(xyz.shape[1] == 3 and len(xyz.shape) == 2)
    nbin = xyz.shape[0]
    na_rows = list(np.where(np.isnan(xyz[:, 0]))[0])
    # print('na_rows:' + str(na_rows))

    # Do interpolation of missing
    # intermediate points first, then takcle the end cases
    interpolate_first = False
    interpolate_last = False
    if (0 in na_rows):
        interpolate_first = True
        # run of bins that we need to do at the fist point
        interpolate_first_bins = [0]
        na_rows.remove(0)
        for b in na_rows.copy():
            if b - max(interpolate_first_bins) == 1:
                interpolate_first_bins.append(b)
                na_rows.remove(b)
    if (nbin - 1 in na_rows):
        interpolate_last = True
        interpolate_last_bins = [nbin - 1]
        na_rows.remove(nbin - 1)
        for b in na_rows.copy()[::-1]:
            if b - min(interpolate_last_bins) == -1:
                interpolate_last_bins.append(b)
                na_rows.remove(b)

    # loop over intermediated bins
    while(len(na_rows) > 0):
        # get run of bins to interpolate
        start_bin = na_rows[0]
        interp_bins = [start_bin]
        na_rows.remove(start_bin)
        for b in na_rows:
            if b - max(start_bin) == 1:
                interp_bins.append(b)
                na_rows.remove(b)

        # print('interp_bins:' + str(interp_bins))
        ninterp = len(interp_bins)

        # do interpolation for this set
        before = xyz[min(interp_bins) - 1, ]
        after = xyz[max(interp_bins) + 1, ]
        addToBefore = (after - before) / (ninterp + 1)
        interpolated = [before + (addToBefore * (x + 1))
                        for x in range(ninterp)]
        # put back into list
        xyz[interp_bins, ] = np.round(interpolated, 3)

    # now do starting and ending cases
    if interpolate_first:
        ninterp = len(interpolate_first_bins)
        first_nonNA = max(interpolate_first_bins) + 1
        calculate_trajectory_from = xyz[first_nonNA:first_nonNA + 2, :]
        add_each = (calculate_trajectory_from[0, :] -
                    calculate_trajectory_from[1, :])
        first_interpolated = [xyz[first_nonNA, :] +
                              (add_each * x) for x in range(ninterp, 0, -1)]
        xyz[interpolate_first_bins] = first_interpolated

    if interpolate_last:
        ninterp = len(interpolate_last_bins)
        last_nonNA = min(interpolate_last_bins) - 1
        calculate_trajectory_from = xyz[last_nonNA - 1:last_nonNA + 1, :]
        add_each = (calculate_trajectory_from[1, :] -
                    calculate_trajectory_from[0, :])
        last_interpolated = [xyz[last_nonNA, :] +
                             (add_each * x) for x in range(1, ninterp + 1)]
        xyz[interpolate_last_bins[::-1]] = last_interpolated
    return(xyz)


def test_interpolation():
    """Test cases for interpolation. TODO: Rigorous testing!"""
    xyz1 = np.array(
        [[0, 0, 0],
         [1, 1, 1],
         [np.nan, np.nan, np.nan],
         [3, 3, 3]])
    print(interpolate_NA(xyz1))
    print()
    xyz2 = np.array(
        [[np.nan, np.nan, np.nan],
         [1, 1, 1],
         [2, 2, 2],
         [3, 3, 3]])
    print(interpolate_NA(xyz2))
    print()
    xyz3 = np.array(
        [[0, 0, 0],
         [1, 1, 1],
         [2, 2, 2],
         [np.nan, np.nan, np.nan]])
    print(interpolate_NA(xyz3))
    print()

    xyz4 = np.array(
        [[np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [1, 1, 1],
         [2, 2, 2],
         [3, 3, 3]])
    print(interpolate_NA(xyz4))
    print()

    xyz5 = np.array(
        [[0, 0, 0],
         [1, 1, 1],
         [2, 2, 2],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan]])
    print(interpolate_NA(xyz5))
    print()

    xyz6 = np.array(
        [[0, 0, 0],
         [np.nan, np.nan, np.nan],
         [2, 2, 2],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan]])
    print(interpolate_NA(xyz6))

    xyz7 = np.array(
        [[np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [1, 1, 1],
         [2, 2, 2],
         [3, 3, 3],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan]])
    print(interpolate_NA(xyz7))
    print()

    xyz8 = np.array(
        [[np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [1, 1, 1],
         [np.nan, np.nan, np.nan],
         [3, 3, 3],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan]])
    print(interpolate_NA(xyz8))
    print()


def interpolate_NA_many(many_struc):
    """Return interpolate_NA across many structures."""
    assert(len(many_struc.shape) == 3 and many_struc.shape[2] == 3)
    to_return = [interpolate_NA(x) for x in many_struc]
    to_return = np.array([a for a in to_return if not np.isnan(a).all()])
    return(to_return)
