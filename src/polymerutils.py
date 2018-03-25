from __future__ import absolute_import, division, print_function, unicode_literals
import six
import warnings
import numpy as np
import joblib
import os
from math import sqrt, sin, cos
import numpy

import scipy, scipy.stats  # @UnusedImport

import base64
import json
import numpy as np
import joblib
import gzip

import io

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert (cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data).decode("ascii")
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return super().default(obj)


def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        todecode = dct['__ndarray__'].encode("ascii")
        data = base64.b64decode(todecode)
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


def joblibToJson(filename, outFilename="auto"):
    data = joblib.load(filename)
    if outFilename == "auto":
        outFilename = filename.replace(".dat", "json.gz")
    json.dump(data, gzip.open("block1.json.gz", 'wb', compresslevel=9), cls=NumpyEncoder)


def loadJson(filename):
    with gzip.open(filename,'rb') as myfile:
        data = myfile.read()
        return json.loads(data.decode(), object_hook=json_numpy_obj_hook)


def scanBlocks(folder, assertContinuous=True):
    if not os.path.exists(folder):
        files = []
    else:
        files = os.listdir(folder)
        files = [i for i in files if i.startswith("block") and i.endswith("dat")]
        files = sorted(files, key=lambda x: int(x[5:-4]))

    keys = np.array([int(i[5:-4]) for i in files])

    if assertContinuous:
        if len(files) > 0:
            assert np.all(np.diff(np.array(keys)) == 1)

    files = [os.path.join(folder, i) for i in files]
    return {"files": files, "keys": keys}


def Cload(filename, center=False):
    """fast polymer loader using weave.inline

    ..warning:: About to be deprecated
    """
    f = open(filename, 'r')
    line = f.readline()
    N = int(line)
    ret = numpy.zeros((3, N), order="C", dtype=numpy.double)
    code = """
    #line 85 "binary_search.py"
    using namespace std;
    FILE *in;
    const char * myfile = filename.c_str();
    in = fopen(myfile,"r");
    int dummy;
    dummy = fscanf(in,"%d",&dummy);
    for (int i = 0; i < N; i++)
    {
    dummy = fscanf(in,"%lf %lf %lf ",&ret[i],&ret[i + N],&ret[i + 2 * N]);
    if (dummy < 3){printf("Error in the file!!!");throw -1;}
    }
    """
    support = """
    #include <math.h>
    """
    from scipy import weave
    try:
        weave.inline(code, ['filename', 'N', 'ret'], extra_compile_args=[
            '-march=native -malign-double'], support_code=support)
    except:
        raise IOError("C code failed to open txt file {0}").format(filename)
    if center == True:
        ret -= numpy.mean(ret, axis=1)[:, None]
    return ret.T


def load(filename, h5dictKey=None):
    """Universal load function for any type of data file"""

    if not os.path.exists(filename):
        raise IOError("File not found :( \n %s" % filename)

    #open and read file at beginning to avoid multiple open/close
    #and to avoid OSerror "too many open files"        
    with open(filename,'rb') as myfile:
        data = myfile.read()
    #data_file = io.StringIO(data)
    data_file= io.BytesIO(data)
    
    try:
        "loading from a joblib file here"
        mydict = dict(joblib.load(data_file))
        data = mydict.pop("data")
        return data

    except:
        pass
    
    
    try:
        "checking for a text file"
        data_file.seek(0)
        line0 = data_file.readline()
        try:
            N = int(line0)
        except (ValueError, UnicodeDecodeError):
            raise TypeError("Cannot read text file... reading pickle file")
        # data = Cload(filename, center=False)
        data = [list(map(float, i.split())) for i in data_file.readlines()]

        if len(data) != N:
            raise ValueError("N does not correspond to the number of lines!")
        return np.array(data)

    except (TypeError, UnicodeDecodeError):
        pass
    

    #try:
    #    data = loadJson(filename)
    #    return data["data"]
    #except:
    #    print("Could not load json")
    #    pass

    #h5dict loading deleted



def save(data, filename, mode="txt", h5dictKey="1", pdbGroups=None):
    data = np.asarray(data, dtype=np.float32)

    h5dictKey = str(h5dictKey)
    mode = mode.lower()

    if mode == "h5dict":
        from mirnylib.h5dict import h5dict
        mydict = h5dict(filename, mode="w")
        mydict[h5dictKey] = data
        del mydict
        return

    elif mode in ["joblib", "json"]:
        metadata = {}
        metadata["data"] = data
        if mode == "joblib":
            joblib.dump(metadata, filename=filename, compress=9)
        else:

            with gzip.open(filename, 'wb') as myfile:
                mystr = json.dumps(metadata,  cls=NumpyEncoder)
                mybytes = mystr.encode("ascii")
                myfile.write(mybytes)

        return

    elif mode == "txt":
        lines = []
        lines.append(str(len(data)) + "\n")

        for particle in data:
            lines.append("{0:.3f} {1:.3f} {2:.3f}\n".format(*particle))
        if filename == None:
            return lines
        elif isinstance(filename, six.string_types):
            with open(filename, 'w') as myfile:
                myfile.writelines(lines)
        elif hasattr(filename, "writelines"):
            filename.writelines(lines)
        else:
            return lines

    elif mode == 'pdb':
        data = data - np.minimum(np.min(data, axis=0), np.zeros(3, float) - 100)[None, :]
        retret = ""

        def add(st, n):
            if len(st) > n:
                return st[:n]
            else:
                return st + " " * (n - len(st) )

        if pdbGroups == None:
            pdbGroups = ["A" for i in range(len(data))]
        else:
            pdbGroups = [str(int(i)) for i in pdbGroups]

        for i, line, group in zip(list(range(len(data))), data, pdbGroups):
            atomNum = (i + 1) % 9000
            segmentNum = (i + 1) // 9000 + 1
            line = [float(j) for j in line]
            ret = add("ATOM", 6)
            ret = add(ret + "{:5d}".format(atomNum), 11)
            ret = ret + " "
            ret = add(ret + "CA", 17)
            ret = add(ret + "ALA", 21)
            ret = add(ret + group[0], 22)
            ret = add(ret + str(atomNum), 26)
            ret = add(ret + "         ", 30)
            #ret = add(ret + "%i" % (atomNum), 30)
            ret = add(ret + ("%8.3f" % line[0]), 38)
            ret = add(ret + ("%8.3f" % line[1]), 46)
            ret = add(ret + ("%8.3f" % line[2]), 54)
            ret = add(ret + (" 1.00"), 61)
            ret = add(ret + str(float(i % 8 > 4)), 67)
            ret = add(ret, 73)
            ret = add(ret + str(segmentNum), 77)
            retret += (ret + "\n")
        with open(filename, 'w') as f:
            f.write(retret)
            f.flush()
    elif mode == "pyxyz":
        with open(filename, 'w') as f:             
            for i in data: 
                filename.write("C {0} {1} {2}".format(*i))
            

    else:
        raise ValueError("Unknown mode : %s, use h5dict, joblib, txt or pdb" % mode)


def rotation_matrix(rotate):
    """Calculates rotation matrix based on three rotation angles"""
    tx, ty, tz = rotate
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))


def msd(data1, data2, rotate=True, N=999999, fullReturn=False):
    """
    An utility to calculate mean square displacement between two polymer conformations
    Parameters
    ----------

    data1, dat2 : Mx3 array
        First conformation
    rotate : bool (true by default)
        Compensate for rotation and displacement between conformation
    N : int (optional)
        If conformations are large, use only M monomers for comparison.
        If N > M (num. of particles), use all monomers (default)
    fullReturn : bool, default=False
        Return information about all displacements

    returns
    -------

    MSD : float
        Diplacement between conformation
    if fullReturn==False: dictionary of results
        See the end of the code for reference
    """
    import scipy.optimize

    warnings.warn("Please use polymerCython.fastMSD; this function will be deprecated")

    def distN(a, b, N=N):

        num = len(a) / N + 1
        newa = a[::num]
        newb = b[::num]
        return numpy.sqrt(numpy.mean((newa - newb) ** 2) * 3)

    def distance(rotate_shift, N=N):
        #    print rotate
        rotate = rotate_shift[:3]
        shift = rotate_shift[3:]
        rotated = numpy.dot(data2, rotation_matrix(rotate))
        shifted = rotated + shift[None, :]
        return distN(data1, shifted, N=N)

    if not rotate:
        return distN(data1, data2, 999999)

    optimal = scipy.optimize.fmin(distance, (0, 0, 0, 0, 0, 0), xtol=1e-4, ftol=1e-4)

    r = numpy.arange(len(data1))
    numpy.random.shuffle(r)
    mydist = distance(optimal, N=999999)
    shuffled = distN(data1, data1[r], N=999999)
    original = distN(data1, data2, N=999999)
    print("shuffled distance = ", shuffled)
    print("original distance =", original)
    print("optimized distance= ", mydist)
    if not fullReturn:
        return mydist
    else:
        return {"optimized": mydist, "original": original,
                "shuffled": shuffled, "angle": optimal[:3],
                "shift": optimal[3:]}


def bondLengths(data):
    bonds = np.diff(data, axis=0)
    return np.sqrt((bonds ** 2).sum(axis=1))


def persistenceLength(data):
    bonds = np.diff(data, axis=0)
    lens = np.sqrt((bonds ** 2).sum(axis=1))
    bondCosines = np.dot(bonds, bonds.T) / lens[:, None] / lens[:, None].T
    avgCosines = np.array([np.diag(bondCosines, i).mean() for i in range(lens.size)])
    truncCosines = avgCosines[:np.where(avgCosines < 1.0 / np.e / np.e)[0][0]]
    slope, intercept, _, _, _ = scipy.stats.linregress(
            list(range(truncCosines.size)), np.log(truncCosines))
    return -1.0 / slope


def generateRandomLooping(length=10000, oneMoverPerBp=1000, numSteps=100):
    N = length
    myarray = np.zeros(N, int)
    movers = []
    onsetRate = length / float(oneMoverPerBp)

    def initMovers():
        for i in movers:
            myarray[i[0]] = 1
            myarray[i[1]] = 1

    def addMovers():
        for _ in range(np.random.poisson(onsetRate)):
            pos = np.random.randint(N - 1)
            if myarray[pos:pos + 2].sum() == 0:
                movers.append((pos, pos + 1))
                myarray[pos:pos + 2] = 1

    def translocateMovers():
        moved = False
        for j, mover in enumerate(movers):
            left, right = mover
            if left > 0:
                if myarray[left - 1] == 0:
                    myarray[left] = 0
                    myarray[left - 1] = 1
                    left = left - 1
                    moved = True
            if right < N - 1:
                if myarray[right + 1] == 0:
                    myarray[right] = 0
                    myarray[right + 1] = 1
                    right = right + 1
                    moved = True
            movers[j] = (left, right)
        return moved

    for _ in range(numSteps):
        addMovers()
        translocateMovers()
    while translocateMovers():
        pass
    return movers


def create_spiral(r1, r2, N):
    """
    Creates a "propagating spiral", often used as a starting conformation.
    Run it with r1=10, r2 = 13, N=5000, and see what it does.
    """
    Pi = 3.141592
    points = []
    finished = [False]

    def rad(phi):
        return phi / (2 * Pi)

    def ang(rad):
        return 2 * Pi * rad

    def coord(phi):
        r = rad(phi)
        return (r * sin(phi), r * cos(phi))

    def fullcoord(phi, z):
        c = coord(phi)
        return [c[0], c[1], z]

    def dist(phi1, phi2):
        c1 = coord(phi1)
        c2 = coord(phi2)
        d = sqrt((c1[1] - c2[1]) ** 2 + (c1[0] - c2[0]) ** 2)
        return d

    def nextphi(phi):
        phi1 = phi
        phi2 = phi + 0.7 * Pi
        mid = phi2
        while abs(dist(phi, mid) - 1) > 0.00001:
            mid = (phi1 + phi2) / 2.
            if dist(phi, mid) > 1:
                phi2 = mid
            else:
                phi1 = mid
        return mid

    def prevphi(phi):

        phi1 = phi
        phi2 = phi - 0.7 * Pi
        mid = phi2

        while abs(dist(phi, mid) - 1) > 0.00001:
            mid = (phi1 + phi2) / 2.
            if dist(phi, mid) > 1:
                phi2 = mid
            else:
                phi1 = mid
        return mid

    def add_point(point, points=points, finished=finished):
        if (len(points) == N) or (finished[0] == True):
            points = np.array(points)
            finished[0] = True
            print("finished!!!")
        else:
            points.append(point)

    z = 0
    forward = True
    curphi = ang(r1)
    add_point(fullcoord(curphi, z))
    while True:
        if finished[0] == True:
            return np.transpose(points)
        if forward == True:
            curphi = nextphi(curphi)
            add_point(fullcoord(curphi, z))
            if (rad(curphi) > r2):
                forward = False
                z += 1
                add_point(fullcoord(curphi, z))
        else:
            curphi = prevphi(curphi)
            add_point(fullcoord(curphi, z))
            if (rad(curphi) < r1):
                forward = True
                z += 1
                add_point(fullcoord(curphi, z))


def create_random_walk(step_size, N, segment_length=1):
    theta = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                      segment_length)
    theta = 2.0 * np.pi * theta[:N]
    u = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                  segment_length)
    u = 2.0 * u[:N] - 1.0
    x = step_size * np.sqrt(1. - u * u) * numpy.cos(theta)
    y = step_size * np.sqrt(1. - u * u) * numpy.sin(theta)
    z = step_size * u
    x, y, z = np.cumsum(x), np.cumsum(y), np.cumsum(z)
    return np.vstack([x, y, z]).T


matlabImported = False


def createFBM(length, H):
    if not matlabImported:
        import mlabwrap
        from mlabwrap import mlab
    x = mlab.wfbm(H, length)
    y = mlab.wfbm(H, length)
    z = mlab.wfbm(H, length)
    result = np.zeros((length, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = z
    return result


def grow_rw(step, size, method="line"):
    """This does not grow a random walk, but the name stuck.

    What it does - it grows a polymer in the middle of the sizeXsizeXsize box.
    It can start with a small ring in the middle (method="standart"),
    or it can start with a line ("method=line").
    If method="linear", then it grows a linearly organized chain from 0 to size.

    step has to be less than size^3

    """
    numpy = np
    t = size // 2
    if method == "standard":
        a = [(t, t, t), (t, t, t + 1), (t, t + 1, t + 1), (t, t + 1, t)]
    elif method == "line":
        a = []
        for i in range(1, size):
            a.append((t, t, i))

        for i in range(size - 1, 0, -1):
            a.append((t, t - 1, i))

    elif method == "linear":
        a = []
        for i in range(0, size + 1):
            a.append((t, t, i))
        if (len(a) % 2) != (step % 2):
            a = a[:-1]

    else:
        raise ValueError("select methon from line, standard, linear")

    b = numpy.zeros((size + 1, size + 1, size + 1), int)
    for i in a:
        b[i] = 1
    for i in range((step - len(a)) // 2):
        # print len(a)
        while True:
            t = numpy.random.randint(0, len(a))
            if t != len(a) - 1:
                c = numpy.abs(numpy.array(a[t]) - numpy.array(a[t + 1]))
                t0 = numpy.array(a[t])
                t1 = numpy.array(a[t + 1])
            else:
                c = numpy.abs(numpy.array(a[t]) - numpy.array(a[0]))
                t0 = numpy.array(a[t])
                t1 = numpy.array(a[0])
            cur_direction = numpy.argmax(c)
            while True:
                direction = numpy.random.randint(0, 3)
                if direction != cur_direction:
                    break
            if numpy.random.random() > 0.5:
                shift = 1
            else:
                shift = -1
            shiftar = numpy.array([0, 0, 0])
            shiftar[direction] = shift
            t3 = t0 + shiftar
            t4 = t1 + shiftar
            if (b[tuple(t3)] == 0) and (b[tuple(t4)] == 0) and (numpy.min(t3) >= 1) and (numpy.min(t4) >= 1) and (
                numpy.max(t3) < size) and (numpy.max(t4) < size):
                a.insert(t + 1, tuple(t3))
                a.insert(t + 2, tuple(t4))
                b[tuple(t3)] = 1
                b[tuple(t4)] = 1
                break
                # print a
    return numpy.array(a)


def _test():
    print("testing save/load")
    a = np.random.random((20000, 3))
    save(a, "bla", mode="txt")
    b = load("bla")
    print(a)
    print(b)
    assert abs(b.mean() - a.mean()) < 0.00001

    save(a, "bla", mode="joblib")
    b = load("bla")
    assert abs(b.mean() - a.mean()) < 0.00001

    #save(a, "bla.json", mode="json")
    #b = loadJson("bla.json")["data"]
    #assert abs(b.mean() - a.mean()) < 0.00001


    #save(a, "bla.json.gz", mode="json")
    #b = load("bla.json.gz")
    #assert abs(b.mean() - a.mean()) < 0.00001

    #save(a, "bla.json", mode="json")
    #b = load("bla.json")
    #assert abs(b.mean() - a.mean()) < 0.00001

    #save(a, "bla", mode="h5dict")
    #b = load("bla")
    #assert abs(b.mean() - a.mean()) < 0.00001

    os.remove("bla")
    os.remove("bla.json.gz")

    print("Finished testing save/load, successful")


def createSpiralRing(N, twist, r=0, offsetPerParticle=np.pi, offset=0):
    """
    Creates a ring of length N. Then creates a spiral
    """
    from mirnylib import numutils
    if not numutils.isInteger(N * offsetPerParticle / (2 * np.pi)):
        print(N * offsetPerParticle / (2 * np.pi))
        raise ValueError("offsetPerParticle*N should be multitudes of 2*Pi")
    totalTwist = twist * N
    totalTwist = np.floor(totalTwist / (2 * np.pi)) * 2 * np.pi
    alpha = np.linspace(0, 2 * np.pi, N + 1)[:-1]
    # print alpha
    twistPerParticle = totalTwist / float(N) + offsetPerParticle
    R = float(N) / (2 * np.pi)
    twist = np.cumsum(np.ones(N, dtype=float) * twistPerParticle) + offset
    # print twist
    x0 = R + r * np.cos(twist)
    z = 0 + r * np.sin(twist)
    x = x0 * np.cos(alpha)
    y = x0 * np.sin(alpha)
    return np.array(np.array([x, y, z]).T, order="C")


def smooth_conformation(conformation, n_avg):
    """Smooth a conformation using moving average.
    """
    if conformation.shape[0] == 3:
        conformation = conformation.T
    new_conformation = np.zeros(shape=conformation.shape)
    N = conformation.shape[0]

    for i in range(N):
        if i < n_avg:
            new_conformation[i] = conformation[:i + n_avg].mean(axis=0)
        elif i >= N - n_avg:
            new_conformation[i] = conformation[-(N - i + n_avg):].mean(axis=0)
        else:
            new_conformation[i] = conformation[i - n_avg:i + n_avg].mean(axis=0)
    return new_conformation


def distance_matrix(d1, d2=None):
    """A brute-force to find a matrix of distances between i-th and j-th
    particles.

    Parameters
    ----------
        d1 : numpy.array
            If the only array supplied, find the pairwise distances.
        d2 : numpy.array
            If supplied, find distances from every point in d1 to
            every point in d2.
    """
    if d2 is None:
        dists = np.zeros(shape=(d1.shape[0], d1.shape[0]))
        for i in range(dists.shape[0]):
            dists[i] = (((d1 - d1[i]) ** 2).sum(axis=1)) ** 0.5
    else:
        dists = np.zeros(shape=(d1.shape[0], d2.shape[0]))
        for i in range(d1.shape[0]):
            dists[i] = (((d2 - d1[i]) ** 2).sum(axis=1)) ** 0.5
    return dists


def endtoend(d):
    """A brute-force method to find average end-to-end distance v.s. separation.
    """
    dists = distance_matrix(d)
    avgdists = np.array([np.diag(dists, i).mean() for i in range(dists.shape[0])])
    return avgdists


def getCloudGeometry(d, frac=0.05, numSegments=1, widthPercentile=50, delta=0):
    """Trace the centerline of an extended cloud of points and determine
    its length and width.

    The function switches to the principal axes of the cloud (e1,e2,e3)
    and applies LOWESS to define the centerline as (x2,x3)=f(x1).
    The length is then determined as the total length of the centerline.
    The width is determined as the median shortest distance from the points of
    clouds to the centerline.
    On top of that, the cloud can be chopped into `numSegments` in the order
    of data entries in `d`. The centerline is then determined independently for
    each segment.

    Parameters
    ----------

    d : np.array, 3xN
        an array of coordinates

    frac : float
        The fraction of all points used to determine the local position and
        slope of the centerline in LOWESS.

    numSegments : int
        The number of segments to split `d` into. The centerline in fit
        independently for each data segment.

    widthPercentile : float
        The width is determined at `widthPercentile` of shortest distances
        from the points to the centerline. The default value is 50, i.e. the
        width is the median distance to the centerline.

    delta : float
        The parameter of LOWESS. According to the documentation:
        "delta can be used to save computations. For each x_i, regressions are
        skipped for points closer than delta. The next regression is fit for the
        farthest point within delta of x_i and all points in between are
        estimated by linearly interpolating between the two regression fits."

    Return
    ------

        (length, width) : (float, float)

    """

    import statsmodels
    import statsmodels.nonparametric
    import statsmodels.nonparametric.smoothers_lowess
    from mirnylib import numutils

    dists = []
    length = 0.0
    for segm in range(numSegments):
        segmd = d[segm * (d.shape[0] // numSegments): (segm + 1) * (d.shape[0] // numSegments)]
        (e1, e2), _ = numutils.PCA(segmd, 2)
        e3 = np.cross(e1, e2)
        xs = np.dot(segmd, e1)
        ys = np.vstack([np.dot(segmd, e2), np.dot(segmd, e3)])
        ys_pred = np.vstack([
            statsmodels.nonparametric.smoothers_lowess.lowess(
                    ys[0], xs, frac=frac, return_sorted=False, delta=10),
            statsmodels.nonparametric.smoothers_lowess.lowess(
                    ys[1], xs, frac=frac, return_sorted=False,
                    delta=10)])
        order = np.argsort(xs)
        fit_d = np.vstack([xs[order],
                           ys_pred[0][order],
                           ys_pred[1][order]]).T

        for i in range(len(xs)):
            dists.append(
                    (((fit_d - np.array([xs[i], ys[0][i], ys[1][i]])) ** 2).sum(axis=1) ** 0.5).min())

        length += (((fit_d[1:] - fit_d[:-1]) ** 2).sum(axis=1) ** 0.5).sum()
    width = np.percentile(dists, widthPercentile)

    return length, width


def _getLinkingNumber(data1, data2, randomOffset=True):
    if len(data1) == 3:
        data1 = numpy.array(data1.T)
    if len(data2) == 3:
        data2 = numpy.array(data2.T)
    if len(data1[0]) != 3:
        raise ValueError
    if len(data2[0]) != 3:
        raise ValueError
    data1 = np.asarray(data1, dtype=np.double, order="C")
    data2 = np.asarray(data2, dtype=np.double, order="C")
    if randomOffset:
        data1 += np.random.random(data1.shape) * 0.0000001
        data2 += np.random.random(data2.shape) * 0.0000001
    olddata = numpy.concatenate([data1, data2], axis=0)
    olddata = numpy.array(olddata, dtype=float, order="C")
    M = len(data1)
    N = len(olddata)
    returnArray = numpy.array([0])
    from scipy import weave

    support = r"""
#include <stdio.h>
#include <stdlib.h>

double *cross(double *v1, double *v2) {
    double *v1xv2 = new double[3];
    v1xv2[0]=-v1[2]*v2[1] + v1[1]*v2[2];
    v1xv2[1]=v1[2]*v2[0] - v1[0]*v2[2];
    v1xv2[2]=-v1[1]*v2[0] + v1[0]*v2[1];
    return v1xv2;
}

double *linearCombo(double *v1, double *v2, double s1, double s2) {
    double *c = new double[3];
    c[0]=s1*v1[0]+s2*v2[0];
    c[1]=s1*v1[1]+s2*v2[1];
    c[2]=s1*v1[2]+s2*v2[2];
        return c;
}

long int intersectValue(double *p1, double *v1, double *p2, double *v2) {
    int x=0;
    double *v2xp2 = cross(v2,p2), *v2xp1 = cross(v2,p1), *v2xv1 = cross(v2,v1);
    double *v1xp1 = cross(v1,p1), *v1xp2 = cross(v1,p2), *v1xv2 = cross(v1,v2);
    double t1 = (v2xp2[2]-v2xp1[2])/v2xv1[2];
    double t2 = (v1xp1[2]-v1xp2[2])/v1xv2[2];
    if(t1<0 || t1>1 || t2<0 || t2>1) {
        free(v2xp2);free(v2xp1);free(v2xv1);free(v1xp1);free(v1xp2);free(v1xv2);
        return 0;
    }
    else {
        if(v1xv2[2]>=0) x=1;
        else x=-1;
    }
    double *inter1 = linearCombo(p1,v1,1,t1), *inter2 = linearCombo(p2,v2,1,t2);
    double z1 = inter1[2];
    double z2 = inter2[2];

    free(v2xp2);free(v2xp1);free(v2xv1);free(v1xp1);free(v1xp2);free(v1xv2);free(inter1);free(inter2);
    if(z1>=z2) return x;
    else return -x;
}
    """

    code = r"""
    #line 1149 "numutils.py"
    double **data = new double*[N];
    long int i,j;
    for(i=0;i<N;i++) {

        data[i] = new double[3];
        data[i][0]=olddata[3*i];
        data[i][1]=olddata[3*i+1];
        data[i][2]=olddata[3*i + 2];
    }

    long int L = 0;
        for(i=0;i<M;i++) {
            for(j=M;j<N;j++) {
                double *v1, *v2;
                if(i<M-1) v1 = linearCombo(data[i+1],data[i],1,-1);
                else v1 = linearCombo(data[0],data[M-1],1,-1);

                if(j<N-1) v2 = linearCombo(data[j+1],data[j],1,-1);
                else v2 = linearCombo(data[M],data[N-1],1,-1);
                L+=intersectValue(data[i],v1,data[j],v2);
                free(v1);free(v2);
            }
        }

    returnArray[0] =  L;

"""
    M, N  # Eclipse warning removal
    weave.inline(code, ['M', 'olddata', 'N', "returnArray"], extra_compile_args=['-march=native -malign-double -O3'],
                 support_code=support)
    return returnArray[0]


def findSimplifiedPolymer(data):
    """a weave.inline wrapper for polymer simplification code
    Calculates a simplified topologically equivalent polymer ring"""

    if len(data) != 3:
        data = numpy.transpose(data)
    if len(data) != 3:
        raise ValueError("Wrong dimensions of data")
    datax = numpy.array(data[0], np.longdouble, order="C")
    datay = numpy.array(data[1], np.longdouble, order="C")
    dataz = numpy.array(data[2], np.longdouble, order="C")
    N = len(datax)
    ret = numpy.array([1])
    datax, datay, dataz, N  # eclipse warning removal
    code = r"""
    #line 290 "binary_search.py"
    int M = 0;
    int k1;
    int sum = 0;
    int t=0,s=0,k=0;
    int turn=0;
    bool breakflag;
    float maxdist;
    int a;
    position=vector<point>(N);
    newposition=vector<point>(N);

    for (i=0;i<N;i++)
    {
    position[i].x = datax[i] +  0.000000000001*(rand()%1000);
    position[i].y = datay[i] + 0.00000000000001*(rand()%1000);
    position[i].z  = dataz[i] +  0.0000000000001*(rand()%1000);
    }
    todelete = vector <int> (N);
    for (i=0;i<N;i++) todelete[i] == -2;
    for (int xxx = 0; xxx < 1000; xxx++)
        {
        maxdist = 0;
        for (i=0;i<N-1;i++)
        {
        if (dist(i,i+1) > maxdist) {maxdist = dist(i,i+1);}
        }
        turn++;
        M=0;
        for (i=0;i<N;i++) todelete[i] = -2;
        for (int j=1;j<N-1;j++)  //going over all elements trying to delete
            {
            breakflag = false; //by default we delete thing

            for (k=0;k<N;k++)  //going over all triangles to check
                {
                long double dd = dist(j,k);
                if (dd  <  2 * maxdist)
                {

                if (k < j-2 || k > j+1)
                    {
                    if (k < N-1) k1 = k+1;
                    else k1 = 0;
                    sum = intersect(position[j-1],position[j],position[
                        j+1],position[k],position[k1]);
                    if (sum!=0)
                        {
                        //printf("intersection at %d,%d\n",j,k);
                        breakflag = true; //keeping thing
                        break;
                        }
                    }
		        }
		else
		{
			k+= max(((int)((float)dd/(float)maxdist )- 3), 0);
		}
                }
            if (breakflag ==false)
            {
            todelete[M++] = j;
            position[j] = (position[j-1] + position[j+1])* 0.5;
            //printf("%d will be deleted at %d\n",j,k);
            j++;
            //break;
            }
            }
        t = 0;//pointer for todelete
        s = 0;//pointer for newposition
        if (M==0)
            {
            break;
            }
        for (int j=0;j<N;j++)
            {
            if (todelete[t] == j)
                {
                t++;
                continue;
                }
            else
                {
                newposition[s++] = position[j];
                }
            }
        N = s;
        M = 0;
        t = 0;
        position = newposition;
        }
    ret[0] = N;

    for (i=0;i<N;i++)
    {
    datax[i]  = position[i].x;
    datay[i]  = position[i].y;
    dataz[i]  = position[i].z;
    }
    """
    support = r"""
#line 400 "binary_search.py"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <ctime>
#include <omp.h>
#include <stdio.h>
using namespace std;
struct point{
    long double x,y,z;
    point operator + (const point &p) const {
        return (point) {x+p.x, y+p.y, z+p.z};
    }
    point operator - (const point &p) const {
        return (point) {x-p.x, y-p.y, z-p.z};
    }
/* cross product */
    point operator * (const point &p) const {
        return (point) {y*p.z - z*p.y,
                        z*p.x - x*p.z,
                        x*p.y - y*p.x};
    }
    point operator * (const long double &d) const {
        return (point) {d*x, d*y, d*z};
    }

    point operator / (const long double &d) const {
        return (point) {x/d, y/d, z/d};
    }
};

vector <point> position;
vector <point> newposition;
vector <int> todelete;
int N;
int i;
long double dist(int i,int j);
long double dotProduct(point a,point b);
int intersect(point t1,point t2,point t3,point r1,point r2);

inline long double sqr(long double x){
    return x*x;
}
inline double dist(int i,int j){
return sqrt(dotProduct((position[i]-position[j]),(position[i]-position[j])));
}

inline long double dist(point a,point b){
    return sqr(a.x-b.x)+sqr(a.y-b.y)+sqr(a.z-b.z);
}

inline long double dotProduct(point a,point b){
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

int intersect(point t1,point t2,point t3,point r1,point r2)
{
point A,B,C,D,n;
int r;
long double det,t,u,v,c1,d1,d2,d3;
B = t2 - t1;
C = t3 - t1;
D = r2 - t1;
A = r2 - r1;

d1 = (B.y*C.z-C.y*B.z);
d2 = (B.x*C.z-B.z*C.x);
d3 = (B.x*C.y-C.x*B.y);
det = A.x*d1-A.y*d2+A.z*d3;
if (det == 0) return 0;
if (det >0){
t = D.x*d1-D.y*d2+D.z*d3;
if (t<0 || t>det) return 0;
u = A.x*(D.y*C.z-C.y*D.z)-A.y*(D.x*C.z-D.z*C.x)+A.z*(D.x*C.y-C.x*D.y);
if (u<0 || u>det) return 0;
v = A.x*(B.y*D.z-D.y*B.z)-A.y*(B.x*D.z-B.z*D.x)+A.z*(B.x*D.y-D.x*B.y);
if (v<0 || v>det || (u+v)>det) return 0;
//printf("\n%lf,%lf,%lf, ",t/det,u/det,v/det);
n = B*C;
c1 = dotProduct(r1-t1,n);
if (c1>0) return 1;
else return -1;
}
else{
t = D.x*d1-D.y*d2+D.z*d3;
if (t>0 || t<det) return 0;
u = A.x*(D.y*C.z-C.y*D.z)-A.y*(D.x*C.z-D.z*C.x)+A.z*(D.x*C.y-C.x*D.y);
if (u>0 || u<det) return 0;
v = A.x*(B.y*D.z-D.y*B.z)-A.y*(B.x*D.z-B.z*D.x)+A.z*(B.x*D.y-D.x*B.y);
if (v>0 || v<det || (u+v)<det) return 0;
//printf("\n%lf,%lf,%lf, ",t/det,u/det,v/det);
n = B*C;
c1 = dotProduct(r1-t1,n);
if (c1>0) return 1;
else return -1;
}
}
//DNA conformation
"""
    from scipy import weave
    weave.inline(code, ['datax', 'datay', 'dataz', 'N', 'ret'],
                 extra_compile_args=['-malign-double'], support_code=support)

    data = numpy.array([datax, datay, dataz]).T

    return data[:ret[0]]


def _mutualSimplify(data1, data2):
    """a weave.inline wrapper for polymer simplification code
    Calculates a simplified topologically equivalent polymer ring"""

    if len(data1) != 3:
        data1 = numpy.transpose(data1)
    if len(data1) != 3:
        raise ValueError("Wrong dimensions of data")
    if len(data2) != 3:
        data2 = numpy.transpose(data2)
    if len(data2) != 3:
        raise ValueError("Wrong dimensions of data")

    datax1 = numpy.array(data1[0], float, order="C")
    datay1 = numpy.array(data1[1], float, order="C")
    dataz1 = numpy.array(data1[2], float, order="C")

    datax2 = numpy.array(data2[0], float, order="C")
    datay2 = numpy.array(data2[1], float, order="C")
    dataz2 = numpy.array(data2[2], float, order="C")

    N1 = len(datax1)
    N2 = len(datax2)

    ret = numpy.array([1, 1])
    datax1, datay1, dataz1, datax2, datay2, dataz2, N1, N2  # eclipse warning removal
    code = r"""
    #line 264 "binary_search.py"
    int M = 0;
    int sum = 0;
    int t=0,s=0,k=0, k1;
    int turn=0;
    bool breakflag;

    int a;
    position1=vector<point>(N1);
    newposition1=vector<point>(N1);

    position2=vector<point>(N2);
    newposition2=vector<point>(N2);


    for (i=0;i<N1;i++)
    {
    position1[i].x = datax1[i] +  0.000000000000001*(rand()%1000);
    position1[i].y = datay1[i] +0.00000000000000001*(rand()%1000);
    position1[i].z  = dataz1[i] +  0.00000000000000001*(rand()%1000);
    }

    for (i=0;i<N2;i++)
    {
    position2[i].x = datax2[i] +  0.000000000000001*(rand()%1000);
    position2[i].y = datay2[i] +0.0000000000000000001*(rand()%1000);
    position2[i].z  = dataz2[i] +  0.0000000000000000001*(rand()%1000);
    }

    todelete1 = vector <int> (N1);
    todelete2 = vector <int> (N2);

    for (i=0;i<N1;i++) todelete1[i] == -2;
    for (i=0;i<N2;i++) todelete2[i] == -2;

    for (int ttt = 0; ttt < 1; ttt++)
        {
        turn++;
        M=0;
        for (i=0;i<N1;i++) todelete1[i] = -2;
        for (i=0;i<N2;i++) todelete2[i] = -2;

        for (int j=1;j<N1-1;j++)  //going over all elements trying to delete
            {

            breakflag = false; //by default we delete thing
            for (k=0;k<N1;k++)  //going over all triangles to check
                {
                if (k < j-2 || k > j+1)
                    {
                    if (k < N1 - 1) k1 = k + 1;
                    else k1 = 0;
                    sum = intersect(position1[j-1],position1[j],position1[
                        j+1],position1[k],position1[k1]);
                    if (sum!=0)
                        {
                        //printf("intersection at %d,%d\n",j,k);
                        breakflag = true; //keeping thing
                        break;
                        }
                    }
                }

            if (breakflag == false)
            {
            for (k=0;k<N2;k++)  //going over all triangles to check
                {
                    if (k < N2 - 1) k1 = k + 1;
                    else k1 = 0;
                    sum = intersect(position1[j-1],position1[j],position1[
                        j+1],position2[k],position2[k1]);
                    if (sum!=0)
                        {
                        //printf("crossintersection at %d,%d\n",j,k);
                        breakflag = true; //keeping thing
                        break;
                        }
                }
             }

            if (breakflag ==false)
            {
            todelete1[M++] = j;
            position1[j] = (position1[j-1] + position1[j+1])* 0.5;
            //printf("%d will be deleted at %d\n",j,k);
            j++;
            //break;
            }
            }
        t = 0;//pointer for todelete
        s = 0;//pointer for newposition
        if (M==0)
            {
            break;
            }
        for (int j=0;j<N1;j++)
            {
            if (todelete1[t] == j)
                {
                t++;
                continue;
                }
            else
                {
                newposition1[s++] = position1[j];
                }
            }
        N1 = s;
        M = 0;
        t = 0;
        position1 = newposition1;
        }

    ret[0] = N1;
    ret[1] = N2;

    for (i=0;i<N1;i++)
    {
    datax1[i]  = position1[i].x;
    datay1[i]  = position1[i].y;
    dataz1[i]  = position1[i].z;
    }
    for (i=0;i<N2;i++)
    {
    datax2[i]  = position2[i].x;
    datay2[i]  = position2[i].y;
    dataz2[i]  = position2[i].z;
    }

    """
    support = r"""
#line 415 "binary_search.py"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <ctime>
#include <omp.h>
#include <stdio.h>
using namespace std;
struct point{
    double x,y,z;
    point operator + (const point &p) const {
        return (point) {x+p.x, y+p.y, z+p.z};
    }
    point operator - (const point &p) const {
        return (point) {x-p.x, y-p.y, z-p.z};
    }
/* cross product */
    point operator * (const point &p) const {
        return (point) {y*p.z - z*p.y,
                        z*p.x - x*p.z,
                        x*p.y - y*p.x};
    }
    point operator * (const double &d) const {
        return (point) {d*x, d*y, d*z};
    }

    point operator / (const double &d) const {
        return (point) {x/d, y/d, z/d};
    }
};

vector <point> position1;
vector <point> newposition1;
vector <int> todelete1;
int N1;
vector <point> position2;
vector <point> newposition2;
vector <int> todelete2;
int N2;


int i;
double dist1(int i,int j);
double dist2(int i,int j);
double dotProduct(point a,point b);
int intersect(point t1,point t2,point t3,point r1,point r2);

inline double sqr(double x){
    return x*x;
}
inline double dist1(int i,int j){
return sqrt(dotProduct((position1[i]-position1[j]),(position1[i]-position1[j])));
}

inline double dist2(int i,int j){
return sqrt(dotProduct((position2[i]-position2[j]),(position2[i]-position2[j])));
}

inline double dist(point a,point b){
    return sqr(a.x-b.x)+sqr(a.y-b.y)+sqr(a.z-b.z);
}

inline double dotProduct(point a,point b){
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

int intersect(point t1,point t2,point t3,point r1,point r2)
{
point A,B,C,D,n;
int r;
double det,t,u,v,c1,d1,d2,d3;
B = t2 - t1;
C = t3 - t1;
D = r2 - t1;
A = r2 - r1;

d1 = (B.y*C.z-C.y*B.z);
d2 = (B.x*C.z-B.z*C.x);
d3 = (B.x*C.y-C.x*B.y);
det = A.x*d1-A.y*d2+A.z*d3;
if (det == 0) return 0;
if (det >0){
t = D.x*d1-D.y*d2+D.z*d3;
if (t<0 || t>det) return 0;
u = A.x*(D.y*C.z-C.y*D.z)-A.y*(D.x*C.z-D.z*C.x)+A.z*(D.x*C.y-C.x*D.y);
if (u<0 || u>det) return 0;
v = A.x*(B.y*D.z-D.y*B.z)-A.y*(B.x*D.z-B.z*D.x)+A.z*(B.x*D.y-D.x*B.y);
if (v<0 || v>det || (u+v)>det) return 0;
//printf("\n%lf,%lf,%lf, ",t/det,u/det,v/det);
n = B*C;
c1 = dotProduct(r1-t1,n);
if (c1>0) return 1;
else return -1;
}
else{
t = D.x*d1-D.y*d2+D.z*d3;
if (t>0 || t<det) return 0;
u = A.x*(D.y*C.z-C.y*D.z)-A.y*(D.x*C.z-D.z*C.x)+A.z*(D.x*C.y-C.x*D.y);
if (u>0 || u<det) return 0;
v = A.x*(B.y*D.z-D.y*B.z)-A.y*(B.x*D.z-B.z*D.x)+A.z*(B.x*D.y-D.x*B.y);
if (v>0 || v<det || (u+v)<det) return 0;
//printf("\n%lf,%lf,%lf, ",t/det,u/det,v/det);
n = B*C;
c1 = dotProduct(r1-t1,n);
if (c1>0) return 1;
else return -1;
}
}
//DNA conformation
"""
    from scipy import weave
    weave.inline(code, ['datax1', 'datay1', 'dataz1', 'N1',
                        'datax2', 'datay2', 'dataz2', 'N2', 'ret'],
                 extra_compile_args=['-malign-double'], support_code=support)

    data1 = numpy.array([datax1, datay1, dataz1]).T
    data2 = numpy.array([datax2, datay2, dataz2]).T

    return data1[:ret[0]], data2[:ret[1]]


def mutualSimplify(a, b, verbose=False):
    if verbose:
        print("Starting mutual simplification of polymers")
    while True:
        la, lb = len(a), len(b)
        if verbose:
            print(len(a), len(b), "before; ", end=' ')
        a, b = _mutualSimplify(a, b)
        if verbose:
            print(len(a), len(b), "after one; ", end=' ')
        b, a = _mutualSimplify(b, a)
        if verbose:
            print(len(a), len(b), "after two; ")

        if (len(a) == la) and (len(b) == lb):
            if verbose:
                print("Mutual simplification finished")
            return a, b


def getLinkingNumber(a, b, randomOffset=True):
    a, b = mutualSimplify(a, b)
    return _getLinkingNumber(a, b, randomOffset)


def _testMutualSimplify():
    for _ in range(10):
        np = numpy
        mat = np.random.random((3, 3))
        a = grow_rw(2000, 14)
        b = grow_rw(2000, 14)
        a = np.dot(a, mat)
        b - np.dot(b, mat)
        a = a + np.random.random(a.shape) * 0.0001
        b = b + np.random.random(b.shape) * 0.0001
        c1 = getLinkingNumber(a, b, False)
        a, b = mutualSimplify(a, b, verbose=False)
        c2 = getLinkingNumber(a, b)
        print("simplified from 2000 to {0} and {1}".format(len(a), len(b)))
        print("Link before: {0}, link after: {1}".format(c1, c2))
        if c1 != c2:
            print("Test failed! Linking numbers are different")
            return -1
    for _ in range(10):
        np = numpy
        mat = np.random.random((3, 3))
        a = create_random_walk(1, 3000)
        b = create_random_walk(1, 1000)

        a = np.dot(a, mat)
        b = np.dot(b, mat)
        a = a + np.random.random(a.shape) * 0.0001
        b = b + np.random.random(b.shape) * 0.0001

        c1 = _getLinkingNumber(a, b, False)
        a, b = mutualSimplify(a, b, verbose=False)
        c2 = _getLinkingNumber(a, b)
        print("simplified from 3000 and 1000 to {0} and {1}".format(len(a), len(b)))
        print("Link before: {0}, link after: {1}".format(c1, c2))
        if c1 != c2:
            print("Test failed! Linking numbers are different")
            return -1

    print('Test finished successfully')


def testLinkingNumber():
    a = np.random.random((3, 1000))
    b = np.random.random((3, 1000))
    for i in range(100):
        mat = np.random.random((3, 3))
        na = np.dot(mat, a)
        nb = np.dot(mat, b)
        print(getLinkingNumber(na, nb, randomOffset=False))

# testLinkingNumber()
