import os
from functools import singledispatch
from pathlib import Path
import imageio
import numpy as np
import scipy
import tensorflow as tf
from deprecated import deprecated  # type: ignore[import]
from multipledispatch import dispatch


def dimString(arr, prefix="", postfix="", keepLength1Dims=False):
    return dimStringFromShape(
        arr.shape, prefix=prefix, postfix=postfix, keepLength1Dims=keepLength1Dims
    )


def dimStringFromShape(shape, prefix="", postfix="", keepLength1Dims=False):
    if keepLength1Dims:
        shape = [x for x in shape]
    else:
        shape = [x for x in shape if x != 1]

    dimStr = ""
    for ii, dim in enumerate(shape[::-1]):
        if ii == 0:
            dimStr += str(dim)
        else:
            dimStr += "x" + str(dim)

    dimStr = prefix + dimStr + postfix

    return dimStr

def convertArrayTypeSafely(arr, dtype, rangeCheck=True):
    if arr.dtype != dtype:
        finfo = np.finfo(dtype)
        assert finfo.min <= np.min(arr), "Value not in the range of {}: {} > {}".format(
            dtype, finfo.min, np.min(arr)
        )
        assert finfo.max >= np.max(arr), "Value not in the range of {}: {} < {}".format(
            dtype, finfo.max, np.max(arr)
        )
        return arr.astype(dtype)
    else:
        return arr


def getDimsOfPath(path, squeeze=False):
    dims = (path.split(".")[-2].split("_")[-1]).split("x")
    dims = [int(dim) for dim in dims[::-1] if dim != ""]
    if squeeze:
        dims = [x for x in dims if x != 1]
    return dims


@deprecated(version="", reason="You should use writeArray")
def writeNumpy(array, path, makeParentsIfNotExist=False, keepLength1DimsInSuffix=False):
    writeArray(
        array,
        path,
        makeParentsIfNotExist=makeParentsIfNotExist,
        keepLength1DimsInSuffix=keepLength1DimsInSuffix,
    )


@dispatch(
    np.ndarray,
    (str, Path),
    makeParentsIfNotExist=bool,
    keepLength1DimsInSuffix=bool,
    dtype=type,
)
def writeArray(
    array, path, makeParentsIfNotExist=False, keepLength1DimsInSuffix=False, dtype=None
):
    path = str(path)

    if path[-4:] != ".raw":
        path += dimString(
            array, prefix="_", postfix=".raw", keepLength1Dims=keepLength1DimsInSuffix
        )

    if dtype is not None:
        array = convertArrayTypeSafely(array, dtype, rangeCheck=True)

    try:
        array.tofile(path)
    except FileNotFoundError as e:
        if makeParentsIfNotExist:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory)
            array.tofile(path)
        else:
            raise e


@dispatch(tf.Tensor, str, makeParentsIfNotExist=bool, keepLength1DimsInSuffix=bool)  # type: ignore[no-redef]
def writeArray(  # noqa: F811
    array, path, makeParentsIfNotExist=False, keepLength1DimsInSuffix=False
):
    writeArray(
        array.numpy(),
        path,
        makeParentsIfNotExist=makeParentsIfNotExist,
        keepLength1DimsInSuffix=keepLength1DimsInSuffix,
    )


def readNumpy(path, dtype=np.float32):
    dims = getDimsOfPath(path)

    return np.fromfile(path, dtype=dtype, count=-1, sep="").reshape(dims)


def NumpyArrayToRaw(array, path, endIsDim=True):
    shape = np.shape(array)
    shape = [x for x in shape if x != 1]

    if endIsDim:
        end = ""
        for dim in range(len(shape)):
            if dim == 0:
                end += "_" + str(shape[-(dim + 1)])
            else:
                end += "x" + str(shape[-(dim + 1)])
        outPath = path + end + ".raw"
    else:
        outPath = path + ".raw"

    array.tofile(outPath)

    return outPath


def numpyToImage_noInterpolation(arr, path, upscaleFactor=4):
    arr = np.repeat(arr, upscaleFactor, axis=0)
    arr = np.repeat(arr, upscaleFactor, axis=1)
    imageio.imwrite(path, arr)


def saveAsNpz(arr, path, keepLength1DimsInSuffix=False, dtype=None):
    if dtype is not None:
        arr = convertArrayTypeSafely(arr, dtype, rangeCheck=True)

    # Append shape information to file name
    dimDesc = dimString(
        arr, prefix="_", postfix="", keepLength1Dims=keepLength1DimsInSuffix
    )

    np.savez_compressed(path + dimDesc, a=arr)


def loadNpz(path, dtype=np.float32):
    try:
        arr_sparse = scipy.sparse.load_npz(path)
        arr = arr_sparse.toarray()

        shape = getDimsOfPath(path)
        arr = arr.reshape(shape)
    except ValueError:
        arr = np.load(path)["a"]

    if dtype is not None:
        if arr.dtype != dtype:
            arr.astype(dtype)

    return arr


def readArray(path, dtype=np.float32):
    path = str(path)
    if path.endswith(".raw"):
        return readNumpy(path, dtype=dtype)
    elif path.endswith(".npz"):
        return loadNpz(path, dtype=dtype)
    else:
        raise ValueError("Cannot determine format of file:\n{}".format(path))