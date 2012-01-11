"""Construct super-resolution reconstruction of a non-zero yaw satellite data-set.

"""

import numpy as np

from supreme.resolve import yaw_solve, initial_guess_avg
from supreme.config import data_path
from supreme.io import load_vgg, imread, imsave
from supreme.transform import homography
from supreme.noise import dwt_denoise
from supreme.photometry import photometric_adjust

import scipy.ndimage as ndi

import matplotlib.pyplot as plt

import osgeo.gdal as gdal

import sys, os
from optparse import OptionParser

usage = "%prog [options] yaw_image"
parser = OptionParser(usage=usage)
parser.add_option('-y', '--yaw', type=float,
                  help='Satellite yaw angle dureing acquisition.')
parser.add_option('-s', '--scale', type=float,
                  help='Resolution improvement required [default: %default]')
parser.add_option('-d', '--damp', type=float,
                  help='Damping coefficient -- '
                       'suppresses oscillations [default: %default]')
parser.add_option('-m', '--method', dest='method',
                  help='`CG`, `LSQR`, `L-BFGS-B` or `descent`. '
                  'Specifies optimisation algorithm [default: %default]')
parser.add_option('-L', '--norm', type=int,
                  help='The norm used to measure errors. [default: %default]')

parser.set_defaults(scale=2,
                    damp=1e-1,
                    method='CG',
                    norm=2)

(options, args) = parser.parse_args()

yaw = options.yaw / 180. * 3.1415 
scale = options.scale

if options.norm not in (1, 2):
    raise ValueError("Only L1 and L2 error norms are supported.")

if options.norm == 1 and options.method == 'L-BFGS-B':
    import warnings
    warnings.warn("Using the 1-norm with L-BFGS-B may lead to non-convergence.")

yaw_image_filename = parser.largs[0]

dsYawImage = gdal.Open(yaw_image_filename)
dsHighRes = gdal.GetDriverByName("GTIFF").Create('/tmp/out.tif', dsYawImage.RasterXSize, dsYawImage.RasterYSize, dsYawImage.RasterCount, gdal.GDT_Float32)

d = options.__dict__
print "Input Parameters"
print "------------------------"
for k in d:
    print '%s: %s' % (k, d[k])
print "------------------------"


for band in range(1,dsYawImage.RasterCount+1):
    print "Band", band
    image = dsYawImage.GetRasterBand(band).ReadAsArray()
    out = yaw_solve(image, yaw, scale=options.scale, tol=0.1,
                    damp=options.damp, iter_lim=None,
                    method=options.method,
                    norm=options.norm)
    dsHighRes.GetRasterBand(band).WriteArray(out)

dsHighRes.FlushCache()
