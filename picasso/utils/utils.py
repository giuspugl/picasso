#
#
#
#
#   date: 2019-08-20
#   author: YUUKI OMORI , GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2019   Giuseppe Puglisi    gpuglisi@stanford.edu
#



import reproject
import numpy as np
import astropy.io.fits as fits


def h2f(hmap,target_header,coord_in='C'):
    """
    Returns a target square submap from a projected  HEALPIX map, given a WCS header,
    using :py:mod:`reproject` package

    **Parameters**

    - ``hmap`` : {array}
        healpix map
    - ``target_header``:
        header defined from :func:`set_header`



    """
    pr,footprint = reproject.reproject_from_healpix(
    (hmap, coord_in), target_header, shape_out=(500,500),
    order='nearest-neighbor', nested=False)
    return pr

def f2h(flat,target_header,nside,coord_in='C'):
    """
    Returns a  HEALPIX map projected  and the footprint of a flat one , given a WCS header,
    using :py:mod:`reproject` package

    **Parameters**

    - ``flat`` : {2D array}
        flat  map
    - ``target_header``:
        header defined from :func:`set_header`
    - ``nside``:{int}
        nside of output healpix map


    """

    pr,footprint = reproject.reproject_to_healpix(
    (flat, target_header),coord_system_out='C', nside=nside ,
    order='nearest-neighbor', nested=False)
    return pr, footprint

def rd2tp(ra,dec):
    """
    Convert ``ra,dec -> theta,phi``
    """
    tht = (-dec+90.0)/180.0*np.pi
    phi = ra/180.0*np.pi
    return tht,phi

def tp2rd(tht,phi):
    """
    Convert ``theta,phi -> ra,dec``
    """
    ra  = phi/np.pi*180.0
    dec = -1*(tht/np.pi*180.0-90.0)
    return ra,dec


def set_header(ra,dec, pixelsize ,Npix=128 ):
    """
    Sets the WCS header needed to perform the projection with :func:`h2f` and :func:`f2h`.

    **Parameters**

    - ``ra,dec`` : {float}
        coordinates of the center of the patch
    - ``pixelsize``:{float}
        the size of the pixels of the reprojected flat map  in units of ``deg/pixel``
    - ``Npix``:{int}
        number of pixels in one side of the flat map


    """

    hdr = fits.Header()
    hdr.set('SIMPLE' , 'T')
    hdr.set('BITPIX' , -32)
    hdr.set('NAXIS'  ,  2)
    hdr.set('NAXIS1' ,  Npix)
    hdr.set('NAXIS2' ,  Npix )
    hdr.set('CRVAL1' ,  ra)
    hdr.set('CRVAL2' ,  dec)
    hdr.set('CRPIX1' ,  Npix/2. +.5)
    hdr.set('CRPIX2' ,  Npix/2. +.5 )
    hdr.set('CD1_1'  , pixelsize )
    hdr.set('CD2_2'  , -pixelsize )
    hdr.set('CD2_1'  ,  0.0000000)
    hdr.set('CD1_2'  , -0.0000000)
    hdr.set('CTYPE1'  , 'RA---ZEA')
    hdr.set('CTYPE2'  , 'DEC--ZEA')
    hdr.set('CUNIT1'  , 'deg')
    hdr.set('CUNIT2'  , 'deg')
    hdr.set('COORDSYS','icrs')
    return hdr


def MinMaxRescale(x,a=0,b=1):
    """
    Performs  a MinMax Rescaling on an array `x` to a generic range :math:`[a,b]`.
    """
    xresc = (b-a)*(x- x.min() )/(x.max() - x.min() ) +a
    return xresc
def StandardizeFeatures(x) :
    """
    Standardizes  an array `x` to have average ``0``, and std. deviation ``1`` .
    """
    return (x - x.mean()) /(x.std()  )

def return_intersection(hist_test, hist_true ):
    """
    Estimate the intersection between two histograms.
    """

    minima = np.minimum(hist_test, hist_true )
    intersection = np.true_divide(np.sum(minima), np.sum(hist_true ))
    return intersection
