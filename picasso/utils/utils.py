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
    #project healpix -> flatsky
    pr,footprint = reproject.reproject_from_healpix(
    (hmap, coord_in), target_header, shape_out=(500,500),
    order='nearest-neighbor', nested=False)
    return pr

def f2h(flat,target_header,nside,coord_in='C'):
    #project flatsky->healpix
    pr,footprint = reproject.reproject_to_healpix(
    (flat, target_header),coord_system_out='C', nside=nside ,
    order='nearest-neighbor', nested=False)
    return pr, footprint 

def rd2tp(ra,dec):
    """
    Convert ra,dec -> tht,phi
    """
    tht = (-dec+90.0)/180.0*np.pi
    phi = ra/180.0*np.pi
    return tht,phi

def tp2rd(tht,phi):
    """
    Convert tht,phi -> ra,dec
    """
    ra  = phi/np.pi*180.0
    dec = -1*(tht/np.pi*180.0-90.0)
    return ra,dec


def set_header(ra,dec, size_patch ,Npix=128 ):
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
    hdr.set('CD1_1'  , size_patch )
    hdr.set('CD2_2'  , -size_patch )
    hdr.set('CD2_1'  ,  0.0000000)
    hdr.set('CD1_2'  , -0.0000000)
    hdr.set('CTYPE1'  , 'RA---ZEA')
    hdr.set('CTYPE2'  , 'DEC--ZEA')
    hdr.set('CUNIT1'  , 'deg')
    hdr.set('CUNIT2'  , 'deg')
    hdr.set('COORDSYS','icrs')
    return hdr


def numpy2png (arr ):
    image  = np.uint8( 255 *  arr )
    # replicate image to the 3 channels
    image = image [:,:,None] * np.ones(3, dtype=int)[None, None, :]
    return image

def setup_input ( fname_masked, seed= 123456789, method = 'Deep-Prior'  ):
    maskdmap=np.load(fname_masked)
    holemask = np.ma.masked_not_equal(maskdmap,0) .mask
    maxval = maskdmap[holemask].max() ; minval = maskdmap[holemask].min()
    maskdmap = np.expand_dims(np.expand_dims( maskdmap, axis=0), axis=-1)
    maskdmap = (maskdmap -minval) / (maxval - minval)
    if  method =='Deep-Prior':
        randstate= np.random.RandomState(seed)
        noisemap =     randstate.uniform( size=maskdmap.shape )
        return [ maskdmap, noisemap  , minval, maxval]
    else :
         return maskdmap,   minval, maxval
