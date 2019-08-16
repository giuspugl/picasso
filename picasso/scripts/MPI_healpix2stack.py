import healpy as hp
import matplotlib
from scipy import special as ss
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gf1
import reproject
from scipy.ndimage.filters import gaussian_filter as gf
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
import argparse
import astropy.io.fits as fits
from mpi4py import MPI


def h2f(hmap,target_header,coord_in='C'):
    #project healpix -> flatsky
    pr,footprint = reproject.reproject_from_healpix(
    (hmap, coord_in), target_header, shape_out=(500,500),
    order='nearest-neighbor', nested=False)
    return pr

#def f2h(flat,target_header,coord_in='C'):
#    #project healpix -> flatsky
#    pr,footprint = reproject.reproject_to_healpix(
#    (flat, target_header),coord_system_out='C', nside=4096,
#    order='nearest-neighbor', nested=False)
#    return pr

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

def ptsrc2heal(ra0,dec0,proj,header,nside):
    #tht,phi = hp.pix2ang(nside,hpix)
    tht0,phi0= rd2tp(ra0,dec0)
    vec      = hp.pixelfunc.ang2vec(tht0,phi0)
    pix      = hp.query_disc(nside,vec,radius=3*0.000290888)
    tht,phi  = hp.pix2ang(nside,pix)
    ra,dec    = tp2rd(tht,phi)

    w         = WCS(header)
    x,y       = w.all_world2pix(ra, dec, 0) #locations to sample from

    tmp       = np.zeros(hp.nside2npix(nside))

    f         = RegularGridInterpolator((np.arange(header.get('NAXIS1')),np.arange(header.get('NAXIS2'))), proj.T, method='linear',fill_value=0,bounds_error=False)

    z         = f(np.c_[x,y])
    tmp[pix] = z
    return tmp

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


def zeropad(cl):
    cl = np.insert(cl,0,0)
    cl = np.insert(cl,0,0)
    return cl



def main(args):

    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()
    glob_ra,glob_dec, _  = np.loadtxt('/scratch/groups/kipac/yomori/MLinpaint/ptsrcS3_2019-08-02.dat',unpack=True)

    localsize = glob_ra.shape[0]/nprocs  ## WARNING:  this MUST  evenly divide!!!!!!

    ra =  glob_ra[slice( rank *localsize ,  (rank +1)* localsize)]
    dec =  glob_dec[slice( rank *localsize ,  (rank +1)* localsize)]

    Nstacks= ra.shape [0]
    Npix = 128
    if args.pol :
        keys = ['T', 'Q', 'U']
        sima = hp.read_map(args.hpxmap  ,field=[0,1,2] )
    else:
        keys = ['T' ]

        sima = [hp.read_map( args.hpxmap) ]

    mask = np.ones_like (sima[0] )
    beam =np.deg2rad( args.beamsize /60.)


    nside = hp.get_nside(sima)
    size_im = {2048: 192.  ,4096 : 64. }
    for i in range(Nstacks):
        sizepatch = size_im[nside]*1. /Npix/60.
        header       = set_header(ra[i],dec[i], sizepatch )

        tht,phi      = rd2tp(ra[i],dec[i])
        vec          = hp.ang2vec( theta = tht,phi =phi )
        pixs         = hp.query_disc(nside,vec,3* beam)
        mask[pixs]   = 0

        for k,j  in  zip(keys, range(len(sima)) ) :
        	np.save(args.stackfile+k+'_{:.5f}_{:.5f}_masked'.format(ra[i],dec[i] ),
                                h2f(mask * sima[j] ,header))
        	np.save(args.stackfile+k+'_{:.5f}_{:.5f}'.format(ra[i],dec[i])  , h2f(sima[j] ,header) )
       
        if i %100 ==0  and rank ==0  :
	    	print("Stacking %d source "%i   )


    comm.Barrier()
    
    if rank ==0 :
        print ("collecting stacks to 1 single file" )
        globT = np.zeros( (Nstacks, 128,128))
        mglobT = np.zeros( (Nstacks, 128,128))
        if args.pol : 
        	globQ = np.zeros( (Nstacks, 128,128))
        	globU = np.zeros( (Nstacks, 128,128))
        	mglobQ = np.zeros( (Nstacks, 128,128))
	        mglobU = np.zeros( (Nstacks, 128,128))

        for i in range(Nstacks):
            globT[i,:,: ] =np.load (args.stackfile+ 'T_{:.5f}_{:.5f}.npy'.format(ra[i],dec[i] ))
            mglobT[i,:,: ] =np.load (args.stackfile+ 'T_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] ))
            if args.pol:
                globQ[i,:,: ] =np.load (args.stackfile+ 'Q_{:.5f}_{:.5f}.npy'.format(ra[i],dec[i] ))
            	mglobQ[i,:,: ] =np.load (args.stackfile+ 'Q_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] ))
            	globU[i,:,: ] =np.load (args.stackfile+ 'U_{:.5f}_{:.5f}.npy'.format(ra[i],dec[i] ))
                mglobU[i,:,: ] =np.load (args.stackfile+ 'U_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] ))
        np.save(args.stackfile+'T_masked',  mglobT)
        np.save(args.stackfile+'T' , globT)
        if args.pol:
            np.save(args.stackfile+'Q_masked',  mglobQ)
            np.save(args.stackfile +'Q', globQ)
            np.save(args.stackfile+'U_masked',  mglobU)
            np.save(args.stackfile+'U' , globU)
    comm.Barrier()

    comm.Disconnect



if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )
	parser.add_argument("--hpxmap" , help='path to the healpix map to be stacked, no extension ' )
	parser.add_argument("--beamsize", help = 'beam size in arcminutes of the input map', type=np.float  )
	parser.add_argument("--stackfile", help='path to the file with stacked maps')
	parser.add_argument("--pol", action="store_true" , default=False )
	args = parser.parse_args()
	main( args)
