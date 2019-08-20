#
#
#
#
#   date: 2019-08-20
#   author: XIRAN BAI, GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2019
#



import numpy as np


class NearestNeighbours():
	def __init__ (self,   verbose = False , Niters= 500  ) :
		self.verbose=  verbose
		self.niter  = Niters

		pass
	def setup_input (self, fname_masked  )	 :
		self.X =np.load(fname_masked)
		self.mask = np.int_ ( np.ma.masked_not_equal(self.X,0) .mask )
		pass

	def predict (self ):
		'''
		mask: binary mask with the hole as 0, and the rest of as 1
		self.X: input nonmasked data (1 channel)
		niter: number of iterations. The saturation iterations depends on the mask size.
		       (a mask size of ~300 takes about ~50 iters)
		'''
		"""
		data = self.X.copy()
	    mask_pos = np.where(self.mask==0)
	    x, y = mask_pos
	    data [ ~self.mask] = np.mean(self.X*self.mask)
	    for i in range(niter):
	        for r,c in zip(x,y):
	            try:
	                data[r,c] = data[(r-1):(r+2),(c-1):(c+2)].mean()
	            except IndexError:
	            	print('Mask index out of range')
	                pass
	    return data
		"""
		p = self.X.copy()
		mask_pos = np.where(self.mask==0)

		x, y = mask_pos
		p[np.logical_not( self.mask )] = np.mean(self.X*self.mask)
		for i in range(self.niter):
			for r,c in zip(x,y):
				try:
					p[r,c] = p[(r-1):(r+2),(c-1):(c+2)].mean()
				except IndexError:
					print('Mask index out of range')
					pass
		return  p
