import numpy as np


class NN():
	
	def NN_fill(mask, data, niter):
		'''
		mask: binary mask with the hole as 0, and the rest of as 1
		data: input nonmasked data (1 channel)
		niter: number of iterations. The saturation iterations depends on the mask size. 
		       (a mask size of ~300 takes about ~50 iters) 
		'''
	    data = data.copy()
	    mask_pos = np.where(mask==0)
	    h, w = data.shape
	    x, y = mask_pos
	    data[~mask] = np.mean(data*mask)
	    for i in range(niter):
	        for r,c in zip(x,y):
	            try:
	                data[r,c] = data[(r-1):(r+2),(c-1):(c+2)].mean()
	            except IndexError:
	            	print('Mask index out of range')
	                pass
	    return data
	