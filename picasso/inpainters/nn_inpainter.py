import numpy as np


class NN():
	
	def NN_fill(mask, input, niter):
		'''
		mask: binary mask with the hole as 0, and the rest of as 1
		input: input nonmasked data
		niter: number of iterations. The saturation iterations depends on the mask size. 
		       (a mask size of ~300 takes about ~50 iters) 
		'''
	    img = img.copy()
	    mask_pos = np.where(mask==0)
	    h, w = img.shape
	    x, y = mask_pos
	    img[~mask] = np.mean(img*mask)
	    for i in range(niter):
	        for r,c in zip(x,y):
	            try:
	                img[r,c] = img[(r-1):(r+2),(c-1):(c+2)].mean()
	            except IndexError:
	            	print('Mask index out of range')
	                pass
	    return img
	