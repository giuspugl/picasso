#	date: 2019-09-02
#   author: XIRAN BAI
#   Copyright (C) 2019   Xiran Bai    xiranbai@stanford.edu
#
#   visualization for contextual attention adopted from:
#    https://github.com/JiahuiYu/generative_inpainting
#
#   date: 2019-08-20
#   author:  Jiahui Yu
#   python3.6
#   Copyright (C) 2018


import numpy as np
import tensorflow as tf
import neuralgym as ng
import healpy as hp
import argparse
import cv2
import os
from matplotlib import pyplot as plt
from generative_inpainting_model import InpaintCAModel
from contextual_attention_gan import ContextualAttention
from generative_inpainting_ops import (
	contextual_attention, make_color_wheel,
	compute_color, flow_to_image,flow_to_image_tf
)


def ca_vis(args):
    rate = 2
    stride = 1
    grid = rate*stride

    i = np.load(args.inpainted)
    i = np.expand_dims(i, 2)
    h, w, _ = i.shape
    i = i[:h//grid*grid, :w//grid*grid, :]
    i = np.expand_dims(i, 0)

    m=np.load(args.masked)
    m = np.expand_dims(m, 2)
    h, w, _ = m.shape
    m = m[:h//grid*grid, :w//grid*grid, :]
    m = np.expand_dims(m, 0)

    mask = np.ma.masked_not_equal(m,0) .mask
    mask  = np.int_(mask )

    with tf.Session() as sess:
        bt = tf.constant(i, dtype=tf.float32)
        ft = tf.constant(m, dtype=tf.float32)
        mask = tf.constant(mask, dtype=tf.float32)
        _, flow = contextual_attention(
            ft, bt, stride=stride, rate=rate,
            training=False, fuse=False, mask=mask)
        flow = sess.run(flow)
        cv2.imwrite(args.output, flow[0]*100)

def flat_map_vis(args):
	inpainted = np.load(args.inpainted)
	masked = np.load(args.masked)
	original = np.load(args.original)
	vmax = np.max([np.max(inpainted),np.max(masked),np.max(original)])
	vmin = np.min([np.min(inpainted),np.min(masked),np.min(original)])
	plt.subplot(1, 3, 1)
	plt.imshow(original, vmin=vmin, vmax=vmax)
	plt.title('original')

	plt.subplot(1, 3, 2)
	plt.imshow(masked, vmin=vmin, vmax=vmax)
	plt.title('masked')

	plt.subplot(1, 3, 3)
	plt.imshow(inpainted, vmin=vmin, vmax=vmax)
	plt.title('inpainted')
	plt.savefig(args.output)

def healpix_map_vis(args):
	if args.inpainted.endswith('.fits')==False:
		print('Please provide in the .fits file for healpix map visualization')
		return

	healpix_map = hp.read_map(args.inpainted)
	hp.mollview(healpix_map,min=90,max=110)
	plt.savefig(args.output)


if __name__=="__main__":
	parser = argparse.ArgumentParser( description='provide visualization for the inpainted map')
	parser.add_argument("--inpainted" , help='path and name to the inpainted map' )
	parser.add_argument("--masked" , help='path and name to the masked map' )
	parser.add_argument("--original" , help='path and name to the original map' )
	parser.add_argument("--output", help = 'visualization output dir and name')
	parser.add_argument("--vis_method", help = 'choose visualization method from flow_map, flat_map, and healpix_map')
	args = parser.parse_args()

	if args.vis_method == 'flow_map':
		ca_vis(args)
	elif args.vis_method == 'flat_map':
		flat_map_vis(args)
	elif args.vis_method == 'healpix_map':
		healpix_map_vis(args)
	else:
		print('Please provide the visualization method from flow_map, flat_map, or healpix_map')
	