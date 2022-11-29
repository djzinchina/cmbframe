#######################################################################
# This file is a part of CMBframe
#
# Cosmic Microwave Background (data analysis) frame(work)
# Copyright (C) 2021  Shamik Ghosh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information about CMBframe please visit 
# <https://github.com/1cosmologist/CMBframe> or contact Shamik Ghosh 
# at shamik@ustc.edu.cn
#
#########################################################################

import numpy as np 
import healpy as hp
from numpy.random import default_rng

# sigma0 of the noise in WMAP maps
# Pixel noise in units of mK may be evaluated from Nobs with the expression sigma = sigma0 / sqrt(Nobs) where
band = ['K','Ka', 'Q', 'V', 'W']#['K',	'Ka',	  'Q',	  'V',	  'W']
sigT = [1.429, 1.466, 2.188, 3.131, 6.544]
sigP = [1.435, 1.472, 2.197, 3.141, 6.560]

def get_wmap_weights(channel):
    try:
        wh = band.index(channel)
        datafile = '/home/doujzh/DATA/WMAP9/wmap_band_iqumap_r9_nineyear_v5/wmap_band_iqumap_r9_9yr_'+channel+'_v5.fits'
        hits = hp.read_map(datafile, field=(0,1,2,3), dtype=np.float64, hdu=2, verbose=False)
        return hits
    except:
        print("ERROR: Channel info not found.")
        exit()

def upscale_hits(hits_map, nside_up):
    hits = np.array(hits_map)
    ndim = hits.ndim 

    nside_wmap = 512
    npix_up = hp.nside2npix(nside_up)

    pix_arr = np.arange(npix_up)
    x, y, z = hp.pix2vec(nside_up, pix_arr)
    pix_wmap = hp.vec2pix(nside_wmap, x, y, z)

    if ndim == 1:
        upscaled_hits = np.zeros((npix_up,))
        upscaled_hits = hits[pix_wmap]
        
    else:
        nmaps = len(hits[:,0])
        upscaled_hits = np.zeros((nmaps, npix_up))
        upscaled_hits = hits[:,pix_wmap]

    return upscaled_hits 


def get_wmap_noise(channel, nside, pol=True, seed=None):
    nside_wmap = 512
    npix_wmap = hp.nside2npix(nside_wmap)

    if nside < nside_wmap:
        nside_o = nside
        nside = nside_wmap
        need_to_downgrade = True
    else:
        need_to_downgrade = False

    npix = hp.nside2npix(nside)

    try:
        wh = band.index(channel)
    except:
        print("ERROR: Check channel selection.")
        exit()
    
    sigmaT = sigT[wh]
    sigmaP = sigP[wh]

    hits = get_wmap_weights(channel)

    pixratio = max(nside / nside_wmap, 1.)

    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)
    if not pol:
        hits_up = upscale_hits(hits[0], nside)
        noise_map = rng.standard_normal(size=(npix,), dtype=np.float64) * (sigmaT * pixratio) / np.sqrt(hits_up)

    else:
        noise_map = rng.standard_normal(size=(3, npix), dtype=np.float64)

        hits_up = upscale_hits(hits[0], nside)

        covar = np.zeros((npix_wmap, 2, 2))
        covar[:,0,0] = np.copy(hits[1])
        covar[:,0,1] = np.copy(hits[2])
        covar[:,1,0] = np.copy(hits[2])
        covar[:,1,1] = np.copy(hits[3])

        covar = np.linalg.inv(covar) * (sigmaP * pixratio)**2

        A = np.linalg.cholesky(covar)

        if nside > nside_wmap:
            A_up = np.zeros((npix, 2, 2))
            A_up[:,0,0] = upscale_hits(A[:,0,0], nside)
            A_up[:,0,1] = upscale_hits(A[:,0,1], nside)
            A_up[:,1,1] = upscale_hits(A[:,1,1], nside)
        else:
            A_up = A

        noise_map[0] *= (sigmaT * pixratio) / np.sqrt(hits_up[0])

        # print(np.transpose(noise_map)[:,1:3].shape)
        QU_noise = np.matmul(A_up, np.transpose(noise_map)[:,1:3].reshape(npix, 2, 1))
        noise_map[1] = QU_noise[:,0,0]
        noise_map[2] = QU_noise[:,1,0]

        del covar, A, A_up, QU_noise

    if need_to_downgrade:
        noise_map = hp.ud_grade(noise_map, nside_o)

    return noise_map
