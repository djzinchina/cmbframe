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

# Note that this simulation method disagrees with without filter main pipeline
# noise simulations at power spectrum level if the higher noise areas are not removed.
# Minimum 10 uK-pixel 150 GHz mask suggested for consistency.

import numpy as np 
import healpy as hp
from numpy.random import default_rng


def get_alicpt_sigma(channel, pol=True, scan='deep'):
    if scan is 'deep':
        datafile_I = '/media/doujzh/AliCPT_data/NoiseVarDC1/I_NOISE_'+channel+'_C_1024.fits'
        datafile_P = '/media/doujzh/AliCPT_data/NoiseVarDC1/P_NOISE_'+channel+'_C_1024.fits'

        sigma_I = hp.read_map(datafile_I, field=0, dtype=np.float64, verbose=False)

        if pol:
            sigma_P = hp.read_map(datafile_P, field=0, dtype=np.float64, verbose=False)
            return np.array([sigma_I, sigma_P, sigma_P])
        else:
            return np.array(sigma_I)
    elif scan is 'wide':
        datafile = '/media/doujzh/AliCPT_data/AliCPT_widescan/20211030/WideScan2/AliCPT_1_'+channel+'GHz_NOISE.fits'

        if pol:
            sigma_I, sigma_Q, sigma_U = hp.read_map(datafile, field=None, dtype=np.float64)
            return np.array([sigma_I, sigma_Q, sigma_U])
        else:
            sigma_I= hp.read_map(datafile, field=0, dtype=np.float64)
            return np.array(sigma_I)

def upscale_sigma(sigma, nside):
    nside_ali = 1024
    npix = hp.nside2npix(nside)

    pix_arr = np.arange(npix)
    x,y,z = hp.pix2vec(nside, pix_arr)
    pix_map = hp.vec2pix(nside_ali, x, y, z)

    pixratio = nside / nside_ali

    if sigma.ndim == 1:
        upscaled_sigma = sigma[pix_map] * pixratio
    else:
        upscaled_sigma = sigma[:, pix_map] * pixratio

    return upscaled_sigma

def get_alicpt_noise(channel, nside, pol=True, scan='deep', seed=None):
    nside_ali = 1024

    if nside < nside_ali:
        nside_o = nside
        nside = nside_ali
        need_to_downgrade = True 
    else:
        need_to_downgrade = False

    npix = hp.nside2npix(nside)

    sigma = get_alicpt_sigma(channel, pol=pol, scan=scan)

    if nside > nside_ali:
        sigma = upscale_sigma(sigma, nside)

    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)
        
    if not pol:
        noise_map = rng.standard_normal(size=(npix,), dtype=np.float64) * sigma
    else:
        noise_map = rng.standard_normal(size=(3, npix), dtype=np.float64) * sigma

    if need_to_downgrade:
        noise_map = hp.ud_grade(noise_map, nside_o)

    return np.array(noise_map)