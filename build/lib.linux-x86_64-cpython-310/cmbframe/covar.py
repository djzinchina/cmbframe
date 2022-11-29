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
import cmbframe.covarifast as co
from . import super_pix as sp 

def get_super_nside(wvlt_nside):
    possible_wav_nside = [16,32,64,128,256,512,1024,2048,4096,8192]
    correspo_sup_nside = [1,1,2,2,4,4,8,8,16,16]

    if wvlt_nside in possible_wav_nside :
        super_nside = correspo_sup_nside[np.where(np.array(possible_wav_nside) == wvlt_nside)[0][0]]
    else :
        print('Case not implemented')

    return super_nside

def get_dgrade_nside(wvlt_nside):
    if wvlt_nside / 4 > 1 :
        return int(wvlt_nside / 4)
    else:
        return int(1)

def localcovar(map1, map2, nside_sup, fwhm):
    cov_12 = hp.ud_grade(map1 * map2, nside_sup)

    # print(cov_12.shape)
    lmax_sht = 3 * nside_sup - 1
    bl = hp.gauss_beam(fwhm, lmax=lmax_sht, pol=False)

    smoothed_alm = hp.almxfl(hp.map2alm(cov_12, lmax=lmax_sht, use_weights=True, datapath='/home/doujzh/Healpix_3.70/data', pol=False), bl)

    return hp.alm2map(smoothed_alm, nside_sup) # This is different from GNILC where output is at the same resolution as map.   
    

def dgrade_covar(wav_maps, wav_band, nu_dim_bias=None, ilc_bias=0.03):
    wav_maps = np.array(wav_maps)
    npix_map = hp.get_map_size(wav_maps[0])
    nside_map = hp.npix2nside(npix_map)
    nu_dim = len(wav_maps)

    if nu_dim_bias == None:
        nu_dim_bias = nu_dim

    ell_fac = 2*np.arange(len(wav_band)) + 1

    # print(nu_dim, len(wav_band))
    
    pps = np.sqrt(npix_map * (nu_dim_bias - 1) / ilc_bias / np.sum(ell_fac * wav_band**2.))
    fwhm = pps * np.sqrt(4. * np.pi / npix_map) 
    # print(npix_map, pps, fwhm, nu_dim_bias, ilc_bias, np.sum(ell_fac * wav_band**2.))
    # print(fwhm, np.rad2deg(fwhm)*60)

    nside_sup = get_dgrade_nside(nside_map)
    # print(nside_sup, hp.nside2npix(nside_sup))
    cov_mat = np.zeros((hp.nside2npix(nside_sup), nu_dim, nu_dim))
    for i in range(nu_dim):
        for j in range(i,nu_dim):
            # print(i,j)
            cov_mat[:,i,j] = cov_mat[:,j,i] = localcovar(wav_maps[i], wav_maps[j], nside_sup, fwhm)
    
    return cov_mat

def supix_covar(wav_maps, nside_sup=None):
    wav_maps = np.array(wav_maps)

    nside_map = hp.npix2nside(len(wav_maps[0]))

    if nside_sup == None:
        nside_sup = get_super_nside(nside_map)
    
    spix_groups_arr = sp.load_neighbour_array(nside_sup, '/media/doujzh/AliCPT_data/NILC_neighbours')
    s2nind = sp.npix2spix_map(nside_map, nside_sup)

    cov_mat = co.compute_cov_mat(spix_groups_arr, s2nind, wav_maps)

    return cov_mat