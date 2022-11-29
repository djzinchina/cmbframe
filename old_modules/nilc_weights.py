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
import matplotlib
matplotlib.use('Agg')
import healpy as hp 
import multiprocessing as mp 
import joblib as jl 
import concurrent.futures as cf

# from . import super_pix as sp 
from . import super_pix2 as sp 

class nilc_weights_wsp_himem:
    '''
        Workspace class for NILC weights calculation workspace.
        This class has lot of memory requirements making it unsuitable for smaller systems.
    '''
    def __init__(self, wvlt_maps, pre_indices=None, super_nside=None):
        possible_wav_nside = [16,32,64,128,256,512,1024,2048,4096,8192]
        correspo_sup_nside = [1,1,2,2,4,4,8,8,16,16]

        self.nu_dim = len(wvlt_maps)
        self.__wav_maps = wvlt_maps

        wvlt_nside = hp.npix2nside(len(wvlt_maps[0]))

        if super_nside == None:
            if wvlt_nside in possible_wav_nside :
                # print(np.where(np.array(possible_wav_nside) == wvlt_nside))
                super_nside = correspo_sup_nside[np.where(np.array(possible_wav_nside) == wvlt_nside)[0][0]]
            else :
                print('Case not implemented')

        self.nside_map = wvlt_nside
        self.nside_sup = super_nside

        if isinstance(pre_indices, tuple):
            self.__ind = pre_indices
        else :
            self.__ind = sp.get_map_indices(self.nside_map, self.nside_sup)

        self.__s2nind = sp.npix2spix_map(self.nside_map, self.nside_sup)
    

    def __get_cov(self, spix, nu_i, nu_j):

        return np.sum(self.__wav_maps[nu_i][self.__ind[spix]]*self.__wav_maps[nu_j][self.__ind[spix]]) - np.sum(self.__wav_maps[nu_i][np.where(self.__s2nind == spix)]*\
                self.__wav_maps[nu_j][np.where(self.__s2nind == spix)])

    def __get_covmat_spix(self, spix):

        nu_vec = np.arange(self.nu_dim)

        cov_mat = np.zeros((self.nu_dim,self.nu_dim))

        get_cov_vec = np.vectorize(self.__get_cov, otypes=[np.float64], cache=False)

        for nu_i in nu_vec:
            cov_mat[nu_i,:] = get_cov_vec(spix, nu_i, nu_vec)

        # for nu_i in nu_vec:
        #     for nu_j in range(nu_i, len(nu_vec)):
        #         cov_mat[nu_i, nu_j] = cov_mat[nu_j, nu_i] = self.__get_cov(spix, nu_i, nu_j)

        return cov_mat

    def get_weights(self, mix_vec=None):
        if isinstance(mix_vec, (np.ndarray, list)):
            if len(mix_vec) == self.nu_dim:
                e_vec = np.array(mix_vec)
            else:
                print("ERROR: Mixing vector length does not match nu_dim.")
        else:
            e_vec = np.ones((self.nu_dim,), dtype=np.float64)

        npix_super = hp.nside2npix(self.nside_sup)

        n_cores = mp.cpu_count()
        covmat_map = jl.Parallel(n_jobs=n_cores)(jl.delayed(self.__get_covmat_spix)(spix) for spix in range(npix_super))

        # covmat_map = np.zeros((npix_super, self.nu_dim, self.nu_dim))
        # covmap_vec = np.vectorize(self.__get_cov, otypes=[np.float64], cache=False)
        # spix_arr = np.arange(npix_super)

        # for i in range(self.nu_dim):
        #     for j  in range(i, self.nu_dim):
        #         covmat_map[:,i,j] = covmat_map[:,j,i] = covmap_vec(spix_arr, i, j)

        covmat_map = np.array(covmat_map)
        # print(covmat_map.shape)

        covinv_map = np.linalg.pinv(covmat_map)
        covinv_map[np.isnan(covinv_map)] = 0.

        del covmat_map

        weights_smap = np.matmul(e_vec, covinv_map) / np.matmul(e_vec, np.matmul(covinv_map, e_vec.T).reshape(npix_super,self.nu_dim,1))

        # print(self.nu_dim, np.array(weights_smap).shape)
        del covinv_map

        w2s_map = sp.npix2spix_map(self.nside_map, self.nside_sup)

        # hp.mollview(w2s_map)
        # plt.show()

        # print(self.nside_map, self.nside_sup)

        weights_wav = np.zeros((self.nu_dim, hp.nside2npix(self.nside_map),))

        # print(weights_wav.shape, w2s_map.shape)

        for spix in range(npix_super) :
            # print(np.where(w2s_map == spix)[0].shape)
            # hp.mollview(w2s_map[np.where(w2s_map == spix)[0]], cmap=plt.cm.plasma)
            # plt.show()
            # print(weights_smap[spix].shape)
            pix_map = list(np.where(w2s_map == spix)[0])
            # print(weights_wav[:,pix_map].shape)
            weights_wav[:,pix_map] = weights_smap[spix].reshape(self.nu_dim,1)*np.ones((self.nu_dim,len(pix_map)))

        del w2s_map, weights_smap

        return weights_wav

    def get_indices(self):
        return self.__ind


class nilc_weights_wsp_slo:
    '''
        NILC workspace to calculate weights. This class is suitable for low RAM systems. 
        This has overhead as some functions are recalculated, instead of being stored in memory.
    '''

    def __init__(self, wvlt_maps, super_nside=None):
        possible_wav_nside = [16,32,64,128,256,512,1024,2048,4096,8192]
        correspo_sup_nside = [1,1,2,2,4,4,8,8,16,16]

        self.nu_dim = len(wvlt_maps)
        self.__wav_maps = wvlt_maps

        wvlt_nside = hp.npix2nside(len(wvlt_maps[0]))

        if super_nside == None:
            if wvlt_nside in possible_wav_nside :
                # print(np.where(np.array(possible_wav_nside) == wvlt_nside))
                super_nside = correspo_sup_nside[np.where(np.array(possible_wav_nside) == wvlt_nside)[0][0]]
            else :
                print('Case not implemented')

        self.nside_map = wvlt_nside
        self.nside_sup = super_nside 

        self.w2s_map = sp.npix2spix_map(self.nside_map, self.nside_sup)

        self.__covmat_map = np.array(np.zeros((hp.nside2npix(super_nside), self.nu_dim, self.nu_dim))).copy()
        self.__covmat_map.setflags(write=True)

    def __get_cov(self, spix, nu_i, nu_j):
        ind = sp.get_mapindx_spix(self.nside_sup, spix, self.w2s_map)
        return np.sum(self.__wav_maps[nu_i][ind]*self.__wav_maps[nu_j][ind])


    def __update_covmat(self, spix):
        nu_vec = np.arange(self.nu_dim)

        get_cov_vec = np.vectorize(self.__get_cov, otypes=[np.float64], cache=False)

        for nu_i in nu_vec:
            self.__covmat_map[spix, nu_i,:] = get_cov_vec(spix, nu_i, nu_vec)
    
    def get_weights(self):
        e_vec = np.ones((self.nu_dim,), dtype=np.float64)

        npix_super = hp.nside2npix(self.nside_sup)

        # n_cores = mp.cpu_count()
        # garbage = jl.Parallel(n_jobs=n_cores)(jl.delayed(self.__update_covmat)(spix) for spix in range(npix_super))

        # del garbage

        with cf.ThreadPoolExecutor(max_workers=47) as executor :
            futures = {executor.submit(self.__update_covmat, spix) for spix in range(npix_super)}
            cf.wait(futures)

        covinv_map = np.linalg.pinv(self.__covmat_map)

        del self.__covmat_map

        weights_smap = np.matmul(e_vec, covinv_map) / np.matmul(e_vec, np.matmul(covinv_map, e_vec.T).reshape(npix_super,self.nu_dim,1))

        # print(self.nu_dim, np.array(weights_smap).shape)
        del covinv_map

        weights_wav = np.zeros((self.nu_dim, hp.nside2npix(self.nside_map),))

        # print(weights_wav.shape, w2s_map.shape)

        for spix in range(npix_super) :
            # print(np.where(w2s_map == spix)[0].shape)
            # hp.mollview(w2s_map[np.where(w2s_map == spix)[0]], cmap=plt.cm.plasma)
            # plt.show()
            # print(weights_smap[spix].shape)
            pix_map = list(np.where(self.w2s_map == spix)[0])
            # print(weights_wav[:,pix_map].shape)
            weights_wav[:,pix_map] = weights_smap[spix].reshape(self.nu_dim,1)*np.ones((self.nu_dim,len(pix_map)))

        del weights_smap

        return weights_wav

def get_projected_wav(wavmap_in, band_weights_in):
    proj_wav = []

    nu_dim = len(wavmap_in)
    nbands = len(wavmap_in[0])

    nbands_wgt = len(band_weights_in)
    nu_dim_wgt = len(band_weights_in[0])

    if nu_dim != nu_dim_wgt:
        print("Error: Shape error -- nu_dim mismatch")
    
    if nbands != nbands_wgt:
        print("Error: Shape error -- nbands mismatch")

    for nu in range(nu_dim):
        if nu == 0 :
            for band in range(nbands):
                proj_wav.append(band_weights_in[band][nu] * wavmap_in[nu][band])           
        else :
            for band in range(nbands):
                proj_wav[band] += band_weights_in[band][nu] * wavmap_in[nu][band]

    return proj_wav