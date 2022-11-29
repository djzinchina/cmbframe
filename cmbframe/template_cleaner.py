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

class Emode_recycler:
    '''
        E-mode recycling method workspace setup.
    '''

    def __init__(self, IQU_in, mask_in, lmax_in=None):

        self.IQU = IQU_in
        
        self.nside = hp.npix2nside(len(self.IQU[0,:]))
        if lmax_in == None :
            self.lmax = 3*self.nside - 1
        else: 
            self.lmax = lmax_in 

        self.msk = np.array(np.copy(mask_in))
        self.__msk_arr = np.array([self.msk, self.msk, self.msk])

        self.__alm = hp.map2alm(np.copy(self.IQU)*self.__msk_arr, lmax=self.lmax, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts/')
        self.B_c = hp.alm2map(np.copy(self.__alm[2]), self.nside, lmax=self.lmax, pol=False, verbose=False) * self.msk

    def compute_template(self):

        TE_lm = np.zeros_like(self.__alm)
        TE_lm[0] = np.copy(self.__alm[0])
        TE_lm[1] = np.copy(self.__alm[1])

        self.IQU_TE = hp.alm2map(TE_lm, self.nside, lmax=self.lmax, pol=True, verbose=False)

        alm_tilde = hp.map2alm(np.copy(self.IQU_TE)*self.__msk_arr, lmax=self.lmax, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')

        self.B_t = hp.alm2map(alm_tilde[2], self.nside, lmax=self.lmax, pol=False, verbose=False) * self.msk

    def __lin_fit(self, x, y, intercept_in=None, slope_in=None):
        # Slope parameter:
        if slope_in == None:
            slope = np.cov(x,y)[0,1] / np.var(x, dtype=np.float64)
        else:
            slope = slope_in

        # Intercept paramter:
        if intercept_in == None:
            intercept = np.mean(y, dtype=np.float64) - slope * np.mean(x, dtype=np.float64)
        else:
            intercept = intercept_in
        return intercept, slope

    def clean_Bmap(self, beta_0=None, beta_1=None, return_fit=False):

        if not hasattr(self, 'B_t'):
            self.compute_template()

        beta_0, beta_1 = self.__lin_fit(np.copy(self.B_t[np.where(self.msk > 0.9)]), np.copy(self.B_c[np.where(self.msk > 0.9)]), intercept_in=beta_0, slope_in=beta_1)

        # print(beta_0,beta_1)

        self.B_f = (self.B_c - beta_0 - (beta_1 * self.B_t))*self.msk 

        if return_fit:
            return beta_0, beta_1

def get_cleanedBmap(map_IQU, mask_bin, lmax_sht=None, beta_0=None, beta_1=None, return_fit=False):
    cleaner = Emode_recycler(map_IQU, mask_bin, lmax_in=lmax_sht)
    cleaner.compute_template()

    if return_fit:
        beta_0, beta_1 = cleaner.clean_Bmap(beta_0=beta_0, beta_1=beta_1, return_fit=True)
        return cleaner.B_f, beta_0, beta_1
    else: 
        cleaner.clean_Bmap(beta_0=beta_0, beta_1=beta_1)
        return cleaner.B_f


def get_residual(recyler, IQU_full, ret_full=False):

    if not hasattr(recyler, 'B_f'):
        recyler.clean_Bmap()

    alm_o = hp.map2alm(np.copy(IQU_full), lmax=recyler.lmax, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
    
    B_o = hp.alm2map(np.copy(alm_o[2]), recyler.nside, lmax=recyler.lmax, pol=False, verbose=False) * recyler.msk

    B_r = np.copy(recyler.B_f) - B_o
    
    if ret_full :
        return B_r, B_o
    else :
        return B_r

def get_leakage(IQU_full, msk_in) :

    nside = hp.npix2nside(len(IQU_full[0,:]))
    lmax = 3*nside - 1

    alm_o = hp.map2alm(np.copy(IQU_full), lmax=lmax, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
    full_TE_lm = np.zeros_like(alm_o)
    full_TE_lm[0] = np.copy(alm_o[0])
    full_TE_lm[1] = np.copy(alm_o[1])

    IQU_TE = hp.alm2map(full_TE_lm, nside, lmax=lmax, pol=True, verbose=False)

    alm_m = hp.map2alm(np.copy(IQU_TE)*[msk_in, msk_in, msk_in], lmax=lmax, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')

    L = hp.alm2map(np.copy(alm_m[2]), nside, lmax=lmax, pol=False, verbose=False) * msk_in

    return L 



        