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

from . import cleaning_utils as cu


class hilc_cleaner:
    '''
        Workspace for harmonic domain ILC method.  
    '''

    def __init__(self, alms_in, beams=None, com_res_beam=None):

        ALM = hp.Alm()
        self.lmax = ALM.getlmax(len(alms_in[0,:]))
        self.nu_dim = len(alms_in[:,0])

        if isinstance(beams, (tuple,np.ndarray)) and isinstance(com_res_beam, np.ndarray):
            self.beam_ratio = cu.beam_ratios(beams, com_res_beam)

        self.alms = []
        for nu in range(self.nu_dim):
            self.alms.append(hp.almxfl(np.copy(alms_in[nu]), self.beam_ratio[nu]))

        self.alms = np.array(self.alms)

    def compute_hilc_weights(self):
        har_cov_ij = np.zeros((self.lmax-1,self.nu_dim,self.nu_dim))
        e_vec = np.ones((1,self.nu_dim))

        har_invcov_ij = np.linalg.pinv(har_cov_ij) 
        
        if np.any(har_invcov_ij == 0.) or np.any(np.isnan(har_invcov_ij)):
            for l in range(0,self.lmax-1):
                for nu_1 in range(0,self.nu_dim):
                    for nu_2 in range(0,self.nu_dim):
                        if har_invcov_ij[l, nu_1, nu_2] == 0.:
                            print("PS Zero for",l, nu_1, nu_2)
                        if np.isnan(har_invcov_ij[l, nu_1, nu_2]):
                            print("PS NaN for", l, nu_1, nu_2)

        self.har_wgts = np.matmul(e_vec, har_invcov_ij) / np.matmul(e_vec,np.matmul(har_invcov_ij, e_vec.T).reshape(self.lmax-1,self.nu_dim,1))

        self.har_wgts = np.swapaxes(np.array(self.har_wgts),0,2)[:,0,:]
        self.har_wgts = np.insert(self.har_wgts, 0, 0., axis=1)
        self.har_wgts = np.insert(self.har_wgts, 0, 0., axis=1)

    def get_cleaned_alms(self):
    
        for nu in range(self.nu_dim):
            if nu == 0 :
                cleaned_alm = hp.almxfl(self.alms[nu], self.har_wgts[nu])

            else :
                cleaned_alm += hp.almxfl(self.alms[nu], self.har_wgts[nu])

        return cleaned_alm

    def get_projected_alms(self, alms_to_proj, adjust_beam=True):

        ALM = hp.Alm()
        lmax_proj = ALM.getlmax(len(alms_to_proj[0,:]))
        nu_dim_proj = len(alms_to_proj[:,0])

        if lmax_proj != self.lmax:
            print("lmax of alms to project does not match lmax of weights.")
            exit()
        elif nu_dim_proj != self.nu_dim:
            print("Number of frequencies of alms to project does not match number of bands in weights.")
            exit()

    
        for nu in range(self.nu_dim):
            if nu == 0 :
                if adjust_beam:
                    cleaned_alm = hp.almxfl(alms_to_proj[nu], self.har_wgts[nu] * self.beam_ratio[nu])
                else:
                    cleaned_alm = hp.almxfl(alms_to_proj[nu], self.har_wgts[nu])

            else :
                if adjust_beam:
                    cleaned_alm += hp.almxfl(alms_to_proj[nu], self.har_wgts[nu] * self.beam_ratio[nu])
                else:
                    cleaned_alm += hp.almxfl(alms_to_proj[nu], self.har_wgts[nu])

        return cleaned_alm

    




