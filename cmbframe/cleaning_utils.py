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

__pdoc__ = {}
__pdoc__['compute_binned_cov'] = False

def compute_beam_ratio(beam_nu, beam_0):
    """
    Computes beam ratio to change the resolution/beam smoothing of a single map/alm.

    Parameters
    ----------
    beam_nu : numpy array
        A numpy 1D array of shape [lmax+1], containing the original/native beam of the data. 
        If polarized beam contains either the E component or the B component depending on 
        which map/alm is being targeted. This represents $$B^{T/E/B}_{\\ell}$$ for the 
        different maps in the set.
    beam_0 : numpy array
        A numpy 1D array of shape [lmax+1] representing the beam of the common resolution 
        that is being targetted.

    Returns
    -------
    numpy array
        A numpy 1D array of shape [lmax+1] that contains multiplicative factors 
        to convert map alms to the common resolution. 
    """

    lmax_beam = len(beam_nu)

    ratio_nu = np.zeros((lmax_beam))

    lmax_nonzero = np.max(np.where(beam_nu>0.))+1
    # print(lmax_nonzero)
    ratio_nu[0:lmax_nonzero] = beam_0[0:lmax_nonzero] / beam_nu[0:lmax_nonzero]

    del lmax_beam, lmax_nonzero, beam_nu, beam_0
    return ratio_nu

def beam_ratios(beams_in, beam_0):
    """
    Computes beam ratios to convert a set of maps/alms from their native beam resolution
    to a common beam smoothing. This is a wrapper for compute_beam_ratio function.

    Parameters
    ----------
    beams_in : numpy ndarray
        A numpy 2D array of shape [nu_dim, lmax+1], where nu_dim is the number of different 
        resolution channels. This contains beams for nu_dim channels. If polarized beam contains 
        either the E component or the B component depending on which map/alm is being targeted. 
        This represents $$B^{T/E/B}_{\\ell}$$ for the different maps in the set.
    beam_0 : numpy array
        A numpy 1D array of shape [lmax+1] representing the beam of the common resolution 
        that is being targetted.

    Returns
    -------
    numpy ndarray
        A numpy 2D array of shape [nu_dim,lmax+1] that contains multiplicative factors 
        to convert map alms to the common resolution. 
    """

    nu_dim = len(beams_in[:,0])
    lmax_beam = len(beams_in[0,:])

    ratios = np.zeros((nu_dim,lmax_beam))

    for nu in range(nu_dim):
        ratios[nu,:] = compute_beam_ratio(beams_in[nu,:], beam_0)

    return ratios

# def beam_ratios(beams_in, beam_0):
#     nu_dim = len(beams_in[:,0])
#     lmax_beam = len(beams_in[0,:])

#     ratios = np.zeros((nu_dim,lmax_beam))

#     for nu in range(nu_dim):
#         lmax_nonzero = np.max(np.where(beams_in[nu,:]>0.))+1
#         ratios[nu,0:lmax_nonzero] = beam_0[0:lmax_nonzero] / beams_in[nu,0:lmax_nonzero]

#     return ratios

def calc_binned_cov(alm1, alm2=None):
    ALM = hp.Alm()
    lmax = ALM.getlmax(len(alm1))

    Cl_1x2 = hp.alm2cl(alm1, alms2=alm2)

    ells = np.arange(lmax+1)
    mode_factor = 2.*ells + 1. 
    Cl_1x2 = mode_factor * Cl_1x2

    Cl_binned = np.zeros((lmax+1,))

    for li in range(2, len(Cl_1x2)) :
        limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-7))), 2)
        limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+7))), lmax-1)
        
        Cl_binned[li] = (np.sum(Cl_1x2[limin:limax]) - Cl_1x2[li]) / np.sum(mode_factor[limin:limax]) #(limax - limin) 

    del Cl_1x2 
    return Cl_binned

# def calc_binned_cov(alm1, alm2=None):
#     ALM = hp.Alm()
#     lmax = ALM.getlmax(len(alm1))

#     Cl_1x2 = hp.alm2cl(alm1, alms2=alm2)

#     ells = np.arange(lmax+1)
#     mode_factor = 2.*ells + 1. 
#     Cl_1x2 = mode_factor * Cl_1x2

#     Cl_binned = np.zeros((lmax+1,))

#     for li in range(2, len(Cl_1x2)) :
#             limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 2)
#             limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), lmax-1)
            
#             Cl_binned[li] = np.sum(Cl_1x2[limin:limax]) / np.sum(mode_factor[limin:limax]) 

#     del Cl_1x2 
#     return Cl_binned

def compute_Ncov(Nlms_in, beams=None, com_res_beam=None):
    ALM = hp.Alm()
    lmax = ALM.getlmax(len(Nlms_in[0,:]))
    nu_dim = len(Nlms_in[:,0])
    Ncov_ij = np.zeros((lmax+1, nu_dim, nu_dim))

    if isinstance(beams, (tuple,np.ndarray)) and isinstance(com_res_beam, np.ndarray):
        beam_ratio = beam_ratios(beams, com_res_beam)

    Nlms = []
    for nu in range(nu_dim):
        Nlms.append(hp.almxfl(np.copy(Nlms_in[nu]), beam_ratio[nu]))

    Nlms = np.array(Nlms)

    for nu_1 in range(0, nu_dim) :
        for nu_2 in range(nu_1, nu_dim) :
            if nu_2 == nu_1:
                Ncov_ij[:,nu_1,nu_1] = calc_binned_cov(Nlms[nu_1])
            else:
                Ncov_ij[:,nu_1,nu_2] = Ncov_ij[:,nu_2,nu_1] = calc_binned_cov(Nlms[nu_1], alm2=Nlms[nu_2])

    del Nlms, Nlms_in, beam_ratio
    return Ncov_ij