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
import pymaster as nmt

def setup_bins(nside, lmax_o, bsz_0=30 , loglims=[[1.3]], fixed_bins=False, is_Dell=True):
# 
# Set scaling as [scaling, lmax_scale] for each choice. 
# If same scaling wanted for entire ell range set only scaling.
# Thus loglims is a tuple if signgle or double element lists: 
# [[sca_1, lmax_sca1],[sca_2, lmax_sca2],...] 
#

    if fixed_bins :
        bin_sz = bsz_0
        b = nmt.NmtBin(nside, nlb=bin_sz, lmax=lmax_o, is_Dell=is_Dell)
        leff = b.get_effective_ells()

        return b, leff
    else :
        i = 0
        bin_sz = []
        ell_min = [] 
        # leff = []
        while True:
            if i == 0:
                ell_min_d = 2
                ell_max_d = bsz_0
            else :
                ell_min_d = ell_max_d + 1
                
                for i in range(len(loglims)):

                    if len(loglims[i]) == 1:
                        ell_max_d = np.int(np.maximum(ell_min_d + bsz_0, np.ceil(loglims[i][0] * ell_min_d)))
                        break
                    elif len(loglims[i]) == 2:
                        if ell_min_d < loglims[i][1]:
                            ell_max_d = np.int(np.maximum(ell_min_d + bsz_0, np.ceil(loglims[i][0] * ell_min_d)))
                            break

            if ell_max_d > lmax_o:
                break
            else:
                ell_min.append(ell_min_d)
                bin_sz.append(np.int(ell_max_d - ell_min_d + 1))
                # leff.append((ell_min_d + ell_max_d) / 2.)
                # print(ell_min[i], ell_max_d, LL[i])
                i = i + 1
        bin_sz = np.array(bin_sz)
        ell_min = np.array(ell_min)
        # leff = np.array(leff)

        ells = np.arange(3 * nside, dtype='int32') 
        wgts = np.zeros(ells.shape)
        bpws = -1 + np.zeros_like(ells)

        for j in range(0, len(bin_sz)):
            dum_i = ell_min[j]
            dum_f = ell_min[j] + bin_sz[j]
            # print(dum_i, dum_f)
            bpws[dum_i:dum_f] = j
            wgts[dum_i:dum_f] = 1. / bin_sz[j]  

        b = nmt.NmtBin(nside=nside, bpws=bpws, ells=ells, weights=wgts, lmax=lmax_o, is_Dell=is_Dell)
        leff = b.get_effective_ells()

        return b, leff, bin_sz, ell_min

def binner(Dell_in, lmax, bin_sz, leff, is_Cell = False, fixed_bins=False):
    ell = np.arange(lmax+1)
    Dell_factor = ell * (ell + 1.) / 2. / np.pi

    Dell = np.copy(Dell_in[0:lmax+1])

    if is_Cell :
        Dell = Dell_factor[0:lmax+1] * Dell    

    Dell_binned = []

    if fixed_bins :
        bsz= bin_sz * np.ones(len(leff), dtype=np.int32)
    else :
        bsz = bin_sz

    # print(bsz)
    lmin_i = 2
    lmax_i = lmin_i + bsz[0]

    for i in range(0,len(leff)):
        # print(lmin_i, lmax_i)                                               
        dummy = np.sum(Dell[lmin_i:lmax_i])  / (lmax_i - lmin_i)
        
        Dell_binned.append(dummy)  

        if i < len(leff)-1:
            lmin_i = lmax_i
            lmax_i = lmin_i + bsz[i+1]    
        
    del dummy, Dell
    return np.array(Dell_binned)

def bin_error(Dl_in, fsky_apo, llmin, bsz, is_Cell = True) :
    if is_Cell :
        lmax_in = len(Dl_in)
        ells = np.arange(lmax_in)
        Dell_factor = ells * (ells + 1.) / 2. / np.pi 
        # print(Dell_factor)
        Dl_ub = Dell_factor * np.copy(Dl_in) / fsky_apo
    else :
        Dl_ub = np.copy(Dl_in) / fsky_apo

    err_Dl = []

    for i in range(len(bsz)) :
        
        err_bin = np.std( Dl_ub[ llmin[i]:llmin[i]+bsz[i] ] )  / np.sqrt( bsz[i] )
        err_Dl.append(err_bin)

        # print(np.std( Dl_ub[ llmin[i]:llmin[i]+bsz[i] ] ), np.std( Dl_ub[ llmin[i]:llmin[i]+bsz[i] ] ) / np.mean( Dl_ub[ llmin[i]:llmin[i]+bsz[i] ] ))
    
    err_Dl = np.array(err_Dl)
    return err_Dl


def map2coupCl_nmt(map_in, mask_in, map_in2=None, beam2=None, lmax_sht=-1, beam=None, masked_on_input=False, bins=None, return_wsp=False):
    field = nmt.NmtField(mask_in, [map_in], beam=beam, lmax_sht=lmax_sht, masked_on_input=masked_on_input)

    if return_wsp:
        wsp = nmt.NmtWorkspace()

    if isinstance(map_in2, (list, tuple, np.ndarray)):
        field2 = nmt.NmtField(mask_in, [map_in2], beam=beam2, lmax_sht=lmax_sht, masked_on_input=masked_on_input)

        coup_Cl = nmt.compute_coupled_cell(field, field2)

        if return_wsp: wsp.compute_coupling_matrix(field, field2, bins)
    else:
        coup_Cl = nmt.compute_coupled_cell(field, field)
        if return_wsp: wsp.compute_coupling_matrix(field, field, bins)

    if return_wsp:

        return coup_Cl, wsp 
    else:
        return coup_Cl

def map2Cl_nmt(map_in, mask_in, bins, map_in2=None, lmax_sht=-1, beam=None, beam2=None, masked_on_input=False, return_wsp=False, reuse_wsp=None, noise_Cl=None, bias_Cl=None):
    if isinstance(reuse_wsp, type(nmt.NmtWorkspace())):
        coup_Cl = map2coupCl_nmt(map_in, mask_in, map_in2=map_in2, beam2=beam2, lmax_sht=lmax_sht, beam=beam, masked_on_input=masked_on_input, bins=bins)
        Cls = reuse_wsp.decouple_cell(coup_Cl, cl_bias=bias_Cl, cl_noise=noise_Cl)

        return Cls
    else:
        coup_Cl, wsp = map2coupCl_nmt(map_in, mask_in, map_in2=map_in2, beam2=beam2, lmax_sht=lmax_sht, beam=beam, masked_on_input=masked_on_input, bins=bins, return_wsp=True)

        Cls = wsp.decouple_cell(coup_Cl, cl_bias=bias_Cl, cl_noise=noise_Cl)

        if return_wsp:
            return Cls, wsp 
        else:
            return Cls