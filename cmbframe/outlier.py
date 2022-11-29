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

def smoothing_filter(fwhm, fsky, mode='HFI', band='100', lmax=None):
    if mode == 'HFI':
        Cl_diffuse = np.loadtxt('/media/doujzh/AliCPT_data/ps_processing_data/Cl_tot_HFI'+band+'_debeamed_fsky-corr_inK_CMB.dat')
    elif mode == 'COM':
        Cl_diffuse = np.loadtxt('/media/doujzh/AliCPT_data/ps_processing_data/Cl_cmb-sync-dust_Commander_debeamed_fsky-corr_inK_CMB.dat')
    else:
        print("ERROR: Only HFI100 (HFI) or Commander (COM) modes are supported presently.")
        exit()

    # Nl = np.loadtxt('/media/doujzh/AliCPT_data/ps_processing_data/Nl_HFI'+band+'_debeamed_fsky-corr_inK_CMB.dat')

    if lmax != None:
        lmax_filter = min(lmax, len(Cl_diffuse)-1)
    else:
        lmax_filter = len(Cl_diffuse)-1

    Cl_tot = ( Cl_diffuse[:lmax_filter+1] ) * fsky  # + Nl[:lmax_filter+1]

    beam = hp.gauss_beam(np.deg2rad(fwhm / 60.), lmax=lmax_filter)

    Wl = np.zeros((lmax_filter+1,))

    index_nonzero = max(np.where(beam>0)[0])
    # print(np.where(beam>0)[0],index_nonzero)
    Wl[2:index_nonzero] = 1. / beam[2:index_nonzero] / Cl_tot[2:index_nonzero]

    return Wl 

def change_beam(map_in, fwhm_in, fwhm_out, lmax=None, EorB=False):
    if np.array(map_in).ndim > 1:
        print("ERROR: Only single map allowed")
        exit()

    nside = hp.get_nside(map_in)
    
    if lmax == None:
        lmax = 3 * nside - 1

    alm = hp.map2alm(map_in, lmax=lmax, pol=False, use_weights=True, datapath='/home/doujzh/Healpix_3.70/data')

    beam_in = hp.gauss_beam(np.deg2rad(fwhm_in / 60.), lmax=lmax, pol=True)
    beam_out = hp.gauss_beam(np.deg2rad(fwhm_out / 60.), lmax=lmax, pol=True)

    if EorB:
        imap = 1
    else:
        imap = 0

    index_nonzero = max(np.where(beam_in[:, imap]>0)[0])
    fl = beam_out[:, imap]
    fl[:index_nonzero] = fl[:index_nonzero] / beam_in[:index_nonzero, imap]

    alm = hp.almxfl(alm, fl)

    map_out = hp.alm2map(alm, nside, lmax=lmax, pol=False, verbose=False)

    return map_out 


def get_filteredmap(map_in, mask_in, fwhm_in, fwhm_out, mode='HFI', band='100', lmax=None):
    nside = hp.get_nside(map_in)
    
    if lmax == None:
        lmax = 3 * nside - 1

    fsky = np.sum(mask_in**2.) / hp.nside2npix(nside)
    Wl = smoothing_filter(fwhm_in, fsky, mode=mode, band=band, lmax=lmax)
    if np.any(np.isnan(Wl)):
        print("Wl is nan")

    alm = hp.map2alm(map_in, lmax=lmax, pol=False, use_weights=True, datapath='/home/doujzh/Healpix_3.70/data')
    if np.any(np.isnan(alm)):
        print("alm unfiltered is nan")

    alm = hp.almxfl(alm, Wl)

    if np.any(np.isnan(alm)):
        print("alm filtered is nan")

    map_out = np.abs(hp.alm2map(alm, nside, lmax=lmax, pol=False, verbose=False))

    if fwhm_in != fwhm_out :
        return change_beam(map_out, fwhm_in, fwhm_out, lmax=lmax)
    else:
        return map_out

    return np.abs(map_out)

def outlier_cut(map_in, mask_in, cut_val=None, cut_sigma=None):
    ps_mask = np.ones_like(mask_in)

    if cut_val != None:
        ps_mask[map_in > cut_val] = 0. 
    elif cut_sigma != None:
        map_masked = map_in[mask_in > 0] 
        mean_value = np.mean(map_masked)
        sigma = np.std(map_masked)

        cut_val = mean_value + cut_sigma*sigma
        ps_mask[map_in > cut_val] = 0.
    else:
        print("ERROR: Either one of cut_val and cut_sigma needs to be provided.")
        exit()

    return ps_mask 