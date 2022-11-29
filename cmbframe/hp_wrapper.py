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

def iqu2teb(map_iqu, mask_in=None, teb='te', lmax_sht=None, return_alm=False):
    nside = hp.get_nside(map_iqu[0])

    if not isinstance(mask_in,(list, np.ndarray)):
        mask_in = np.ones_like((hp.nside2npix(nside),))

    mask_arr = [mask_in, mask_in, mask_in]
    alms = hp.map2alm(map_iqu * mask_arr, lmax=lmax_sht, use_weights=True, datapath='/home/doujzh/Healpix_3.70/data')

    mask_bin = np.ones_like(mask_in)
    mask_bin[mask_in == 0.] = 0.

    teb_maps = []
    if ('t' in teb) or ('T' in teb) :
        if return_alm:
            teb_maps.append(alms[0])
        else:
            teb_maps.append(hp.alm2map(alms[0], nside, lmax=lmax_sht, pol=False, verbose=False) * mask_bin)
    if ('e' in teb) or ('E' in teb) :
        if return_alm:
            teb_maps.append(alms[1])
        else:
            teb_maps.append(hp.alm2map(alms[1], nside, lmax=lmax_sht, pol=False, verbose=False) * mask_bin)
    if ('b' in teb) or ('B' in teb) :
        if return_alm:
            teb_maps.append(alms[2])
        else:
            teb_maps.append(hp.alm2map(alms[2], nside, lmax=lmax_sht, pol=False, verbose=False) * mask_bin)

    return np.array(teb_maps)

def calc_binned_Cl(alm1, alm2=None):
    ALM = hp.Alm()
    lmax = ALM.getlmax(len(alm1))

    Cl_1x2 = hp.alm2cl(alm1, alms2=alm2)

    # ells = np.arange(lmax+1)
    # mode_factor = 2.*ells + 1. 
    # Cl_1x2 = mode_factor * Cl_1x2

    Cl_binned = np.zeros((lmax+1,))

    for li in range(2, len(Cl_1x2)) :
            limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 2)
            limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), lmax-1)
            # li = li - 2
            # if li < len(leff):
            #     limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 0)
            #     limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), len(leff)-1)
            Cl_binned[li] = (np.sum(Cl_1x2[limin:limax])) / (limax - limin) #) 

    del Cl_1x2 

    Cl_binned = np.reshape(Cl_binned,(1,len(Cl_binned)))
    return Cl_binned

def roll_bin_Cl(Cl_in, fmt_nmt=False):
    Cl_in = np.array(Cl_in)

    if Cl_in.ndim > 2:
        print("ERROR: Upto 2-d Cl arrays supported in form [ndim, lmax+1]")
        exit()
    elif Cl_in.ndim == 2:
        # Assume that Cl_in is [nmaps, lmax+1] in size
        lmax = len(Cl_in[0]) - 1
        nmaps = len(Cl_in)
        Cl_1x2 = np.copy(Cl_in)

        Cl_binned = np.zeros((nmaps, lmax+1))

        for li in range(2, lmax+1) :
            limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 2)
            limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), lmax-1)
            # li = li - 2
            # if li < len(leff):
            #     limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 0)
            #     limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), len(leff)-1)

            Cl_binned[:,li] = (np.sum(np.copy(Cl_1x2)[:,limin:limax], axis=1)) / (limax - limin) #)

        del Cl_1x2

        if fmt_nmt:
            Cl_binned = np.reshape(Cl_binned,(nmaps, 1, lmax+1))
    else:
        lmax = len(Cl_in) - 1

        Cl_1x2 = np.copy(np.array(Cl_in))

        # ells = np.arange(lmax+1)
        # mode_factor = 2.*ells + 1. 
        # Cl_1x2 = mode_factor * Cl_1x2

        Cl_binned = np.zeros((lmax+1,))

        for li in range(2, len(Cl_1x2)) :
            limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 2)
            limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), lmax-1)
            # li = li - 2
            # if li < len(leff):
            #     limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 0)
            #     limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), len(leff)-1)
            Cl_binned[li] = (np.sum(Cl_1x2[limin:limax])) / (limax - limin) #) 

        del Cl_1x2 

        if fmt_nmt:
            Cl_binned = np.reshape(Cl_binned,(1,len(Cl_binned)))

    return Cl_binned

def harmonic_udgrade(map_in, nside_out=None, fwhm_in=None, fwhm_out=None, beam_in=None, beam_out=None, pixadj_in=False, pixadj_out=False, lmax_sht=None, pol_only=False, scal_only=False):
    map_to_grd = np.array(map_in)

    if map_to_grd.ndim == 1 :
        nside_in = hp.get_nside(map_to_grd)
        nmaps = 1
    else:
        nside_in = hp.get_nside(map_to_grd[0])
        nmaps = len(map_to_grd[:,0])

    if nmaps > 3:
        print("ERROR: NMAPS > 3 not supported at this moment")
        exit()

    if nside_out == None:
        nside_out = nside_in 

    if lmax_sht == None:
        lmax = 3 * min(nside_in, nside_out) - 1
    else:
        lmax = min(3 * min(nside_in, nside_out) - 1, lmax_sht)

    if isinstance(beam_in, (list, np.ndarray, tuple)):
        beam_in = np.array(beam_in)
        if beam_in.ndim == 1:
            nbeams_i = 1
        else:
            nbeams_i = len(beam_in[0])
    elif fwhm_in != None:
        if scal_only:
            beam_in = hp.gauss_beam(np.deg2rad(fwhm_in / 60.), lmax=lmax, pol=False)
            nbeams_i = 1
        elif pol_only:
            beam_in = hp.gauss_beam(np.deg2rad(fwhm_in / 60.), lmax=lmax, pol=True)[:,2]
            nbeams_i = 1
        else:
            if nmaps > 1:
                beam_in = hp.gauss_beam(np.deg2rad(fwhm_in / 60.), lmax=lmax, pol=True)
                nbeams_i = 3
            else:
                beam_in = hp.gauss_beam(np.deg2rad(fwhm_out / 60.), lmax=lmax)
                nbeams_i = 1
    else: 
        # Assume that the beam is 3 x pixel size

        fwhm_in = 3 * hp.nside2resol(nside_in) 
        if scal_only:
            beam_in = hp.gauss_beam(fwhm_in, lmax=lmax, pol=False)
            nbeams_i = 1
        elif pol_only:
            beam_in = hp.gauss_beam(fwhm_in, lmax=lmax, pol=True)[:,2]
            nbeams_i = 1
        else:
            beam_in = hp.gauss_beam(fwhm_in, lmax=lmax, pol=True)
            nbeams_i = 3
    
    if isinstance(beam_out, (list, np.ndarray, tuple)):
        beam_out = np.array(beam_out)
        if beam_out.ndim == 1:
            nbeams_o = 1
        else:
            nbeams_o = len(beam_in[0])
    elif fwhm_out != None:
        if scal_only:
            beam_out = hp.gauss_beam(np.deg2rad(fwhm_out / 60.), lmax=lmax, pol=False)
            nbeams_o = 1
        elif pol_only:
            beam_out = hp.gauss_beam(np.deg2rad(fwhm_out / 60.), lmax=lmax, pol=True)[:,2]
            nbeams_o = 1
        else:
            if nmaps > 1:
                beam_out = hp.gauss_beam(np.deg2rad(fwhm_out / 60.), lmax=lmax, pol=True)
                nbeams_o = 3
            else:
                beam_out = hp.gauss_beam(np.deg2rad(fwhm_out / 60.), lmax=lmax)
                nbeams_o = 1
    else: 
        # Assume that the beam is 3 x pixel size

        fwhm_out = 3 * hp.nside2resol(nside_out) 
        if scal_only:
            beam_out = hp.gauss_beam(fwhm_out, lmax=lmax, pol=False)
            nbeams_o = 1
        elif pol_only:
            beam_out = hp.gauss_beam(fwhm_out, lmax=lmax, pol=True)[:,2]
            nbeams_o = 1
        else:
            if nmaps > 1:
                beam_out = hp.gauss_beam(fwhm_out, lmax=lmax, pol=True)
                nbeams_o = 3
            else:
                beam_out = hp.gauss_beam(fwhm_out, lmax=lmax)
                nbeams_o = 1

    
    if pixadj_in :
        pixwin_in = hp.pixwin(nside_in, lmax=lmax) 

    if pixadj_out :
        pixwin_out = hp.pixwin(nside_out, lmax=lmax)

    if scal_only or pol_only:
        if nbeams_i != 1:
            print("ERROR: Expecting only one input beam for scal_only/pol_only mode")
            exit()
        if nbeams_o != 1:
            print("ERROR: Expecting only one output beam for scal_only/pol_only mode")
            exit()
        if pixadj_in: 
            beam_i = beam_in * pixwin_in
        else:
            beam_i = beam_in

        if pixadj_out:
            beam_o = beam_out * pixwin_out
        else:
            beam_o = beam_out

        beam_ratio = cu.compute_beam_ratio(beam_i, beam_o)

        alms = hp.map2alm(map_to_grd, lmax=lmax, use_weights=True, datapath='/home/doujzh/Healpix_3.70/data')

        if nmaps == 1:
            alm_fl = hp.almxfl(alms, beam_ratio)
            map_out = hp.alm2map(alm_fl, nside_out, lmax=lmax, verbose=False)
        else:
            map_out = []
            for imap in range(nmaps):
                alm_fl = hp.almxfl(alms[imap], beam_ratio)
                map_out.append(hp.alm2map(alm_fl, nside_out, lmax=lmax, verbose=False))
            
    else: 
        if nmaps == 1:
            if pixadj_in: 
                beam_i = beam_in * pixwin_in
            else:
                beam_i = beam_in

            if pixadj_out:
                beam_o = beam_out * pixwin_out
            else:
                beam_o = beam_out
            

            beam_ratio = cu.compute_beam_ratio(beam_i, beam_o)
            alms = hp.map2alm(map_to_grd, lmax=lmax, use_weights=True, datapath='/home/doujzh/Healpix_3.70/data')
            alm_fl = hp.almxfl(alms, beam_ratio)
            map_out = hp.alm2map(alm_fl, nside_out, lmax=lmax, verbose=False)
        
        elif nmaps == 3:
            alms = hp.map2alm(map_to_grd, lmax=lmax, pol=True, use_weights=True, datapath='/home/doujzh/Healpix_3.70/data')
            map_out = []

            alms_out = []
            for imap in range(nmaps):
                if nbeams_i == 1:
                    if pixadj_in:
                        beam_i = beam_in * pixwin_in
                    else:
                        beam_i = beam_in
                elif nbeams_i == 2:
                    if imap == 0: 
                        if pixadj_in:
                            beam_i = beam_in[:,0] * pixwin_in
                        else: 
                            beam_i = beam_in[:,0]
                    if imap > 0:
                        if pixadj_in:
                            beam_i = beam_in[:,1] * pixwin_in
                        else:
                            beam_i = beam_in[:,1]
                elif nbeams_i == nmaps:
                    if pixadj_in:
                        beam_i = beam_in[:,imap] * pixwin_in
                    else:
                        beam_i = beam_in[:,imap]


                if nbeams_o == 1:
                    if pixadj_out:
                        beam_o = beam_out * pixwin_out
                    else:
                        beam_o = beam_out
                elif nbeams_o == 2:
                    if imap == 0: 
                        if pixadj_out:
                            beam_o = beam_out[:,0] * pixwin_out
                        else: 
                            beam_o = beam_out[:,0]
                    if imap > 0:
                        if pixadj_out:
                            beam_o = beam_out[:,1] * pixwin_out
                        else:
                            beam_o = beam_out[:,1]
                elif nbeams_o == nmaps:
                    if pixadj_out:
                        beam_o = beam_out[:,imap] * pixwin_out
                    else:
                        beam_o = beam_out[:,imap]

                beam_ratio = cu.compute_beam_ratio(beam_i, beam_o)
                alms_out.append(hp.almxfl(alms[imap], beam_ratio))

            map_out = hp.alm2map(alms_out, nside_out, lmax=lmax, pol=True, verbose=False)
        
        else:
            print("ERROR: Cannot process mixed TP maps without IQU format. If not in IQU format try T and P map separately with scal_only or pol_only set to True.")
            exit()

    return np.array(map_out)

def mask_udgrade(mask_in, nside_out, cut_val=0.9):
    nside_in = hp.get_nside(mask_in)
    if nside_out != nside_in:
        mask_out = hp.ud_grade(mask_in, nside_out)
    else:
        mask_out = np.copy(mask_in)
        
    mask_out[mask_out > cut_val] = 1.
    mask_out[mask_out <= cut_val] = 0.

    return mask_out


def alm_fort2c(alm_in):
    # Assume alm shape to be [lmax, mmax] for nmaps = 1 and [nmaps, lmax, mmax] >= 1

    alm_fort = np.array(alm_in)

    alm_dim = alm_fort.ndim

    if alm_dim == 3:
        nmaps = len(alm_fort[:,0,0])
        lmax = len(alm_fort[0,:,0]) - 1
        mmax = len(alm_fort[0,0,:]) - 1
    elif alm_dim == 2:
        lmax = len(alm_fort[:,0]) - 1
        mmax = len(alm_fort[0,:]) - 1
    else:
        print("ERROR: Fortran-type alm has wrong dimensions. Only [nmaps, lmax, mmax] or [lmax, mmax] supported")
        exit()

    ALM = hp.Alm()
    c_alm_size = ALM.getsize(lmax,mmax)
    ls, ms = ALM.getlm(lmax)

    idx_arr = np.arange(c_alm_size)

    if alm_dim == 3:
        alm_c = np.zeros((nmaps, c_alm_size), dtype=np.complex128)
        alm_c[:,idx_arr] = alm_fort[:, ls, ms]
    else:
        alm_c = np.zeros((c_alm_size,), dtype=np.complex128)
        alm_c[idx_arr] = alm_fort[ls, ms]

    return alm_c
    

def alm_c2fort(alm_in):
    # Assume alm shape to be [midx,] for nmaps = 1 and [nmaps, midx] >= 1

    alm_c = np.array(alm_in)

    alm_dim = alm_c.ndim

    if alm_dim == 2:
        nmaps = len(alm_c[:,0]) 
        midx = len(alm_c[0,:])
    elif alm_dim == 1:
        midx = len(alm_c[:])
    else:
        print("ERROR: C-type alm has wrong dimensions. Only [nmaps, midx] or [midx] supported")
        exit()
    
    ALM = hp.Alm()
    lmax = ALM.getlmax(midx)
    mmax = lmax 

    idx_arr = np.arange(midx)

    ls, ms = ALM.getlm(lmax, i=idx_arr)

    if alm_dim == 2:
        alm_fort = np.zeros((nmaps, lmax+1, mmax+1), dtype=np.complex128)
        alm_fort[:,ls, ms] = alm_c[:,idx_arr]
    else:
        alm_fort = np.zeros((lmax+1, mmax+1), dtype=np.complex128)
        alm_fort[ls, ms] = alm_c[idx_arr]

    return alm_fort 