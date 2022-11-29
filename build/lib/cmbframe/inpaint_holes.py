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

from healpy.pixelfunc import vec2ang
import numpy as np 
import healpy as hp 
from scipy.interpolate import Rbf, CloughTocher2DInterpolator, griddata, NearestNDInterpolator 
from astropy.wcs import WCS
import multiprocessing as mp 
import joblib as jl 

from . import group_adjacent_pix as gap

def vstack_cross(vec1, vec2_stack):
    out_vec = np.zeros_like(vec2_stack)

    out_vec[0] = vec1[1]*vec2_stack[2] - vec1[2]*vec2_stack[1]
    out_vec[1] = vec1[2]*vec2_stack[0] - vec1[0]*vec2_stack[2]
    out_vec[2] = vec1[0]*vec2_stack[1] - vec1[1]*vec2_stack[0]

    return out_vec

def vec_rot(vec_list, rot_vec, rot_ang):
    # First normalize the rotation vector

    new_vec = np.zeros_like(vec_list)

    norm = np.sqrt(np.sum(rot_vec**2.))
    rotation_vector = rot_vec / norm

    # Compute the new vector
    cos_angle = np.cos(rot_ang)
    
    sec_term = (1. - cos_angle) * np.dot(rotation_vector, vec_list)

    cross_vec = vstack_cross(rotation_vector, vec_list)
    new_vec[0] = cos_angle*vec_list[0] + sec_term*rotation_vector[0] + np.sin(rot_ang)*cross_vec[0]
    new_vec[1] = cos_angle*vec_list[1] + sec_term*rotation_vector[1] + np.sin(rot_ang)*cross_vec[1]
    new_vec[2] = cos_angle*vec_list[2] + sec_term*rotation_vector[2] + np.sin(rot_ang)*cross_vec[2]

    return new_vec

def gnomic_local_transform(vec_list, mean_vec):
    z_axis = hp.ang2vec(0., 0., lonlat=True)

    rot_ang = hp.rotator.angdist(mean_vec, z_axis)

    rot_vec = - np.array([z_axis[1] * mean_vec[2] - z_axis[2] * mean_vec[1], \
                          z_axis[2] * mean_vec[0] - z_axis[0] * mean_vec[2], \
                          z_axis[0] * mean_vec[1] - z_axis[1] * mean_vec[0] ])

    rot_lon, rot_lat = hp.vec2ang(np.transpose(vec_rot(vec_list, rot_vec, rot_ang)), lonlat=True)
    
    return rot_lon, rot_lat 

class inpaint_workspace:
    '''
        Inpaint workspace class is used to set up a parallel compute of hole inpainting,
        based on local area surface fit.
    '''
    
    def __init__(self, map_in, ps_mask_in, mask_in=None, coherence_scale='short', interpolator='rbf'):
        self.nside = hp.get_nside(ps_mask_in)
        self.pix_size = hp.nside2resol(self.nside)

        self.coh_len = coherence_scale
        if self.coh_len == 'short':
            self.min_disc_rad = 2.5 * self.pix_size
        elif self.coh_len == 'long':
            self.min_disc_rad = 3.2 * self.pix_size

        self.ps_msk = np.copy(ps_mask_in)
        self.map_to_inp = np.array(np.copy(map_in))

        # self.inpainted_map = np.array(np.copy(map_in))

        # print(self.inpainted_map.flags)

        self.interpolator = interpolator
        if isinstance(mask_in, (list, tuple, np.ndarray)):
            self.ps_msk[np.argwhere(mask_in == 0).flatten()] = 1.

        self.ps_pix = np.argwhere(self.ps_msk == 0).flatten()

        self.groups = gap.group_adjacent_pixels(self.ps_msk)

        self.ngp = np.max(self.groups)

    def fill_a_hole(self, gr):
        whg = np.argwhere(self.groups == gr).flatten()
        nwhg = len(whg) 

        bad_pixels = self.ps_pix[whg]
        bad_vecs = np.array(hp.pix2vec(self.nside, bad_pixels))

        # print(bad_pixels[25], bad_vecs[:,25])
        if nwhg > 1:
            mvec = np.mean(bad_vecs, axis=1)
            # dist = np.arccos(np.dot(mvec, bad_vecs))
            dist = hp.rotator.angdist(mvec, bad_vecs)
            rad_hole = np.max(dist)

        elif nwhg == 1:
            # print('here')
            mvec = np.copy(bad_vecs[:,0])
            rad_hole = self.pix_size 

        if self.coh_len == 'short':
            rad_disc = 1.5 * rad_hole
        elif self.coh_len == 'long':
            rad_disc = 2.0 * rad_hole

        good_pixels = hp.query_disc(self.nside, mvec, max(self.min_disc_rad, rad_disc))

        good_pixels = good_pixels[self.ps_msk[good_pixels] == 1]
        good_vecs = np.array(hp.pix2vec(self.nside, good_pixels))

        bad_lon, bad_lat = gnomic_local_transform(bad_vecs, mvec)
        good_lon, good_lat = gnomic_local_transform(good_vecs, mvec)

        # print(bad_lon[25], bad_lat[25])
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [0., 0.]
        # wcs.wcs.cdelt = [1./3600., 1./3600.]
        # wcs.wcs.crval = [23.2334, 45.2333]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        x_out, y_out = wcs.wcs_world2pix(bad_lon, bad_lat, 1)
        x, y = wcs.wcs_world2pix(good_lon, good_lat, 1)

        if (np.isnan(x_out).any()) or (np.isnan(y_out).any()):
            print("Inpaint ERROR: badpix NaN", gr)
        if (np.isnan(x).any()) or (np.isnan(y).any()):
            print("Inpaint ERROR: goodpix NaN", gr)

        # return x,y, x_out, y_out
        # print(bad_lon, bad_lat)
        # print(x_out, y_out)
        # print(bad_lon[25], bad_lat[25], wcs.wcs_world2pix(bad_lon[25], bad_lat[25], 1))

        if self.map_to_inp.ndim == 1:
            data = self.map_to_inp[good_pixels]

            if self.interpolator == 'rbf':
                interpol_surf = Rbf(x,y,data, function='thin_plate')#SmoothBivariateSpline(x,y,z)
                # self.inpainted_map[bad_pixels] = interpol_surf(x_out, y_out)
                return bad_pixels, interpol_surf(x_out, y_out)

            elif self.interpolator == 'cti':
                interpol_surf = CloughTocher2DInterpolator(list(zip(x, y)), data)
                # self.inpainted_map[bad_pixels] = interpol_surf(list(zip(x_out, y_out)))
                return bad_pixels, interpol_surf(list(zip(x_out, y_out)))
            elif self.interpolator == 'nnd':
                    interpol_surf = NearestNDInterpolator(list(zip(x, y)), data)
                    # self.inpainted_map[bad_pixels] = interpol_surf(list(zip(x_out, y_out)))
                    return bad_pixels, interpol_surf(list(zip(x_out, y_out)))
            elif self.interpolator == 'grd':
                # self.inpainted_map[bad_pixels] = griddata((x, y), data, (x_out, y_out), method='cubic')
                return bad_pixels, griddata((x, y), data, (x_out, y_out), method='cubic')

            # print(inpainted_map[bad_pixels], Rbf_surface(x_out, y_out))
            # del data, interpol_surf
        
        elif self.map_to_inp.ndim == 2:
            nmaps = len(self.map_to_inp[:,0])
            inpainted_array = []
            for imap in range(nmaps):
                data = self.map_to_inp[imap, good_pixels]

                if self.interpolator == 'rbf':
                    interpol_surf = Rbf(x,y,data, function='thin_plate')#SmoothBivariateSpline(x,y,z)
                    # self.inpainted_map[imap, bad_pixels] = interpol_surf(x_out, y_out)
                    inpainted_array.append(interpol_surf(x_out, y_out))
                elif self.interpolator == 'cti':
                    interpol_surf = CloughTocher2DInterpolator(list(zip(x, y)), data)
                    self.inpainted_map[imap, bad_pixels] = interpol_surf(list(zip(x_out, y_out)))
                elif self.interpolator == 'nnd':
                    interpol_surf = NearestNDInterpolator(list(zip(x, y)), data)
                    self.inpainted_map[imap, bad_pixels] = interpol_surf(list(zip(x_out, y_out)))
                elif self.interpolator == 'grd':
                    self.inpainted_map[imap, bad_pixels] = griddata((x, y), data, (x_out, y_out), method='cubic')

            return bad_pixels, inpainted_array

def inpaint_holes(map_in, ps_mask_in, mask_in=None, interpolator='rbf', coherence_scale='short'):
    
    inp_wsp = inpaint_workspace(map_in, ps_mask_in, mask_in=mask_in, coherence_scale=coherence_scale, interpolator=interpolator)

    n_cores = mp.cpu_count()
    inpainted_holes = jl.Parallel(n_jobs=n_cores)(jl.delayed(inp_wsp.fill_a_hole)(gr_no) for gr_no in range(inp_wsp.ngp+1))

    inpainted_map = np.array(np.copy(map_in))

    for gr in range(inp_wsp.ngp+1):
        if inpainted_map.ndim > 1:
            inpainted_map[:,inpainted_holes[gr][0]] = np.array(inpainted_holes[gr][1])
        elif inpainted_map.ndim == 1:
            inpainted_map[inpainted_holes[gr][0]] = inpainted_holes[gr][1]
        else:
            print("ERROR:Map_in.ndim <1 should not exist.")

    return inpainted_map