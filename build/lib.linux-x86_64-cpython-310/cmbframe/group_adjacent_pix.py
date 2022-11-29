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

from . import border_finder as bf 


def group_cont_pix(pix, listpix, checkedpix, nside):
    

    nip = hp.get_all_neighbours(nside, pix)

    intersect_lp, subip, sublp = np.intersect1d(nip, listpix, return_indices=True)

    # print(len(intersect_lp), nip, checkedpix)

    if len(intersect_lp) == 0 :
        # print('Here!')
        return checkedpix
    
    
    else:

        ip = intersect_lp

        intersect_cp, checkip, subcp = np.intersect1d(ip, checkedpix, return_indices=True)

        if len(intersect_cp) != 0 :
            ip[checkip] = -1
        
        wh = np.where(ip > 0)[0]
        nwh = len(wh)

        # print(intersect_lp, "nwh=",nwh)

        if nwh > 0 :

            ip = ip[wh]
            checkedpix = np.append(checkedpix, ip)
            # print(checkedpix, ip)

            for iip in range(len(ip)):

                # print('iip=', iip, ip[iip], checkedpix)
                checkedpix = group_cont_pix(ip[iip], listpix, checkedpix, nside)
                
        return checkedpix

def group_adjacent_pixels_classic(pixels, nside):
    lp = np.copy(pixels)

    groups = -np.ones_like(pixels)

    i_group = 0

    nwh = len(lp)

    while nwh > 0:
        g = group_cont_pix(lp[0], lp, lp[0], nside)

        intersect_pix, subg, subp = np.intersect1d(g, pixels, return_indices=True) 
        groups[subp] = i_group

        intersect_lp, subg, sublp = np.intersect1d(g, lp, return_indices=True)
        lp[sublp] = -1

        wh = np.where(lp > 0)[0]
        nwh = len(wh)

        # print(nwh)
        if nwh > 0 :
            lp = lp[wh]

        i_group += 1

    return groups

def vec_rot(vec, rot_vec, angle):
    # First normalize the rotation vector

    norm = np.sqrt(np.sum(rot_vec**2.))
    rotation_vector = rot_vec / norm

    # Compute the new vector
    cos_angle = np.cos(angle)
    # print(angle, cos_angle)
    sec_term = (1. - cos_angle) * np.sum(rotation_vector * vec)

    # print(cos_angle, np.sum(rotation_vector * vec), sec_term)

    new_vec = cos_angle*vec + sec_term*rotation_vector + np.sin(angle)*np.cross(rotation_vector, vec)

    return new_vec

# vector_rotation = np.vectorize(vec_rot, otypes=[np.float64])

def group_adjacent_pixels(ps_mask):

    # npix = hp.get_map_size(ps_mask)
    nside = hp.get_nside(ps_mask)

    ps_pixels = np.where(ps_mask == 0)[0]

    groups = -np.ones_like(ps_pixels)

    p0, p1, np0, np1 = bf.get_mask_border(ps_mask, need_nos=True)
    # print(p0)

    pgp = group_adjacent_pixels_classic(p0, nside)

    # print(p0)

    npgp = np.max(pgp)

    pix_map = np.zeros_like(ps_mask)

    pixsize = hp.nside2resol(nside)
    igroup = 0

    for i in range(npgp+1):
        wh = np.argwhere(pgp == i).flatten()

        # print(np.argwhere(pgp == i).flatten())
        # print(np.where(pgp == i)[0])
        nwh = len(wh)

        # print(nwh, wh, p0, p0[1], p0[2])

        # print(wh, nwh)

        if nwh > 1:
            wh = p0[wh]

            pix_map[p0] = 1
            pix_map[wh] = 0

            vector = np.array(hp.pix2vec(nside, wh))
            # print(vector, wh)

            mvec = np.mean(vector, axis=1)

            # print(np.linalg.norm(mvec))

            # print(mvec.shape)
            # print(mvec, vector)
            # dist = np.arccos(np.dot(mvec, vector))
            dist = hp.rotator.angdist(mvec, vector)
            rad_hole = np.max(dist)
            imaxv = np.where(dist == rad_hole)[0]
            # print(dist, imaxv)
            maxv = vector[:,imaxv[0]]

            # print(vector.shape, maxv.shape)

            rotvec = - np.array([maxv[1]*mvec[2]-maxv[2]*mvec[1], \
                                 maxv[2]*mvec[0]-maxv[0]*mvec[2], \
                                 maxv[0]*mvec[1]-maxv[1]*mvec[0]])

            new_vector= vec_rot(maxv, rotvec, pixsize)

            tip = hp.vec2pix(nside, new_vector[0], new_vector[1], new_vector[2])

            if (ps_mask[tip] != 0):
        
                listpix = hp.query_disc(nside, mvec, rad_hole*1.05)

        else:
            # print(i, np.argwhere(pgp == i).flatten(), wh, nwh)
            listpix = p0[wh]

        if (nwh == 0) or (ps_mask[tip] != 0):
            if np.sum(pix_map[listpix]) == 0 :
                intersect_lp, subp, sublp = np.intersect1d(ps_pixels, listpix, return_indices=True)
                groups[subp] = igroup
                
                igroup += 1
        
    subpix = np.where(groups == -1)[0]

    if len(subpix > 0):
        extra_groups = group_adjacent_pixels_classic(ps_pixels[subpix], nside)

        groups[subpix] = extra_groups + igroup

    return groups