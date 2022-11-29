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

def get_neighbours(nside, ipix, nnearest) : 
    pixlist_new = set([ipix])
    nearness = []
    pixlist_old = set()
    # print(pixlist_new)
    for near in range(0, nnearest):
        if len(pixlist_new) != 0 :
            pixlist_iter = hp.get_all_neighbours(nside, list(pixlist_new.copy()))
            # print(near, 'output=',pixlist_iter.flatten())

            len_o = len(pixlist_old)
            pixlist_old.update(pixlist_new.copy())
            len_n = len(pixlist_old)
            nearness.extend(list(near for _ in range(len_o, len_n)))
            
            pixlist_new = set(pixlist_iter.flatten())
            pixlist_new.difference_update(pixlist_old)
            pixlist_new.discard(-1)

        # print(near, pixlist_old)
        # print(near, pixlist_new)
        # print(near, nearness)

    if len(pixlist_new) != 0 :
        len_o = len(pixlist_old)
        pixlist_old.update(pixlist_new.copy())
        len_n = len(pixlist_old)
        nearness.extend(list(nnearest for _ in range(len_o, len_n)))

    # pixlist_old.discard(ipix)     #For ILC bias removal

    # print(nnearest, pixlist_old)
    # print(nnearest, nearness)

    pixlist = list(pixlist_old)
    if len(pixlist) != len(nearness) :
        print("Lengths don't match!")
    return pixlist#,  nearness


def get_super_neighbours(nside_super):
    n_cores = mp.cpu_count()

    npix_super = hp.nside2npix(nside_super)

    # params = [nside_super, 4]
    
    neighbour_list = jl.Parallel(n_jobs=n_cores)(jl.delayed(get_neighbours)(nside_super, super_pix, 4) for super_pix in np.arange(npix_super))

    # print(len(neighbour_list), np.array(neighbour_list).shape)

    return neighbour_list

def npix2spix_map(nside_map, nside_super):
    
    npix_map = np.arange(hp.nside2npix(nside_map))

    spix = hp.nside2npix(nside_super)

    x, y, z = hp.pix2vec(nside_map, npix_map)

    spix_map = hp.vec2pix(nside_super, x,y,z)

    return spix_map

def neighbour2index(npix_spix_map, neighbours):
    return np.where(np.in1d(npix_spix_map, np.array(neighbours)))


def get_map_indices(wvlt_nside, super_nside):

    npix_super = hp.nside2npix(super_nside)
    super_neighbours = get_super_neighbours(super_nside)
    if len(super_neighbours) != npix_super:
        print("Super lengths wrong!")

    wvlt_spix_map = npix2spix_map(wvlt_nside, super_nside)

    n_cores = mp.cpu_count()

    indices = jl.Parallel(n_jobs=n_cores)(jl.delayed(neighbour2index)(wvlt_spix_map, super_neighbours[i]) for i in range(npix_super))

    return indices

# nl = get_super_neighbours(1)
# index = get_map_indices(1024,32)
# print(index)


def get_mapindx_spix(super_nside, super_ipix, npix_mapped_spix):
    spix_neighbours = get_neighbours(super_nside, super_ipix, 4)

    map_indices = neighbour2index(npix_mapped_spix, spix_neighbours)

    return map_indices

