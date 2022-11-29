import numpy as np
import matplotlib
matplotlib.use('Agg')
import healpy as hp 
import multiprocessing as mp 
import joblib as jl 

def get_disc(nside_super, vec, radius):
    disc_pix = hp.query_disc(nside_super, vec, radius, inclusive=True)
    return np.array(disc_pix)

def get_super_neighbours(nside_super):
    n_cores = mp.cpu_count()

    super_pix_arr = np.arange(hp.nside2npix(nside_super))
    super_pix_vec = np.array(hp.pix2vec(nside_super, super_pix_arr))

    radius = 4.5* hp.nside2resol(nside_super)
    
    neighbour_list = jl.Parallel(n_jobs=n_cores)(jl.delayed(get_disc)(nside_super, super_pix_vec[:, ivec], radius) for ivec in super_pix_arr)

    return neighbour_list

def npix2spix_map(nside_map, nside_super):
    
    npix_map = np.arange(hp.nside2npix(nside_map))

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