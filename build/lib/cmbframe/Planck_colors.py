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
"""
Planck colors module produces matplotlib color map using PSM Planck colors
divergent color scheme.

Attributes
----------
gaussian
    A ``matplotlib`` color map for use with divergent CMB maps. Account for normal contrast.
foreground
    A ``matplotlib`` color map for use with high contrast foreground maps.
"""
import numpy as np
import matplotlib as mpl

# For gaussian signal, diverging colormap

tone = np.array([   0,  42,  85, 127, 170, 212, 255])
Rtab = np.array([   0,   0,   0, 255, 255, 255, 100])
Gtab = np.array([   0, 112, 221, 237, 180,  75,   0])
Btab = np.array([ 255, 255, 255, 217,   0,   0,   0])

n_tones = 256.

tones_full = np.arange(n_tones)

Rval_full_g = np.interp(tones_full, tone, Rtab) / (n_tones-1.)
Gval_full_g = np.interp(tones_full, tone, Gtab) / (n_tones-1.)
Bval_full_g = np.interp(tones_full, tone, Btab) / (n_tones-1.)
Aval_full_g = np.ones_like(tones_full)

RGBA_full_g = np.transpose(np.vstack((Rval_full_g, Gval_full_g, Bval_full_g, Aval_full_g)))

# print(RGBA_full_g, np.any(RGBA_full_g > 1.))
gaussian = mpl.colors.ListedColormap(RGBA_full_g)

del tone, Rtab, Gtab, Btab
# For high dynamic fields like foregrounds

tone = np.array([   0,  13,  26,  39,  52,  65,  76,  77,  88, 101, 114, 127, 140, 153, 166, 179, 192, 205, 218, 231, 255])
Rtab = np.array([   0,  10,  30,  80, 191, 228, 241, 241, 245, 248, 249.9, 242.25,204, 165, 114, 127.5, 178.5, 204, 229.5, 242.25, 252.45])
Gtab = np.array([   0,  20, 184, 235, 239, 240, 241, 241, 240, 235, 204,  153,  76.5,  32,  0, 127.5, 178.5, 204, 229.5, 242.25, 252.45])
Btab = np.array([ 255, 255, 255, 255, 250, 245, 212, 212, 175, 130, 38.25, 12.75,  0,   32,  32, 153,  204,  229.5, 242.25, 249.9, 255])

Rval_full_f = np.interp(tones_full, tone, Rtab) / (n_tones - 1.)
Gval_full_f = np.interp(tones_full, tone, Gtab) / (n_tones - 1.)
Bval_full_f = np.interp(tones_full, tone, Btab) / (n_tones - 1.)
Aval_full_f = np.ones_like(tones_full)

RGBA_full_f = np.transpose(np.vstack((Rval_full_f, Gval_full_f, Bval_full_f, Aval_full_f)))
foreground = mpl.colors.ListedColormap(RGBA_full_f)
