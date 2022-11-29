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

from setuptools import setup
from numpy.distutils.core import Extension, setup
#

pkgs = {'cmbframe' : 'cmbframe'}
#
# exts = list()
# exts.append(
#     Extension(name='cmbframe.covarifast',
#               sources=['cmbframe/f90/covarifast.f90'],
#               extra_compile_args=['-fopenmp'],
#               extra_link_args=["-lgomp"],
#               libraries=['gomp'],
#               extra_f90_compile_args=["-fopenmp", "-lgomp"],
#             #   f2py_options=['--debug-capi']
#               ))
    ###
    ### 

setup(
      name = 'CMBframe',
      version='0.4.0',
      author='Shamik Ghosh',
      author_email='thequarkexpress@gmail.com',
      # ext_modules = exts,
      packages = list(pkgs.keys()),
      url='https://github.com/1cosmologist/CMBframe',
      license='LICENSE',
      description='CMB data analysis framework',
      long_description=open('README.md').read(),
      install_requires=[
      "numpy",
      "healpy",
      "matplotlib",
      "scipy",
      "joblib",
      "pymaster",
      "tqdm"
      ],
      package_dir = pkgs,
    #   libraries=['covarifast', sources=['cmbframef90/covarifast.f90']]
    #   include_package_data=True,
    #   zip_safe=False
      )

# cov_f90 = Extension(name = 'covarifast',
#                     libraries = [
#                     'gomp',
#                     #'blas',
#                     ],
#                  extra_compile_args = ['-fopenmp'],
#                  sources = ['cmbframe/covarifast.f90'],
#                  extra_f90_compile_args=["-fopenmp", "-lgomp"]
#                  )

# # excluded = ['old_modules']

# setup(
#    name='CMBframe',
#    version='0.4.0',
#    author='Shamik Ghosh',
#    author_email='thequarkexpress@gmail.com',
#    packages=['cmbframe'],
#    url='https://github.com/1cosmologist/CMBframe',
#    license='LICENSE',
#    description='CMB data analysis framework',
#    long_description=open('README.md').read(),
#    install_requires=[
#        "numpy",
#        "healpy",
#        "matplotlib",
#        "scipy",
#        "joblib",
#        "pymaster"
#    ],
#    ext_package = 'cmbframe',
#    ext_modules = [cov_f90]
# )


