# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "Network",
        ["Network.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTI",
        ["OGSTI.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTII",
        ["OGSTII.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTII_Repeat",
        ["OGSTII_Repeat.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
]
setup(
    name="Extreme_Evol_EAS",
    ext_modules=cythonize(
        extensions,
        language_level="3",
        build_dir="build",
    ),
)
