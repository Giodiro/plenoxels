from setuptools import setup
import os.path as osp

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules = [
    CUDAExtension('plenoxels.c_ext',
        sources=[
            'plenoxels/csrc/svox.cu',
            # 'plenoxels/csrc/octree_common.cu',
            # 'csrc/svox_kernel.cu',
            # 'csrc/rt_kernel.cu',
        ],
        include_dirs=[
            'plenoxels/csrc',
            'plenoxels/csrc/include',
        ],
        extra_compile_args={
            "cxx": ["-std=c++14", "-g"],
            "nvcc": ["-std=c++14", "--ptxas-options=-v", "-keep", "-lineinfo"]
        },
    )
]

setup(
    name='svox',
    version='0.1',
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description='Sparse voxel N^3-tree data structure using CUDA',
    long_description='Sparse voxel N^3-tree data structure PyTorch extension, using CUDA',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    packages=['plenoxels'],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)},
    zip_safe=False,
)
