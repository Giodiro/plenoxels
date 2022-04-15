from setuptools import setup
import os.path as osp

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

try:
    ext_modules = [
        CUDAExtension('csrc', [
            'csrc/svox.cpp',
            'csrc/octree.cu',
            # 'csrc/svox_kernel.cu',
            # 'csrc/rt_kernel.cu',
        ], include_dirs=[osp.join(ROOT_DIR, "csrc", "include")],
                      optional=True),
    ]
except:
    import warnings

    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

setup(
    name='svox',
    version='0.1',
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description='Sparse voxel N^3-tree data structure using CUDA',
    long_description='Sparse voxel N^3-tree data structure PyTorch extension, using CUDA',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    packages=['csrc'],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
)
