from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import sys
import os

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# Find where conda installed Eigen
conda_prefix = os.environ.get('CONDA_PREFIX', os.path.abspath(os.path.join(os.path.dirname(__file__), 'env3.6')))
eigen_include_dirs = [
    os.path.join(conda_prefix, 'Library', 'include', 'eigen3'),  # Windows conda path
    os.path.join(conda_prefix, 'include', 'eigen3'),             # Alternative location
    "/usr/include/eigen3"                                        # Original Unix path
]

ext_modules = [
    Extension(
        "one_robot_cnndqn.utils.inverse_sensor_model",
        ["one_robot_cnndqn/utils/inverse_sensor_model.cpp"],
        
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            *eigen_include_dirs  # Use our enhanced list of Eigen paths
        ],
        language='c++'
    ),


    Extension(
        "two_robot_dueling_dqn_attention.utils.inverse_sensor_model",
        ["two_robot_dueling_dqn_attention/utils/inverse_sensor_model.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            *eigen_include_dirs
        ],
        define_macros=[('_USE_MATH_DEFINES', None)],
        language='c++'
    ),

    Extension(
        "two_robot_cnndqn_attention.utils.inverse_sensor_model",
        ["two_robot_cnndqn_attention/utils/inverse_sensor_model.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            *eigen_include_dirs
        ],
        define_macros=[('_USE_MATH_DEFINES', None)],
        language='c++'
    ),
]

# C++11 compiler settings
class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = []
        if ct == 'unix':
            opts.append('-std=c++11')
            opts.append('-O3')
        elif ct == 'msvc':  # Add MSVC-specific options
            opts.append('/EHsc')
            opts.append('/O2')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name="frontier_exploration",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'scikit-image>=0.17.0',
        'pybind11>=2.6.0',
    ],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)