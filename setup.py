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

ext_modules = [
    Extension(
        "one_robot_cnndqn.utils.inverse_sensor_model",
        ["one_robot_cnndqn/utils/inverse_sensor_model.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            "/usr/include/eigen3"  # 添加 Eigen 
        ],
        language='c++'
    ),
    
]

# 添加 C++11
class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = []
        if ct == 'unix':
            opts.append('-std=c++11')
            opts.append('-O3')
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
