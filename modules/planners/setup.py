# python3 setup.py build_ext --inplace

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "dijkstra_bind",
        ["dijkstra.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-std=c++17", "-O3", "-fno-operator-names"],
    ),
    Pybind11Extension(
        "astar_bind",
        ["astar.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-std=c++17", "-O3", "-fno-operator-names"],
    ),
]

setup(
    name="path_planners",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
