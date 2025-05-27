"""
Loads the c graphlib library.
"""
import sys
import imp
import ctypes
import _ctypes
import os

def find_path():
    """
    Find the path to the c graphlib library.
    :return:
    """
    if sys.platform == "linux2" or sys.platform == "linux":
        extension = ".so"
    elif sys.platform == "darwin":
        extension = ".dylib"
    elif sys.platform == "win32":
        extension = ".dll"
    else:
        print("Unknown system type!")
        return (True,0,0)

    path_lgc = imp.find_module('localgraphclustering')[1]
    return path_lgc+"/src/lib/graph_lib_test/libgraph"+extension

def load_library():
    """
    Load the c graphlib library.
    :return: the c library as a python package
    """
    #load library
    lib=ctypes.cdll.LoadLibrary(find_path())
    return lib

_graphlib = load_library()
#assert(is_loaded(_graphlib))


from .aclpagerank_cpp import *
from .dcpagerank_cpp import *
from .dcpagerank_weighted_cpp import *
from .aclpagerank_weighted_cpp import *
from .capacity_releasing_diffusion_cpp import *
from .densest_subgraph_cpp import *
from .MQI_cpp import *
from .MQI_weighted_cpp import *
from .proxl1PRaccel import *
from .proxl1PRrand_cpp import *
from .SimpleLocal_cpp import *
from .SimpleLocal_weighted_cpp import *
from .sweepcut_cpp import *
from .triangleclusters_cpp import *
from .set_scores_cpp import *
