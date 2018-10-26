#!/usr/bin/env python

import numpy as np
import glob
import sys
import os
import tempfile as tmp
from pathlib import Path
import io
import argparse
from time import sleep
import subprocess as sp
# path_to_rust = (Path(__file__).parent / "target/release/lumberjack_1").resolve()
# path_to_tree_reader = Path(__file__).resolve()
# sys.path.append(path_to_tree_reader)
import tree_reader as tr

import numpy as np
# import matplotlib.pyplot as plt

def main(target,**kwargs):
    # print("Tree reader?")
    # print(path_to_tree_reader)
    print("Running main")
    print("Trying to load")
    print(target)
    counts = np.loadtxt(target)
    print("Loaded counts")
    print(counts)
    fit_return = context(counts,**kwargs)
    print(fit_return)

def context(targets,header=None,**kwargs):

    targets = targets.T

    print("Setting context")

    tmp_dir = tmp.TemporaryDirectory()
    output = tmp_dir.name + "/tmp"

    np.savetxt(output + ".truth",targets)

    features = targets.shape[1]

    if header is None:
        np.savetxt(output + ".header", np.arange(features))

    print("CHECK TRUTH")
    print(tmp_dir.name)
    print(os.listdir(tmp_dir.name))

    print("Generating trees")

    fit(targets,output,**kwargs)

    print("CHECK OUTPUT")
    print(os.listdir(tmp_dir.name))

    forest = tr.Forest.load(output,prefix="/tmp.*.compact",header=".header",truth=".truth")

    tmp_dir.cleanup()

    return forest


def fit(targets,location, **kwargs):

    targets = "\n".join(["\t".join([str(y) for y in x]) for x in targets]) + "\n"

    path_to_rust = (Path(__file__).parent / "../target/release/lumberjack_1").resolve()

    print("Running " + str(path_to_rust))

    arg_list = [str(path_to_rust),"generate","-stdin","-o",location ,"-auto"]

    for arg in kwargs.keys():
        arg_list.append("-" + str(arg))
        arg_list.append(str(kwargs[arg]))

    print("Command: " + " ".join(arg_list))

    # cp = sp.run(arg_list,input=targets,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)

    cp = sp.Popen(arg_list,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)
    try:
        cp.communicate(input=targets,timeout=1)
    except:
        print("Communicated input")
    #
    # with sp.Popen(arg_list,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True) as cp:
    #     try:
    #         cp.communicate(input=targets,timeout=1)
    #     except:
    #         pass
    #     while True:
    #         print("Trying to readline")
    #         # sleep(1)
    #         rc = cp.poll()
    #         if rc is not None:
    #             print(cp.stdout.read())
    #             print(cp.stderr.read())
    #             break
    #         output = cp.stdout.readline()
    #         print("Read line")
    #         print(output)


    while cp.poll() is None:
        sys.stdout.flush()
        # sys.stdout.write("Constructing trees: %s" % str(len(glob.glob(location + "/tmp.*.compact"))) + "\r")
        sys.stdout.write(os.listdir(location))
        sleep(1)

    # print(cp.stdout)

    print(cp.stderr)

if __name__ == "__main__":
    kwargs = {x.split("=")[0]:x.split("=")[1] for x in sys.argv[2:]}
    main(sys.argv[1],**kwargs)


# ,feature_sub=None,distance=None,sample_sub=None,scaling=None,merge_distance=None,refining=False,error_dump=None,convergence_factor=None,smoothing=None,locality=None)
