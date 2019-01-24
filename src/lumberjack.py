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
# import asyncio as aio
# import multiprocessing as mp
import subprocess as sp
# path_to_rust = (Path(__file__).parent / "target/release/lumberjack_1").resolve()
# path_to_tree_reader = Path(__file__).resolve()
# sys.path.append(path_to_tree_reader)
import tree_reader as tr

import numpy as np
# import matplotlib.pyplot as plt

def main(location,input,output=None,header=None,**kwargs):
    # print("Tree reader?")
    # print(path_to_tree_reader)
    if output is None:
        output = input
    print("Running main")
    print("Trying to load")
    print(input)
    print(output)
    input_counts = np.loadtxt(input)
    output_counts = np.loadtxt(output)
    if header is not None:
        header = np.loadtxt(header,dtype=str)
    print("Loaded counts")
    print(input)
    fit_return = save_trees(location,input_counts,output_counts=output_counts,**kwargs)
    print(fit_return)

def save_trees(location,input_counts,output_counts=None,header=None,**kwargs):

    if output_counts is None:
        output_counts = input_counts

    np.savetxt(location + "input.counts",input_counts)
    np.savetxt(location + "output.counts",output_counts)

    if header is None:
        np.savetxt(location + "tmp.header", np.arange(output_counts.shape[1],dtype=int),fmt='%u')
    else:
        np.savetxt(location + "tmp.header", header,fmt="%s")

    print("Generating trees")

    inner_fit(input_counts,output_counts,location,header=(location + "tmp.header"),**kwargs)


def fit(input_counts,output_counts=None,header=None,**kwargs):

    if output_counts is None:
        output_counts = input_counts

    print("Setting context")

    print("Input:" + str(input_counts.shape))
    print("Output:" + str(output_counts.shape))

    tmp_dir = tmp.TemporaryDirectory()
    output = tmp_dir.name + "/"

    np.savetxt(output + "input.counts",input_counts)
    np.savetxt(output + "output.counts", output_counts)

    input_features = input_counts.shape[1]
    output_features = output_counts.shape[1]

    if header is None:
        np.savetxt(output + "tmp.i.header", np.arange(output_features,dtype=int),fmt='%u')
        np.savetxt(output + "tmp.o.header", np.arange(output_features,dtype=int),fmt='%u')
    else:
        np.savetxt(output + "tmp.i.header", header,fmt="%s")
        np.savetxt(output + "tmp.o.header", header,fmt="%s")

    print("CHECK TRUTH")
    print(tmp_dir.name)
    print(os.listdir(tmp_dir.name))

    print("Generating trees")

    inner_fit(input_counts,output_counts,output,ifh=output+"tmp.i.header",ofh=output+"tmp.o.header",**kwargs)

    print("CHECK OUTPUT")
    print(os.listdir(tmp_dir.name))

    forest = tr.Forest.load(output,prefix="tmp.*.compact",header="tmp.o.header",truth="output.counts")

    tmp_dir.cleanup()

    return forest


def inner_fit(input_counts,output_counts,location, **kwargs):

    # targets = "\n".join(["\t".join([str(y) for y in x]) for x in targets]) + "\n"

    path_to_rust = (Path(__file__).parent / "../target/release/lumberjack_1").resolve()

    print("Running " + str(path_to_rust))

    arg_list = [str(path_to_rust),"generate","-ic",location + "input.counts","-oc",location + "output.counts","-o",location + "tmp","-auto"]

    for arg in kwargs.keys():
        arg_list.append("-" + str(arg))
        arg_list.append(str(kwargs[arg]))

    print("Command: " + " ".join(arg_list))

    # cp = sp.run(arg_list,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)

    # cp = sp.Popen(arg_list,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)
    # try:
    #     output,error = cp.communicate(input=targets,timeout=1)
    # except:
    #     print("Communicated input")
    #
    with sp.Popen(arg_list,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True) as cp:
        # try:
        #     cp.communicate(input=targets,timeout=1)
        # except:
        #     pass
        print("Trying to readline")
        while True:
            # sleep(0.1)
            rc = cp.poll()
            if rc is not None:
                print(cp.stdout.read())
                print(cp.stderr.read())
                break
            output = cp.stdout.readline()
            # print("Read line")
            print(output.strip())


    # while cp.poll() is None:
    #     sys.stdout.flush()
    #     sys.stdout.write("Constructing trees: %s" % str(len(glob.glob(location + "tmp.*.compact"))) + "\r")
    #     # sys.stdout.write(str(os.listdir(location)))
    #     sleep(1)

    # print(cp.stdout.read())
    #
    # print(cp.stderr.read())

if __name__ == "__main__":
    kwargs = {x.split("=")[0]:x.split("=")[1] for x in sys.argv[3:]}
    main(sys.argv[1],sys.argv[2],**kwargs)


# ,feature_sub=None,distance=None,sample_sub=None,scaling=None,merge_distance=None,refining=False,error_dump=None,convergence_factor=None,smoothing=None,locality=None)
