import numpy as np
import sys
import os
import tempfile as tmp
from pathlib import Path
import io
import argparse
from time import sleep
import subprocess as sp
import tree_reader as tr

import numpy as np
# import matplotlib.pyplot as plt

def main():
    print("Running main")
    counts = np.loadtxt(sys.argv[1])
    generate(counts)

def generate(targets,header=None, **args):

# ,feature_sub=None,distance=None,sample_sub=None,scaling=None,merge_distance=None,refining=False,error_dump=None,convergence_factor=None,smoothing=None,locality=None)

    features = targets.shape[1]

    # np.array(targets)
    tmp_dir = tmp.TemporaryDirectory()
    output = tmp_dir.name + "/tmp."

    np.savetxt(output + "truth",targets)

    if header is None:
        np.savetxt(output + "header", np.arange(features))

    targets = "\n".join(["\t".join([str(y) for y in x]) for x in targets]) + "\n"

    # print(targets)

    # table_pipe = io.StringIO(targets)
    #
    # table_pipe.seek(0)

    path_to_rust = (Path(__file__).parent / "target/release/lumberjack_1").resolve()

    print("Running " + str(path_to_rust))

    arg_list = [path_to_rust,"generate","-stdin","-o",output ,"-auto"]

    for arg,val in args.iteritems():
        arg_list.append("-" + str(arg))
        if type(val) is not bool:
            arg_list.append(str(val))

    # arg_list = [str(path_to_rust),"generate"]
    # arg_list.extend(["-stdin"])
    # arg_list.extend(["-o",output])
    # if sample_sub is not None:
    #     arg_list.extend(["-ss",str(sample_sub)])
    # if feature_sub is not None:
    #     arg_list.extend(["-fs",str(feature_sub)])
    # if scaling is not None:
    #     arg_list.extend(["-sf",str(scaling)])
    # if merge_distance is not None:
    #     arg_list.extend(["-m",str(merge_distance)])
    # if error_dump is not None:
    #     arg_list.extend(["-error",str(error_dump)])
    # if convergence_factor is not None:
    #     arg_list.extend(["-convergence",str(convergence_factor)])
    # if locality is not None:
    #     arg_list.extend(["-l",str(locality)])
    # if smoothing is not None:
    #     arg_list.extend(["-smoothing",str(smoothing)])
    # if distance is not None:
    #     arg_list.extend(["-d",str(distance)])
    # if refining:
    #     arg_list.extend(["-refining"])

    print("Command: " + " ".join(arg_list))

    cp = sp.run(arg_list,input=targets,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)

    for stdout_line in iter(sp.stdout.readline, ""):
        yield stdout_line
    sp.stdout.close()

    print(sp.stderr)

    let forest = forest.load(output,prefix="/tmp.*.compact",header="/tmp.header",truth="/tmp.truth")

    tmp_dir.cleanup()

    return(list(map(lambda x: int(x),cp.stdout.split())))


if __name__ == "__main__":
    main()
