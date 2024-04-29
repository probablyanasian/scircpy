import copy
import glob
import argparse
import functools
import itertools
from typing import Callable


class Node:
    def __init__(self, name: str, voltage: int = 5):
        self.name = name
        self.value = False  # default low should be okay...
        self.voltage = voltage

    def set(self, value: bool | None):
        # cast to bool since it's possible that a Node is passed in due to reduce
        self.value = bool(value)

    def __repr__(self):
        return f"{self.name}: {self.value}"

    def __bool__(self):
        return self.value


class Operation:
    def __init__(self, name: str, type: str, inputs: list[Node], output: Node, op: Callable[[Node], bool]):
        self.name = name
        self.type = type
        self.inputs = inputs
        self.output = output
        self.op = op

    def reassign(self, op: Callable[[Node], bool]):
        self.op = op

    def compute(self) -> bool:
        new_value = self.op(self.inputs)
        # cast for clarity
        old_value = bool(self.output)
        self.output.set(new_value)
        return old_value ^ new_value

    def __repr__(self):
        return f"{self.name}: {self.inputs} -> {self.output}"


def _and(inputs: list[Node]) -> bool:
    return functools.reduce(lambda x, y: x and y, inputs[1:], inputs[0])


def _or(inputs: list[Node]) -> bool:
    return functools.reduce(lambda x, y: x or y, inputs[1:], inputs[0])


def _nand(inputs: list[Node]) -> bool:
    return not functools.reduce(lambda x, y: x and y, inputs[1:], inputs[0])


def _nor(inputs: list[Node]) -> bool:
    return not functools.reduce(lambda x, y: x or y, inputs[1:], inputs[0])


def _not(inputs: list[Node]) -> bool:
    return not inputs[0]


def main():
    args = parser.parse_args()
    nodal_map = {}
    op_map = {}
    # network_map = {}
    probe_list = []
    defined_ops = {"AND": _and, "OR": _or, "NAND": _nand, "NOR": _nor, "NOT": _not}
    if args.all_files:
        scirc_files = glob.glob("*.scirc")
    else:
        print(args.filenames)
        scirc_files = args.filenames
    unique_counter = itertools.count()

    # load in each file, mangle such that we can do duplication and linkage of components
    for file in scirc_files:
        file_prefix = file.rsplit(".", 1)[0]
        print(f">>> loading scirc file: {file}")
        with open(file, "r") as inf:
            for line in inf:
                kw, *args = line.rstrip().split(" ")
                if kw in {"WIRE", "WIRES", "NODE", "NODES"}:
                    for wire in args:
                        nodal_map[f"{file_prefix}_{wire}"] = Node(
                            f"{file_prefix}_{wire}"
                        )
                elif kw == "MEASURE":
                    for wire in args:
                        probe_list.append(nodal_map[f"{file_prefix}_{wire}"])
                elif kw == "IMPORT":
                    pass
                elif kw == "EXPORT":
                    pass
                elif (
                    kw.upper() in defined_ops
                ):  # begin definition of logic gates. TODO: make this extensible
                    output, *inputs = args
                    op_inputs = [nodal_map[f"{file_prefix}_{n}"] for n in inputs]
                    op_output = nodal_map[f"{file_prefix}_{output}"]
                    op_name = f"{file_prefix}_{kw}_{next(unique_counter)}"
                    op_map[op_name] = Operation(op_name, kw, op_inputs, op_output, defined_ops[kw])
    print(op_map)

    runtime_ops = {"AUTOEXEC": True}
    while True:
        print(not nodal_map[f"{file_prefix}_a"])
        op_map["or_OR_0"].compute()
        print(op_map)
        break


parser = argparse.ArgumentParser(
    description="Small, extensible circuit design and simulation language",
    epilog="",
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-a",
    "--all",
    dest="all_files",
    action="store_true",
    help="Load all .scirc files in the directory",
)
group.add_argument(
    "-f",
    dest="filenames",
    action="append",
    metavar="filename",
    help="source files to be loaded directly",
)
parser.add_argument(
    "-l",
    "--allow-dependency-loop",
    dest="dep_loop",
    action="store_true",
    help="allow loops in the dependency tree",
)
parser.add_argument(
    "-d",
    "--depth",
    dest="dep_loop",
    action="store",
    metavar="N",
    default=1000,
    type=int,
    help="Number of ops/computations allowed before convergence error (default: 1000)",
)

if __name__ == "__main__":
    main()
