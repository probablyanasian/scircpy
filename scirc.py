import copy
import glob
import argparse
import functools
import itertools
from collections import deque
from typing import Callable


class Node:
    def __init__(self, name: str, voltage: int = 5):
        self.name = name
        self.value = False # could have cyclic circuits like SR Latch, so sim cold start.
        self.voltage = voltage

    def set(self, value: bool | None): # debate on removing `| None`
        # cast to bool since it's possible that a Node is passed in due to how reduce works
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

    def execute(self) -> bool:
        new_value = bool(self.op(self.inputs))
        # cast for clarity
        if self.output.value is not None:
            old_value = bool(self.output)
        else:
            old_value = not new_value # changed = True

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

class ScircError(Exception):
    pass

def main():
    proc_args = parser.parse_args()
    nodal_map: dict[str, Node] = {} # name: Node
    op_map: dict[str, Operation] = {} # name: Op
    dependency_dict: dict[Node, set[Operation]] = {} # Node: [Ops], input: ops
    reserved_keywords = {"clk", "clock"}
    input_set = set()
    input_name_set = set()
    # network_map = {}
    probe_list = []
    defined_ops = {"AND": _and, "OR": _or, "NAND": _nand, "NOR": _nor, "NOT": _not}
    if proc_args.all_files:
        scirc_files = glob.glob("*.scirc")
    else:
        print(proc_args.filenames)
        scirc_files = proc_args.filenames
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
                        if wire in reserved_keywords:
                            raise ScircError(f"{wire} is a reserved keyword and cannot be declared as a node.")
                        new_node = Node(f"{file_prefix}_{wire}")
                        nodal_map[f"{file_prefix}_{wire}"] = new_node
                        input_set.add(new_node)
                        input_name_set.add(f"{file_prefix}_{wire}")
                        dependency_dict[new_node] = set()
                elif kw in {"PROBE", "MEASURE"}:
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
                    input_set.discard(op_output) # is an output value
                    op_name = f"{file_prefix}_{kw}_{next(unique_counter)}"
                    op_map[op_name] = Operation(op_name, kw, op_inputs, op_output, defined_ops[kw])
                    for node in op_inputs:
                        dependency_dict[node].add(op_map[op_name])

    runtime_ops = {"AUTOEXEC": True}
    execution_queue: deque[Operation] = deque()
    for node in input_set:
        if node.value is None:
            node.set(False)
        execution_queue.extend(dependency_dict[node])

    # do initial computation of the whole circuit to establish ground state.
    seen = set()
    depth_counter = 0
    while len(execution_queue) > 0:
            cur = execution_queue.popleft()
            if any(n.value is None for n in cur.inputs):
                execution_queue.append(cur) # delay exec until inputs are known
                continue
            if cur.execute() or cur not in seen:
                execution_queue.extend(dependency_dict[cur.output])
                seen.add(cur)
            depth_counter += 1
            if depth_counter >= proc_args.max_depth:
                raise ScircError(f"Exceeded maximum allowed depth. (Currently: {proc_args.max_depth})")
    # TODO: optimize by delaying execution of duplicate nodes on the execution queue(?)
    # validate above logic
    while True:
        depth_counter = 0
        while len(execution_queue) > 0:
            cur = execution_queue.popleft()
            if any(n.value is None for n in cur.inputs):
                execution_queue.append(cur) # delay exec until inputs are known
                continue
            if cur.execute():
                execution_queue.extend(dependency_dict[cur.output])
            depth_counter += 1
            if depth_counter >= proc_args.max_depth:
                raise ScircError(f"Exceeded maximum allowed depth. (Currently: {proc_args.max_depth})")
        user_input = input("> ").strip().lower()
        if user_input in {"exit", "quit", "q"}:
            break
        elif user_input in {"show", "s"}:
            print(probe_list)
        elif user_input in {"show all", "sa"}:
            print({n for n in nodal_map.values()})
        elif user_input in {"help", "h"}:
            print("Available Commands:")
            print("show (s): \tShow logic level for currently probed nodes")
            print("show all (sa): \tShow logic level for all nodes (Potentially long output)")
            print("quit (q): \tExit the program")
        elif user_input.startswith("set"):
            _, node, val = user_input.split(" ")
            if node not in nodal_map:
                print("Node is not known.")
                continue
            if node not in input_name_set:
                print("Node is Not an input node.")
                continue
            if val.lower() in {"1", "true"}:
                s_node = nodal_map[node]
                s_node.set(True)
                execution_queue.extend(dependency_dict[s_node])
            elif val.lower() in {"0", "false"}:
                s_node = nodal_map[node]
                s_node.set(False)
                execution_queue.extend(dependency_dict[s_node])
            else:
                print("Unknown set value")
        else:
            print("Unknown Command")


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
    "--max-depth",
    dest="max_depth",
    action="store",
    metavar="N",
    default=1000,
    type=int,
    help="Number of ops/computations allowed before convergence error (default: 1000)",
)

if __name__ == "__main__":
    main()
