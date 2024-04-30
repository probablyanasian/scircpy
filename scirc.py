import copy
import glob
import argparse
import functools
import itertools
from collections import deque
from typing import Callable


class Node:
    def __init__(self, name: str, voltage: int = 5) -> None:
        self.name = name
        self.value = (
            False  # could have cyclic circuits like SR Latch, so sim cold start.
        )
        self.voltage = voltage

    def set(self, value: bool | None) -> None:  # debate on removing `| None`
        # cast to bool since it's possible that a Node is passed in due to how reduce works
        self.value = bool(value)

    def __repr__(self) -> str:
        return f"{self.name}: {self.value}"

    def __bool__(self) -> bool:
        return self.value


class Operation:
    def __init__(
        self,
        name: str,
        type: str,
        inputs: list[Node],
        output: Node,
        op: Callable[[Node], bool],
    ):
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
            old_value = not new_value  # changed = True

        self.output.set(new_value)
        return old_value ^ new_value

    def __repr__(self) -> str:
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


class HexProbe:
    def __init__(self, wires: list[Node]) -> None:
        self.wires = wires

    def __repr__(self) -> str:
        return hex(int("".join(["1" if bool(n) else "0" for n in self.wires]), 2))


class NetworkMap:
    def __init__(self, name, parent_chain: set[str]) -> None:
        self.parent_chain = (
            parent_chain.copy()
        )  # strings are primitives so this is okay
        self.parent_chain.add(name)
        self.name = name
        self.nodal_map: dict[str, Node] = {}  # name: Node
        self.op_map: dict[str, Operation] = {}  # name: Op
        self.dependency_dict: dict[Node, set[Operation]] = {}  # Node: [Ops], input: ops
        self.input_set = set()
        self.input_name_set = set()
        self.extended_ops = {}
        self.unique_counter = itertools.count()


def scirc_parse(net: NetworkMap, filename: str) -> None:
    defined_ops = {"AND": _and, "OR": _or, "NAND": _nand, "NOR": _nor, "NOT": _not}
    reserved_keywords = {"clk", "clock"}
    file_prefix = filename.rsplit(".", 1)[0]
    print(f">>> loading scirc file: {filename}")
    with open(filename, "r") as inf:
        for line in inf:
            kw, *args = line.rstrip().split(" ")
            if kw in {"WIRE", "WIRES", "NODE", "NODES"}:
                for wire in args:
                    if wire in reserved_keywords:
                        raise ScircError(
                            f"{wire} is a reserved keyword and cannot be declared as a node."
                        )
                    new_node = Node(f"{file_prefix}_{wire}")
                    net.nodal_map[f"{file_prefix}_{wire}"] = new_node
                    net.input_set.add(new_node)
                    net.input_name_set.add(f"{file_prefix}_{wire}")
                    net.dependency_dict[new_node] = set()
            elif kw in {"PROBE", "MEASURE"}:
                fmt, *wires = args
                if fmt in {"BITS", "BIT", "B"}:
                    for wire in wires:
                        net.probe_list.append(net.nodal_map[f"{file_prefix}_{wire}"])
                elif fmt in {"HEX", "H"}:
                    net.probe_list.append(
                        HexProbe(
                            [net.nodal_map[f"{file_prefix}_{wire}"] for wire in wires]
                        )
                    )
            elif kw == "IMPORT":
                pass
            elif kw == "EXPORT":
                pass
            elif (
                kw.upper() in defined_ops or kw.upper() in net.extended_ops
            ):  # begin definition of logic gates. TODO: make this extensible
                output, *inputs = args
                op_inputs = [net.nodal_map[f"{file_prefix}_{n}"] for n in inputs]
                op_output = net.nodal_map[f"{file_prefix}_{output}"]
                net.input_set.discard(op_output)  # is an output value
                op_name = f"{file_prefix}_{kw}_{next(net.unique_counter)}"
                net.op_map[op_name] = Operation(
                    op_name, kw, op_inputs, op_output, defined_ops[kw]
                )
                for node in op_inputs:
                    net.dependency_dict[node].add(net.op_map[op_name])


def main():
    proc_args = parser.parse_args()
    gnm = NetworkMap(proc_args.filename, set())

    extended_ops = {}
    if proc_args.all_files:
        scirc_file = glob.glob("*.scirc")
        raise ScircError("Unimplemented Error: Not supported yet.")
    else:
        print(proc_args.filename)
        # load in file, mangle such that we can do duplication and linkage of components
        scirc_parse(gnm, proc_args.filename)

    runtime_ops = {"AUTOEXEC": True}
    execution_queue: deque[Operation] = deque()
    for node in gnm.input_set:
        if node.value is None:
            node.set(False)
        execution_queue.extend(gnm.dependency_dict[node])

    # do initial computation of the whole circuit to establish ground state.
    seen = set()
    depth_counter = 0
    while len(execution_queue) > 0:
        cur = execution_queue.popleft()
        if any(n.value is None for n in cur.inputs):
            execution_queue.append(cur)  # delay exec until inputs are known
            continue
        if cur.execute() or cur not in seen:
            execution_queue.extend(gnm.dependency_dict[cur.output])
            seen.add(cur)
        depth_counter += 1
        if depth_counter >= proc_args.max_depth:
            raise ScircError(
                f"Exceeded maximum allowed depth. (Currently: {proc_args.max_depth})"
            )

    # TODO: optimize by delaying execution of duplicate nodes on the execution queue(?)
    # validate above logic
    while True:
        depth_counter = 0
        while len(execution_queue) > 0:
            cur = execution_queue.popleft()
            if any(n.value is None for n in cur.inputs):
                execution_queue.append(cur)  # delay exec until inputs are known
                continue
            if cur.execute():
                execution_queue.extend(gnm.dependency_dict[cur.output])
            depth_counter += 1
            if depth_counter >= proc_args.max_depth:
                raise ScircError(
                    f"Exceeded maximum allowed depth. (Currently: {proc_args.max_depth})"
                )
        user_input = input("> ").strip().lower()
        if user_input in {"exit", "quit", "q"}:
            break
        elif user_input in {"show", "s"}:
            print(gnm.probe_list)
        elif user_input in {"show all", "sa"}:
            print({n for n in gnm.nodal_map.values()})
        elif user_input in {"help", "h"}:
            print("Available Commands:")
            print("show (s): \tShow logic level for currently probed nodes")
            print(
                "show all (sa): \tShow logic level for all nodes (Potentially long output)"
            )
            print("quit (q): \tExit the program")
        elif user_input.startswith("set"):
            input_split = user_input.split(" ")
            if len(input_split) != 3:
                print("Invalid number of inputs")
            _, node, val = input_split
            if node not in gnm.nodal_map:
                print("Node is not known.")
                continue
            if node not in gnm.input_name_set:
                print("Node is Not an input node.")
                continue
            if val.lower() in {"1", "true"}:
                s_node = gnm.nodal_map[node]
                s_node.set(True)
                execution_queue.extend(gnm.dependency_dict[s_node])
            elif val.lower() in {"0", "false"}:
                s_node = gnm.nodal_map[node]
                s_node.set(False)
                execution_queue.extend(gnm.dependency_dict[s_node])
            else:
                print("Unknown set value")
        else:
            print("Unknown Command")


parser = argparse.ArgumentParser(
    description="A small, extensible circuit design and simulation language",
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
    dest="filename",
    action="store",
    metavar="filename",
    help="source file to be loaded",
)
parser.add_argument(
    "-l",
    "--allow-dependency-loop",
    dest="dep_loop",
    action="store_true",
    help="allow loops in a file's dependency tree",
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
