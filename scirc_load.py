def parser():
    file_prefix = file.rsplit(".", 1)[0]
    print(f">>> loading scirc file: {file}")
    with open(file, "r") as inf:
        for line in inf:
            kw, *args = line.rstrip().split(" ")
            if kw in {"WIRE", "WIRES", "NODE", "NODES"}:
                for wire in args:
                    if wire in reserved_keywords:
                        raise ScircError(
                            f"{wire} is a reserved keyword and cannot be declared as a node."
                        )
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
                input_set.discard(op_output)  # is an output value
                op_name = f"{file_prefix}_{kw}_{next(unique_counter)}"
                op_map[op_name] = Operation(
                    op_name, kw, op_inputs, op_output, defined_ops[kw]
                )
                for node in op_inputs:
                    dependency_dict[node].add(op_map[op_name])