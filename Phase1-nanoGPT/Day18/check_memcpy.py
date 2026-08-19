import onnx
g = onnx.load('memcpy_evidence_before_optimized_graph.onnx')

memcpy_nodes = [n for n in g.graph.node if 'Memcpy' in n.op_type]
for n in memcpy_nodes:
    print(n.op_type, n.name, '<-', list(n.input))

target_input = memcpy_nodes[0].input[0]
for n in g.graph.node:
    if target_input in n.output:
        print('producer:', n.op_type, n.name)
