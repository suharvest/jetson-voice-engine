#!/usr/bin/env python3
"""Replace an exporter-generated dynamic If with its batch-1 Squeeze branch.

TensorRT 10.3 rejects the original If because ONNX shape inference describes
its branches as [1, 1024] and [1, 1024, 1]. Qwen3-TTS voice cloning uses a
batch-1 mel input and the encoder's final Conv has a singleton last dimension,
so the Squeeze branch is the applicable path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()

    model = onnx.load(args.source, load_external_data=True)
    matches = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.op_type == "If" and node.name == "/enc/If"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one /enc/If node, found {len(matches)}")

    index, conditional = matches[0]
    then_branch = next(
        attribute.g for attribute in conditional.attribute if attribute.name == "then_branch"
    )
    squeeze = next(node for node in then_branch.node if node.op_type == "Squeeze")
    if list(squeeze.input[:1]) != ["/enc/fc/Conv_output_0"]:
        raise RuntimeError(f"unexpected Squeeze input: {list(squeeze.input)}")

    axes_name = "/enc/If_trt_squeeze_axes"
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([2], dtype=np.int64), name=axes_name)
    )
    replacement = helper.make_node(
        "Squeeze",
        inputs=["/enc/fc/Conv_output_0", axes_name],
        outputs=list(conditional.output),
        name="/enc/If_trt_squeeze",
    )
    del model.graph.node[index]
    model.graph.node.insert(index, replacement)

    onnx.checker.check_model(model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    print(f"replaced /enc/If with batch-1 Squeeze axis=2: {output}")


if __name__ == "__main__":
    main()
