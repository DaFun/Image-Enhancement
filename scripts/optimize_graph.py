#!/usr/bin/env python
# encoding: utf-8
# Copyright 2018 Fei Cheng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import tensorflow as tf
import hdrnet.models as models
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.pywrap_tensorflow import TransformGraphWithStringInputs
from tensorflow.python.util import compat

def TransformGraph(input_graph_def, inputs, outputs, transforms):
    """Python wrapper for the Graph Transform Tool.

    Gives access to all graph transforms available through the command line tool.
    See documentation at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
    for full details of the options available.

    Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    inputs: List of node names for the model inputs.
    outputs: List of node names for the model outputs.
    transforms: List of strings containing transform names and parameters.

    Returns:
    New GraphDef with transforms applied.
    """

    input_graph_def_string = input_graph_def.SerializeToString()
    inputs_string = compat.as_bytes(",".join(inputs))
    outputs_string = compat.as_bytes(",".join(outputs))
    transforms_string = compat.as_bytes(" ".join(transforms))
    with errors.raise_exception_on_not_ok_status() as status:
        output_graph_def_string = TransformGraphWithStringInputs(
            input_graph_def_string, inputs_string, outputs_string,
            transforms_string, status)
    output_graph_def = graph_pb2.GraphDef()
    output_graph_def.ParseFromString(output_graph_def_string)
    return output_graph_def


def load_graph(frozen_graph_path):
    # load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def write_trans_graph(output_graph, output_graph_def):
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def main(args):
    graph_def = load_graph(args.frozen_path)
    out = TransformGraph(graph_def, args.input_nodes, args.output_nodes,
                         ['strip_unused_nodes', 'remove_nodes(op=Identity, op=CheckNumerics)', 'merge_duplicate_nodes',
                          'fold_constants(ignore_errors=true)', 'fold_batch_norms', 'sort_by_execution_order',
                          'strip_unused_nodes'])
    write_trans_graph(args.optimized_path, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frozen_path', default=None, help='path to the saved frozen graph')
    parser.add_argument('optimized_path', default=None, help='path to output optimized graph')
    parser.add_argument('input_nodes', nargs='+', help='input nodes names of the graph')
    parser.add_argument('output_nodes', nargs='+', help='output nodes names of the graph')

    args = parser.parse_args()
    main(args)