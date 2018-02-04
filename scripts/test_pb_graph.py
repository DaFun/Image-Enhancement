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

import tensorflow as tf
import hdrnet.models as models
import cv2
import numpy as np
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import argparse


def load_graph(pb_graph_file):
    # load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(pb_graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def main(args):
    input_path = args.input_image
    im_input = cv2.imread(input_path, -1)  # -1 means read as is, no conversions.
    if im_input.shape[2] == 4:
        im_input = im_input[:, :, :3]

    im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB.
    im_input = skimage.img_as_float(im_input)

    lowres_input = skimage.transform.resize(im_input, [256, 256], order=0)
    im_input = im_input[np.newaxis, :, :, :]
    lowres_input = lowres_input[np.newaxis, :, :, :]

    graph = load_graph(args.pb_file)

    # nodes names need to be customized if graph changed
    fullres = graph.get_tensor_by_name('fullres_input:0')
    lowres = graph.get_tensor_by_name('lowres_input:0')
    out = graph.get_tensor_by_name('output_img:0')

    with tf.Session(graph=graph) as sess:
        feed_dict = {
            fullres: im_input,
            lowres: lowres_input
        }
        # run the inference
        y_out = sess.run(out, feed_dict=feed_dict)

    img = Image.fromarray(y_out, 'RGB')
    img.save(args.output_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pb_file', default=None, help='path to the optimized graph')
    parser.add_argument('input_image', default=None, help='input image path')
    parser.add_argument('output_image', default=None, help='output image path')

    args = parser.parse_args()
    main(args)