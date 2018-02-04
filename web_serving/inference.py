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
import numpy as np
import skimage
import skimage.io
import skimage.transform
import base64
import cv2
from PIL import Image

class Hdrnet(object):
    def __init__(self, checkpoint):
        self.mask_id = 5800
        self.checkpoint = checkpoint
        self.sess = tf.Session()
        self.graph = self.load_graph(checkpoint)

    def load_graph(self, graph):
        # load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(graph, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
        return graph

    def preprocess(self, file):
        data = base64.b64decode(file)
        return data


    def infer(self, file):
        """ Perform inferencing.  In other words, generate a paraphrase
        for the source sentence.

        Args:
            file : input buffer from memory

        Returns:
            new_image: numpy array
        """

        # img = self.preprocess(file)
        im_input = cv2.imdecode(file, -1)  # -1 means read as is, no conversions.
        if im_input.shape[2] == 4:
            im_input = im_input[:, :, :3]

        im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB.
        im_input = skimage.img_as_float(im_input)

        lowres_input = skimage.transform.resize(im_input, [256, 256], order=0)
        im_input = im_input[np.newaxis, :, :, :]
        lowres_input = lowres_input[np.newaxis, :, :, :]

        fullres = self.graph.get_tensor_by_name('fullres_input:0')
        lowres = self.graph.get_tensor_by_name('lowres_input:0')
        out = self.graph.get_tensor_by_name('output_img:0')

        feed_dict = {
            fullres: im_input,
            lowres: lowres_input
        }

        y_out = self.sess.run(out, feed_dict=feed_dict)
        return y_out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='optimized graph path')
    parser.add_argument('input_image', type=str, help='input image file')
    parser.add_argument('output_image', type=str, help='output image path')
    args = parser.parse_args()
    hdrnet = Hdrnet(args.checkpoint)

    with open(args.image_file, 'rb') as f:
        img = f.read()
        new_image = hdrnet.infer(img)

    return new_image
    # img = Image.fromarray(new_image, 'RGB')
    # img.save(args.output_image)

if __name__ == '__main__':
    main()