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

from bottle import route, run, template, request, response, BaseRequest
BaseRequest.MEMFILE_MAX = 256000000
from inference import Hdrnet
import base64

# set up different model def graph
hdrnet_face = Hdrnet('optimized_graph.pb', 'face')
hdrnet_edge = Hdrnet('optimized_edge.pb', 'edge')
hdrnet_hdr = Hdrnet('optimized_hdr.pb', 'hdr')


@route('/')
def index():
    return template('index')

@route('/infer', method='POST')
def infer():
    file = request.forms.get('data')
    mode = request.forms.get('mode')
    if mode == 'face':
        data = hdrnet_face.infer(file)
    elif mode == 'edge':
        data = hdrnet_edge.infer(file)
    elif mode == 'hdr':
        data = hdrnet_hdr.infer(file)

    response.content_type = 'text/json'
    return {'data': base64.b64encode(data)}

# set inet address
run(host='10.64.25.231', port=9999, reload=True)
