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

from bottle import route, run, template, request
from inference import Hdrnet

# set the optimized graph path
hdrnet = Hdrnet('/home/models/optimized_graph.pb')

@route('/')
def index():
    return template('index')

@route('/infer', method='POST')
def infer():
    print(request.forms.get('image'))
    print(hdrnet.infer())

run(host='184.105.86.228', port=9999, reload=True)
