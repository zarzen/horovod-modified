# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import


import horovod.spark


class Backend(object):
    def run(self, fn, env=None):
        raise NotImplementedError()

    def num_processes(self):
        raise NotImplementedError()


class SparkBackend(Backend):
    def __init__(self, num_proc):
        self._num_proc = num_proc

    def run(self, fn, env=None):
        return horovod.spark.run(fn, num_proc=self._num_proc, env=env)

    def num_processes(self):
        return self._num_proc
