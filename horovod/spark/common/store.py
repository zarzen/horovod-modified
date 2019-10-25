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
from __future__ import print_function

import os


class Store(object):
    def get_train_data_path(self):
        raise NotImplementedError()

    def get_val_data_path(self):
        raise NotImplementedError()

    def exists(self, path):
        raise NotImplementedError()

    def get_petastorm_path_generator(self):
        raise NotImplementedError()


class LocalStore(Store):
    def __init__(self, prefix_path):
        self.prefix_path = prefix_path

    def get_train_data_path(self):
        return self._get_path('train_data')

    def get_val_data_path(self):
        return self._get_path('val_data')

    def exists(self, path):
        return os.path.exists(path)

    def get_petastorm_path_generator(self):
        def get_path(path):
            return 'file://' + path
        return get_path

    def _get_path(self, key):
        return os.path.join(self.prefix_path, key)


class HDFSStore(Store):
    def __init__(self, prefix_path, spark):
        # spark context need to be present
        self.prefix_path = prefix_path
        self.spark = spark

    def get_train_data_path(self):
        return self._get_path('train_data')

    def get_val_data_path(self):
        return self._get_path('val_data')

    def exists(self, path):
        fs = self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(self.spark._jsc.hadoopConfiguration())
        return fs.exists(self.spark._jvm.org.apache.hadoop.fs.Path(path))

    def get_petastorm_path_generator(self):
        def get_path(path):
            import pydoop.hdfs as hdfs
            return hdfs.path.abspath(path)
        return get_path

    def _get_path(self, key):
        return os.path.join(self.prefix_path, key)



