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

import horovod.spark.common._namedtuple_fix

import io
import math
import numbers
import os
import sys

import h5py
import numpy as np
import tensorflow as tf

from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.util import MLWritable, MLReadable
from pyspark.ml.param.shared import Param, Params

from horovod.run.common.util import codec

from horovod.spark.common import util
from horovod.spark.common.params import EstimatorParams, ModelParams
from horovod.spark.common.serialization import \
    HorovodParamsWriter, HorovodParamsReader
from horovod.spark.keras import optimizer

PETASTORM_HDFS_DRIVER = 'libhdfs'

_training_cache = None

BARE_KERAS = 'keras'
TF_KERAS = 'tf_keras'


def _is_instance_of_bare_keras_optimizer(opt):
    import keras
    return isinstance(opt, keras.optimizers.Optimizer)


def _is_instance_of_bare_keras_model(model):
    import keras
    return isinstance(model, keras.models.Model)


def _serialize_keras_model(model, save_model_fn):
    """Serialize model into byte array encoded into base 64."""
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        save_model_fn(model, f)
    return codec.dumps_base64(bio.getvalue())


def _deserialize_keras_model_generator():
    _deserialize = util.deserialize_generator()

    def deserialize_keras_model_fn(model_bytes, load_model_fn):
        """Deserialize model from byte array encoded in base 64."""
        model_bytes = _deserialize(model_bytes)
        bio = io.BytesIO(model_bytes)
        with h5py.File(bio, 'r') as f:
            return load_model_fn(f)

    return deserialize_keras_model_fn


def _deserialize_keras_model(model_bytes, load_model_fn):
    _deserialize_keras_model_fn = _deserialize_keras_model_generator()
    return _deserialize_keras_model_fn(model_bytes, load_model_fn)


def _serialize_param_value(param_name, param_val, serialize_model_fn, serialize_opt_fn):
    if param_val is None:
        return param_val

    if param_name in [EstimatorParams.backend.name, EstimatorParams.store.name]:
        # We do not serialize backend and store. These params have to be regenerated for each
        # run of the pipeline
        return None
    elif param_name == EstimatorParams.model.name:
        return serialize_model_fn(param_val)
    if param_name == KerasEstimator.optimizer.name:
        return serialize_opt_fn(param_val)
    else:
        return codec.dumps_base64(param_val)


class TFKerasUtil(object):
    type = TF_KERAS

    @staticmethod
    def get_horovod():
        import horovod.tensorflow.keras as hvd
        return hvd

    @staticmethod
    def keras():
        import tensorflow.keras as tf_keras
        return tf_keras

    @staticmethod
    def serialize_optimizer(*args, **kwargs):
        return optimizer.serialize_tf_keras_optimizer(*args, **kwargs)

    @staticmethod
    def deserialize_optimizer(*args, **kwargs):
        return optimizer.deserialize_tf_keras_optimizer(*args, **kwargs)

    @staticmethod
    def serialize_model(*args, **kwargs):
        def serialize_keras_model(x):
            return _serialize_keras_model(x, TFKerasUtil.keras().models.save_model)

        return serialize_keras_model(*args, **kwargs)

    @staticmethod
    def deserialize_model(*args, **kwargs):
        return _deserialize_keras_model(*args, **kwargs)

    @staticmethod
    def serialize_param_value(*args, **kwargs):
        def _serialize_param(x, y):
            return _serialize_param_value(x, y,
                                          serialize_model_fn=TFKerasUtil.serialize_model,
                                          serialize_opt_fn=TFKerasUtil.serialize_optimizer)

        return _serialize_param(*args, **kwargs)


class BareKerasUtil(object):
    type = BARE_KERAS

    @staticmethod
    def get_horovod():
        import horovod.keras as hvd
        return hvd

    @staticmethod
    def keras():
        import keras
        return keras

    @staticmethod
    def serialize_optimizer(*args, **kwargs):
        return optimizer.serialize_bare_keras_optimizer(*args, **kwargs)

    @staticmethod
    def deserialize_optimizer(*args, **kwargs):
        return optimizer.deserialize_bare_keras_optimizer(*args, **kwargs)

    @staticmethod
    def serialize_model(*args, **kwargs):
        def serialize_keras_model(x):
            return _serialize_keras_model(x, BareKerasUtil.keras().models.save_model)

        return serialize_keras_model(*args, **kwargs)

    @staticmethod
    def deserialize_model(*args, **kwargs):
        return _deserialize_keras_model(*args, **kwargs)

    @staticmethod
    def serialize_param_value(*args, **kwargs):
        def _serialize_param(x, y):
            return _serialize_param_value(x, y,
                                          serialize_model_fn=BareKerasUtil.serialize_model,
                                          serialize_opt_fn=BareKerasUtil.serialize_optimizer)

        return _serialize_param(*args, **kwargs)


class KerasEstimatorParamsWriter(HorovodParamsWriter):

    def saveImpl(self, path):
        keras_utils = self.instance._get_keras_utils()
        # Write the parameters
        HorovodParamsWriter.saveMetadata(self.instance, path, self.sc,
                                         param_serializer_fn=keras_utils.serialize_param_value)


class KerasEstimatorParamsWritable(MLWritable):

    def write(self):
        return KerasEstimatorParamsWriter(self)


class KerasEstimatorParamsReader(HorovodParamsReader):

    def _deserialize_dict(self, dict):
        def _param_deserializer_fn(name, param_val, keras_utils, custom_objects):
            if param_val is None:
                return param_val

            if name == EstimatorParams.model.name:
                def load_model_fn(x):
                    with keras_utils.keras().utils.custom_object_scope(custom_objects):
                        return keras_utils.keras().models.load_model(x, compile=True)

                return keras_utils.deserialize_model(param_val,
                                                     load_model_fn=load_model_fn)
            elif name == KerasEstimator.optimizer.name:
                opt_base64_encoded = codec.loads_base64(param_val)
                return keras_utils.deserialize_optimizer(opt_base64_encoded)
            else:
                return codec.loads_base64(param_val)

        # In order to deserialize the model, we need to deserialize the custom_objects param
        # first.
        keras_utils = None
        if KerasEstimator._keras_pkg_type.name in dict:
            keras_pkg_type = _param_deserializer_fn(KerasEstimator._keras_pkg_type.name,
                                                    dict[KerasEstimator._keras_pkg_type.name],
                                                    None, None)
            if keras_pkg_type == BARE_KERAS:
                keras_utils = BareKerasUtil
            elif keras_pkg_type == TF_KERAS:
                keras_utils = TFKerasUtil

        custom_objects = {}
        if KerasEstimator.custom_objects.name in dict:
            custom_objects = _param_deserializer_fn(KerasEstimator.custom_objects.name,
                                                    dict[KerasEstimator.custom_objects.name],
                                                    None, None)

        for key, val in dict.items():
            dict[key] = _param_deserializer_fn(key, val, keras_utils, custom_objects)
        return dict


class KerasEstimatorParamsReadable(MLReadable):

    @classmethod
    def read(cls):
        """Returns a KerasEstimatorParamsReader instance for this class."""
        return KerasEstimatorParamsReader(cls)


def calculate_shuffle_buffer_size_generator():
    TOTAL_BUFFER_MEMORY_CAP = util.TOTAL_BUFFER_MEMORY_CAP
    ONE_GB = util.ONE_GB

    def calculate_shuffle_buffer_size(hvd, avg_row_size, train_row_count_per_worker):
        """
        Determines the shuffling buffer size such that each worker gets at most 1GB for shuffling buffer
        such that on a single machine, among all the workers on that machine, at most memory_cap_gb GB
        are allocated for shuffling buffer. Also, it ensures that the buffer size is identical among
        all the workers.

        example 1:
        memory_cap_gb = 4
        machine1: 8 workers
        machine2: 3 workers
        shuffle_buffer_size = 0.5 GB

        example 2:
        memory_cap_gb = 4
            machine1: 2 workers
            machine2: 3 workers
        shuffle_buffer_size = 1 GB

        example 3:
        memory_cap_gb = 4
            machine1: 2 workers
            machine2: 8 workers
            machine3: 5 workers
        shuffle_buffer_size = 0.5 GB
        """
        local_size = hvd.local_size()
        print(hvd.__path__)
        local_sizes = [1] #hvd.allgather([local_size])
        max_local_size = max(local_sizes)

        if max_local_size > TOTAL_BUFFER_MEMORY_CAP:
            shuffle_buffer_size = TOTAL_BUFFER_MEMORY_CAP * ONE_GB / avg_row_size / max_local_size
        else:
            shuffle_buffer_size = ONE_GB / avg_row_size

        return int(min(shuffle_buffer_size, train_row_count_per_worker))

    return calculate_shuffle_buffer_size


def custom_sparse_to_dense_generator():
    # TODO (Fardin): ask petastorm team about codecs for sparse and dense vectors and see if that is
    # a better solution
    def custom_sparse_to_dense(custom_sparse_vec, dense_shape):
        # original sparse vector:   v = {1:2.0, 3:.4.5, 5:7.1}
        # custom sparse vector:     v = [3, 1, 3, 5, 2.0, 4.5, 7.1]
        # dense vector:             v = [0, 2.0, 0, 4.5, 0, 7.1]

        # Get the first element from custom_sparse_vec. This element is the size of
        # non-zero elements in the original sparse vector.
        sparse_vector_size = tf.cast(tf.gather(custom_sparse_vec, 0, axis=0), tf.int32)
        sparse_vector_size = tf.reshape(sparse_vector_size, [1])

        # get the first sparse_vector_size elements of the custom_sparse_vec which are the
        # indices
        indices_1d = tf.to_int64(
            tf.slice(custom_sparse_vec, begin=tf.constant([1]), size=sparse_vector_size))
        indices_reshaped = tf.reshape(indices_1d,
                                      tf.concat([sparse_vector_size, tf.constant([1])], 0))
        # have to pad the indices to match the expected format by the SparseTensor
        indices = tf.pad(indices_reshaped, [[0, 0], [1, 0]], "CONSTANT")

        # get the second sparse_vector_size elements of the custom_sparse_vec which are
        # the values
        begin_index = sparse_vector_size + tf.constant(1)
        values = tf.slice(custom_sparse_vec, begin=begin_index, size=sparse_vector_size)

        # construct a sparse vector with the indices and values
        dense_shape = [1, dense_shape]
        sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values,
                                               dense_shape=dense_shape)
        # convert the sparse vector into a dense vector
        return tf.sparse.to_dense(sparse_tensor)

    return custom_sparse_to_dense


def convert_custom_sparse_to_dense_bare_keras_generator():
    def convert_custom_sparse_to_dense(row, shape):
        size = int(row[0])
        dense_row = np.zeros(shape)
        dense_row[row[1:size + 1].astype(int)] = row[size + 1:2 * size + 1]
        return dense_row

    return convert_custom_sparse_to_dense


def prepare_data_bare_keras_generator(metadata):
    convert_custom_sparse_to_dense = convert_custom_sparse_to_dense_bare_keras_generator()
    CUSTOM_SPARSE = util.CUSTOM_SPARSE

    def prepare_data_bare_keras(rows, col, shape):
        intermediate_format = metadata[col]['intermediate_format']
        if intermediate_format != CUSTOM_SPARSE:
            return rows.reshape(shape)

        dense_rows = []
        shape_1d = metadata[col]['shape']
        for row in rows:
            dense_row = convert_custom_sparse_to_dense(row, shape_1d)
            dense_rows.append(dense_row)
        return np.array(dense_rows).reshape(shape)

    return prepare_data_bare_keras


def batch_generator_generator(feature_columns, label_columns, sample_weight_col,
                              input_shapes, output_shapes, batch_size,
                              metadata):
    prepare_data_bare_keras_fn = prepare_data_bare_keras_generator(metadata)

    cols = feature_columns + label_columns
    if sample_weight_col:
        cols.append(sample_weight_col)

    def batch_generator(reader, shuffle_buffer_size):

        while True:
            num_rows_read_sofar = 0
            data = None
            while num_rows_read_sofar < shuffle_buffer_size:
                # Each call to next reads one row group at a time. reader is an infinite
                # generator and never ends
                row_group_data = next(reader)
                if not data:
                    data = {col: getattr(row_group_data, col) for col in cols}
                else:
                    for col in cols:
                        data[col] = np.concatenate((data[col],
                                                    getattr(row_group_data, col)))
                num_rows_read_sofar += row_group_data[0].shape[0]

            # Create a permutation of len of data and use it to shuffle each numpy array
            perm = np.random.permutation(num_rows_read_sofar)
            inputs = [prepare_data_bare_keras_fn(data[col][perm], col, shape) for col, shape
                      in zip(feature_columns, input_shapes)]
            labels = [prepare_data_bare_keras_fn(data[col][perm], col, shape) for col, shape
                      in zip(label_columns, output_shapes)]

            num_outputs = len(label_columns)
            sample_weights = None
            if sample_weight_col:
                sample_weights = data[sample_weight_col][perm]

            batch_count = int(len(inputs[0]) / batch_size)
            for i in range(0, batch_count):
                if sample_weight_col:
                    # We use the same sample weight for all the outputs of the sample
                    sample_weight = sample_weights[i * batch_size:(i + 1) * batch_size]
                    sample_weight_for_batch = [sample_weight for i in range(num_outputs)]

                    yield (
                        [input[i * batch_size:(i + 1) * batch_size] for input in inputs],
                        [label[i * batch_size:(i + 1) * batch_size] for label in labels],
                        sample_weight_for_batch)
                else:
                    yield (
                        [input[i * batch_size:(i + 1) * batch_size] for input in inputs],
                        [label[i * batch_size:(i + 1) * batch_size] for label in labels])

    return batch_generator


def reshape_genrator(sample_weight_col, feature_columns, label_columns, metadata):
    CUSTOM_SPARSE = util.CUSTOM_SPARSE
    custom_sparse_to_dense = custom_sparse_to_dense_generator()

    def reshape(row):
        new_row = {}
        if sample_weight_col:
            new_row[sample_weight_col] = getattr(row, sample_weight_col)

        for col in feature_columns + label_columns:
            v = getattr(row, col)
            intermediate_format = metadata[col]['intermediate_format']
            if intermediate_format == CUSTOM_SPARSE:
                reshaped_v = tf.reshape(v, [metadata[col]['max_size'] * 2 + 1])
                v = custom_sparse_to_dense(reshaped_v, metadata[col]['shape'])

            new_row[col] = v
        return new_row

    return reshape


def prep_data_tf_keras_generator(has_sparse_col, sample_weight_col,
                                 feature_columns, label_columns, input_shapes,
                                 output_shapes, output_names):
    def _get_from_dict(row, col):
        return row[col]

    def _get_from_named_tuple(row, col):
        return getattr(row, col)

    if has_sparse_col:
        get_col_from_row_fn = _get_from_dict
    else:
        get_col_from_row_fn = _get_from_named_tuple

    num_inputs = len(feature_columns)
    num_labels = len(label_columns)

    def prep(row):
        if sample_weight_col:
            sample_weight = get_col_from_row_fn(row, sample_weight_col)
            return (
                tuple(
                    tf.reshape(get_col_from_row_fn(row, feature_columns[i]), input_shapes[i]) for i
                    in range(num_inputs)),
                tuple(tf.reshape(get_col_from_row_fn(row, label_columns[j]), output_shapes[j]) for j
                      in range(num_labels)),
                {name: tf.reshape(sample_weight, [-1]) for name in output_names}
            )
        else:
            return (
                tuple(
                    tf.reshape(get_col_from_row_fn(row, feature_columns[i]), input_shapes[i]) for i
                    in range(num_inputs)),
                tuple(tf.reshape(get_col_from_row_fn(row, label_columns[j]), output_shapes[j]) for j
                      in range(num_labels))
            )

    return prep


class KerasEstimator(Estimator, EstimatorParams, KerasEstimatorParamsReadable,
                     KerasEstimatorParamsWritable):
    custom_objects = Param(Params._dummy(), 'custom_objects', 'custom objects')
    _keras_pkg_type = Param(Params._dummy(), '_keras_pkg_type', 'keras package type')

    @keyword_only
    def __init__(self,
                 model=None,
                 backend=None,
                 store=None,
                 custom_objects=None,
                 optimizer=None,
                 loss=None,
                 loss_weights=None,
                 sample_weight_col=None,
                 compression=None,
                 metrics=None,
                 feature_cols=None,
                 label_cols=None,
                 validation_col=None,
                 callbacks=None,
                 batch_size=None,
                 epochs=None,
                 validation_split=None,
                 verbose=None,
                 shuffle_buffer_size=None):

        super(KerasEstimator, self).__init__()

        self._setDefault(optimizer=None,
                         custom_objects={},
                         _keras_pkg_type=None)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def _get_keras_utils(self):
        # This function determines the keras package type of the Estimator based on the passed
        # optimizer and model and updates _keras_pkg_type parameter.

        model_type = None
        model = self.getModel()
        if model:
            if isinstance(model, tf.keras.Model):
                model_type = TF_KERAS
            elif _is_instance_of_bare_keras_model(model):
                model_type = BARE_KERAS
            else:
                raise ValueError(
                    "model has to be an instance of tensorflow.keras.Model or keras.Model")

        optimizer_type = None
        optimizer = self.getOptimizer()
        if optimizer:
            if isinstance(optimizer, str):
                optimizer_type = None
            elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
                optimizer_type = TF_KERAS
            elif _is_instance_of_bare_keras_optimizer(optimizer):
                optimizer_type = BARE_KERAS
            else:
                raise ValueError("invalid optimizer type")

        types = set([model_type, optimizer_type])
        types.discard(None)

        if len(types) > 1:
            raise ValueError('mixed keras and tf.keras values for optimizers and model')
        elif len(types) == 1:
            pkg_type = types.pop()
            super(KerasEstimator, self)._set(_keras_pkg_type=pkg_type)

            if pkg_type == TF_KERAS:
                return TFKerasUtil
            elif pkg_type == BARE_KERAS:
                return BareKerasUtil
            else:
                raise ValueError("invalid keras type")

    def setCustomObjects(self, value):
        return self._set(custom_objects=value)

    def getCustomObjects(self):
        return self.getOrDefault(self.custom_objects)

    def _fit(self, df):
        backend = self.getBackend()
        store = self.getStore()
        label_columns = self.getLabelCols()
        feature_columns = self.getFeatureCols()
        validation_split = self.getValidationSplit()
        user_callbacks = self.getCallbacks()
        user_verbose = self.getVerbose()
        batch_size = self.getBatchSize()
        epochs = self.getEpochs()
        sample_weight_col = self.getSampleWeightCol()
        custom_objects = self.getCustomObjects()
        validation_col = self.getValidationCol()
        keras_utils = self._get_keras_utils()

        train_rows, val_rows, metadata, avg_row_size = \
            util.prepare_data(backend,
                              store,
                              df,
                              label_columns=label_columns,
                              feature_columns=feature_columns,
                              validation_col=validation_col,
                              validation_split=validation_split,
                              sample_weight_col=sample_weight_col)

        # Check if any of the columns are only SparseVector
        has_sparse_col = any(metadata[col_name]['training_data_type'] == util.SPARSE_VECTOR
                             for col_name in label_columns + feature_columns)

        train_data_path = store.get_train_data_path()
        val_data_path = store.get_val_data_path()
        should_validate = self._should_validate()
        user_shuffle_buffer_size = self.getShufflingBufferSize()

        # Compile the model with all the parameters
        model = self.getModel()
        loss = self.getLoss()
        loss_weights = self.getLossWeights()

        if not model:
            raise ValueError('Model parameter is required')

        if not loss:
            raise ValueError('Loss parameter is required for the model to compile')

        optimizer = self.getOptimizer()
        if not optimizer:
            optimizer = model.optimizer

        if not optimizer:
            raise ValueError('Optimizer must be provided either as a parameter or as part of a '
                             'compiled model')

        # Get input and output shapes and dtypes
        input_shapes = [[dim if dim else -1 for dim in input.shape.as_list()]
                        for input in model.inputs]
        input_dtypes = [type(model_input.dtype.as_numpy_dtype())
                        for model_input in model.inputs]

        output_shapes = [[dim if dim else -1 for dim in output.shape.as_list()]
                         for output in model.outputs]
        output_names = model.output_names

        # # Check for model and input type incompatibility. Feature columns must have the same size
        # # (total number of elements) and data types of the corresponding inputs.
        # # Same applies to label columns and outputs for shape, but type only needs to be castable.
        # util.check_model_compatibility(metadata, feature_columns, label_columns,
        #                                input_shapes, output_shapes, input_dtypes)
        # metrics = self.getMetrics()
        # compression = self.getCompression()
        # optimizer_weight_values = optimizer.get_weights()
        #
        # dist_optimizer_args = dict(optimizer=optimizer)
        # if compression:
        #     dist_optimizer_args['compression'] = compression
        #
        # # Horovod: wrap optimizer with DistributedOptimizer.
        # dist_optimizer = keras_utils.get_horovod().DistributedOptimizer(**dist_optimizer_args)
        # model.compile(optimizer=dist_optimizer,
        #               loss=loss,
        #               loss_weights=loss_weights,
        #               metrics=metrics)
        #
        # if optimizer_weight_values:
        #     model.optimizer.set_weights(optimizer_weight_values)
        #
        # serialized_model = keras_utils.serialize_model(model)
        # keras_type = keras_utils.type
        #
        # get_petastorm_path = store.get_petastorm_path_generator()
        #
        # keras_module = keras_utils.keras()
        # floatx = keras_module.backend.floatx()
        #
        # # Functions:
        # serialize = util.serialize_generator()
        # deserialize_keras_model_fn = _deserialize_keras_model_generator()
        # calculate_shuffle_buffer_size_fn = calculate_shuffle_buffer_size_generator()
        # batch_generator_fn = batch_generator_generator(feature_columns, label_columns,
        #                                                sample_weight_col, input_shapes,
        #                                                output_shapes,
        #                                                batch_size, metadata)
        # reshape_fn = reshape_genrator(sample_weight_col, feature_columns, label_columns, metadata)
        # prep_data_tf_keras_fn = \
        #     prep_data_tf_keras_generator(has_sparse_col, sample_weight_col,
        #                                  feature_columns, label_columns, input_shapes,
        #                                  output_shapes, output_names)

        # serialized_model = _serialize_keras_model(model, save_model_fn=tf.keras.models.save_model)

        # bio = io.BytesIO()
        with h5py.File('/tmp/model', 'w') as f:
            tf.keras.models.save_model(model, f)
        # serialized_model = codec.dumps_base64(bio.getvalue())

        def train():
            import tempfile
            import shutil
            import contextlib
            from petastorm import make_batch_reader
            from petastorm.tf_utils import make_petastorm_dataset

            import tensorflow as tf

            import horovod.tensorflow.keras as hvd
            hvd.init()
            local_size = hvd.local_size()
            print(hvd.__path__)
            import sys
            for m in [x for x in sorted(sys.modules.keys()) if '.' not in x]:
                print(m)

            local_sizes = hvd.allgather([local_size])
            print(local_sizes)
            return None, None, 1

        # def train():
        #     import tempfile
        #     import shutil
        #     import contextlib
        #     from petastorm import make_batch_reader
        #     from petastorm.tf_utils import make_petastorm_dataset
        #
        #     def _get_keras(type):
        #         if type == BARE_KERAS:
        #             import keras
        #             return keras
        #         else:
        #             import tensorflow.keras as tf_keras
        #             return tf_keras
        #
        #     def _get_hvd(type):
        #         if type == BARE_KERAS:
        #             import horovod.keras as hvd
        #             return hvd
        #         else:
        #             import horovod.tensorflow.keras as hvd
        #             return hvd
        #
        #     @contextlib.contextmanager
        #     def _tempdir():
        #         dirpath = tempfile.mkdtemp()
        #         try:
        #             yield dirpath
        #         finally:
        #             shutil.rmtree(dirpath)
        #
        #     # k = _get_keras(keras_type)
        #     # k.backend.set_floatx(floatx)
        #
        #     # hvd = _get_hvd(keras_type)
        #
        #     import horovod.tensorflow.keras as hvd
        #     hvd.init()
        #
        #     local_size = hvd.local_size()
        #     print(hvd.__path__)
        #     import sys
        #     for m in [x for x in sorted(sys.modules.keys()) if '.' not in x]:
        #         print(m)
        #
        #     local_sizes = hvd.allgather([local_size])
        #
        #     # config = tf.ConfigProto()
        #     # config.gpu_options.allow_growth = True
        #     # config.gpu_options.visible_device_list = str(hvd.local_rank())
        #     # k.backend.set_session(tf.Session(config=config))
        #     #
        #     # if not user_shuffle_buffer_size:
        #     #     shuffle_buffer_size = \
        #     #         calculate_shuffle_buffer_size_fn(hvd, avg_row_size, train_rows / hvd.size())
        #     # else:
        #     #     shuffle_buffer_size = user_shuffle_buffer_size
        #
        #     return None, serialized_model, 1

            # # needs to be deserialized in the with scope
            # with k.utils.custom_object_scope(custom_objects):
            #     model = deserialize_keras_model_fn(serialized_model, lambda x: hvd.load_model(x))

            # # Horovod: adjust learning rate based on number of processes.
            # k.backend.set_value(model.optimizer.lr,
            #                     k.backend.get_value(model.optimizer.lr) * hvd.size())
            #
            # # Verbose mode 1 will print a progress bar
            # verbose = user_verbose if hvd.rank() == 0 else 0
            #
            # callbacks = [
            #     # Horovod: broadcast initial variable states from rank 0 to all other processes.
            #     # This is necessary to ensure consistent initialization of all workers when
            #     # training is started with random weights or restored from a checkpoint.
            #     hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),
            #
            #     # Horovod: average metrics among workers at the end of every epoch.
            #     #
            #     # Note: This callback must be in the list before the ReduceLROnPlateau,
            #     # TensorBoard, or other metrics-based callbacks.
            #     hvd.callbacks.MetricAverageCallback(),
            # ]
            # callbacks += user_callbacks
            #
            # steps_per_epoch = int(math.ceil(train_rows / batch_size / hvd.size()))
            # # math.ceil because if val_rows is smaller than batch_size we still get the at least
            # # one step. float(val_rows) because val_rows/batch_size evaluates to zero before
            # # math.ceil
            # validation_steps = int(math.ceil(float(val_rows) / batch_size / hvd.size()))
            #
            # schema_fields = feature_columns + label_columns
            # if sample_weight_col:
            #     schema_fields.append(sample_weight_col)
            #
            # with _tempdir() as ckpt_dir:
            #     # Model checkpoint location.
            #     ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')
            #
            #     # Horovod: save checkpoints only on the first worker to prevent other workers from
            #     # corrupting them.
            #     if hvd.rank() == 0:
            #         callbacks.append(k.callbacks.ModelCheckpoint(ckpt_file))
            #
            #     # Petastorm: read data from the store with the correct shard for this rank
            #     with make_batch_reader(get_petastorm_path(train_data_path),
            #                            shuffle_row_groups=True,
            #                            # setting num_epochs=None will cause an infinite iterator
            #                            num_epochs=None,
            #                            cur_shard=hvd.rank(),
            #                            shard_count=hvd.size(),
            #                            hdfs_driver=PETASTORM_HDFS_DRIVER,
            #                            schema_fields=schema_fields) as train_reader:
            #
            #         if keras_type == TF_KERAS:
            #             if has_sparse_col:
            #                 train_ds = make_petastorm_dataset(train_reader) \
            #                     .apply(tf.data.experimental.unbatch()) \
            #                     .shuffle(shuffle_buffer_size) \
            #                     .batch(1) \
            #                     .map(reshape_fn) \
            #                     .batch(batch_size) \
            #                     .map(prep_data_tf_keras_fn)
            #             else:
            #                 train_ds = make_petastorm_dataset(train_reader) \
            #                     .apply(tf.data.experimental.unbatch()) \
            #                     .shuffle(shuffle_buffer_size) \
            #                     .batch(batch_size) \
            #                     .map(prep_data_tf_keras_fn)
            #
            #             if should_validate:
            #                 # setting num_epochs=None will cause an infinite iterator and enables
            #                 # ranks to perform training and validation with unequal number of
            #                 # samples
            #                 with make_batch_reader(
            #                         get_petastorm_path(val_data_path),
            #                         num_epochs=None,
            #                         cur_shard=hvd.rank(),
            #                         shard_count=hvd.size(),
            #                         hdfs_driver=PETASTORM_HDFS_DRIVER,
            #                         schema_fields=schema_fields) as val_reader:
            #
            #                     if has_sparse_col:
            #                         val_ds = make_petastorm_dataset(val_reader) \
            #                             .apply(tf.data.experimental.unbatch()) \
            #                             .batch(1) \
            #                             .map(reshape_fn) \
            #                             .batch(batch_size) \
            #                             .map(prep_data_tf_keras_fn)
            #                     else:
            #                         val_ds = make_petastorm_dataset(val_reader) \
            #                             .apply(tf.data.experimental.unbatch()) \
            #                             .batch(batch_size) \
            #                             .map(prep_data_tf_keras_fn)
            #
            #                     history = model.fit(
            #                         train_ds,
            #                         validation_data=val_ds,
            #                         steps_per_epoch=steps_per_epoch,
            #                         validation_steps=validation_steps,
            #                         callbacks=callbacks,
            #                         verbose=verbose,
            #                         epochs=epochs)
            #             else:
            #                 history = model.fit(
            #                     train_ds,
            #                     steps_per_epoch=steps_per_epoch,
            #                     callbacks=callbacks,
            #                     verbose=verbose,
            #                     epochs=epochs)
            #
            #         elif keras_type == BARE_KERAS:
            #
            #             if should_validate:
            #                 # setting num_epochs=None will cause an infinite iterator and enables
            #                 # ranks to perform training and validation with unequal number of
            #                 # samples
            #                 with make_batch_reader(get_petastorm_path(val_data_path),
            #                                        num_epochs=None,
            #                                        cur_shard=hvd.rank(),
            #                                        shard_count=hvd.size(),
            #                                        hdfs_driver=PETASTORM_HDFS_DRIVER,
            #                                        schema_fields=schema_fields) as val_reader:
            #                     history = model.fit_generator(
            #                         generator=batch_generator_fn(train_reader, shuffle_buffer_size),
            #                         steps_per_epoch=steps_per_epoch,
            #                         validation_data=batch_generator_fn(val_reader,
            #                                                            shuffle_buffer_size),
            #                         validation_steps=validation_steps,
            #                         callbacks=callbacks,
            #                         verbose=verbose,
            #                         epochs=epochs)
            #
            #             else:
            #                 history = model.fit_generator(generator=batch_generator_fn(
            #                     train_reader, shuffle_buffer_size),
            #                     steps_per_epoch=steps_per_epoch,
            #                     callbacks=callbacks,
            #                     verbose=verbose,
            #                     epochs=epochs)
            #
            #     # Dataset API usage currently displays a wall of errors upon termination.
            #     # This global model registration ensures clean termination.
            #     # Tracked in https://github.com/tensorflow/tensorflow/issues/24570
            #     globals()['_DATASET_FINALIZATION_HACK'] = model
            #
            #     if hvd.rank() == 0:
            #         with open(ckpt_file, 'rb') as f:
            #             return history.history, serialize(f.read()), hvd.size()

        # Workaround:
        # https://stackoverflow.com/questions/50583056/is-there-any-way-to-set-java-opts-for-tensorflow-process/50615570
        env = {'LIBHDFS_OPTS': '-Xms2048m -Xmx2048m'}

        ret = backend.run(train, env=env, stdout=sys.stdout, stderr=sys.stdout)
        history, serialized_model, hvd_size = ret[0]

        def load_model_fn(x):
            with keras_module.utils.custom_object_scope(custom_objects):
                return keras_module.models.load_model(x)

        deserialized_model = \
            keras_utils.deserialize_model(serialized_model, load_model_fn=load_model_fn)

        # Here, learning rate is scaled down with the number of horovod workers.
        # This is important the retraining of the model. User may retrain the model with
        # different number of workers and we need the raw learning rate to adjust with the
        # new number of workers.
        scaled_lr = keras_module.backend.get_value(deserialized_model.optimizer.lr)
        keras_module.backend.set_value(deserialized_model.optimizer.lr, scaled_lr / hvd_size)
        return self._create_model(history, deserialized_model, metadata, floatx)

    def _create_model(self, history, model, metadata, floatx):
        return KerasModel(history=history,
                          model=model,
                          feature_columns=self.getFeatureCols(),
                          label_columns=self.getLabelCols(),
                          custom_objects=self.getCustomObjects(),
                          _metadata=metadata,
                          _floatx=floatx)


class KerasModel(Model, ModelParams, KerasEstimatorParamsReadable,
                 KerasEstimatorParamsWritable):
    custom_objects = Param(Params._dummy(), 'custom_objects', 'custom objects')

    # Setting _keras_pkg_type parameter helps us determine the type of keras package during
    # deserializing the transformer
    _keras_pkg_type = Param(Params._dummy(), '_keras_pkg_type', 'keras package type')

    _floatx = Param(Params._dummy(), '_floatx', 'keras default float type')

    @keyword_only
    def __init__(self,
                 history=None,
                 model=None,
                 feature_columns=None,
                 label_columns=None,
                 custom_objects=None,
                 _metadata=None,
                 _floatx=None):

        super(KerasModel, self).__init__()

        if label_columns:
            self.setOutputCols([col + '__output' for col in label_columns])

        self._setDefault(custom_objects={})

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setCustomObjects(self, value):
        return self._set(custom_objects=value)

    def getCustomObjects(self):
        return self.getOrDefault(self.custom_objects)

    def _get_keras_utils(self, model=None):
        # infer keras package from model
        model = self.getModel()
        if model:
            if isinstance(model, tf.keras.Model):
                pkg_type = TF_KERAS
            elif _is_instance_of_bare_keras_model(model):
                pkg_type = BARE_KERAS
            else:
                raise ValueError(
                    "model has to be an instance of tensorflow.keras.Model or keras.Model")

            super(KerasModel, self)._set(_keras_pkg_type=pkg_type)

            if pkg_type == TF_KERAS:
                return TFKerasUtil
            elif pkg_type == BARE_KERAS:
                return BareKerasUtil
            else:
                raise ValueError("invalid keras type")

        raise ValueError("model is not set")

    def _get_floatx(self):
        return self.getOrDefault(self._floatx)

    # To run locally on OS X, need export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    def _transform(self, df):
        keras_utils = self._get_keras_utils()
        floatx = self._get_floatx()
        serialized_model = keras_utils.serialize_model(self.getModel())

        label_cols = self.getLabelColumns()
        output_cols = self.getOutputCols()
        feature_cols = self.getFeatureColumns()
        custom_objects = self.getCustomObjects()
        metadata = self._get_metadata()

        def predict(rows):
            import tensorflow as tf
            from pyspark import Row
            from pyspark.ml.linalg import DenseVector, SparseVector

            k = keras_utils.keras()
            k.backend.set_floatx(floatx)

            # Do not use GPUs for prediction, use single CPU core per task.
            config = tf.ConfigProto(device_count={'GPU': 0})
            config.inter_op_parallelism_threads = 1
            config.intra_op_parallelism_threads = 1
            k.backend.set_session(tf.Session(config=config))

            def load_model_fn(x):
                with k.utils.custom_object_scope(custom_objects):
                    return k.models.load_model(x)

            model = keras_utils.deserialize_model(serialized_model,
                                                  load_model_fn=load_model_fn)

            input_shapes = [[dim if dim else -1 for dim in input.shape.as_list()]
                            for input in model.inputs]

            def convert_to_array(item):
                if type(item) in [DenseVector or SparseVector]:
                    return item.toArray()
                else:
                    return np.array(item)

            # Perform predictions.
            for row in rows:
                fields = row.asDict().copy()
                preds = model.predict_on_batch(
                    [convert_to_array(row[feature_cols[i]]).reshape(input_shapes[i])
                     for i in range(len(feature_cols))])

                for label_col, output_col, pred, in zip(label_cols, output_cols, preds):
                    meta = metadata[label_col]
                    col_type = meta['dtype']
                    # dtype for dense and spark tensor is always np.float64
                    if col_type == DenseVector:
                        shape = meta['shape']
                        flattened_pred = pred.reshape(shape, )
                        field = DenseVector(flattened_pred)
                    elif col_type == SparseVector:
                        shape = meta['shape']
                        flattened_pred = pred.reshape(shape, )
                        nonzero_indices = flattened_pred.nonzero()[0]
                        field = SparseVector(shape, nonzero_indices,
                                             flattened_pred[nonzero_indices])
                    else:
                        # If the column is scalar type, int, float, etc.
                        value = pred[0]
                        if issubclass(col_type, numbers.Integral):
                            value = round(value)
                        field = col_type(value)

                    fields[output_col] = field

                yield Row(**fields)

        return df.rdd.mapPartitions(predict).toDF()
