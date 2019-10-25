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

import sys

import numpy as np
import pyspark.sql.functions as f

from horovod.run.common.util import codec
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.sql.types import (IntegerType, StringType, FloatType,
                               BinaryType, DoubleType, LongType, BooleanType)

_training_cache = None

ARRAY = 'array'
CUSTOM_SPARSE = 'custom_sparse_format'
NOCHANGE = 'nochange'

MIXED_SPARSE_DENSE_VECTOR = 'mixed_sparse_dense_vector'
SPARSE_VECTOR = 'sparse_vector'
DENSE_VECTOR = 'dense_vector'

TOTAL_BUFFER_MEMORY_CAP = 4
ONE_GB = 1073741824


def data_type_to_str(dtype):
    if dtype == VectorUDT:
        return 'Vector'
    elif dtype == IntegerType:
        return 'Int'
    elif dtype == StringType:
        return 'String'
    elif dtype == FloatType:
        return 'Float'
    elif dtype == BinaryType:
        return 'Binary'
    elif dtype == DoubleType:
        return 'Double'
    elif dtype == LongType:
        return 'Long'
    elif dtype == BooleanType:
        return 'Boolean'
    else:
        raise ValueError('Unrecognized data type: {}'.format(dtype))


def numpy_type_to_str(dtype):
    if dtype == np.int32:
        return 'Int'
    elif dtype == np.float32:
        return 'Float'
    elif dtype == np.uint8:
        return 'Binary'
    elif dtype == np.float64:
        return 'Double'
    elif dtype == np.int64:
        return 'Long'
    elif dtype == np.bool:
        return 'Boolean'
    else:
        raise ValueError('Cannot convert numpy data type to Spark string: {}'.format(dtype))


def data_type_to_numpy(dtype):
    if dtype == VectorUDT:
        return np.float64
    elif dtype == IntegerType:
        return np.int32
    elif dtype == StringType:
        return np.uint8
    elif dtype == FloatType:
        return np.float32
    elif dtype == BinaryType:
        return np.uint8
    elif dtype == DoubleType:
        return np.float64
    elif dtype == LongType:
        return np.int64
    elif dtype == BooleanType:
        return np.bool
    else:
        raise ValueError('Unrecognized data type: {}'.format(dtype))


def check_model_compatibility(metadata, feature_columns, label_columns,
                              input_shapes, output_shapes=None,
                              input_dtypes=None, output_dtypes=None):
    feature_count = len(feature_columns)
    if feature_count != len(input_shapes):
        raise ValueError('Feature column count {features} must equal '
                         'model inputs count {inputs}'
                         .format(features=feature_count, inputs=len(input_shapes)))

    for idx, col, input_shape in zip(range(feature_count), feature_columns, input_shapes):
        col_size = metadata[col]['shape']
        input_size = abs(np.prod(input_shape))
        if col_size != input_size:
            raise ValueError('Feature column \'{col}\' with size {feature} must equal that of the '
                             'model input at index {idx} with size {input}'
                             .format(col=col, feature=col_size, idx=idx, input=input_size))

    label_count = len(label_columns)
    if output_shapes is not None:
        if label_count != len(output_shapes):
            raise ValueError('Label column count {labels} must equal '
                             'model outputs count {outputs}'
                             .format(labels=label_count, outputs=len(output_shapes)))

        for idx, col, output_shape in zip(range(label_count), label_columns, output_shapes):
            col_size = metadata[col]['shape']
            output_size = abs(np.prod(output_shape))
            if col_size != output_size:
                raise ValueError('Label column \'{col}\' with size {label} must equal that of the '
                                 'model output at index {idx} with size {output}'
                                 .format(col=col, label=col_size, idx=idx, output=output_size))

    if input_dtypes is not None:
        for idx, col, input_dtype in zip(range(feature_count), feature_columns, input_dtypes):
            col_dtype = data_type_to_numpy(metadata[col]['spark_data_type'])
            if col_dtype != input_dtype:
                raise ValueError('Feature column \'{col}\' with data type {feature} must be equal '
                                 'to data type {input} for model input at index {idx}'
                                 .format(col=col, feature=col_dtype, input=input_dtype, idx=idx))

    if output_dtypes is not None:
        for idx, col, output_dtype in zip(range(label_count), label_columns, output_dtypes):
            col_dtype = data_type_to_numpy(metadata[col]['spark_data_type'])
            if col_dtype != output_dtype:
                raise ValueError('Label column \'{col}\' with data type {label} must be equal '
                                 'to data type {output} for model output at index {idx}'
                                 .format(col=col, label=col_dtype, output=output_dtype, idx=idx))


def _get_col_info(df):
    """
    Infer the type and shape of all the columns.
    """

    def get_meta(row):
        row_dict = row.asDict()
        row_schema = []
        for col_name, data_col in row_dict.items():
            dtype = type(data_col)
            if dtype == DenseVector:
                # shape and size of dense vector are the same
                shape = data_col.array.shape[0]
                size = shape
            elif dtype == SparseVector:
                # shape is the total size of vector
                shape = data_col.size
                # size is the number of nonzero elements in the sparse vector
                size = data_col.indices.shape[0]
            else:
                shape = 1
                size = 1
            row_schema.append((col_name, (set([dtype]), set([shape]), set([size]))))
        return row_schema

    def merge(x, y):
        dtypes = x[0]
        dtypes.update(y[0])
        shapes = x[1]
        shapes.update(y[1])
        sizes = x[2]
        sizes.update(y[2])
        return dtypes, shapes, sizes

    raw_col_info_list = df.rdd.flatMap(get_meta).reduceByKey(merge).collect()

    all_col_types = {}
    col_shapes = {}
    col_max_sizes = {}

    for col_info in raw_col_info_list:
        col_name = col_info[0]

        all_col_types[col_name] = col_info[1][0]
        col_shapes[col_name] = col_info[1][1]
        col_max_sizes[col_name] = col_info[1][2]

    # all the rows of each columns must have the same shape
    for col in df.schema.names:
        shape_set = col_shapes[col]
        if len(shape_set) != 1:
            raise ValueError(
                'col {col} does not have uniform shapes. shape set: {shapes_set}'.format(col=col,
                                                                                         shapes_set=shape_set))
        col_shapes[col] = shape_set.pop()

    for col in df.schema.names:
        sizes = col_max_sizes[col]
        if len(sizes) > 1 and not (SparseVector in all_col_types[col]):
            raise ValueError(
                "rows of column {col} have varying sizes. This is only allowed if datatype is "
                "SparseVector or a mix of Sparse and DenseVector.".format(col=col))
        col_max_sizes[col] = max(sizes)

    return all_col_types, col_shapes, col_max_sizes


def _get_metadata(df):
    """
    Infer the type and shape of all the columns and determines if what intermedite format they
    need to be converted to in case they are a vector.

    Example return value:
    {
    'col1': {
        'dtype': <type 'float'>,
        'intermediate_format': 'nochange',
        'max_size': 1,
        'shape': 1
        },
     'col2': {
        'dtype': <type 'float'>,
        'intermediate_format': 'nochange',
        'max_size': 1,
        'shape': 1
        },
     'col3': {
        'dtype': <class 'pyspark.ml.linalg.SparseVector'>,
        'intermediate_format': 'custom_sparse_format',
        'max_size': 37,
        'shape': 56
        }
    }
    """
    all_col_types, col_shapes, col_max_sizes = _get_col_info(df)

    metadata = dict()
    for field in df.schema.fields:
        col = field.name
        spark_data_type = type(field.dataType)
        col_types = all_col_types[col].copy()
        if DenseVector in col_types and SparseVector in col_types:
            # If a col has DenseVector type (whether it is mixed sparse and dense vector or just
            # DenseVector), convert all of the values to dense vector
            training_data_type = MIXED_SPARSE_DENSE_VECTOR
            output_data_type = DenseVector
            convert_to_target = ARRAY
        elif DenseVector in col_types:
            # If a col has DenseVector type (whether it is mixed sparse and dense vector or just
            # DenseVector), convert all of the values to dense vector
            training_data_type = DENSE_VECTOR
            output_data_type = DenseVector
            convert_to_target = ARRAY
        elif SparseVector in col_types:
            # If a col has only sparse vectors, convert all the data into custom dense vectors
            training_data_type = SPARSE_VECTOR
            output_data_type = SparseVector
            convert_to_target = CUSTOM_SPARSE
        else:
            col_type = col_types.pop()

            training_data_type = col_type
            output_data_type = col_type
            convert_to_target = NOCHANGE

        metadata[col] = {'dtype': output_data_type,
                         'spark_data_type': spark_data_type,
                         'training_data_type': training_data_type,
                         'shape': col_shapes[col],
                         'intermediate_format': convert_to_target,
                         'max_size': col_max_sizes[col]}

    return metadata


def to_petastorm_generator(schema_cols, metadata):
    # Convert Spark Vectors into arrays so Petastorm can read them
    def to_petastorm(row):
        import numpy as np
        from pyspark import Row

        fields = row.asDict().copy()
        for col in schema_cols:
            col_data = row[col]
            intermediate_format = metadata[col]['intermediate_format']
            if intermediate_format == ARRAY:
                fields[col] = col_data.toArray().tolist()
            elif intermediate_format == CUSTOM_SPARSE:
                # Currently petastorm does not support reading pyspark sparse vector. We put
                # the indices and values into one array. when consuming the data, we re-create
                # the vector from this format.
                size = len(col_data.indices)
                padding_zeros = 2 * (metadata[col]['max_size'] - len(col_data.indices))

                fields[col] = np.concatenate(
                    (np.array([size]), col_data.indices, col_data.values,
                     np.zeros(padding_zeros))).tolist()

        return Row(**fields)

    return to_petastorm


def prepare_data(backend, store, df, label_columns, feature_columns,
                 validation_col=None, validation_split=None, sample_weight_col=None):
    if validation_split and validation_col:
        raise ValueError("can only specify one of validation_col and validation_split")

    num_processes = backend.num_processes()
    num_partitions = num_processes * 10
    print('num_partitions={}'.format(num_partitions))

    train_data_path = store.get_train_data_path()
    val_data_path = store.get_val_data_path()
    print('train_data_path={}'.format(train_data_path))
    print('val_data_path={}'.format(val_data_path))

    for col in label_columns:
        if col not in df.columns:
            raise ValueError('Label column {} does not exist in this DataFrame'.format(col))

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in set(label_columns)]

    hash_code = df.__hash__()

    global _training_cache
    if _training_cache and _training_cache.is_cached(hash_code, validation_split, validation_col,
                                                     train_data_path, val_data_path):
        train_rows = _training_cache.train_rows
        val_rows = _training_cache.val_rows
        metadata = _training_cache.metadata
        avg_row_size = _training_cache.avg_row_size

        print('using cached dataframes')
        print('train_rows={}'.format(train_rows))
        print('val_rows={}'.format(val_rows))
    else:
        print('writing dataframes')

        schema_cols = feature_columns + label_columns
        if sample_weight_col:
            schema_cols.append(sample_weight_col)
        if validation_col:
            schema_cols.append(validation_col)
        df = df[schema_cols]

        metadata = _get_metadata(df)

        to_petastorm = to_petastorm_generator(schema_cols, metadata)
        train_df = df.rdd.map(to_petastorm).toDF()

        val_df = None
        if validation_split > 0:
            train_df, val_df = train_df.randomSplit([1.0 - validation_split, validation_split])
        elif validation_col:
            val_df = train_df.filter(f.col(validation_col) > 0).drop(validation_col)
            train_df = train_df.filter(f.col(validation_col) == 0).drop(validation_col)

        print('get train_partitions')
        train_partitions = max(int(num_partitions * (1.0 - validation_split)), 10)
        print('train_partitions={}'.format(train_partitions))

        train_df \
            .coalesce(train_partitions) \
            .write \
            .mode('overwrite') \
            .parquet(train_data_path)

        train_rows = train_df.count()
        print('train_rows={}'.format(train_rows))

        # Get an estimate of each row of the dataset
        sample_ratio = 0.01 if train_rows > 100 else 1.0
        sample_count = train_rows * 0.01 + 1  # +1 to ensure it is not zero
        for i in range(5):
            sample_size = train_df.sample(sample_ratio).rdd.map(lambda x: sys.getsizeof(x)).sum()
            avg_row_size = sample_size / sample_count
            if avg_row_size != 0:
                break

        if avg_row_size == 0:
            raise ValueError("average row size is 0 bytes")

        val_rows = 0
        if val_df:
            val_rows = val_df.count()
            if val_rows == 0:
                raise ValueError(
                    "{validation_col} col does not any validation samples".format(
                        validation_col=validation_col))

            if validation_split:
                _validation_split = validation_split
            elif validation_col:
                _validation_split = val_rows / (val_rows + train_rows)

            val_partitions = max(int(num_partitions * _validation_split), 10)
            print('val_partitions={}'.format(val_partitions))

            val_df \
                .coalesce(val_partitions) \
                .write \
                .mode('overwrite') \
                .parquet(val_data_path)

            print('val_rows={}'.format(val_rows))

    _training_cache = TrainingDataCache(store, hash_code, validation_split, validation_col,
                                        train_rows, val_rows, metadata, avg_row_size)

    return train_rows, val_rows, metadata, avg_row_size


def is_module_available(module_name):
    _is_module_available = is_module_available_generator()
    return _is_module_available(module_name)


def is_module_available_generator():
    def _is_module_available(module_name):
        if sys.version_info < (3, 0):
            # python 2
            import pkgutil
            torch_loader = pkgutil.find_loader(module_name)
        elif sys.version_info <= (3, 3):
            # python 3.0 to 3.3
            import pkgutil
            torch_loader = pkgutil.find_loader(module_name)
        elif sys.version_info >= (3, 4):
            # python 3.4 and above
            import importlib
            torch_loader = importlib.util.find_spec(module_name)

        return torch_loader is not None

    return _is_module_available


def serialize_generator():
    def _serialize(model):
        """Serialize model into byte array encoded into base 64."""
        functional_module_is_set = False
        if is_module_available('torch'):
            import torch
            sys.modules["torch._C._nn"] = torch.nn.functional
            functional_module_is_set = True

        serialized_obj = codec.dumps_base64(model)
        if functional_module_is_set:
            del sys.modules["torch._C._nn"]

        return serialized_obj

    return _serialize


def deserialize_generator():
    is_module_available = is_module_available_generator()

    def _deserialize(model_bytes_base64):
        """Deserialize model from byte array encoded in base 64."""
        functional_module_is_set = False
        if is_module_available('torch'):
            import torch
            sys.modules["torch._C._nn"] = torch.nn.functional
            functional_module_is_set = True

        obj = codec.loads_base64(model_bytes_base64)
        if functional_module_is_set:
            del sys.modules["torch._C._nn"]
        return obj

    return _deserialize


def deserialize(model_bytes_base64):
    _deserialize = deserialize_generator()
    return _deserialize(model_bytes_base64)


def serialize(model):
    _serialize = serialize_generator()
    return _serialize(model)


class TrainingDataCache:
    def __init__(self, store, dataframe_hash, validation_split, validation_col,
                 train_rows, val_rows, metadata, avg_row_size):
        self.store = store
        self.dataframe_hash = dataframe_hash
        self.validation_split = validation_split
        self.validation_col = validation_col
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.metadata = metadata
        self.avg_row_size = avg_row_size

    def is_cached(self, dataframe_hash, validation_split, validation_col, train_data_path,
                  val_data_path):
        return \
            self.dataframe_hash == dataframe_hash and \
            self.validation_split == validation_split and \
            self.validation_col == validation_col and \
            self.store.exists(train_data_path) and \
            (validation_split == 0.0 or self.store.exists(val_data_path))
