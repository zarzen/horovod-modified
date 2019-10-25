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

import math
import numbers
import os
import sys

import torch
import torch.utils.data
import torch.utils.data
from horovod.run.common.util import codec
from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import MLWritable, MLReadable

from horovod.spark.common import util
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.params import EstimatorParams, ModelParams
from horovod.spark.common.serialization import \
    HorovodParamsWriter, HorovodParamsReader

PETASTORM_HDFS_DRIVER = 'libhdfs'


def _torch_param_serialize(param_name, param_val):
    if param_name in [EstimatorParams.backend.name, EstimatorParams.store.name]:
        # We do not serialize backend and store. These params have to be regenerated for each
        # run of the pipeline
        return None

    if param_val is None:
        return None

    return codec.dumps_base64(param_val)


class TorchEstimatorParamsWriter(HorovodParamsWriter):

    def saveImpl(self, path):
        # Write the parameters
        HorovodParamsWriter.saveMetadata(self.instance, path, self.sc,
                                         param_serializer_fn=_torch_param_serialize)


class TorchEstimatorParamsWritable(MLWritable):

    def write(self):
        return TorchEstimatorParamsWriter(self)


class TorchEstimatorParamsReader(HorovodParamsReader):
    def _deserialize_dict(self, dict_values):
        deserialized_dict = dict()
        for key, val in dict_values.items():
            if val is None:
                deserialized_dict[key] = None
            else:
                deserialized_dict[key] = codec.loads_base64(val)
        return deserialized_dict


class TorchEstimatorParamsReadable(MLReadable):

    @classmethod
    def read(cls):
        """Returns a DefaultParamsReader instance for this class."""
        return TorchEstimatorParamsReader(cls)


def get_optimizer_with_unscaled_lr_generator():
    def get_optimizer_with_unscaled_lr(hvd, current_optimizer, optimizer_cls, model):
        optimizer_state = current_optimizer.state_dict()
        # scale down the learning rate with the number of horovod workers
        for i in range(len(optimizer_state['param_groups'])):
            optimizer_state['param_groups'][i]['lr'] = \
                optimizer_state['param_groups'][i]['lr'] / hvd.size()
        optimizer = optimizer_cls(model.parameters(), lr=1)
        optimizer.load_state_dict(optimizer_state)
        return optimizer

    return get_optimizer_with_unscaled_lr


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
        local_sizes = hvd.allgather(torch.tensor([local_size]))
        max_local_size = torch.max(local_sizes).item()

        if max_local_size > TOTAL_BUFFER_MEMORY_CAP:
            shuffle_buffer_size = TOTAL_BUFFER_MEMORY_CAP * ONE_GB / avg_row_size / max_local_size
        else:
            shuffle_buffer_size = ONE_GB / avg_row_size
        return int(min(shuffle_buffer_size, train_row_count_per_worker))

    return calculate_shuffle_buffer_size


def construct_metric_value_holders_generator():
    def construct_metric_value_holders(metric_class, metric_fn_groups, label_columns, hvd):
        metric_values = []
        for group_number, metric_group in enumerate(metric_fn_groups):
            metric_group_val = []
            for label_col in label_columns:
                metric_group_val.append(
                    metric_class('group_' + str(group_number) + '_' + label_col, hvd))

            metric_values.append(metric_group_val)
        return metric_values

    return construct_metric_value_holders


def metric_class_generator():
    # Horovod: average metrics from distributed training.
    class Metric(object):
        def __init__(self, name, hvd):
            self.name = name
            self.sum = torch.tensor(0.)
            self.n = torch.tensor(0.)
            self.hvd = hvd

        def update(self, val):
            self.sum += self.hvd.allreduce(val.detach().cpu(), name=self.name)
            self.n += 1

        @property
        def avg(self):
            return self.sum / self.n

    return Metric


def prepare_np_data_generator():
    CUSTOM_SPARSE = util.CUSTOM_SPARSE

    def prepare_np_data(rows, col_name, metadata):
        intermediate_format = metadata[col_name]['intermediate_format']
        if intermediate_format != CUSTOM_SPARSE:
            return rows

        shape = metadata[col_name]['shape']
        num_rows = rows.shape[0]
        dense_rows = torch.zeros([num_rows, shape])
        for r in range(num_rows):
            size = rows[r][0].long()
            dense_rows[r][rows[r][1:size + 1].long()] = \
                rows[r][size + 1:2 * size + 1]
        return dense_rows

    return prepare_np_data


def get_metric_avgs_generator():
    def get_metric_avgs(metric_value_groups):
        all_metric_groups_values = []
        for metric_value_group in metric_value_groups:
            metric_avgs = {}
            for metric in metric_value_group:
                metric_avgs[metric.name] = metric.avg.item()
            all_metric_groups_values.append(metric_avgs)
        return all_metric_groups_values

    return get_metric_avgs


def update_metrics_generator(metric_fn_groups):
    def update_metrics(metric_value_groups, outputs, labels):
        """
        metric_value_groups is a list of metric functions. For example, for a model with 3
        outputs, we can define these two metric groups
        [
            [metric_fn1],
            [metric_fn21,metric_fn22,metric_fn23],
        ]

        In this example, first metric group provides only one metric function. This
        function will be used to calculate the metric on all of the model outputs. Second
        metric groups, however, defines one metric function per output.
        """

        num_outputs = len(outputs)
        for metric_fn_group, metric_value_group in zip(metric_fn_groups, metric_value_groups):
            if len(metric_fn_group) == 1:
                _metric_fn_group = [metric_fn_group[0] for _ in range(num_outputs)]
            else:
                _metric_fn_group = metric_fn_group

            for metric_val, metric_fn, output_group, label_group in \
                    zip(metric_value_group, _metric_fn_group, outputs, labels):
                metric_val.update(metric_fn(output_group, label_group))

        return metric_value_groups

    return update_metrics


def calculate_loss_generator():
    def calculate_loss(outputs, labels, loss_weights, loss_fns, sample_weights=None):
        # If only one loss function is passed by user, use it to calculate the loss on all
        # the outputs
        if len(loss_fns) == 1:
            _loss_fns = [loss_fns[0] for _ in range(len(outputs))]
        else:
            _loss_fns = loss_fns
        if sample_weights is not None:
            # when reduction='none', loss function returns the value of all the losses
            # from all the samples. We multiply each sample's weight to its loss and
            # then take the mean of the weight adjusted losses from all the samples in the
            # batch. Note that this approach is not "weighted average" because the sum of
            # the sample weights in each batch does not necessarily add up to one. If we add
            # the weights and divide the sum to the sum of weights, the impact of two
            # samples with identical weights but in different batches will not be equal on
            # the calculated gradients.
            losses = []
            for output, label, loss_fn, loss_weight in zip(outputs, labels,
                                                           _loss_fns, loss_weights):
                weight_adjusted_sample_losses = \
                    loss_fn(output, label, reduction='none').flatten() * sample_weights
                output_loss = weight_adjusted_sample_losses.mean()
                losses.append(output_loss * loss_weight)
        else:
            losses = [loss_fn(output, label) * loss_weight for
                      output, label, loss_fn, loss_weight in
                      zip(outputs, labels, _loss_fns, loss_weights)]

        loss = sum(losses)
        return loss

    return calculate_loss


class TorchEstimator(Estimator, EstimatorParams, TorchEstimatorParamsWritable,
                     TorchEstimatorParamsReadable):
    input_shapes = Param(Params._dummy(), 'input_shapes', 'input layer shapes')
    loss_constructor = Param(Params._dummy(), 'loss_constructor',
                             'functions that construct the loss')

    @keyword_only
    def __init__(self,
                 num_proc=None,
                 model=None,
                 backend=None,
                 store=None,
                 optimizer=None,
                 loss=None,
                 loss_constructor=None,
                 metrics=None,
                 loss_weights=None,
                 sample_weight_col=None,
                 compression=None,
                 feature_cols=None,
                 input_shapes=None,
                 validation_col=None,
                 label_cols=None,
                 callbacks=None,
                 batch_size=None,
                 epochs=None,
                 validation_split=None,
                 verbose=1,
                 shuffle_buffer_size=None,
                 partitions_per_process=None):
        super(TorchEstimator, self).__init__()
        self._setDefault(loss_constructor=None)

        kwargs = self._input_kwargs

        if EstimatorParams.loss_weights.name not in kwargs and EstimatorParams.label_cols.name in kwargs:
            # If loss_wight is not provided use equal weights for all the losses
            num_outpus = len(kwargs[EstimatorParams.label_cols.name])
            kwargs[EstimatorParams.loss_weights.name] = [1 / float(num_outpus) for _ in
                                                         range(num_outpus)]

        if EstimatorParams.loss.name in kwargs and TorchEstimator.loss_constructor.name in kwargs:
            raise ValueError("only one of loss_constructor and loss parameters can be specified.")

        if EstimatorParams.loss.name in kwargs and not \
                (isinstance(loss, list) or isinstance(loss, tuple)):
            kwargs[EstimatorParams.loss.name] = [kwargs[EstimatorParams.loss.name]]

        if TorchEstimator.loss_constructor.name in kwargs and not \
                (isinstance(loss_constructor, list) or isinstance(loss_constructor, tuple)):
            kwargs[TorchEstimator.loss_constructor.name] = [
                kwargs[TorchEstimator.loss_constructor.name]]

        self.setParams(**kwargs)

    def setInputShapes(self, value):
        return self._set(input_shapes=value)

    def getInputShapes(self):
        return self.getOrDefault(self.input_shapes)

    def getLossConstructors(self):
        return self.getOrDefault(self.loss_constructor)

    def _get_optimizer(self):
        return self.getOrDefault(self.optimizer)

    def getOptimizer(self):
        model = self.getModel()
        if model:
            optimizer = self._get_optimizer()
            optimizer_cls = optimizer.__class__
            optimizer_state = optimizer.state_dict()
            optimzer = optimizer_cls(model.parameters(), lr=1)
            optimzer.load_state_dict(optimizer_state)
            return optimzer
        else:
            return self._get_optimizer()

    def _check_model_compatibility(self, metadata):
        util.check_shape_compatibility(metadata,
                                       self.getFeatureCols(),
                                       self.getLabelCols(),
                                       input_shapes=self.getInputShapes())

    def _fit(self, df):
        optimizer = self._get_optimizer()
        loss_fns_pre_train = self.getLoss()
        loss_constructors = self.getLossConstructors()
        compression = self.getCompression()
        backend = self.getBackend()
        store = self.getStore()
        loss_weights = self.getLossWeights()
        input_shapes = self.getInputShapes()
        label_columns = self.getLabelCols()
        feature_columns = self.getFeatureCols()
        validation_split = self.getValidationSplit()
        batch_size = self.getBatchSize()
        epochs = self.getEpochs()
        sample_weight_col = self.getSampleWeightCol()
        validation_col = self.getValidationCol()
        metric_fn_groups = self.getMetrics()
        user_shuffle_buffer_size = self.getShufflingBufferSize()
        partitions_per_process = self.getPartitionsPerProcess()

        num_processes = self.getNumProc()
        if (num_processes is None) == (backend is None):
            raise ValueError('Exactly one of parameters "num_processes" and "backend" '
                             'must be specified')
        elif backend is None:
            backend = SparkBackend(num_processes)
        elif num_processes is None:
            num_processes = backend.num_processes()

        train_rows, val_rows, metadata, avg_row_size = \
            util.prepare_data(num_processes,
                              store,
                              df,
                              label_columns=label_columns,
                              feature_columns=feature_columns,
                              validation_split=validation_split,
                              validation_col=validation_col,
                              sample_weight_col=sample_weight_col,
                              partitions_per_process=partitions_per_process)

        self._check_model_compatibility(metadata)

        train_data_path = store.get_train_data_path()
        validation_data_path = store.get_val_data_path()
        should_validate = self._should_validate()

        optimizer_cls = optimizer.__class__
        optimizer_state = optimizer.state_dict()

        get_petastorm_path = store.get_petastorm_path_generator()
        user_verbose = self.getVerbose()

        # Class generators
        metric_class = metric_class_generator()

        # Function generators
        get_optimizer_with_unscaled_lr_fn = get_optimizer_with_unscaled_lr_generator()
        calculate_shuffle_buffer_size_fn = calculate_shuffle_buffer_size_generator()
        construct_metric_value_holders_fn = construct_metric_value_holders_generator()
        prepare_np_data_fn = prepare_np_data_generator()
        get_metric_avgs_fn = get_metric_avgs_generator()
        update_metrics_fn = update_metrics_generator(metric_fn_groups)
        calculate_loss_fn = calculate_loss_generator()
        serialize_fn = util.serialize_generator()
        deserialize_fn = util.deserialize_generator()

        model_pre_train = self.getModel()
        serialized_model = util.serialize(model_pre_train)

        def train():
            import io
            import tempfile
            import shutil
            import contextlib
            import torch
            import horovod.torch as hvd
            from petastorm import make_batch_reader
            from petastorm.pytorch import DataLoader

            @contextlib.contextmanager
            def _tempdir():
                dirpath = tempfile.mkdtemp()
                try:
                    yield dirpath
                finally:
                    shutil.rmtree(dirpath)

            # Deserializing objects
            model = deserialize_fn(serialized_model)

            if loss_fns_pre_train:
                loss_fns = loss_fns_pre_train
            if loss_constructors:
                local_vars = locals()
                loss_fns = [loss_constructor(**local_vars) for loss_constructor in loss_constructors]

            # Horovod: initialize library.
            hvd.init()

            if not user_shuffle_buffer_size:
                shuffle_buffer_size = \
                    calculate_shuffle_buffer_size_fn(hvd, avg_row_size, train_rows / hvd.size())
            else:
                shuffle_buffer_size = user_shuffle_buffer_size

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                # Horovod: pin GPU to local rank.
                torch.cuda.set_device(hvd.local_rank())
                # Move model to GPU.
                model.cuda()

            # Optimizer object needs to be re-instantiated. Internally, it uses memory addresses of
            # objects as their identity and therefore it cannot be serialized and then
            # deserialized. The deserialized optimizer object stores the names of the parameters
            # with their old memory addresses but in reality those are different than the
            # reconstructed deserialized object and that creates problem.
            # Learning rate is a required parameters in SGD optimizer. It will be overridden with
            # load_state_dict.
            optimizer = optimizer_cls(model.parameters(), lr=1)
            # scale the learning rate with the number of horovod workers
            for i in range(len(optimizer_state['param_groups'])):
                optimizer_state['param_groups'][i]['lr'] = \
                    optimizer_state['param_groups'][i]['lr'] * hvd.size()

            optimizer.load_state_dict(optimizer_state)

            # Horovod: broadcast parameters & optimizer state.
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

            dist_optimizer_args = dict(optimizer=optimizer,
                                       named_parameters=model.named_parameters())
            if compression:
                # Pass the compression arg only if it is specified by the user.
                dist_optimizer_args['compression'] = compression
            # Horovod: wrap optimizer with DistributedOptimizer.
            dist_optimizer = hvd.DistributedOptimizer(**dist_optimizer_args)

            # This function takes the current optimizer and constructs a new optimizer with the
            # same state except with learning rate scaled down with the number of horovod workers.
            # This is important the retraining of the model. User may retrain the model with
            # different number of workers and we need the raw learning rate to adjust with the
            # new number of workers.

            schema_fields = feature_columns + label_columns
            if sample_weight_col:
                schema_fields.append(sample_weight_col)

            with _tempdir() as ckpt_dir:
                # Model checkpoint location.
                ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')

                # Petastorm: read data from the store with the correct shard for this rank
                with make_batch_reader(get_petastorm_path(train_data_path),
                                       # setting num_epochs=None will cause an infinite iterator
                                       # and enables ranks to perform training and validation with
                                       # unequal number of samples
                                       num_epochs=None,
                                       cur_shard=hvd.rank(),
                                       shard_count=hvd.size(),
                                       hdfs_driver=PETASTORM_HDFS_DRIVER,
                                       schema_fields=schema_fields,
                                       ) as train_reader:

                    train_loader = DataLoader(train_reader,
                                              batch_size=batch_size,
                                              shuffling_queue_capacity=shuffle_buffer_size)
                    train_loader_iter = iter(train_loader)
                    steps_per_epoch = int(math.ceil(float(train_rows) / batch_size / hvd.size()))

                    def _train():
                        model.train()
                        train_loss = metric_class('loss', hvd)

                        metric_value_groups = construct_metric_value_holders_fn(metric_class,
                                                                                metric_fn_groups,
                                                                                label_columns,
                                                                                hvd)

                        # iterate on one epoch
                        for batch_idx in range(steps_per_epoch):
                            row = next(train_loader_iter)
                            inputs = [
                                prepare_np_data_fn(row[col].float(), col, metadata).reshape(shape)
                                for col, shape in zip(feature_columns, input_shapes)]
                            labels = [prepare_np_data_fn(row[col].float(), col, metadata) for col in
                                      label_columns]

                            sample_weights = row.get(sample_weight_col, None)

                            if cuda_available:
                                inputs = [input.cuda() for input in inputs]
                                labels = [label.cuda() for label in labels]

                            dist_optimizer.zero_grad()
                            outputs = model(*inputs)
                            if type(outputs) != tuple and type(outputs) != list:
                                outputs = [outputs]

                            # reshape labels to match the output shape of the model
                            if hasattr(outputs[0], 'shape'):
                                labels = [label.reshape(output.shape) for label, output in
                                          zip(labels, outputs)]

                            loss = calculate_loss_fn(outputs, labels, loss_weights, loss_fns,
                                                     sample_weights)
                            train_loss.update(loss)
                            loss.backward()
                            dist_optimizer.step()
                            update_metrics_fn(metric_value_groups, outputs, labels)

                        all_metric_groups_values = get_metric_avgs_fn(metric_value_groups)
                        return {train_loss.name: train_loss.avg.item(),
                                'all_metrics': all_metric_groups_values}

                    if not should_validate:
                        history = []
                        for epoch in range(epochs):
                            epoch_metrics = {'epoch': epoch}
                            epoch_metrics['train'] = _train()
                            if user_verbose > 0:
                                print(epoch_metrics)
                            history.append(epoch_metrics)
                            # Save model after every epoch
                            if hvd.rank() == 0:
                                with open(ckpt_file, 'w') as f:
                                    _model_serialized = serialize_fn(model)
                                    f.write(_model_serialized)

                    else:
                        # Petastorm: read data from the store with the correct shard for this rank
                        with make_batch_reader(get_petastorm_path(validation_data_path),
                                               # setting num_epochs=None will cause an infinite
                                               # iterator and enables ranks to perform training
                                               # and validation with unequal number of samples
                                               num_epochs=None,
                                               cur_shard=hvd.rank(),
                                               shard_count=hvd.size(),
                                               hdfs_driver=PETASTORM_HDFS_DRIVER,
                                               schema_fields=schema_fields,
                                               ) as val_reader:

                            val_loader = DataLoader(val_reader,
                                                    batch_size=batch_size)

                            val_loader_iter = iter(val_loader)
                            validation_steps = int(
                                math.ceil(float(val_rows) / batch_size / hvd.size()))

                            def _validate():
                                model.eval()
                                val_loss = metric_class('loss', hvd)
                                metric_value_groups = \
                                    construct_metric_value_holders_fn(metric_class,
                                                                      metric_fn_groups,
                                                                      label_columns, hvd)
                                # iterate on one epoch
                                for batch_idx in range(validation_steps):
                                    row = next(val_loader_iter)

                                    inputs = [prepare_np_data_fn(row[col].float(), col,
                                                                 metadata).reshape(shape)
                                              for col, shape in zip(feature_columns, input_shapes)]
                                    labels = [prepare_np_data_fn(row[col].float(), col, metadata)
                                              for col in label_columns]

                                    sample_weights = row.get(sample_weight_col, None)
                                    if cuda_available:
                                        inputs = [inpt.cuda() for inpt in inputs]
                                        labels = [label.cuda() for label in labels]
                                    outputs = model(*inputs)
                                    if type(outputs) != tuple and type(outputs) != list:
                                        outputs = [outputs]

                                    # reshape labels to match the output shape of the model
                                    if hasattr(outputs[0], 'shape'):
                                        labels = [label.reshape(output.shape) for label, output in
                                                  zip(labels, outputs)]

                                    loss = calculate_loss_fn(outputs, labels, loss_weights,
                                                             loss_fns, sample_weights)
                                    val_loss.update(loss)
                                    update_metrics_fn(metric_value_groups, outputs, labels)

                                all_metric_groups_values = get_metric_avgs_fn(metric_value_groups)
                                return {val_loss.name: val_loss.avg.item(),
                                        'all_metrics': all_metric_groups_values}

                            history = []
                            for epoch in range(epochs):
                                epoch_metrics = {'epoch': epoch}
                                epoch_metrics['train'] = _train()
                                epoch_metrics['validation'] = _validate()
                                if user_verbose > 0:
                                    print(epoch_metrics)
                                history.append(epoch_metrics)
                                # Save model after every epoch
                                if hvd.rank() == 0:
                                    with open(ckpt_file, 'wb') as f:
                                        _model_serialized = serialize_fn(model)
                                        f.write(_model_serialized)
            if hvd.rank() == 0:
                with open(ckpt_file, 'r') as f:
                    model_serialized = f.read()
                    loaded_model = deserialize_fn(model_serialized)
                    # need to move the model to cpu before serialization. Otherwise,
                    # deserialization will fail if the machine on which the deserialization
                    # is happening does not have gpu.
                    loaded_model.cpu()

                optimizer_with_unscaled_lr = \
                    get_optimizer_with_unscaled_lr_fn(hvd, dist_optimizer, optimizer_cls, model)

                bio_opt = io.BytesIO()
                torch.save(optimizer_with_unscaled_lr, bio_opt)
                bio_opt.seek(0)

                return history, serialize_fn(loaded_model), serialize_fn(bio_opt)

        history, serialized_model, serialized_optimizer = backend.run(train, env={})[0]

        trained_model = codec.loads_base64(serialized_model)

        # torch.load correctly moves all the optimizer state values to cpu
        # before creating the object.
        optimizer_bio = codec.loads_base64(serialized_optimizer)
        opt = torch.load(optimizer_bio, map_location=torch.device('cpu'))

        return self._create_model(history, trained_model, opt, metadata)

    def _create_model(self, history, model, optimizer, metadata):
        return TorchModel(history=history,
                          model=model,
                          optimizer=optimizer,
                          feature_columns=self.getFeatureCols(),
                          input_shapes=self.getInputShapes(),
                          label_columns=self.getLabelCols(),
                          _metadata=metadata)


class TorchModel(Model, ModelParams, TorchEstimatorParamsWritable, TorchEstimatorParamsReadable):
    optimizer = Param(Params._dummy(), 'optimizer', 'optimizer')
    input_shapes = Param(Params._dummy(), 'input_shapes', 'input layer shapes')

    @keyword_only
    def __init__(self,
                 history=None,
                 model=None,
                 feature_columns=None,
                 input_shapes=None,
                 label_columns=None,
                 optimizer=None,
                 _metadata=None):
        super(TorchModel, self).__init__()

        if label_columns:
            self.setOutputCols([col + '__output' for col in label_columns])
        kwargs = self._input_kwargs

        self.setParams(**kwargs)

    def setInputShapes(self, value):
        return self._set(input_shapes=value)

    def getInputShapes(self):
        return self.getOrDefault(self.input_shapes)

    def setOptimizer(self, value):
        return self._set(optimizer=value)

    def _get_optimizer(self):
        return self.getOrDefault(self.optimizer)

    def getOptimizer(self):
        model = self.getModel()
        if model:
            _optimizer = self._get_optimizer()
            optimizer_cls = _optimizer.__class__
            optimizer_state = _optimizer.state_dict()
            optimzer = optimizer_cls(model.parameters(), lr=1)
            optimzer.load_state_dict(optimizer_state)
            return optimzer
        else:
            return self._get_optimizer()

    # To run locally on OS X, need export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    def _transform(self, df):

        model_pre_predict = self.getModel()
        model_pre_predict.eval()
        serialized_model = util.serialize(model_pre_predict)

        input_shapes = self.getInputShapes()
        label_cols = self.getLabelColumns()
        output_cols = self.getOutputCols()
        feature_cols = self.getFeatureColumns()
        metadata = self._get_metadata()

        # Functions
        deserialize_fn = util.deserialize_generator()

        def predict(rows):
            from pyspark import Row
            from pyspark.ml.linalg import DenseVector, SparseVector

            model = deserialize_fn(serialized_model)
            # Perform predictions.
            for row in rows:
                fields = row.asDict().copy()

                # Note: if the col is SparseVector, torch.tensor(col) correctly converts it to a
                # dense torch tensor.
                data = [torch.tensor([row[col]]).reshape(shape) for
                        col, shape in zip(feature_cols, input_shapes)]

                preds = model(*data)
                if not isinstance(preds, list) and not isinstance(preds, tuple):
                    preds = [preds]

                for label_col, output_col, pred in zip(label_cols, output_cols, preds):
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
                        value = pred.item()
                        if issubclass(col_type, numbers.Integral):
                            value = round(value)
                        field = col_type(value)

                    fields[output_col] = field

                yield Row(**fields)

        return df.rdd.mapPartitions(predict).toDF()
