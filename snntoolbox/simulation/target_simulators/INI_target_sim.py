# -*- coding: utf-8 -*-
"""Building and simulating spiking neural networks using INIsim.

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os

import keras
import numpy as np
from future import standard_library

from snntoolbox.simulation.backends.inisim.inisim import bias_relaxation
from snntoolbox.simulation.utils import AbstractSNN

standard_library.install_aliases()

remove_classifier = False


class SNN(AbstractSNN):
    """
    The compiled spiking neural network, using layers derived from
    Keras base classes (see `snntoolbox.simulation.backends.inisim.inisim`).

    Aims at simulating the network on a self-implemented Integrate-and-Fire
    simulator using a timestepped approach.

    Attributes
    ----------

    snn: keras.models.Model
        Keras model. This is the output format of the compiled spiking model
        because INI simulator runs networks of layers that are derived from
        Keras layer base classes.
    """

    def __init__(self, config, queue=None):

        AbstractSNN.__init__(self, config, queue)

        self.snn = None
        self._spiking_layers = {}
        self._input_images = None
        self._binary_activation = None
        self.avg_rate = None

    @property
    def is_parallelizable(self):
        return True

    def add_input_layer(self, input_shape):
        self._input_images = keras.layers.Input(batch_shape=input_shape)
        self._spiking_layers[self.parsed_model.layers[0].name] = \
            self._input_images

    def add_layer(self, layer):
        from snntoolbox.parsing.utils import get_type
        spike_layer_name = getattr(self.sim, 'Spike' + get_type(layer))
        inbound = [self._spiking_layers[inb.name] for inb in
                   layer.inbound_nodes[0].inbound_layers]
        if len(inbound) == 1:
            inbound = inbound[0]
        layer_kwargs = layer.get_config()
        layer_kwargs['config'] = self.config

        # Check if layer uses binary activations. In that case, we will want to
        # tell the following to MaxPool layer because then we can use a
        # cheaper operation.
        if 'Conv' in layer.name and 'binary' in layer.activation.__name__:
            self._binary_activation = layer.activation.__name__

        if 'MaxPool' in layer.name and self._binary_activation is not None:
            layer_kwargs['activation'] = self._binary_activation
            self._binary_activation = None

        # Replace activation from kwargs by 'linear' before initializing
        # superclass, because the relu activation is applied by the spike-
        # generation mechanism automatically. In some cases (quantized
        # activation), we need to apply the activation manually. This
        # information is taken from the 'activation' key during conversion.
        activation_str = str(layer_kwargs.pop(str('activation'), None))

        spike_layer = spike_layer_name(**layer_kwargs)
        spike_layer.activation_str = activation_str
        self._spiking_layers[layer.name] = spike_layer(inbound)

    def build_dense(self, layer):
        pass

    def build_convolution(self, layer):
        pass

    def build_pooling(self, layer):
        pass

    def compile(self):
        self.snn = keras.models.Model(
            self._input_images,
            self._spiking_layers[self.parsed_model.layers[-1].name])
        self.snn.compile('sgd', 'categorical_crossentropy', ['accuracy'])
        self.snn.set_weights(self.parsed_model.get_weights())
        for layer in self.snn.layers:
            if hasattr(layer, 'bias'):
                # Adjust biases to time resolution of simulator.
                layer.bias.set_value(layer.bias.get_value() * self._dt)
                if bias_relaxation:  # Experimental
                    layer.b0.set_value(layer.bias.get_value())

    def simulate(self, **kwargs):

        from snntoolbox.utils.utils import echo
        from snntoolbox.simulation.utils import get_layer_synaptic_operations

        input_b_l = kwargs[str('x_b_l')] * self._dt

        output_b_l_t = np.zeros((self.batch_size, self.num_classes,
                                 self._num_timesteps), 'int32')

        # Loop through simulation time.
        self.avg_rate = 0
        self._input_spikecount = 0
        for sim_step_int in range(self._num_timesteps):
            sim_step = (sim_step_int + 1) * self._dt
            self.set_time(sim_step)

            # Generate new input in case it changes with each simulation step.
            if self._poisson_input:
                input_b_l = self.get_poisson_frame_batch(kwargs[str('x_b_l')])
            elif self._dataset_format == 'aedat':
                input_b_l = kwargs[str('dvs_gen')].next_eventframe_batch()

#            self.scale_first_layer_parameters(sim_step_int, input_b_l)
            new_input = np.concatenate([input_b_l, np.array(input_b_l > 0, 'float32')]) if self.config.getboolean('conversion', 'use_isi_code') else input_b_l
            # Main step: Propagate input through network and record output
            # spikes.
            out_spikes = self.snn.predict_on_batch(new_input)

            # Add current spikes to previous spikes.
            if remove_classifier:  # Need to flatten output.
                output_b_l_t[:, :, sim_step_int] = np.argmax(np.reshape(
                    out_spikes.astype('int32'), (out_spikes.shape[0], -1)), 1)
            else:
                output_b_l_t[:, :, sim_step_int] = out_spikes.astype('int32')

            # Record neuron variables.
            i = j = 0
            for layer in self.snn.layers:
                # Excludes Input, Flatten, Concatenate, etc:
                if hasattr(layer, 'spiketrain') \
                        and layer.spiketrain is not None:
                    spiketrains_b_l = layer.spiketrain.get_value()
                    self.avg_rate += np.count_nonzero(spiketrains_b_l)
                    if self.spiketrains_n_b_l_t is not None:
                        self.spiketrains_n_b_l_t[i][0][
                            Ellipsis, sim_step_int] = spiketrains_b_l
                    if self.synaptic_operations_b_t is not None:
                        self.synaptic_operations_b_t[:, sim_step_int] += \
                            get_layer_synaptic_operations(spiketrains_b_l,
                                                          self.fanout[i + 1])
                    if self.neuron_operations_b_t is not None:
                        self.neuron_operations_b_t[:, sim_step_int] += \
                            self.num_neurons_with_bias[i + 1]
                    i += 1
                if hasattr(layer, 'mem') and self.mem_n_b_l_t is not None:
                    self.mem_n_b_l_t[j][0][Ellipsis, sim_step_int] = \
                        layer.mem.get_value()
                    j += 1
                # if hasattr(layer, 'v_thresh') and '00Conv' in layer.name:
                #     print(layer.v_thresh.get_value())
            if 'input_b_l_t' in self._log_keys:
                self.input_b_l_t[Ellipsis, sim_step_int] = input_b_l
            if self._poisson_input or self._dataset_format == 'aedat':
                if self.synaptic_operations_b_t is not None:
                    self.synaptic_operations_b_t[:, sim_step_int] += \
                        get_layer_synaptic_operations(input_b_l, self.fanout[0])
            else:
                if self.neuron_operations_b_t is not None:
                    if sim_step_int == 0:
                        self.neuron_operations_b_t[:, 0] += self.fanin[1] * \
                            self.num_neurons[1] * np.ones(self.batch_size) * 2

            if self.config.getint('output', 'verbose') > 0 \
                    and sim_step % 1 == 0:
                if self.config.getboolean('conversion', 'use_isi_code'):
                    first_spiketimes_b_l = np.argmax(output_b_l_t, 2)
                    first_spiketimes_b_l[np.nonzero(np.sum(
                        output_b_l_t, 2) == 0)] = self._num_timesteps
                    guesses_b = np.argmin(first_spiketimes_b_l, 1)
                else:
                    guesses_b = np.argmax(np.sum(output_b_l_t, 2), 1)
                echo('{:.2%}_'.format(np.mean(kwargs[str('truth_b')] ==
                                              guesses_b)))

            if self.config.getboolean('conversion', 'use_isi_code') and \
                    all(np.count_nonzero(output_b_l_t, (1, 2)) >= self.top_k):
                print("Finished early.")
                break

        if self.config.getboolean('conversion', 'use_isi_code'):
            for b in range(self.batch_size):
                for l in range(self.num_classes):
                    spike = 0
                    for t in range(self._num_timesteps):
                        if output_b_l_t[b, l, t] != 0:
                            spike = 1
                        output_b_l_t[b, l, t] = spike

        self.avg_rate /= self.batch_size * np.sum(self.num_neurons) * \
            self._num_timesteps

        print("Average spike rate: {} spikes per simulation time step.".format(
            self.avg_rate))

        return np.cumsum(np.asarray(output_b_l_t, bool), 2)

    def reset(self, sample_idx):

        for layer in self.snn.layers[1:]:  # Skip input layer
            layer.reset(sample_idx)

    def end_sim(self):
        pass

    def save(self, path, filename):

        filepath = os.path.join(path, filename + '.h5')
        print("Saving model to {}...\n".format(filepath))
        self.snn.save(filepath, self.config.getboolean('output', 'overwrite'))

    def load(self, path, filename):

        from snntoolbox.simulation.backends.inisim.inisim import custom_layers

        filepath = os.path.join(path, filename + '.h5')

        try:
            self.snn = keras.models.load_model(filepath, custom_layers)
        except KeyError:
            raise NotImplementedError(
                "Loading SNN for INIsim is not supported yet.")
            # Loading does not work anymore because the configparser object
            # needed by the custom layers is not stored when saving the model.
            # Could be implemented by overriding Keras' save / load methods, but
            # since converting even large Keras models from scratch is so fast,
            # there's really no need.

    def get_poisson_frame_batch(self, x_b_l):
        """Get a batch of Poisson input spikes.

        Parameters
        ----------

        x_b_l: ndarray
            The input frame. Shape: (`batch_size`, ``layer_shape``).

        Returns
        -------

        input_b_l: ndarray
            Array of Poisson input spikes, with same shape as ``x_b_l``.

        """

        if self._input_spikecount < self._num_poisson_events_per_sample \
                or self._num_poisson_events_per_sample < 0:
            spike_snapshot = np.random.random_sample(x_b_l.shape) \
                             * self.rescale_fac * np.max(x_b_l)
            input_b_l = (spike_snapshot <= np.abs(x_b_l)).astype('float32')
            self._input_spikecount += \
                np.count_nonzero(input_b_l) / self.batch_size
            # For BinaryNets, with input that is not normalized and
            # not all positive, we stimulate with spikes of the same
            # size as the maximum activation, and the same sign as
            # the corresponding activation. Is there a better
            # solution?
            input_b_l *= np.max(x_b_l) * np.sign(x_b_l)
        else:  # No more input spikes if _input_spikecount exceeded limit.
            input_b_l = np.zeros(x_b_l.shape)

        return input_b_l

    def set_time(self, t):
        """Set the simulation time variable of all layers in the network.

        Parameters
        ----------

        t: float
            Current simulation time.
        """

        for layer in self.snn.layers[1:]:
            if layer.get_time() is not None:  # Has time attribute
                layer.set_time(np.float32(t))

    def set_spiketrain_stats_input(self):
        # Added this here because PyCharm complains about not all abstract
        # methods being implemented (even though this is not abstract).
        AbstractSNN.set_spiketrain_stats_input(self)

    def get_spiketrains_input(self):
        # Added this here because PyCharm complains about not all abstract
        # methods being implemented (even though this is not abstract).
        AbstractSNN.get_spiketrains_input(self)

    def scale_first_layer_parameters(self, t, input_b_l, tau=1):
        w, b = self.snn.layers[0].get_weights()
        alpha = (self._duration + tau) / (t + tau)
        beta = b + tau * (self._duration - t) / (t + tau) * w * input_b_l
        self.snn.layers[0].kernel.set_value(alpha * w)
        self.snn.layers[0].bias.set_value(beta)
