# -*- coding: utf-8 -*-
"""INI spiking neuron simulator.

This module defines the layer objects used to create a spiking neural network
for our built-in INI simulator
:py:mod:`~snntoolbox.simulation.target_simulators.INI_target_sim`.

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
from future import standard_library
from keras import backend as k
from keras.layers import Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from keras.layers import Layer, Concatenate

standard_library.install_aliases()

bias_relaxation = False


class SpikeLayer(Layer):
    """Base class for layer with spiking neurons."""

    def __init__(self, **kwargs):
        self.config = kwargs.pop(str('config'), None)
        self.layer_type = self.class_name
        self.dt = self.config.getfloat('simulation', 'dt')
        self.duration = self.config.getint('simulation', 'duration')
        self.tau_refrac = self.config.getfloat('cell', 'tau_refrac')
        self._v_thresh = self.config.getfloat('cell', 'v_thresh')
        self.v_thresh = None
        self.time = None
        self.mem = self.spiketrain = self.impulse = None
        self.refrac_until = None

        if self.config.getboolean('conversion', 'use_isi_code'):
            self.last_spiketimes = None
            self.prospective_spikes = None
            self.missing_impulse = None

        allowed_kwargs = {'input_shape',
                          'batch_input_shape',
                          'batch_size',
                          'dtype',
                          'name',
                          'trainable',
                          'weights',
                          'input_dtype',  # legacy
                          }
        for kwarg in kwargs.copy():
            if kwarg not in allowed_kwargs:
                kwargs.pop(kwarg)
        Layer.__init__(self, **kwargs)
        self.stateful = True

    def reset(self, sample_idx):
        """Reset layer variables."""

        self.reset_spikevars(sample_idx)

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__

    def update_neurons(self):
        """Update neurons according to activation function."""

        # Update membrane potentials.
        new_mem = self.get_new_mem()

        # Generate spikes.
        output_spikes = self.linear_activation(new_mem)

        # Reset membrane potential after spikes.
        self.set_reset_mem(new_mem, output_spikes)

        # Store refractory period after spikes.
        if self.tau_refrac > 0:
            new_refractory = k.T.set_subtensor(
                self.refrac_until[output_spikes.nonzero()],
                self.time + self.tau_refrac)
            self.add_update([(self.refrac_until, new_refractory)])

        if self.config.getboolean('conversion', 'use_isi_code'):
            masked_impulse = k.T.set_subtensor(
                self.impulse[k.T.nonzero(self.refrac_until > self.time)], 0.)
            self.add_update([
                (self.v_thresh, k.cast(self._v_thresh * np.ones_like(
                    self.mem, k.floatx()) + self.missing_impulse, k.floatx())),
                (self.prospective_spikes, k.cast(k.greater(masked_impulse, 0),
                                                 k.floatx()))])

        if self.spiketrain is not None:
            self.add_update([(self.spiketrain,
                              self.time * k.not_equal(output_spikes, 0))])

        # Compute post-synaptic potential.
        psp = self.get_psp(output_spikes)

        return k.cast(psp, k.floatx())

    def linear_activation(self, mem):
        """Linear activation."""
        return k.T.mul(k.greater_equal(mem, self.v_thresh), self.v_thresh)

    def get_new_mem(self):
        """Add input to membrane potential."""

        # Destroy impulse if in refractory period
        masked_impulse = self.impulse if self.tau_refrac == 0 else \
            k.T.set_subtensor(
                self.impulse[k.T.nonzero(self.refrac_until > self.time)], 0.)

        new_mem = self.mem + masked_impulse

        if self.config.getboolean('cell', 'leak'):
            # Todo: Implement more flexible version of leak!
            new_mem = k.T.inc_subtensor(
                new_mem[k.T.nonzero(k.T.gt(new_mem, 0))], -0.1 * self.dt)

        return new_mem

    def set_reset_mem(self, mem, spikes):
        """
        Reset membrane potential ``mem`` array where ``spikes`` array is
        nonzero.
        """

        spike_idxs = k.T.nonzero(spikes)
        new = k.T.set_subtensor(mem[spike_idxs], 0.)
        self.add_update([(self.mem, new)])

    def get_psp(self, output_spikes):
        if self.config.getboolean('conversion', 'use_isi_code'):
            new_spiketimes = k.T.set_subtensor(self.last_spiketimes[
                k.T.nonzero(output_spikes)], self.get_time())
            self.add_update([(self.last_spiketimes, new_spiketimes)])
            # psp = k.maximum(0, k.T.true_div(self.dt, self.last_spiketimes))
            psp = k.T.set_subtensor(output_spikes[k.T.nonzero(
                self.last_spiketimes > 0)], self.dt)
            return psp
        else:
            return output_spikes

    def get_time(self):
        """Get simulation time variable.

            Returns
            -------

            time: float
                Current simulation time.
            """

        return self.time.get_value()

    def set_time(self, time):
        """Set simulation time variable.

        Parameters
        ----------

        time: float
            Current simulation time.
        """

        self.time.set_value(time)

    def init_membrane_potential(self, output_shape=None, mode='zero'):
        """Initialize membrane potential.

        Helpful to avoid transient response in the beginning of the simulation.
        Not needed when reset between frames is turned off, e.g. with a video
        data set.

        Parameters
        ----------

        output_shape: Optional[tuple]
            Output shape
        mode: str
            Initialization mode.

            - ``'uniform'``: Random numbers from uniform distribution in
              ``[-thr, thr]``.
            - ``'bias'``: Negative bias.
            - ``'zero'``: Zero (default).

        Returns
        -------

        init_mem: ndarray
            A tensor of ``self.output_shape`` (same as layer).
        """

        if output_shape is None:
            output_shape = self.output_shape

        if mode == 'uniform':
            init_mem = k.random_uniform(output_shape,
                                        -self._v_thresh, self._v_thresh)
        elif mode == 'bias':
            init_mem = np.zeros(output_shape, k.floatx())
            if hasattr(self, 'b'):
                b = self.get_weights()[1]
                for i in range(len(b)):
                    init_mem[:, i, Ellipsis] = -b[i]
        else:  # mode == 'zero':
            init_mem = np.zeros(output_shape, k.floatx())
        return init_mem

    def reset_spikevars(self, sample_idx):
        """
        Reset variables present in spiking layers. Can be turned off for
        instance when a video sequence is tested.
        """

        mod = self.config.getint('simulation', 'reset_between_nth_sample')
        mod = mod if mod else sample_idx + 1
        do_reset = sample_idx % mod == 0
        if do_reset:
            self.mem.set_value(self.init_membrane_potential())
        self.time.set_value(np.float32(self.dt))
        zeros_output_shape = np.zeros(self.output_shape, k.floatx())
        if self.tau_refrac > 0:
            self.refrac_until.set_value(zeros_output_shape)
        if self.spiketrain is not None:
            self.spiketrain.set_value(zeros_output_shape)
        if self.config.getboolean('conversion', 'use_isi_code'):
            self.last_spiketimes.set_value(zeros_output_shape - 1)
            self.v_thresh.set_value(zeros_output_shape + self._v_thresh)
            self.prospective_spikes.set_value(zeros_output_shape)
            self.missing_impulse.set_value(zeros_output_shape)

    def init_neurons(self, input_shape):
        """Init layer neurons."""

        from snntoolbox.bin.utils import get_log_keys, get_plot_keys

        output_shape = self.compute_output_shape(input_shape)
        self.v_thresh = k.variable(self._v_thresh)
        self.mem = k.variable(self.init_membrane_potential(output_shape))
        self.time = k.variable(self.dt)
        # To save memory and computations, allocate only where needed:
        if self.tau_refrac > 0:
            self.refrac_until = k.zeros(output_shape)
        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations', 'operations',
                'synaptic_operations_b_t', 'neuron_operations_b_t',
                'spiketrains_n_b_l_t'} & (get_plot_keys(self.config) |
               get_log_keys(self.config))):
            self.spiketrain = k.zeros(output_shape)
        if self.config.getboolean('conversion', 'use_isi_code'):
            self.last_spiketimes = k.variable(-np.ones(output_shape))
            self.v_thresh = k.variable(self._v_thresh * np.ones(output_shape))
            self.prospective_spikes = k.variable(np.zeros(output_shape))
            self.missing_impulse = k.variable(np.zeros(output_shape))

    def get_layer_idx(self):
        """Get index of layer."""

        label = self.name.split('_')[0]
        layer_idx = None
        for i in range(len(label)):
            if label[:i].isdigit():
                layer_idx = int(label[:i])
        return layer_idx


def spike_call(call):
    def decorator(self, x):

        batch_size = self.config.getint('simulation', 'batch_size')
        input_psp = x[:batch_size]
        prospective_input_spikes = x[batch_size:]

        self.impulse = call(self, input_psp)

        # Should try different ways here: Take absolute weight values, reverse
        # sign, or do not change them at all. Also, take values of positive
        # input psps instead of prospective input spikes of unit size.
        if len(self.weights) > 0:
            weights, bias = self.get_weights()
            self.set_weights([np.abs(weights), bias])
            missing_impulse = call(self, prospective_input_spikes)
            self.set_weights([weights, bias])
            self.add_update([(self.missing_impulse, missing_impulse)])
        elif 'AveragePooling' in self.name:
            missing_impulse = call(self, prospective_input_spikes)
            self.add_update([(self.missing_impulse, missing_impulse)])

        return k.concatenate([self.update_neurons(), self.prospective_spikes])

    return decorator


class SpikeConcatenate(Concatenate):
    """Spike merge layer"""

    def __init__(self, axis, **kwargs):
        kwargs.pop(str('config'))
        Concatenate.__init__(self, axis, **kwargs)

    def _merge_function(self, inputs):
        return self._merge_function(inputs)

    @staticmethod
    def get_time():
        return None

    @staticmethod
    def reset(sample_idx):
        """Reset layer variables."""

        pass

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeFlatten(Flatten):
    """Spike flatten layer."""

    def __init__(self, **kwargs):
        kwargs.pop(str('config'))
        Flatten.__init__(self, **kwargs)
        self.prospective_spikes = None

    def build(self, input_shape):
        Flatten.build(self, input_shape)
        self.prospective_spikes = k.variable(np.zeros(self.compute_output_shape(
            input_shape)))

    def call(self, x, mask=None):

        input_psp = x[0]
        prospective_input_spikes = x[1]

        self.prospective_spikes = Flatten.call(self, prospective_input_spikes)

        psp = k.cast(super(SpikeFlatten, self).call(input_psp), k.floatx())

        return k.concatenate([psp, self.prospective_spikes])

    @staticmethod
    def get_time():
        return None

    def reset(self, sample_idx):
        """Reset layer variables."""

        self.prospective_spikes.set_value(np.zeros(self.output_shape,
                                                   k.floatx()))

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeDense(Dense, SpikeLayer):
    """Spike Dense layer."""

    def build(self, input_shape):
        """Creates the layer neurons and connections.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        Dense.build(self, input_shape)
        self.init_neurons(input_shape)

    @spike_call
    def call(self, x, **kwargs):

        return Dense.call(self, x)


class SpikeConv2D(Conv2D, SpikeLayer):
    """Spike 2D Convolution."""

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        Conv2D.build(self, input_shape)
        self.init_neurons(input_shape)

    @spike_call
    def call(self, x, mask=None):

        return Conv2D.call(self, x)


class SpikeAveragePooling2D(AveragePooling2D, SpikeLayer):
    """Average Pooling."""

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        AveragePooling2D.build(self, input_shape)
        self.init_neurons(input_shape)

    @spike_call
    def call(self, x, mask=None):

        return AveragePooling2D.call(self, x)


class SpikeMaxPooling2D(MaxPooling2D, SpikeLayer):
    """Spiking Max Pooling."""

    def build(self, input_shape):
        """Creates the layer neurons and connections..

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        MaxPooling2D.build(self, input_shape)
        self.init_neurons(input_shape)

    @spike_call
    def call(self, x, mask=None):
        """Layer functionality."""

        return MaxPooling2D.call(self, x)


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate}
