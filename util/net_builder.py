# MIT License
#
# Copyright (c) 2017 BingZhang Hu
#
# Permission is hereby granted, free of charge, to any person obtaiinput_dimg a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NOinput_dimFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *
import tensorflow.contrib.slim as slim

def conv(input_tensor, input_dim, output_dim, kernel_H, kernel_W, stride_H, stride_W, padType, name, phase_train=True,
         use_batch_norm=True, weight_decay=0.0):
    with tf.variable_scope(name):
        l2_regularizer = lambda t: l2_loss(t, weight=weight_decay)
        kernel = tf.get_variable("weights", [kernel_H, kernel_W, input_dim, output_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=1e-1),
                                 regularizer=l2_regularizer, dtype=input_tensor.dtype)
        cnv = tf.nn.conv2d(input_tensor, kernel, [1, stride_H, stride_W, 1], padding=padType)

        if use_batch_norm:
            conv_bn = batch_norm(cnv, phase_train)
        else:
            conv_bn = cnv
        biases = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(),
                                 dtype=input_tensor.dtype)
        bias = tf.nn.bias_add(conv_bn, biases)
        conv1 = tf.nn.relu(bias)
        variable_summaries(kernel,name)
        variable_summaries(biases,name)
    return conv1


def affine(input_tensor, input_dim, output_dim, name, weight_decay=0.0):
    with tf.variable_scope(name):
        l2_regularizer = lambda t: l2_loss(t, weight=weight_decay)
        weights = tf.get_variable("weights", [input_dim, output_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=1e-1),
                                  regularizer=l2_regularizer, dtype=input_tensor.dtype)
        biases = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(),
                                 dtype=input_tensor.dtype)
        affine1 = tf.nn.relu_layer(input_tensor, weights, biases)
    return affine1


def l2_loss(tensor, weight=1.0, scope=None):
    """Define a L2Loss, useful for regularize, i.e. weight decay.
    Args:
      tensor: tensor to regularize.
      weight: an optional weight to modulate the loss.
      scope: Optional scope for op_scope.
    Returns:
      the L2 loss op.
    """
    with tf.name_scope(scope):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.nn.l2_loss(tensor), name='value')
    return loss


def lppool(input_tensor, pnorm, kernel_H, kernel_W, stride_H, stride_W, padding, name):
    with tf.variable_scope(name):
        if pnorm == 2:
            pwr = tf.square(input_tensor)
        else:
            pwr = tf.pow(input_tensor, pnorm)

        subsamp = tf.nn.avg_pool(pwr,
                                 ksize=[1, kernel_H, kernel_W, 1],
                                 strides=[1, stride_H, stride_W, 1],
                                 padding=padding)
        subsamp_sum = tf.multiply(subsamp, kernel_H * kernel_W)

        if pnorm == 2:
            out = tf.sqrt(subsamp_sum)
        else:
            out = tf.pow(subsamp_sum, 1 / pnorm)

    return out


def mpool(input_tensor, kernel_H, kernel_W, stride_H, stride_W, padding, name):
    with tf.variable_scope(name):
        maxpool = tf.nn.max_pool(input_tensor,
                                 ksize=[1, kernel_H, kernel_W, 1],
                                 strides=[1, stride_H, stride_W, 1],
                                 padding=padding)
    return maxpool


def apool(input_tensor, kernel_H, kernel_W, stride_H, stride_W, padding, name):
    with tf.variable_scope(name):
        avgpool = tf.nn.avg_pool(input_tensor,
                                 ksize=[1, kernel_H, kernel_W, 1],
                                 strides=[1, stride_H, stride_W, 1],
                                 padding=padding)
    return avgpool


def batch_norm(x, phase_train):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates traiinput_dimg phase
        scope:       string, variable scope
        affn:      whether to affn-transform outputs
    Return:
        normed:      batch-normalized maps
    Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
    """
    name = 'batch_norm'
    with tf.variable_scope(name):
        phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
        n_out = int(x.get_shape()[3])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                           name=name + '/beta', trainable=True, dtype=x.dtype)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                            name=name + '/gamma', trainable=True, dtype=x.dtype)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def inception(input_tensor, input_dim, stride_size, conv1_output_dim, conv2a_output_dim, conv2_output_dim,
              conv3a_output_dim, conv3_output_dim, pool_kernel_size, conv4_output_dim, pool_stride_size, poolType, name,
              phase_train=True, use_batch_norm=True, weight_decay=0.0):
    print('name = ', name)
    print('inputSize = ', input_dim)
    print('kernelSize = {3,5}')
    print('kernelStride = {%d,%d}' % (stride_size, stride_size))
    print('outputSize = {%d,%d}' % (conv2_output_dim, conv3_output_dim))
    print('reduceSize = {%d,%d,%d,%d}' % (conv2a_output_dim, conv3a_output_dim, conv4_output_dim, conv1_output_dim))
    print('pooling = {%s, %d, %d, %d, %d}' % (
    poolType, pool_kernel_size, pool_kernel_size, pool_stride_size, pool_stride_size))
    if (conv4_output_dim > 0):
        o4 = conv4_output_dim
    else:
        o4 = input_dim
    print('outputSize = ', conv1_output_dim + conv2_output_dim + conv3_output_dim + o4)
    print('\n\n')

    net = []

    with tf.variable_scope(name):
        with tf.variable_scope('branch1_1x1'):
            if conv1_output_dim > 0:
                conv1 = conv(input_tensor, input_dim, conv1_output_dim, 1, 1, 1, 1, 'SAME', 'conv1x1',
                             phase_train=phase_train,
                             use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv1)

        with tf.variable_scope('branch2_3x3'):
            if conv2a_output_dim > 0:
                conv2a = conv(input_tensor, input_dim, conv2a_output_dim, 1, 1, 1, 1, 'SAME', 'conv1x1',
                              phase_train=phase_train,
                              use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv2 = conv(conv2a, conv2a_output_dim, conv2_output_dim, 3, 3, stride_size, stride_size, 'SAME',
                             'conv3x3', phase_train=phase_train,
                             use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv2)

        with tf.variable_scope('branch3_5x5'):
            if conv3a_output_dim > 0:
                conv3a = conv(input_tensor, input_dim, conv3a_output_dim, 1, 1, 1, 1, 'SAME', 'conv1x1',
                              phase_train=phase_train,
                              use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv3 = conv(conv3a, conv3a_output_dim, conv3_output_dim, 5, 5, stride_size, stride_size, 'SAME',
                             'conv5x5', phase_train=phase_train,
                             use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv3)

        with tf.variable_scope('branch4_pool'):
            if poolType == 'MAX':
                pool = mpool(input_tensor, pool_kernel_size, pool_kernel_size, pool_stride_size, pool_stride_size,
                             'SAME', 'pool')
            elif poolType == 'L2':
                pool = lppool(input_tensor, 2, pool_kernel_size, pool_kernel_size, pool_stride_size, pool_stride_size,
                              'SAME', 'pool')
            else:
                raise ValueError('Invalid pooling type "%s"' % poolType)

            if conv4_output_dim > 0:
                pool_conv = conv(pool, input_dim, conv4_output_dim, 1, 1, 1, 1, 'SAME', 'conv1x1',
                                 phase_train=phase_train,
                                 use_batch_norm=use_batch_norm, weight_decay=weight_decay)
            else:
                pool_conv = pool
            net.append(pool_conv)

        concatenated = array_ops.concat(net, 3, name=name)
    return concatenated


def variable_summaries(var,name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name+'/summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn1_forward_propagation(images, phase_train=True, weight_decay=0.0):
    endpoints = {}
    weight_decay = 0.0
    phase_train = True
    net = conv(images, 3, 64, 7, 7, 2, 2, 'SAME', 'conv1_7x7', phase_train=phase_train,
               use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv1'] = net
    net = mpool(net, 3, 3, 2, 2, 'SAME', 'pool1')
    endpoints['pool1'] = net
    net = conv(net, 64, 64, 1, 1, 1, 1, 'SAME', 'conv2_1x1', phase_train=phase_train, use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv2_1x1'] = net
    net = conv(net, 64, 192, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv3_3x3'] = net
    net = mpool(net, 3, 3, 2, 2, 'SAME', 'pool3')
    endpoints['pool3'] = net
    net = inception(net, 192, 1, 64, 96, 128, 16, 32, 3, 32, 1, 'MAX', 'incept3a', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3a'] = net
    net = inception(net, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, 'MAX', 'incept3b', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3b'] = net
    net = inception(net, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2, 'MAX', 'incept3c', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3c'] = net
    net = inception(net, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1, 'MAX', 'incept4a', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4a'] = net
    net = inception(net, 640, 1, 224, 112, 224, 32, 64, 3, 128, 1, 'MAX', 'incept4b', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4b'] = net
    net = inception(net, 640, 1, 192, 128, 256, 32, 64, 3, 128, 1, 'MAX', 'incept4c', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4c'] = net
    net = inception(net, 640, 1, 160, 144, 288, 32, 64, 3, 128, 1, 'MAX', 'incept4d', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4d'] = net
    net = inception(net, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2, 'MAX', 'incept4e', phase_train=phase_train,
                    use_batch_norm=True)
    endpoints['incept4e'] = net
    net = inception(net, 1024, 1, 384, 192, 384, 48, 128, 3, 128, 1, 'MAX', 'incept5a', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept5a'] = net
    net = inception(net, 1024, 1, 384, 192, 384, 48, 128, 3, 128, 1, 'MAX', 'incept5b', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept5b'] = net
    net = apool(net, 7, 7, 2, 2, 'VALID', 'pool6')
    endpoints['pool6'] = net
    net = tf.reshape(net, [-1, 1024])
    endpoints['prelogits'] = net
    net = tf.nn.dropout(net, 1)
    endpoints['dropout'] = net
    return net, endpoints


# Adience age and gender recognition net
def nn2_forward_propagation(images, phase_train=True, weight_decay=0.0):
    endpoints = {}
    weight_decay = 0.0
    phase_train = True
    net = conv(images, 3, 96, 7, 7, 4, 4, 'SAME', 'conv1_7x7', phase_train=phase_train,
               use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv1'] = net

    net = mpool(net, 3, 3, 2, 2, 'SAME', 'pool1')
    endpoints['pool1'] = net

    net = tf.nn.local_response_normalization(net,5,alpha=0.0001,beta=0.75,name='norm1')
    endpoints['norm1'] = net

    net = conv(net, 96, 256, 5, 5, 1, 1, 'SAME', 'conv2_5x5', phase_train=phase_train, use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv2_1x1'] = net

    net = mpool(net,3,3,2,2,'SAME','pool2')
    endpoints['pool2'] = net


    net = conv(net, 256, 384, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv3_3x3'] = net

    net = mpool(net, 3, 3, 2, 2, 'SAME', 'pool3')
    endpoints['pool3'] = net

    net = tf.reshape(net, [-1, 8*8*384])
    endpoints['flat'] = net

    net = fully_connected(net,512,scope='fc1')
    endpoints['fc1']=net

    net = tf.nn.dropout(net,keep_prob=1,name='drop1')
    endpoints['drop1']=net

    net = fully_connected(net, 512, scope='fc2')
    endpoints['fc2'] = net

    net = tf.nn.dropout(net, keep_prob=1, name='drop2')
    endpoints['drop2'] = net

    return net, endpoints

def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Renset-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net

def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                        tower_conv2_2, tower_pool], 3)
    return net

def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet(images, is_training=phase_train,
                                   dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size,
                                   reuse=reuse)


def inception_resnet(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnet'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnet', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net

                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net

                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net

                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                net = block8(net, activation_fn=None)

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    # pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)

                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')

                    end_points['PreLogitsFlatten'] = net

                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                                           scope='Bottleneck', reuse=False)

    return net, end_points
