from model import Tower
from utils import model_property
import tensorflow as tf
import utils as digits


class UserModel(Tower):

    resnet_settings = {
        'conv2': [64, 3],
        'conv3': [128, 3],
        'conv4': [256, 5],
        'conv5': [512, 2]
    }

    @model_property
    def inference(self):
        # expecting 224x224x3
        model = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        weight, bias = self.create_conv_vars([7, 7, self.input_shape[2], UserModel.resnet_settings['conv2'][0]], 'conv1')
        model = self.conv_layer_with_relu(model, weight, bias, 2)

        model = self.max_pool(model, 3, 2)

        model = self.res_segment(model, UserModel.resnet_settings['conv2'][0], UserModel.resnet_settings['conv2'][1], 'conv2')

        model = self.transition_segment(model, UserModel.resnet_settings['conv2'][0], UserModel.resnet_settings['conv3'][0], 'conv2_to_conv3')
        model = self.res_segment(model, UserModel.resnet_settings['conv3'][0], UserModel.resnet_settings['conv3'][1], 'conv3')

        model = self.transition_segment(model, UserModel.resnet_settings['conv3'][0], UserModel.resnet_settings['conv4'][0], 'conv3_to_conv4')
        model = self.res_segment(model, UserModel.resnet_settings['conv4'][0], UserModel.resnet_settings['conv4'][1], 'conv4')

        model = self.transition_segment(model, UserModel.resnet_settings['conv4'][0], UserModel.resnet_settings['conv5'][0], 'conv4_to_conv5')
        model = self.res_segment(model, UserModel.resnet_settings['conv5'][0], UserModel.resnet_settings['conv5'][1], 'conv5')

        model = self.avg_pool(model, 7, 1, "VALID")
        
        fc_weight, fc_bias = self.create_fc_vars([512, self.nclasses], 'fc')
        model = self.fully_connect(model, fc_weight, fc_bias)

        return model
        
    @model_property
    def loss(self):
        model = self.inference
        loss = digits.classification_loss(model, self.y)
        accuracy = digits.classification_accuracy(model, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss

    def res_segment(self, model, layers, stacks, name):
        model
        for i in range(0, stacks):
            model = self.res_block(model, layers, name + "_" + str(i))
            
        return model
        
    def res_block(self, model, layers, name):
        weight_1, bias_1 = self.create_conv_vars([3, 3, layers, layers], name + "_1")
        new_model = self.conv_layer(model, weight_1, bias_1, 1)
        new_model = self.batch_norm(new_model, layers, name + "_1")
        new_model = self.activation_func(new_model)

        weight_2, bias_2 = self.create_conv_vars([3, 3, layers, layers], name + "_2")
        new_model = self.conv_layer(new_model, weight_2, bias_2, 1)
        new_model = self.batch_norm(new_model, layers, name + "_2")
        new_model = tf.add(model, new_model)
        new_model = self.activation_func(new_model)


        return new_model

    def transition_segment(self, model, in_layers, out_layers, name):
        shrunk_model = self.shrink_model(model, in_layers, out_layers, 2, name)
        
        weight_1, bias_1 = self.create_conv_vars([3, 3, in_layers, out_layers], name + "_transition")
        new_model = self.conv_layer(model, weight_1, bias_1, 2)
        new_model = self.batch_norm(new_model, out_layers, name + "_transition")
        new_model = self.activation_func(new_model)

        weight_2, bias_2 = self.create_conv_vars([3, 3, out_layers, out_layers,], name + "_after_conv")
        new_model = self.conv_layer(new_model, weight_2, bias_2, 1)
        new_model = self.batch_norm(new_model, out_layers, name + "_after_conv")
        new_model = tf.add(shrunk_model, new_model)
        new_model = self.activation_func(new_model)

        return new_model

    def shrink_model(self, model, in_layers, out_layers, strides, name):
        weight_1, bias_1 = self.create_conv_vars([1, 1, in_layers, out_layers], name + "_shrink")
        shrunk_model = self.conv_layer_with_relu(model, weight_1, bias_1, 2)

        return shrunk_model
    
    def conv_layer(self, model, weights, biases, stride_size, padding='SAME'):
        new_model = tf.nn.conv2d(model, weights, strides=[1, stride_size, stride_size, 1], padding=padding)
        new_model = tf.nn.bias_add(new_model, biases)
        return new_model

    def conv_layer_with_relu(self, model, weights, biases, stride_size, padding='SAME'):
        new_model = self.conv_layer(model, weights, biases, stride_size, padding)
        new_model = self.activation_func(new_model)
        return new_model

    def deconv_layer_with_relu(self, model, weights, biases, output, stride_size, padding='SAME'):
        new_model = tf.nn.conv2d_transpose(model, weights, output, strides=[1, stride_size, stride_size, 1], padding=padding)
        new_model = tf.nn.bias_add(new_model, biases)
        new_model = tf.nn.relu(new_model)
        return new_model
        
    def activation_func(self, model, alpha=0.001):
        new_model = tf.nn.relu(model)
        return new_model

    def max_pool(self, model, kernal_size, stride_size, padding='SAME'):
        new_model = tf.nn.max_pool(model, ksize=[1, kernal_size, kernal_size, 1],
                                   strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def avg_pool(self, model, kernal_size, stride_size, padding='SAME'):
        new_model = tf.nn.avg_pool(model, ksize=[1, kernal_size, kernal_size, 1],
                                   strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def fully_connect(self, model, weights, biases):
        fc_model = tf.reshape(model, [-1, weights.get_shape().as_list()[0]])
        fc_model = tf.matmul(fc_model, weights)
        fc_model = tf.add(fc_model, biases)
        fc_model = tf.nn.relu(fc_model)
        return fc_model

    def batch_norm(self, model, channels, name):
        mean, variance = tf.nn.moments(model, [0, 1, 2], keep_dims=True)
        beta = self.create_var(channels, name + '_beta', tf.constant_initializer(0.0, tf.float32))
        gamma = self.create_var(channels, name + '_gamma', tf.constant_initializer(1.0, tf.float32))
        normed = tf.nn.batch_normalization(model, mean, variance, beta, gamma, 0.001)
        return normed

    def create_conv_vars(self, size, name):
        weight = self.create_weight(size, name + '_W')
        bias = self.create_bias(size[3], name + '_b')
        return weight, bias

    def create_deconv_vars(self, size, name):
        weight = self.create_weight(size, name + '_W')
        bias = self.create_bias(size[2], name + '_b')  # because deconv filter is [h, w, out, in]
        return weight, bias

    def create_fc_vars(self, size, name):
        weight = self.create_weight(size, name + '_W')
        bias = self.create_bias(size[1], name + '_b')
        return weight, bias

    def create_weight(self, size, name):
        weight = tf.get_variable(name, size, initializer=tf.contrib.layers.xavier_initializer())
        return weight

    def create_bias(self, size, name):
        bias = tf.get_variable(name, [size], initializer=tf.constant_initializer(0.2))
        return bias

    def create_var(self, size, name, init):
        weight = tf.get_variable(name, size, initializer=init)
        return weight