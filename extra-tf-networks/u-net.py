from model import Tower
from utils import model_property
import tensorflow as tf
import utils as digits


class UserModel(Tower):

    # [input_layers, output_layers]
    unet_conv_settings = {
        'left_1': [1, 64],
        'left_2': [64, 128],
        'left_3': [128, 256],
        'left_4': [256, 512],
        'middle': [512, 1024],
        'right_4': [1024, 512],
        'right_3': [512, 256],
        'right_2': [256, 128],
        'right_1': [128, 64]
    }

    # [image_dimension, input_layers, output_layers]
    unet_right_side_settings = {
        'middle_to_right_4': [56, 1024, 512],
        'right_4_to_right_3': [104, 512, 256],
        'right_3_to_right_2': [200, 256, 128],
        'right_2_to_right_1': [392, 128, 64]
    }

    @model_property
    def inference(self):
        # We expect 572 x 572 x 1
        model = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        left_1 = self.segment_operation(model, UserModel.unet_conv_settings['left_1'], 'left_1')

        left_2_before = self.max_pool(left_1, 2, 2)
        left_2 = self.segment_operation(model, UserModel.unet_conv_settings['left_2'], 'left_2')

        left_3_before = self.max_pool(left_2, 2, 2)
        left_3 = self.segment_operation(model, UserModel.unet_conv_settings['left_3'], 'left_3')

        left_4_before = self.max_pool(left_3, 2, 2)
        left_4 = self.segment_operation(model, UserModel.unet_conv_settings['left_4'], 'left_4')

        middle_before = self.max_pool(left_4, 2, 2)
        middle = self.segment_operation(model, UserModel.unet_conv_settings['middle'], 'middle')

        right_4_before = self.create_right_before(middle, left_4, UserModel.unet_right_side_settings['middle_to_right_4'], 'middle_to_right_4')
        right_4 = self.segment_operation(model, UserModel.unet_conv_settings['right_4'], 'right_4')

        right_3_before = self.create_right_before(right_4, left_3, UserModel.unet_right_side_settings['right_4_to_right_3'], 'right_4_to_right_3')
        right_3 = self.segment_operation(model, UserModel.unet_conv_settings['right_3'], 'right_3')

        right_2_before = self.create_right_before(right_3, left_2, UserModel.unet_right_side_settings['right_3_to_right_2'], 'right_3_to_right_2')
        right_2 = self.segment_operation(model, UserModel.unet_conv_settings['right_2'], 'right_2')

        right_1_before = self.create_right_before(right_2, left_1, UserModel.unet_right_side_settings['right_3_to_right_1'], 'right_3_to_right_1')
        right_1 = self.segment_operation(model, UserModel.unet_conv_settings['right_1'], 'right_1')

        output_weights, output_biases = self.create_conv_vars([1, 1, UserModel.unet_conv_settings['right_1'][1], 2], 'output_conv')
        output = self.conv_layer_with_relu(right_1, output_weights, output_biases, 1)

        return output
        
    @model_property
    def loss(self):


    def segment_operation(self, model, settings, segment_name):
        weights, biases = create_segment_weights(settings, segment_name)
        
        first_conv = self.conv_layer_with_relu(model, weights['first_layer'], biases['first_layer'], 1, 'VALID')
        second_conv = self.conv_layer_with_relu(model, weights['second_layer'], biases['second_layer'], 1, 'VALID')

        return second_conv

    def create_right_before(self, model, left_model, settings, name):
        weights, biases = create_deconv_vars([2, 2, settings[1], settings[2]], name)
        model_before = self.deconv_layer_with_relu(model, weights, biases, [-1, settings[0], settings[0], settings[2]], 2, 'VALID')
        model_before = tf.concat([left_model, model_before], 3)
        
        return model_before
        
    def create_segment_weights(self, settings, segment_name):
        first_layer_w, first_layer_b = self.create_conv_vars([3, 3, settings[0], settings[1]], segment_name + '_conv_layer_1')
        second_layer_w, second_layer_b = self.create_conv_vars([3, 3, settings[1], settings[2]], segment_name + '_conv_layer_2')

        weights = {
            'first_layer': first_layer_w,
            'second_layer': second_layer_w
        }

        biases = {
            'first_layer': first_layer_b,
            'second_layer': second_layer_b
        }

        return weights, biases

    def conv_layer_with_relu(self, model, weights, biases, stride_size, padding='SAME'):
        new_model = tf.nn.conv2d(model, weights, strides=[1, stride_size, stride_size, 1], padding=padding)
        new_model = tf.nn.bias_add(new_model, biases)
        new_model = tf.nn.relu(new_model)
        return new_model

    def deconv_layer_with_relu(self, model, weights, biases, output, stride_size, padding='SAME'):
        new_model = tf.nn.conv2d_transpose(model, weights, output, strides=[1, stride_size, stride_size, 1], padding=padding)
        new_model = tf.nn.bias_add(new_model, biases)
        new_model = tf.nn.relu(new_model)
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
