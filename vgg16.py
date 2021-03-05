import tensorflow as tf
import time

#图像预处理 参数写死  图像归一化处理  官方给的
VGG_MEAN = [103.939,116.779,123.68]

class VGGNet(object):

    #初始化
    def __init__(self,params_dict):
        self.params_dict = params_dict

    #得到卷积层某层参数 w
    def get_conv_kernel(self,name):
        return tf.constant(self.params_dict[name][0])

    # 得到全连接某层参数 b
    def get_fc_weight(self, name):
        return tf.constant(self.params_dict[name][0])

    # 得到偏置项
    def get_bias(self,name):
        return tf.constant(self.params_dict[name][1])


    # 计算卷积层
    def conv_layer(self, inputs, name):
        # 名字作用域 防止冲突
        with tf.name_scope(name):
            conv_w = self.get_conv_kernel(name)
            conv_b = self.get_bias(name)

            # 调用api  卷积计算
            conv_result = tf.nn.conv2d(input=inputs, filters=conv_w, strides=[1, 1, 1, 1], padding='SAME',data_format='NHWC', name=name)

            #加偏置项
            conv_result = tf.nn.bias_add(conv_result, conv_b)
            #relu函数激活
            conv_result = tf.nn.relu(conv_result)

            return conv_result

    def pooling_layer(self,inputs, name):
        return tf.nn.max_pool(inputs,[1,2,2,1],[1,2,2,1],padding='SAME',name=name)

    def build_net(self,input_rgb):
        '''
        :param input_rgb:
        :return:
        '''
        start_time = time.time()
        print('building start...')
        start = time.time()

        # 分离  （输入，分离个数 ，维度）
        r, g, b = tf.split(input_rgb,[1, 1, 1], axis=3)
        x_bgr = tf.concat(values=[
            b - VGG_MEAN[0],
            g - VGG_MEAN[1],
            r - VGG_MEAN[2],
        ], axis=3)
        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # 构建
        # stage 1
        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')

        # stage 2
        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        # stage 3
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        # stage 4
        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        # stage 5
        self.conv5_1 = self.conv_layer(self.pool4, 'conv4_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv4_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv4_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        print("构建完成，共花费:%4ds" %(time.time() - start))