import tensorflow as tf
import numpy as np
from PIL import Image
from vgg16 import VGGNet
import os
import config as config
import time

'''
模块
    1 定义输入文件与输出目录
    2 管理模型的超参
    3 数据的提供    (内容图像 风格图像 随机初始化的图像）
    4 构建计算图    (数据流图 定义loss train_op
    5 训练执行过程   (会话中执行 设置cpu gpu
'''
class image_style_transfer(object) :

    def __init__(self):

    # 1.定义输入文件与输出目录
        self.content_image_path = config.content_img_path
        self.style_image_path = config.style_img_path
        self.vgg_path = config.vgg_model_path
        self.outputdir = config.output_dir

    # 2.初始化模型超参数
        self.learning_rate = config.learning_rate
        self.lambda_content_loss = config.lamba_content_loss
        self.lambda_style_loss = config.lamba_style_loss

    # 3. 数据提供(读入图像)
        self.content_img_arr_value = self.read_image(self.content_image_path)
        self.style_img_arr_value = self.read_image(self.style_image_path)
        self.result_img = self.initial_image([1, 224, 224, 3], 127.5, 30)
        self.steps = config.steps

    # 4.构建计算图

        # 获取占位符
        self.content_image = tf.placeholder(tf.float32,[1, 224, 224, 3])
        self.style_image = tf.placeholder(tf.float32, [1, 224, 224, 3])

        #定义三个模型,构建网络
        vgg16_params_dict = np.load(self.vgg_path, encoding='latin1', allow_pickle=True).item()

        self.vgg_16_for_content = VGGNet(vgg16_params_dict)
        self.vgg_16_for_style = VGGNet(vgg16_params_dict)
        self.vgg_16_for_result = VGGNet(vgg16_params_dict)

        self.vgg_16_for_content.build_net(self.content_image)
        self.vgg_16_for_style.build_net(self.style_image)
        self.vgg_16_for_result.build_net(self.result_img)

    def read_image(self, path):
        image = Image.open(path)
        image = image.resize((224, 224))
        np_img = np.array(image)
        np_img = np.asarray([np_img], dtype=np.int32)
        return np_img

    def initial_image(self, shape, mean, stddev):
        initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
        return tf.Variable(initial)
        pass

     # Gram 矩阵 得到相关性的度量
    def gram_matrix(self,x):
        '''
        gram 矩阵计算   k个feature_map  两两之间相关性  相似度的计算  k*k矩阵
        :param x:  shape  [NHWC] [1,height, widths, channles]
        :return:
        '''
        b, w, h, c = x.get_shape().as_list()
        # features  x,shape=[b,h*w,c])

        features = tf.reshape(x, shape=[b, h * w, c])
        # [c,c] = [c,h*w] 矩阵乘法  [h*w,c]
        gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(w * b * c, tf.float32)
        # gram shape : [c,c]  [k,k]
        return gram
        pass

    def losses(self):
        #内容图像的内容特征提取 越低层效果越好
        content_features = [
            self.vgg_16_for_content.conv1_2,
            #self.vgg_16_for_content.conv2_1,
            # vgg_16_for_content_img.conv3_1,
            # vgg_16_for_content_img.conv4_1,
            # vgg_16_for_content_img.conv5_1,
        ]

        #结果图像特征提取
        result_content_features = [
            self.vgg_16_for_result.conv1_2,
            #self.vgg_16_for_result.conv2_1,
            # vgg_16_for_result_img.conv3_1,
            # vgg_16_for_result_img.conv4_1,
            # vgg_16_for_result_img.conv5_1,
        ]

        #风格图像特征提取
        style_features = [
            # vgg_16_for_style_img.conv1_1,
            # vgg_16_for_style_img.conv2_1,
            # vgg_16_for_style_img.conv3_1,
            # vgg_16_for_style_img.conv4_1,
            self.vgg_16_for_style.conv4_3,
            #self.vgg_16_for_style.conv5_3,
        ]

        result_style_features=[
            # vgg_16_for_style_img.conv1_1,
            # vgg_16_for_style_img.conv2_1,
            # vgg_16_for_style_img.conv3_1,
            # vgg_16_for_style_img.conv4_1,
            self.vgg_16_for_result.conv4_3,
            #self.vgg_16_for_style.conv5_3,
        ]


        #内容损失
        content_loss = tf.zeros(1, tf.float32)
        for c, c_result in zip(content_features, result_content_features):
            # c c_result 为四维
            content_loss += tf.reduce_mean(tf.square(c - c_result), axis=[1, 2, 3])

        # 风格损失
        style_gram = [self.gram_matrix(feature) for feature in style_features]
        result_style_gram = [self.gram_matrix(feature) for feature in result_style_features]
        style_loss = tf.zeros(1, tf.float32)

        for s, s_result in zip(style_gram, result_style_gram):
            style_loss += tf.reduce_mean(tf.square(s-s_result),axis=[1,2])

        loss = self.lambda_content_loss * content_loss  +  self.lambda_style_loss * style_loss
        return  content_loss, style_loss , loss
    pass

    def start(self):
        content_loss, style_loss, loss = self.losses()
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        init_op = tf.global_variables_initializer()

        # 输出文件夹不存在则新建
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        with tf.Session() as sess:
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            sess.run(init_op)
            for step in range(self.steps):
                fetches = [loss, content_loss, style_loss, train_op]
                loss_value, content_loss_value, style_loss_value, _    = sess.run(fetches=fetches, feed_dict={
                    self.content_image: self.content_img_arr_value,
                    self.style_image: self.style_img_arr_value
                })
                print('Step: %d ，loss_value：%8.4f  ,content_loss_value:%8.4f  ,style_loss_value:%8.4f' %
                      ((step + 1), loss_value[0], content_loss_value[0], style_loss_value[0]))
                # 每 10 步保存一下
                if (step+1) % 10 == 0:
                    result_img_path = os.path.join("./images/output_imgs", "result-%05d.jpg" %(step+1))
                    print(result_img_path)
                    # result_img  shape: [1,224,224,3]
                    result_img_value = self.result_img.eval(sess)[0]
                    # result_image  shape: [224,224,3]
                    # 图像的像素值必须在0-255之间
                    result_img_value = np.clip(result_img_value, 0, 255)
                    #转换为图像
                    img_arr = np.array(result_img_value, np.uint8)
                    img = Image.fromarray(img_arr)
                    img.save(result_img_path)
        pass
start = time.time()
s = image_style_transfer()
s.start()
print('共计 %d'%(time.time()-start))