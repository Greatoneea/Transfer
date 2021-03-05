





# VGG16的模型存放路径
vgg_model_path = "moudel/vgg16.npy"

# 内容图片
content_img_path = "images/dog.jpg"

# 风格图片
style_img_path = "images/img.png"

# 输出路径
output_dir = 'images/output_imgs'


# 训练配置
steps = 250

learning_rate = 10

lamba_content_loss = 1

# lambda_s= 500
lamba_style_loss= 20