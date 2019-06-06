# 导入需要的包
import os  # 读取路径下文件
import shutil  # 递归删除文件
import tensorflow as tf  # 编写神经网络
import numpy as np  # 矩阵运算操作
from skimage.io import imsave  # 保存影像

# 图像的size为(28, 28, 1)
image_height = 28
image_width = 28
image_size = image_height * image_width

flags = tf.app.flags

flags.DEFINE_string("data", "mnist", "mnist or ...")
flags.DEFINE_string("data_dir", "./data", 'data dir')
flags.DEFINE_boolean("train", True, "train or test")
flags.DEFINE_boolean("restore", False, "whether to store model")
flags.DEFINE_string("output_path", "./output/", "the path to save output")
flags.DEFINE_integer("max_epoch", 500, "# of epoch in an model")
flags.DEFINE_integer("batch_size", 256, "# of step in an epoch")
flags.DEFINE_integer("h1_size", 256, "hidden size 1")
flags.DEFINE_integer("h2_size", 512, "hidden size 2")
flags.DEFINE_integer("z_size", 128, "generator input parameter")

conf = flags.FLAGS

# 主函数
def main(_):
	if conf.train:
		train()

# 定义GAN的生成器
def generator(z_prior):
	'''
	函数功能：生成影像，参与训练过程
	输入：z_prior,       #输入tf格式，size为（batch_size, z_size）的数据
	输出：x_generate,    #生成图像
		 g_params,      #生成图像的所有参数
	'''
	# 第一个链接层
	# 以2倍标准差stddev的截断的正态分布中生成大小为[z_size, h1_size]的随机值，权值weight初始化。
	w1 = tf.Variable(tf.truncated_normal([conf.z_size, conf.h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
	# 生成大小为[h1_size]的0值矩阵，偏置bias初始化
	b1 = tf.Variable(tf.zeros([conf.h1_size]), name="g_b1", dtype=tf.float32)
	# 通过矩阵运算，将输入z_prior传入隐含层h1。激活函数为relu
	h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)

	# 第二个链接层
	# 以2倍标准差stddev的截断的正态分布中生成大小为[h1_size, h2_size]的随机值，权值weight初始化。
	w2 = tf.Variable(tf.truncated_normal([conf.h1_size, conf.h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
	# 生成大小为[h2_size]的0值矩阵，偏置bias初始化
	b2 = tf.Variable(tf.zeros([conf.h2_size]), name="g_b2", dtype=tf.float32)
	# 通过矩阵运算，将h1传入隐含层h2。激活函数为relu
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

	# 第三个链接层
	# 以2倍标准差stddev的截断的正态分布中生成大小为[h2_size, image_size]的随机值，权值weight初始化。
	w3 = tf.Variable(tf.truncated_normal([conf.h2_size, image_size], stddev=0.1), name="g_w3", dtype=tf.float32)
	# 生成大小为[image_size]的0值矩阵，偏置bias初始化
	b3 = tf.Variable(tf.zeros([image_size]), name="g_b3", dtype=tf.float32)
	# 通过矩阵运算，将h2传入隐含层h3。
	h3 = tf.matmul(h2, w3) + b3
	# 利用tanh激活函数，将h3传入输出层
	x_generate = tf.nn.tanh(h3)

	# 将所有参数合并到一起
	g_params = [w1, b1, w2, b2, w3, b3]

	return x_generate, g_params


# 定义GAN的判别器
def discriminator(x_data, x_generated, keep_prob):
	'''
	函数功能：对输入数据进行判断，并保存其参数
	输入：x_data,        #输入的真实数据 
		x_generated,     #生成器生成的虚假数据
		keep_prob，      #dropout率，防止过拟合
	输出：y_data,        #判别器对batch个数据的处理结果
		y_generated,     #判别器对余下数据的处理结果
		d_params，       #判别器的参数
	'''

	# 合并输入数据，包括真实数据x_data和生成器生成的假数据x_generated
	x_in = tf.concat([x_data, x_generated], 0)

	# 第一个链接层
	# 以2倍标准差stddev的截断的正态分布中生成大小为[image_size, h2_size]的随机值，权值weight初始化。
	w1 = tf.Variable(tf.truncated_normal([image_size, conf.h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
	# 生成大小为[h2_size]的0值矩阵，偏置bias初始化
	b1 = tf.Variable(tf.zeros([conf.h2_size]), name="d_b1", dtype=tf.float32)
	# 通过矩阵运算，将输入x_in传入隐含层h1.同时以一定的dropout率舍弃节点，防止过拟合
	h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)

	# 第二个链接层
	# 以2倍标准差stddev的截断的正态分布中生成大小为[h2_size, h1_size]的随机值，权值weight初始化。
	w2 = tf.Variable(tf.truncated_normal([conf.h2_size, conf.h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
	# 生成大小为[h1_size]的0值矩阵，偏置bias初始化
	b2 = tf.Variable(tf.zeros([conf.h1_size]), name="d_b2", dtype=tf.float32)
	# 通过矩阵运算，将h1传入隐含层h2.同时以一定的dropout率舍弃节点，防止过拟合
	h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)

	# 第三个链接层
	# 以2倍标准差stddev的截断的正态分布中生成大小为[h1_size, 1]的随机值，权值weight初始化。
	w3 = tf.Variable(tf.truncated_normal([conf.h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
	# 生成0值，偏置bias初始化
	b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
	# 通过矩阵运算，将h2传入隐含层h3
	h3 = tf.matmul(h2, w3) + b3

	# 从h3中切出batch_size张图像
	y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [conf.batch_size, -1], name=None))
	# 从h3中切除余下的图像
	y_generated = tf.nn.sigmoid(tf.slice(h3, [conf.batch_size, 0], [-1, -1], name=None))

	# 判别器的所有参数
	d_params = [w1, b1, w2, b2, w3, b3]

	return y_data, y_generated, d_params


# 显示结果的函数
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
	'''
	函数功能：输入相关参数，将运行结果以图片的形式保存到当前路径下
	输入：batch_res,       #输入数据
		fname,             #输入路径
		grid_size=(8, 8),  #默认输出图像为8*8张
		grid_pad=5，       #默认图像的边缘留白为5像素
	输出：无
	'''

	# 将batch_res进行值[0, 1]归一化，同时将其reshape成（batch_size, image_height, image_width）
	batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], image_height, image_width)) + 0.5
	# 重构显示图像格网的参数
	img_h, img_w = batch_res.shape[1], batch_res.shape[2]
	grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
	grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
	img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
	for i, res in enumerate(batch_res):
		if i >= grid_size[0] * grid_size[1]:
			break
		img = (res) * 255.
		img = img.astype(np.uint8)
		row = (i // grid_size[0]) * (img_h + grid_pad)
		col = (i % grid_size[1]) * (img_w + grid_pad)
		img_grid[row:row + img_h, col:col + img_w] = img
	# 保存图像
	imsave(fname, img_grid)


# 定义训练过程
def train():
	'''
	函数功能：训练整个GAN网络，并随机生成手写数字
	输入：无
	输出：sess.saver()
	'''

	# 加载数据
	DATA_DIR = os.path.join(conf.data_dir, conf.data)
	if conf.data == 'mnist':
		from tensorflow.examples.tutorials.mnist import input_data
		train_data = input_data.read_data_sets(DATA_DIR).train.images
	size = train_data.shape[0]

	# 构建模型---------------------------------------------------------------------
	# 定义GAN网络的输入，其中x_data为[batch_size, image_size], z_prior为[batch_size, z_size]
	x_data = tf.placeholder(tf.float32, [conf.batch_size, image_size], name="x_data")  # (batch_size, image_size)
	z_prior = tf.placeholder(tf.float32, [conf.batch_size, conf.z_size], name="z_prior")  # (batch_size, z_size)
	# 定义dropout率
	keep_prob = tf.placeholder(tf.float32, name="keep_prob")
	global_step = tf.Variable(0, name="global_step", trainable=False)

	# 利用生成器生成数据x_generated和参数g_params
	x_generated, g_params = generator(z_prior)
	# 利用判别器判别生成器的结果
	y_data, y_generated, d_params = discriminator(x_data, x_generated, keep_prob)

	# 定义判别器和生成器的loss函数
	d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
	g_loss = - tf.log(y_generated)

	# 设置学习率为0.0001，用AdamOptimizer进行优化
	optimizer = tf.train.AdamOptimizer(0.0001)

	# 判别器discriminator 和生成器 generator 对损失函数进行最小化处理
	d_trainer = optimizer.minimize(d_loss, var_list=d_params)
	g_trainer = optimizer.minimize(g_loss, var_list=g_params)
	# 模型构建完毕--------------------------------------------------------------------

	# 全局变量初始化
	init = tf.global_variables_initializer()

	# 启动会话sess
	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(init)

	# 判断是否需要存储
	if conf.restore:
		# 若是，将最近一次的checkpoint点存到outpath下
		chkpt_fname = tf.train.latest_checkpoint(conf.output_path)
		saver.restore(sess, chkpt_fname)
	else:
		# 若否，判断目录是存在，如果目录存在，则递归的删除目录下的所有内容，并重新建立目录
		if os.path.exists(conf.output_path):
			shutil.rmtree(conf.output_path)
		os.mkdir(conf.output_path)

	# 利用随机正态分布产生噪声影像，尺寸为(batch_size, z_size)
	z_sample_val = np.random.normal(0, 1, size=(conf.batch_size, conf.z_size)).astype(np.float32)

	# 逐个epoch内训练
	for i in range(sess.run(global_step), conf.max_epoch):
		# 图像每个epoch内可以放(size // batch_size)个size
		for j in range(size // conf.batch_size):
			if j % 20 == 0:
				print("epoch:%s, iter:%s" % (i, j))

			# 训练一个batch的数据
			batch_end = j * conf.batch_size + conf.batch_size
			if batch_end >= size:
				batch_end = size - 1
			x_value = train_data[j * conf.batch_size: batch_end]
			# 将数据归一化到[-1, 1]
			# x_value = x_value / 255.
			x_value = 2 * x_value - 1

			# 以正太分布的形式产生随机噪声
			z_value = np.random.normal(0, 1, size=(conf.batch_size, conf.z_size)).astype(np.float32)
			# 每个batch下，输入数据运行GAN，训练判别器
			sess.run(d_trainer,
					 feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
			# 每个batch下，输入数据运行GAN，训练生成器
			if j % 1 == 0:
				sess.run(g_trainer,
						 feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
		# 每一个epoch中的所有batch训练完后，利用z_sample测试训练后的生成器
		x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
		# 每一个epoch中的所有batch训练完后，显示生成器的结果，并打印生成结果的值
		show_result(x_gen_val, os.path.join(conf.output_path, "sample%s.jpg" % i))
		print(x_gen_val)
		# 每一个epoch中，生成随机分布以重置z_random_sample_val
		z_random_sample_val = np.random.normal(0, 1, size=(conf.batch_size, conf.z_size)).astype(np.float32)
		# 每一个epoch中，利用z_random_sample_val生成手写数字图像，并显示结果
		x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
		show_result(x_gen_val, os.path.join(conf.output_path, "random_sample%s.jpg" % i))
		# 保存会话
		sess.run(tf.assign(global_step, i + 1))
		saver.save(sess, os.path.join(conf.output_path, "model"), global_step=global_step)

if __name__ == '__main__':
	tf.app.run()