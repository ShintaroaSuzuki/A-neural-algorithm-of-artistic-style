import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import time

VGG_MODEL = "models/imagenet-vgg-verydeep-16.mat"


CONTENT_IMG = 'kyoto1.jpg'  # コンテンツ画像
STYLE_IMG = 'rembrandt.jpg'  # スタイル画像

if os.path.splitext(CONTENT_IMG)[-1] == '.png':
    input_im = Image.open(CONTENT_IMG)
    rgb_im = input_im.convert('RGB')
    rgb_im.save(os.path.splitext(CONTENT_IMG)[0] + ".jpg",quality=100)
    CONTENT_IMG = os.path.splitext(CONTENT_IMG)[0] + ".jpg"

if os.path.splitext(STYLE_IMG)[-1] == '.png':
    input_im = Image.open(STYLE_IMG)
    rgb_im = input_im.convert('RGB')
    rgb_im.save(os.path.splitext(STYLE_IMG)[0] + ".jpg",quality=100)
    STYLE_IMG = os.path.splitext(STYLE_IMG)[0] + ".jpg"

OUTPUT_DIR = 'results'  # 生成画像ディレクトリ
OUTPUT_IMG = os.path.splitext(os.path.basename(CONTENT_IMG))[0] + '_by_' + os.path.splitext(os.path.basename(STYLE_IMG))[0] + '_vgg16_2.png'  # 生成画像ファイル


IMAGE_W = 750
IMAGE_H = 500

#  入力画像から平均画素値を引くための定数(reshapeでそのまま引けるようにする)
MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))

history = {
    'contents_loss': [],
    'style_loss': [],
    'total_loss': []
}

def build_net(ntype, nin, rwb=None):
    """
    ネットワークの各層をTensorFlowで定義する関数
    : param ntype: ネットワークの層のタイプ(ここでは、畳み込み層もしくは、プーリング層)
    : param nin: 前の層
    : param rwb: VGGの最適化された値
    """
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, rwb[0], strides=[1, 1, 1, 1], padding='SAME') + rwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i):
    """
    VGGの各層の最適化された重みとバイアスを取得する関数
    : param vgg_layers: ネットワークの層
    : param i:
    """
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias

def build_vgg16(path):
    """
    TensorFlowでVGGネットワークを構成する関数
    : param path: VGGの学習済みモデルのファイルのパス
    """
    net = {}
    vgg_rawnet = scipy.io.loadmat(path)
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
    net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
    net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
    net['pool1']   = build_net('pool',net['conv1_2'])
    net['conv2_1'] = build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5))
    net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
    net['pool2']   = build_net('pool',net['conv2_2'])
    net['conv3_1'] = build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10))
    net['conv3_2'] = build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12))
    net['conv3_3'] = build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14))
    net['pool3']   = build_net('pool',net['conv3_3'])
    net['conv4_1'] = build_net('conv',net['pool3'],get_weight_bias(vgg_layers,17))
    net['conv4_2'] = build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,19))
    net['conv4_3'] = build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,21))
    net['pool4']   = build_net('pool',net['conv4_3'])
    net['conv5_1'] = build_net('conv',net['pool4'],get_weight_bias(vgg_layers,24))
    net['conv5_2'] = build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,26))
    net['conv5_3'] = build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,28))
    net['pool5']   = build_net('pool',net['conv5_3'])
    return net

def build_content_loss(p, x):
    """
    コンテンツと出力の誤差
    """
    M = p.shape[1]*p.shape[2]
    N = p.shape[3]
    loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((x - p),2))
    return loss

def gram_matrix(x, area, depth):
    """
    個々のフィルタ出力の相関をグラム行列で表現
    """
    x1 = tf.reshape(x,(area,depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g

def gram_matrix_val(x, area, depth):
    """
    スタイル自体もグラム行列で表現
    """
    x1 = x.reshape(area,depth)
    g = np.dot(x1.T, x1)
    return g

def build_style_loss(a, x):
    """
    スタイルと出力の誤差
    """
    M = a.shape[1]*a.shape[2]
    N = a.shape[3]
    A = gram_matrix_val(a, M, N )
    G = gram_matrix(x, M, N )
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
    return loss

def read_image(path):
    """
    画像を読み込む関数
    """
    image = imageio.imread(path)
    image = scipy.misc.imresize(image,(IMAGE_H,IMAGE_W))
    image = image[np.newaxis,:,:,:]
    image = image - MEAN_VALUES
    return image

def write_image(path, image):
    """
    生成された画像を保存する関数
    """
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    imageio.imsave(path, image)


# VGG16モデルの作成
net = build_vgg16(VGG_MODEL)
# ホワイトノイズ
noise_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
# 画像の読み込み
content_img = read_image(CONTENT_IMG)
style_img = read_image(STYLE_IMG)


# 初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

"""
画風変換の出力を調節するにはここを変更
"""
# 各種パラメータの設定
INI_NOISE_RATIO = 1. # ホワイトノイズの重み
STYLE_STRENGTH = 10 # スタイルの強さ
ITERATION = 5000 # 最適化回数

# コンテンツ画像と出力画像で誤差を取る層
CONTENT_LAYERS =[('conv4_2', 1.)]
# スタイル画像と出力画像で誤差を取る層
STYLE_LAYERS=[('conv1_1', .2), ('conv2_1', .2), ('conv3_1', .2), ('conv4_1', .2), ('conv5_1', .2)]

sess.run([net['input'].assign(content_img)])
cost_content = sum(map(lambda l,: l[1]*build_content_loss(sess.run(net[l[0]]) ,  net[l[0]]), CONTENT_LAYERS))

sess.run([net['input'].assign(style_img)])
cost_style = sum(map(lambda l: l[1]*build_style_loss(sess.run(net[l[0]]) ,  net[l[0]]), STYLE_LAYERS))

cost_total = cost_content + STYLE_STRENGTH * cost_style
optimizer = tf.train.AdamOptimizer(4.0)


train = optimizer.minimize(cost_total)
sess.run( tf.global_variables_initializer())
sess.run(net['input'].assign( INI_NOISE_RATIO* noise_img + (1.-INI_NOISE_RATIO) * content_img))

# 保存先ディレクトリが存在しないときは作成
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

start = time.time()

for i in range(ITERATION):
    sess.run(train)
    # 100回ごとに経過を表示、画像を保存
    if i%100 ==0:
        result_img = sess.run(net['input'])
        history['contents_loss'].append(sess.run(cost_content))
        history['style_loss'].append(sess.run(cost_style))
        history['total_loss'].append(sess.run(cost_total))
        print ("ITERATION: ",i,", ",sess.run(cost_total))
        write_image(os.path.join(OUTPUT_DIR,'%s.png'%(str(i).zfill(4))),result_img)
else:
    history['contents_loss'].append(sess.run(cost_content))
    history['style_loss'].append(sess.run(cost_style))
    history['total_loss'].append(sess.run(cost_total))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print(history['contents_loss'][-1])
print(history['style_loss'][-1])
print(history['total_loss'][-1])

write_image(os.path.join(OUTPUT_DIR,OUTPUT_IMG),result_img)

sns.set()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

if ITERATION == 5000:
    x = np.arange(0, 51)
else:
    x = np.arange(0, 101)

ax.plot(x, history['contents_loss'], linestyle='--', color='b', label='contents loss')
ax.plot(x, history['style_loss'], linestyle='--', color='#e46409', label='style loss')
ax.plot(x, history['total_loss'], linestyle='-', color='g', label='total loss')

# x axis
if ITERATION == 5000:
    plt.xlim([-1, 51])
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(['0', '1000', '2000', '3000', '4000', '5000'])
    ax.set_xlabel('epochs')
else:
    plt.xlim([-1, 101])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(['0', '2000', '4000', '6000', '8000', '10000'])
    ax.set_xlabel('epochs')

# y axis
plt.ylim([0, 10000000])
# ax.set_yticks()
ax.set_ylabel('losses')

# legend and title
ax.legend(loc='best')
ax.set_title('contents loss & style loss & total loss')

# save as png
plt.savefig(os.path.splitext(os.path.basename(CONTENT_IMG))[0] + '_by_' + os.path.splitext(os.path.basename(STYLE_IMG))[0] + '_vgg16_2-loss.png')

"""
# gif作成
gif_list = [Image.open('results/%s.png'%(str(i*500).zfill(4))) for i in range(10)]
gif_list.append(Image.open('results/result.png'))
gif_list[0].save('results/result.gif', save_all=True, append_images=gif_list[1:], duration=400, loop=0)
"""
