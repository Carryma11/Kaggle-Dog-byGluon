import numpy as np
from mxnet import image, nd
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data import vision

ctx = mx.gpu()
data_dir = './data'
train_dir = 'train'
test_dir = 'test'
valid_dir = 'valid'
input_dir = 'train_valid_test'
train_valid_dir = 'train_valid'
input_str = data_dir + '/' + input_dir + '/'
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


def evaluate_loss(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss / steps, acc / steps


def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation='relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(120))

    net.initialize(ctx=ctx)
    return net


def transform_train_224(data, label):
    im1 = image.imresize(data.astype('float32') / 255, 224, 224)
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224), resize=0,
                                     rand_crop=False, rand_resize=False, rand_mirror=True,
                                     mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]),
                                     brightness=0, contrast=0,
                                     saturation=0, hue=0,
                                     pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist1:
        im1 = aug(im1)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im1 = nd.transpose(im1, (2, 0, 1))
    return im1, nd.array([label]).asscalar().astype('float32')


def transform_train_299(data, label):
    im2 = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299), resize=0,
                                     rand_crop=False, rand_resize=False, rand_mirror=True,
                                     mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]),
                                     brightness=0, contrast=0,
                                     saturation=0, hue=0,
                                     pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist2:
        im2 = aug(im2)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im2 = nd.transpose(im2, (2, 0, 1))
    return im2, nd.array([label]).asscalar().astype('float32')


def transform_test_224(data, label):
    im1 = image.imresize(data.astype('float32') / 255, 224, 224)
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224),
                                     mean=np.array([0.485, 0.456, 0.406]),
                                     std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist1:
        im1 = aug(im1)

    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im1 = nd.transpose(im1, (2, 0, 1))
    return im1, nd.array([label]).asscalar().astype('float32')


def transform_test_299(data, label):
    im2 = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299),
                                     mean=np.array([0.485, 0.456, 0.406]),
                                     std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist2:
        im2 = aug(im2)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im2 = nd.transpose(im2, (2, 0, 1))
    return im2, nd.array([label]).asscalar().astype('float32')


def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()


train_ds_224 = vision.ImageFolderDataset(input_str + train_dir, flag=1,
                                         transform=transform_train_224)
train_ds_299 = vision.ImageFolderDataset(input_str + train_dir, flag=1,
                                         transform=transform_train_299)
valid_ds_224 = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                         transform=transform_test_224)
valid_ds_299 = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                         transform=transform_test_299)
train_valid_ds_224 = vision.ImageFolderDataset(input_str + train_valid_dir,
                                               flag=1, transform=transform_train_224)
train_valid_ds_299 = vision.ImageFolderDataset(input_str + train_valid_dir,
                                               flag=1, transform=transform_train_299)
batch_size = 32

loader = gluon.data.DataLoader
train_iter_224 = loader(train_ds_224, batch_size, shuffle=True, last_batch='keep')
train_iter_299 = loader(train_ds_299, batch_size, shuffle=True, last_batch='keep')
valid_iter_224 = loader(valid_ds_224, batch_size, shuffle=True, last_batch='keep')
valid_iter_299 = loader(valid_ds_299, batch_size, shuffle=True, last_batch='keep')
train_valid_iter_224 = loader(train_valid_ds_224, batch_size, shuffle=True,
                              last_batch='keep')
train_valid_iter_299 = loader(train_valid_ds_299, batch_size, shuffle=True,
                              last_batch='keep')
