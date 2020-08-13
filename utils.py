from mxnet import image, nd,init
from mxnet import gluon
from mxnet.gluon import nn, model_zoo
import numpy as np
import mxnet as mx


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

class  ConcatNet(nn.HybridBlock):
    def __init__(self,net1,net2,**kwargs):
        super(ConcatNet,self).__init__(**kwargs)
        self.net1 = nn.HybridSequential()
        self.net1.add(net1)
        self.net1.add(nn.GlobalAvgPool2D())
        self.net2 = nn.HybridSequential()
        self.net2.add(net2)
        self.net2.add(nn.GlobalAvgPool2D())
    def hybrid_forward(self,F,x1,x2):
        return F.concat(*[self.net1(x1),self.net2(x2)])


class  OneNet(nn.HybridBlock):
    def __init__(self,features,output,**kwargs):
        super(OneNet,self).__init__(**kwargs)
        self.features = features
        self.output = output
    def hybrid_forward(self,F,x1,x2):
        return self.output(self.features(x1,x2))


class Net():
    def __init__(self,ctx,nameparams=None):
        inception = model_zoo.vision.inception_v3(pretrained=True,ctx=ctx).features
        resnet = model_zoo.vision.resnet152_v1(pretrained=True,ctx=ctx).features
        self.features = ConcatNet(resnet,inception)
        self.output = self.__get_output(ctx,nameparams)
        self.net = OneNet(self.features,self.output)
    def __get_output(self,ctx,ParamsName=None):
        net = nn.HybridSequential("output")
        with net.name_scope():
            net.add(nn.Dense(256,activation='relu'))
            net.add(nn.Dropout(.5))
            net.add(nn.Dense(120))
        if ParamsName is not None:
            net.collect_params().load(ParamsName,ctx)
        else:
            net.initialize(init = init.Xavier(),ctx=ctx)
        return net

