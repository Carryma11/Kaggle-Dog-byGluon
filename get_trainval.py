from mxnet.gluon.data import vision,DataLoader
from utils import transform_train_224,transform_train_299
from mxnet import nd
import tqdm
import mxnet as mx
import os

input_dir = 'train_valid_test'
train_valid_dir = 'train_valid'
data_dir = './data'
input_str = data_dir + '/' + input_dir + '/'

train_valid_ds_224 = vision.ImageFolderDataset(input_str + train_valid_dir,
                                               flag=1, transform=transform_train_224)
train_valid_ds_299 = vision.ImageFolderDataset(input_str + train_valid_dir,
                                               flag=1, transform=transform_train_299)
batch_size = 32
train_valid_iter_224 = DataLoader(train_valid_ds_224, batch_size, shuffle=True,
                              last_batch='keep')
train_valid_iter_299 = DataLoader(train_valid_ds_299, batch_size, shuffle=True,
                              last_batch='keep')


def save_trainval(data,net,name):
    # 不能直接concat两个网络的features，因为在data_iter中是按照随机读取的数据，labels不相同
    x =[]
    y =[]
    print('提取特征 %s' % name)
    for fear1,fear2,label in tqdm(data):
        fear1 = fear1.as_in_context(mx.gpu())
        fear2 = fear2.as_in_context(mx.gpu())
        out = net(fear1,fear2).as_in_context(mx.cpu())
        x.append(out)
        y.append(label)
    x = nd.concat(*x,dim=0)
    y = nd.concat(*y,dim=0)
    print('保存特征 %s' % name)
    nd.save(name,[x,y])
if not os.path.exists('trainval.nd'):
    save_trainval(train_valid_iter_299, train_valid_iter_224, 'trainval.nd')