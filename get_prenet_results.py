from mxnet import autograd
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import time, os, tqdm
from utils import *

train_ds_224 = vision.ImageFolderDataset(input_str + train_dir, flag=1,
                                         transform=transform_train_224)
train_ds_299 = vision.ImageFolderDataset(input_str + train_dir, flag=1,
                                         transform=transform_train_299)
valid_ds_224 = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                         transform=transform_test_224)
valid_ds_299 = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                         transform=transform_test_299)

batch_size = 32

loader = gluon.data.DataLoader
train_iter_224 = loader(train_ds_224, batch_size, shuffle=True, last_batch='keep')
train_iter_299 = loader(train_ds_299, batch_size, shuffle=True, last_batch='keep')
valid_iter_224 = loader(valid_ds_224, batch_size, shuffle=True, last_batch='keep')
valid_iter_299 = loader(valid_ds_299, batch_size, shuffle=True, last_batch='keep')

def save_features(data_iter, model, type: str, ignore=False):
    if os.path.exists('features_%s_%s.nd' % (type, model)) and ignore:
        return
    x = []
    y = []
    net = models.get_model(model, pretrained=True, ctx=ctx)
    if 'squeezenet' in model:   #squeenzenet最后没有全局池化层,需要手动添加,否则参数文件将过大
        net.output = gluon.nn.HybridSequential(prefix='')
        net.output.add(gluon.nn.GlobalAvgPool2D())
        net.output.add(gluon.nn.Flatten())
    else:
        net = net.features
    print('提取特征:%s %s' % (type, model))
    for fear, label in tqdm(data_iter):
        fear = fear.as_in_context(ctx)
        outputs = net(fear).as_in_context(ctx)
        x.append(outputs)
        y.append(label)
    x = nd.concat(*x, dim=0)
    y = nd.concat(*y, dim=0)
    nd.save('features_%s_%s.nd' % (type, model), [x,y])  # [x,y]
    print('保存特征%s %s成功' % (type, model))


#from mxnet.gluon.model_zoo.model_store import _model_sha1
#sorted(_model_sha1.keys()):
model_list = ['alexnet', 'densenet161', 'inceptionv3', 'resnet101_v1', 'resnet152_v1', 'vgg11',
              'vgg13', 'vgg16', 'vgg19']

def save_models(iter_name299, iter_name244, type):
    for model in model_list:
        print(model)
        if model == 'inceptionv3':
            save_features(iter_name299, model, type)
        else:
            save_features(iter_name244, model, type)


save_models(valid_iter_299, valid_iter_224, 'val')
save_models(train_iter_299,train_iter_224,'train')

def get_train_iter(model_name, batch_size):
    train_nd = nd.load('features_train_%s.nd' % model_name)
    return gluon.data.DataLoader(gluon.data.ArrayDataset(train_nd[0], train_nd[1]), batch_size=batch_size, shuffle=True)


def get_val_iter(model_name, batch_size):
    val_nd = nd.load('features_val_%s.nd' % model_name)
    return gluon.data.DataLoader(gluon.data.ArrayDataset(val_nd[0], val_nd[1]), batch_size=batch_size, shuffle=True)



def get_prenet_result(model_name):
    epochs = 50
    batch_size = 128
    train_iter = get_train_iter(model_name, batch_size)
    val_iter = get_val_iter(model_name, batch_size)
    net = Net(ctx).output
    net.hybridize()

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-4})
    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc, = 0., 0.
        steps = len(train_iter)
        for data, label in train_iter:
            data, label = data.as_in_context(ctx), label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)

        val_loss, val_acc = evaluate_loss(net, val_iter)

    time_s = "time %.2f sec" % (time.time() - start_time)
    print("Epoch%d, loss:%.4f, acc:%.2f%%,val_loss %.4f, val_acc %.2f%%, model:%s, time:%s" % (
        epoch + 1, train_loss / steps, train_acc / steps * 100, val_loss, val_acc * 100, model_name, time_s))

    # return val_loss



# losses = []

for model_name in model_list:
    get_prenet_result(model_name)
    # losses.append((model_name, val_loss))

