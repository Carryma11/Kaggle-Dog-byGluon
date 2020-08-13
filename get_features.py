import os
from mxnet.gluon.model_zoo import vision as models
from tqdm import tqdm
from utils import *




def save_features(data_iter, model, type, ignore=True):
    if os.path.exists('features_%s_%s.nd' % (type, model)) and ignore:
        return
        # if os.path.exists('features_test_%s.nd' % name):
    x = []
    y = []
    net = models.get_model(model, pretrained=True, ctx=ctx, root='./pretrained models')
    if 'squeezenet' in model:
        net.output = gluon.nn.HybridSequential(prefix='')
        net.output.add(gluon.nn.GlobalAvgPool2D())
        net.output.add(gluon.nn.Flatten())
    else:
        net = net.features
    print('提取特征:%s %s' % (type, model))
    for fear, label in tqdm(data_iter):
        fear = fear.as_in_context(ctx)
        outputs = net(fear).as_in_context(mx.cpu())
        x.append(outputs)
        y.append(label)
    x = nd.concat(*x, dim=0)
    y = nd.concat(*y, dim=0)
    nd.save('features_%s_%s.nd' % (type, model), [x, y])
    print('保存特征%s %s成功' % (type, model))


"""
from mxnet.gluon.model_zoo.model_store import _model_sha1
sorted(_model_sha1.keys()):
"""


def save_models(data_iter_299, data_iter_244, type,model_list):
    for model in model_list:
        print(model)
        if model == 'inceptionv3':
            save_features(data_iter_299, model, type)
        else:
            save_features(data_iter_244, model, type)


