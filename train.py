from utils import *
from mxnet import autograd
import math

epochs = 150
batch_size = 128
train_size = 1  #∈(0，1]


def get_trainval_iter(batch_size, train_size):
    features = nd.load('trainval.nd')[0].as_in_context(ctx)
    labels = nd.load('trainval.nd')[1].as_in_context(ctx)

    n_train = int(features.shape[0] * train_size)

    X_train = features[:n_train]
    y_train = labels[:n_train]
    data_iter_val = None
    if train_size < 1:
        X_val = features[n_train:]
        y_val = labels[n_train:]
        dataset_val = gluon.data.ArrayDataset(X_val, y_val)
        data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)
    dataset_train = gluon.data.ArrayDataset(X_train,y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)


    return data_iter_train, data_iter_val



net = Net(ctx).output
net.hybridize()
data_iter_train, data_iter_val = get_trainval_iter(batch_size=batch_size,train_size=train_size)
'''
lr_sch = mx.lr_scheduler.FactorScheduler(step=500, factor=0.9)
trainer = gluon.Trainer(net.collect_params(), 'adam',
                        {'learning_rate': 1e-4, 'wd': 1e-4, 'lr_scheduler': lr_sch})
'''
lr = 0.001
momentum = 0.9
wd = 0.0001
lr_factor = 0.99
num_batch = len(data_iter_train)
iterations_per_epoch = math.ceil(num_batch)
lr_steps = 250
schedule = mx.lr_scheduler.FactorScheduler(step=lr_steps, factor=lr_factor, base_lr=lr)
sgd_optimizer = mx.optimizer.SGD(learning_rate=lr, lr_scheduler=schedule, momentum=momentum, wd=wd)
trainer = gluon.Trainer(net.collect_params(), optimizer=sgd_optimizer)

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)

        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    if  data_iter_val is not None:
        val_loss, val_acc = evaluate_loss(net, data_iter_val)
        print("Epoch %d. train_loss: %.4f, train_acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
            epoch + 1, train_loss / steps, train_acc / steps * 100, val_loss, val_acc * 100))
    else:
        print("Epoch %d. loss: %.4f, acc: %.2f%%" % (
            epoch + 1, train_loss / steps, train_acc / steps * 100))
        net.export("result", epochs)


