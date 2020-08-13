from utils import *
from get_features import save_models
from mxnet import autograd

def get_trainval_iter(model_list, batch_size=128, train_size=0.9):
    features =[]
    labels =[]
    for model_name in model_list:
        x = nd.load('features_train_%s.nd' % model_list)[0].as_in_context(ctx)
        features.append(x)

    features = nd.concat(*features, dim=1)
    labels = nd.load('labels.nd')[0].as_in_context(ctx)

    n_train = int(features.shape[0] * train_size)

    X_train = features[:n_train]
    y_train = labels[:n_train]

    X_val = features[n_train:]
    y_val = labels[n_train:]

    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    dataset_val = gluon.data.ArrayDataset(X_val, y_val)

    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)

    return data_iter_train, data_iter_val

model_list =['inceptionv3','resnet152_v1']
save_models(valid_iter_299, valid_iter_224, 'trainval',model_list)
epochs = 150
batch_size = 128
net =get_net()
data_iter_train, data_iter_val = get_trainval_iter(model_list, batch_size=batch_size)
lr_sch = mx.lr_scheduler.FactorScheduler(step=50, factor=0.75)
trainer = gluon.Trainer(net.collect_params(), 'adam',
                        {'learning_rate': 1e-3, 'wd': 1e-4, 'lr_scheduler': lr_sch})

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

    val_loss, val_acc = evaluate_loss(net, data_iter_val)

    print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
        epoch+1, train_loss/steps, train_acc/steps*100, val_loss, val_acc*100))
