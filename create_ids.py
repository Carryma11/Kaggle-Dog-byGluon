from mxnet.gluon import data as gdata
import os
data_dir = './data/train_valid_test'
#input_dir = 'train_valid_test'
train_valid_dir = 'train_valid'
train_valid_path =os.path.join(data_dir, train_valid_dir)
train_valid_ds = gdata.vision.ImageFolderDataset(train_valid_path,
                                           flag=1)

#ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
with open('ids_synset.txt','w') as f:
    for label in train_valid_ds.synsets:
        f.write(label.strip() + '\n')
    f.close()
print("create index successfully!")
