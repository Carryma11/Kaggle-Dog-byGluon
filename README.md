# 使用gluon预测120种狗的类别
## Environment
python:3.6.10    
mxnet-cu100  
tqdm
## 实现步骤
1.在kaggle网站下载数据集并解压到data目录下  
2.运行reorg_data.py,切分数据集成train，trainval，val  
3.(可省略)运行get_prenet_result.py,查看哪种预训练网络适合提取该数据的特征  
4.(1)通过3可知，inceptionv3和resnet152_v1的效果最好,故使用模型融合构造新网络.  
(2)运行get_trainval.py生成我们要用的data_iter.  
(3)运行train.py即可输出结果:json文件是网络结构,params是权重参数.  
备注：create_ids生成用于inference用的类别,utils里面是一些预处理函数.




