# coding:utf-8
__author__ = 'liangz14'
from mpl_lz import MPL
import theano
import theano.tensor as T
import numpy as np
print("reading data...")
source = list()
target = list()
test = list()
with open("train.csv") as train_file:
    lines = train_file.readlines()
    for i,line in enumerate(lines[1:]):

        tmp_str = line.split(",")
        target.append(int(tmp_str[0]))
        source.append([int(c) for c in tmp_str[1:]])

with open("test.csv") as test_file:
    lines = test_file.readlines()
    for i,line in enumerate(lines[1:]):

        tmp_str = line.split(",")
        test.append([int(c) for c in tmp_str])

print("construct model ...")

#参数设置
learning_rate=0.01
L1_reg=0.00
L2_reg=0.0001
n_epochs=40
batch_size=20
n_hidden=500

val_source = np.asarray(source[-12000:],dtype=theano.config.floatX)
val_target = np.asarray(target[-12000:],dtype=np.int64)
tr_source = np.asarray(source[0:-12000],dtype=theano.config.floatX)
tr_target = np.asarray(target[0:-12000],dtype=np.int64)
test = np.asarray(test,dtype=theano.config.floatX)

print tr_source.shape
print tr_source.dtype

#训练集
train_set_x = theano.shared(tr_source, borrow=True)
train_set_y = theano.shared(tr_target, borrow=True)
train_set_y = T.cast(train_set_y, 'int32')

#测试集
valid_set_x = theano.shared(val_source, borrow=True)
valid_set_y = theano.shared(val_target, borrow=True)
valid_set_y = T.cast(valid_set_y, 'int32')
#结果
test_set_x = theano.shared(test, borrow=True)

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

#输入时一组矩阵（minibantch）,输出是预测结果
index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')
classifier = MPL(x, 28*28, n_hidden, 10)

#定义代价函数。用来进行梯度下降。
cost = (classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2
    )

gparams = [T.grad(cost, param) for param in classifier.params]

#更新梯度
updates = [
    (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

#构造计算函数：
#三个计算模型：训练，测试，验证。
train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

result_model = theano.function(
        inputs=[index],
        outputs=classifier.y_pred,
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size]
        }
    )

validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )


print("compute begin ...")

patience = 10000
patience_increase = 2
improvement_threshold = 0.995
#保证每次迭代验证一次
validation_frequency = min(n_train_batches, patience /2)

best_validation_loss = np.inf
best_iter = 0
test_score = 0.

epoch = 0
done_looping = False

#进行多次迭代
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    #每一次访问一个minibatch
    for minibatch_index in xrange(n_train_batches):
        #通过index使用第一组训练数据
        minibatch_avg_cost = train_model(minibatch_index)

        #当前迭代总次数
        iter = (epoch - 1) * n_train_batches + minibatch_index
        '''
        #对所有数据集合进行验证。求误差均值。
        validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)

                输出提示


        print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
        '''

        #只有在验证的时候，有更好的效果才会保存。
        if (iter + 1) % validation_frequency == 0:
            #对所有数据集合进行验证。求误差均值。
            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            #有更好的效果
            if this_validation_loss < best_validation_loss:
                #如果改善的效果比较明显，则可以多训练几次。通过patience设置
                if (
                    this_validation_loss < best_validation_loss *
                    improvement_threshold
                ):
                    patience = max(patience, iter * patience_increase)
                #记录当前的最佳结果的次数。
                best_validation_loss = this_validation_loss

                '''
                    输出提示
                '''
                print(('epoch %i, minibatch %i/%i, loss %d %%') %
                        (epoch, minibatch_index + 1, n_train_batches,best_validation_loss*100))

            #终止条件，改善效率过低
            if patience <= iter:
                done_looping = True
                break
#计算在测试模型上的分数
with open('result.csv','w') as r:
    resultt=[]
    for i in xrange(n_test_batches):
        resultt.extend(result_model(i))
    r.write("ImageId,Label\n")
    for i,n in enumerate(resultt):
        r.write("%d,%d\n" % (i+1,n))

