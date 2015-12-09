# coding:utf-8
__author__ = 'liangz14'
import theano
import theano.tensor as T
import numpy as np

#构造隐藏层(sigmod层)
#生成W，b学习参数，从输入input（计算符号）构造输出output(计算符号)
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out ,activation=T.tanh):

        #构造当前层的参数W和b
        rng = np.random.RandomState(1234) #用来生成随机数
        W_values = np.asarray(   #生成随机的W矩阵
            rng.uniform(
                low = -np.sqrt(6./(n_in + n_out)), #正态分布下界
                high = np.sqrt(6./(n_in + n_out)),  #正态分布上界
                size = (n_in, n_out)  #W矩阵的大小
            ),
            dtype=theano.config.floatX
        )
        b_values = np.asarray(   #生成初始化为0的b
            np.zeros(shape=(n_out,)),
            dtype=theano.config.floatX
        )
        W_values *= 4
        #对于borrow和shaded的官方解释见：
        # http://deeplearning.net/software/theano/tutorial/aliasing.html?highlight=borrow#borrowing-when-creating-shared-variables
        W = theano.shared(W_values,borrow=True)
        b = theano.shared(b_values,borrow=True)

        #记录当前层的各项参数
        self.input = input #符号变量！
        self.W = W #生成的参数shared
        self.b = b #生成的参数shared

        #构造输出：焊好理解～ y = Wx+b,只是要注意
        tmp_output = T.dot(input, self.W) + self.b
        #多层感知网络的隐藏层输出需要经过激活函数（tangh或者singmod）进行限定：
        self.output = T.nnet.sigmoid(tmp_output)
        #or
        #self.output = T.tanh(lin_output)

        #保存参数，方便之后访问。
        self.params = [self.W, self.b]

#构造输出层：
class LogisticRegression(object):
    #初始化也是从输入的到输出，同事也需要计算一个误差
    def __init__(self,input ,n_in,n_out):
        #回归层的系数为0即可
        self.W = theano.shared(
                    value=np.zeros((n_in,n_out),dtype=theano.config.floatX),
                    borrow=True
                )
        self.b = theano.shared(
                    value=np.zeros((n_out,),dtype=theano.config.floatX),
                    borrow=True
                )
        tmp_output = T.dot(input, self.W) + self.b
        #归一化
        self.p_y_given_x = T.nnet.softmax(tmp_output)
        #得到最大值的索引：return the index of the maximum value along a given axis
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input
        #定义似然函数

    #这个就是用实际的y提取出计算出的值
    def negative_log_likelihood(self, y):
        #p_y_given_x就是当前各个结果的似然函数。这里提取除了结果对应的几个，求最小（因为取了负数，本身应该去最小）
        #numpy支持特殊的数组访问模式
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    #定义误差函数
    def errors(self,y):
        #暂时不做多余检查。误差：就是有几何结果和原来不同。
        return T.mean(T.neq(self.y_pred, y))

#构造多层感知网络
class MPL(object):
    def __init__(self, input, n_int, n_hidden, n_out):
        #网络构造完成，给出input就能计算出self.logRegresslonLayer.output
        self.hiddenLayer = HiddenLayer(input,n_int,n_hidden)
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        #防止过拟合的约束L1和L2
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2 = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        #输入就是整个网络的输入。
        self.input = input
        self.y_pred = self.logRegressionLayer.y_pred

        #似然概率直接用输出层的。
        self.negative_log_likelihood =self.logRegressionLayer.negative_log_likelihood
        #误差也直接使用输出层的。
        self.errors = self.logRegressionLayer.errors
#简单的使用一下
def test():
    x = T.dmatrix()
    my_mpl = MPL(x,3,2,3)
    fun = theano.function([x],x[[0,1,2],[0,2,1]])
    print fun([[1,2,3],[1,2,3],[1,2,3]])






