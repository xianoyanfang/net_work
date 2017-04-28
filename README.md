# net_work
神经网络___ufldl(http://ufldl.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95)
## mnist_net.py
这是一个进行识别手写数字的过程，我们设置神经网络层数为 (274,100,10) ,将数据放入训练函数 net_works_train，可已得到权值和偏置 W 和 b ，
再将其放入测试函数 net_works_test 对测试集进行分类，求解出准确率
## networks_train.py
net_works_train(x,y,num,times,e)<br>
* x 表示训练数据属性<br>
* y 表示训练数据标签<br>
* num 可以是元组和列表，表示网络层数，如： (784,100,10) (784,100,50,10)...<br>
* times 表示训练权重和偏置时的最大轮数<br>
* e 表示终止误差<br>
## networks_test.py
net_works_test(x,W,b)<br>
* x 表示测试数据属性<br>
* W 表示网络每一层的权重<br>
* b 表示网络每一层的偏置<br>
