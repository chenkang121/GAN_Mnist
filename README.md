# GAN_Mnist
Mnist数据集生成测试
一、模型结构
==========
使用的gan模型由两个部分（生成器和判别器）组成，项目的目的是通过训练生成器和判别器，使得`生成器`能够根据`一个随机数`生成出类似训练集 `手写数字`的样本，实现数据扩充的效果。<br>

# 1.生成器
## 输入
生成器的输入是通过tf.random.normal(size)生成的符合正太分布的随机数（均值为0，标准差为1）。
```python
noise = tf.random.normal([BATCH_SIZE, noise_dim])
```
## 中间过程
生成器的中间过程取决于使用的模型结构，本例中使用的结构为三层的**全连接层**。
```python
def generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(256, input_shape=(100,), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense((28*28*1), use_bias=False, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((28, 28, 1)))
    return model
```
## 输出
生成器的输出为一副图像，与Mnist数据集中的样本的结构相同。中间过程中的最后一行定义了输出的类型。
```python
model.add(layers.Reshape((28, 28, 1)))
```
## 损失函数
采用CrossEntropy做为损失函数，生成器输出了之后需要经过判别器判断之后再确定损失函数值，（生成器的目的是产生出判别器无法区分的图像），若判别器判断正确的越多说明生成器需要改进的越多，损失函数的值也应越大。

``` python
def generator_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out), fake_out)
```

# 2.判别器
## 输入
判别器的输入为一副来自**原始数据集**或是来自生成器**生成数据集**的图片。
```python
real_out = discriminator(images, training=True)
fake_out = discriminator(generator_image, training=True)
```
## 中间过程
判别器的中间过程也可以根据不同任务需求进行修改，本例中使用三层**全连接层**.
```python
def discriminator_model():
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))
    return model
```
## 输出
判别器的输出为图片属于**原始数据集**或是**生成数据集**的0-1信息。对一张从**原始数据集**来的样本，其输出应为1，而当输入样本来自于**生成数据集**时其输出的值应为0。
```python
real_out = discriminator(images, training=True)
fake_out = discriminator(generator_image, training=True)
```
## 损失函数
判别器的目的是尽可能多的识别出来自不同域的样本。

``` python
def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss
```
# 总结
对抗生成网络GAN在进行数据扩容任务时，需要同时训练两个模型，确定了batch_size之后，对同一个batch输入需要准备好<br>
1.一个batch数量的随机数（生成器的输入）<br>
2.一个batch数量的Mnist手写数字图片（real_image）<br>
首先及那个该batch的随机数输入生成器，得到生成器的输出一个batch的fake_image，然后将这些fake_image输入到判别器（discriminator）中得到fake_out，同时将准备好的real_image输入判别器中得到输出real_out，一次两个模型的前向传播完成。
<br>
在获得了fake_out，与real_out之后可以计算，generator和discriminator的损失函数，generator生成fake_image之后希望fake_image能够被discriminator判断为1（来自**原始数据集**），故损失函数为输出与1之间的差值。而discriminator希望可以判断出所有的样本的真实来源域，故用fake_out与0的距离，real_out与1的距离之和作为损失函数。


