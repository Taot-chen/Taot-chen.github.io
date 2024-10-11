---
layout: post
title: flax_study
date: 2024-05-21
tags: [flax]
author: taot
---


## flax 框架理解

深度学习框架有很多，所有框架都要回答下面的几个基本问题：
* 如何定义网络？
* 如何初始化网络参数？
* 如何计算反向传播？
* 如何更新网络参数？
* 如何管理训练状态？

pytorch 作为越来越受欢迎的框架，以上几个问题的解决无疑是接近完美的，flax 相对于 pytorch，又是如何面临这几个问题的呢？

### 1 网络定义

flax采取就地定义，就地使用的方式，使用时再定义。
```python
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from sqlalchemy import false
import tensorflow_datasets as tfds

class CNN(nn.Module):
  @nn.compact
  def __call__(self, x,is_training:bool=True):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x
```

### 2 初始化网络参数

使用网络的 `init()` 方法初始化网络参数，方法的参数需要输入数据的形状
```python
  cnn = CNN()
  variables=cnn.init(rng, jnp.ones([1, 28, 28, 1]))
  params = variables['params']
  batch_stats=variables['batch_stats']
```

### 3 管理训练状态

TrainState.create创建训练状态，三个参数：前向传播函数，网络参数，优化器
```python
tx = optax.sgd(0.01, 0.99)
state=train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
```

### 4 计算反向传播

* 先定义梯度计算函数`grad_fn = jax.value_and_grad(loss_fn, has_aux=True)`，实际就是grad函数和损失函数的复合函数；
* 调用`grad_fn`得到梯度，函数`grad_fn` 的参数与`loss_fn`一致，返回值就是grads ，结构与loss_fn的第一个参数params一致。
* 有两个函数可以计算梯度：`jax.value_and_grad`和`jax.grad`。

```python
@jax.jit
def apply_model(state, images, labels,old_batch_stats):
  def loss_fn(params,old_batch_stats):
    logits,mutated_vars = state.apply_fn({'params': params,"batch_stats":old_batch_stats}, images,is_training=True, mutable=['batch_stats'])
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, (logits,mutated_vars['batch_stats'])
    
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits,new_batch_stats)), grads = grad_fn(state.params,old_batch_stats)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy,new_batch_stats
```

### 5 更新网络参数

训练状态state更新自己
```python
state=state.apply_gradients(grads=grads)
```


完整代码：
```python
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from sqlalchemy import false
import tensorflow_datasets as tfds
import flax


class CNN(nn.Module):
  @nn.compact
  def __call__(self, x,is_training:bool=True):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x



@jax.jit
def apply_model(state, images, labels,old_batch_stats):
  def loss_fn(params,old_batch_stats):
    logits,mutated_vars = state.apply_fn({'params': params,"batch_stats":old_batch_stats}, images,is_training=True, mutable=['batch_stats'])
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, (logits,mutated_vars['batch_stats'])
    
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits,new_batch_stats)), grads = grad_fn(state.params,old_batch_stats)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy,new_batch_stats


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng,batch_stats):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []
  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy ,batch_stats= apply_model(state, batch_images, batch_labels,batch_stats)
    
    state = update_model(state, grads)
    
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy,batch_stats


def get_datasets():
  ds_builder = tfds.builder('fashion_mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds


def create_train_state(rng):
  cnn = CNN()
  variables=cnn.init(rng, jnp.ones([1, 28, 28, 1]))
  params = variables['params']
  batch_stats=variables['batch_stats']
  tx = optax.sgd(0.01, 0.99)
  state=train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
  
  return state,batch_stats

@jax.jit
def predict(state, batch_stats,image_i):
  logits= state.apply_fn({'params': state.params,"batch_stats":batch_stats},image_i,is_training=False)
  return logits
def test(state, batch_stats,test_ds):
  images = test_ds['image']
  labels = test_ds['label']
  batchs=1000
  accuracy=0
  for i in range(0,len(images),batchs):
    image_i=images[i:i+batchs]
    label_i=labels[i:i+batchs]

    logits= predict(state, batch_stats,image_i)
    accuracy += jnp.sum(jnp.argmax(logits, -1) == label_i)
  
  return accuracy/len(images)

def train_and_evaluate() -> train_state.TrainState:
  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)

  
  rng, init_rng = jax.random.split(rng)
  state,batch_stats = create_train_state(init_rng)
  
  for epoch in range(1, 100 + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy,batch_stats = train_epoch(state, train_ds,
                                                    100,
                                                    input_rng,batch_stats)
    
    
    print(test(state, batch_stats,test_ds),end=" ")
  
  return state

train_and_evaluate() 
```
