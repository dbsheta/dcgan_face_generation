import os
from matplotlib import pyplot as plt
import time
import math

import mxnet as mx
from mxnet import nd
from mxnet import autograd
import numpy as np
from v2.models import Generator, Discriminator
from v2.data_utils import Dataset


# In[4]:


def time_since(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


data_path = r"C:\Users\dhoomil.sheta\Downloads\Pix2Code Data\web\all_data"

epochs = 10  # Set low by default for tests, set higher when you actually run this code.
batch_size = 64
z_dims = 100
img_dims = 256

use_gpu = False
ctx = mx.gpu() if use_gpu else mx.cpu()

lr = 0.0002
beta1 = 0.5
beta2 = 0.999


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')


netG = Generator()
netD = Discriminator()
netG.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.initialize(mx.init.Normal(0.02), ctx=ctx)

loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainerG = mx.gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2})
trainerD = mx.gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2})

real_label = nd.ones((batch_size,), ctx=ctx)
fake_label = nd.zeros((batch_size,), ctx=ctx)

img_list = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.endswith('png')]
train_data = Dataset(img_list, batch_size=batch_size)


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


metric = mx.metric.CustomMetric(facc)


def train(epochs):
    netD.hybridize()
    netG.hybridize()
    print(f"Training for {epochs} epochs...")
    start = time.time()
    for epoch in range(epochs):
        train_data.reset()
        iteration = 0
        while train_data.has_next():
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            data = train_data.next()
            latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, z_dims), ctx=ctx)

            with autograd.record():
                # train with real image
                output = netD(data)
                errD_real = loss(output, real_label)
                metric.update([real_label, ], [output, ])

                # train with fake image
                fake = netG(latent_z)
                print(fake.shape)
                output = netD(fake.detach())
                errD_fake = loss(output, fake_label)
                errD = errD_real + errD_fake
                errD.backward()
                metric.update([fake_label, ], [output, ])

            trainerD.step(batch_size)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            with autograd.record():
                output = netD(fake)
                errG = loss(output, real_label)
                errG.backward()

            trainerG.step(batch_size)

            if iteration % 10 == 0:
                _, acc = metric.get()
                print(f'epoch {epoch}: iter {iteration} d_loss = {nd.mean(errD).asscalar()}, '
                      f'generator loss = {nd.mean(errG).asscalar()}, training acc = {acc}')

        iteration = iteration + 1
        name, acc = metric.get()
        print(f'epoch {epoch}: iter {iteration} d_loss = {nd.mean(errD).asscalar()}, '
              f'generator loss = {nd.mean(errG).asscalar()}, training acc = {acc}')
        print(f"Time: {time_since(start)}")
        metric.reset()
    print(f"Time: {time_since(start)}")


train(2)
