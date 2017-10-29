import os
from matplotlib import pyplot as plt
import time
import math
import argparse

import mxnet as mx
from mxnet import nd
from mxnet import autograd
import numpy as np
from models import Generator, Discriminator
from data_utils import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", help="No. of epochs to run for training", type=int)
parser.add_argument("--resume", "-c", help="Continue training?1:Yes, 0:False", type=int)

args = parser.parse_args()


def time_since(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


data_path = r"/Users/dhoomilbsheta/Development/datasets/pix2code_datasets/web/all_data"

epochs = 10  # Set low by default for tests, set higher when you actually run this code.
batch_size = 64
z_dims = 100
img_dims = 128

use_gpu = False
ctx = mx.gpu() if use_gpu else mx.cpu()

lr = 0.0002
beta1 = 0.5
beta2 = 0.999


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')


netG = Generator(name="dcgan_g_html")
netD = Discriminator(name="dcgan_d_html")

loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

real_label = nd.ones((batch_size,), ctx=ctx)
fake_label = nd.zeros((batch_size,), ctx=ctx)

img_list = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.endswith('png')]
train_data = Dataset(img_list, img_dims, batch_size=batch_size)


def init_params():
    netG.initialize(mx.init.Normal(0.02), ctx=ctx)
    netD.initialize(mx.init.Normal(0.02), ctx=ctx)


def load_weights():
    netG.load_params(ctx=ctx)
    netD.load_params(ctx=ctx)


def init_optimizers():
    trainerG = mx.gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2})
    trainerD = mx.gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2})
    return trainerG, trainerD


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


metric = mx.metric.CustomMetric(facc)


def train(epochs, resume=False):
    if resume:
        load_weights()
        print("Loading pretrained weights")
    else:
        init_params()
        print("Initializing parameters")
    trainerG, trainerD = init_optimizers()

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
        print(f'epoch {epoch} last iteration d_loss = {nd.mean(errD).asscalar()}, '
              f'generator loss = {nd.mean(errG).asscalar()}, training acc = {acc}')
        print(f"Time: {time_since(start)}")
        metric.reset()
        netG.save_params()
        netD.save_params()
    print(f"Time: {time_since(start)}")
    netG.save_params()
    netD.save_params()


train(args.epochs, bool(args.resume))
