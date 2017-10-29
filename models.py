from mxnet import gluon


class Generator(gluon.HybridBlock):
    def __init__(self, n_dims=64, name="dcgan_g", **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.module_name = name

        self.fc1 = gluon.nn.Dense(n_dims * 16 * 4 * 4)

        self.deconv1 = gluon.nn.Conv2DTranspose(n_dims * 8, 4, 2, 1, use_bias=False)
        self.deconv2 = gluon.nn.Conv2DTranspose(n_dims * 4, 4, 2, 1, use_bias=False)
        self.deconv3 = gluon.nn.Conv2DTranspose(n_dims * 2, 4, 2, 1, use_bias=False)
        self.deconv4 = gluon.nn.Conv2DTranspose(n_dims, 4, 2, 1, use_bias=False)
        self.deconv5 = gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False)

        self.fc1_bnorm = gluon.nn.BatchNorm()
        self.deconv1_bnorm = gluon.nn.BatchNorm()
        self.deconv2_bnorm = gluon.nn.BatchNorm()
        self.deconv3_bnorm = gluon.nn.BatchNorm()
        self.deconv4_bnorm = gluon.nn.BatchNorm()

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.fc1(x)
        x = F.reshape(x, shape=[-1, self.n_dims * 16, 4, 4])
        x = F.relu(self.fc1_bnorm(x))
        x = F.relu(self.deconv1_bnorm(self.deconv1(x)))
        x = F.relu(self.deconv2_bnorm(self.deconv2(x)))
        x = F.relu(self.deconv3_bnorm(self.deconv3(x)))
        x = F.relu(self.deconv4_bnorm(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x

    def save_params(self, filename="dcgan_g"):
        self.collect_params().save(self.module_name)

    def load_params(self, ctx, allow_missing=False, ignore_extra=False, filename="dcgan_g"):
        self.collect_params().load(self.module_name, ctx, allow_missing, ignore_extra)


class Discriminator(gluon.HybridBlock):
    def __init__(self, n_dims=64, name="dcgan_d", **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.module_name = name

        self.conv1 = gluon.nn.Conv2D(n_dims, 6, 4, 1, use_bias=False)  # 32 # 64
        self.conv2 = gluon.nn.Conv2D(n_dims * 1, 4, 2, 1, use_bias=False)  # 16
        self.conv3 = gluon.nn.Conv2D(n_dims * 2, 4, 2, 1, use_bias=False)  # 8
        self.conv4 = gluon.nn.Conv2D(n_dims * 4, 4, 2, 1, use_bias=False)  # 4
        self.conv5 = gluon.nn.Conv2D(1, 4, 1, 0, use_bias=False)

        self.batchnorm2 = gluon.nn.BatchNorm()
        self.batchnorm3 = gluon.nn.BatchNorm()
        self.batchnorm4 = gluon.nn.BatchNorm()

        self.flatten = gluon.nn.Flatten()

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = F.LeakyReLU(x, slope=0.2)
        x = self.batchnorm2(self.conv2(x))
        x = F.LeakyReLU(x, slope=0.2)
        x = self.batchnorm3(self.conv3(x))
        x = F.LeakyReLU(x, slope=0.2)
        x = self.batchnorm4(self.conv4(x))
        x = F.LeakyReLU(x, slope=0.2)
        x = self.flatten(self.conv5(x))
        return x

    def save_params(self, filename="dcgan_d"):
        self.collect_params().save(self.module_name)

    def load_params(self, ctx, allow_missing=False, ignore_extra=False, filename="dcgan_d"):
        self.collect_params().load(self.module_name, ctx, allow_missing, ignore_extra)
