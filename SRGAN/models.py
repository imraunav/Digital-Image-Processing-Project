import tensorflow as tf
from tensorflow.keras import utils, layers, Model, optimizers, Input, callbacks
# from tensorflow.python.keras.layers import
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


def build_generator(lr_shape=(None, None, 3), b_residual_blocks=16):
    """
    Builds a genrator networks according to specs descibed by SRGAN.
    The network takes in a low resolution image and generates a corresponding high resolution image.
    """

    def pixel_shuffle(scale):
        return lambda x: tf.nn.depth_to_space(x, scale)
    
    def residual_block(layer_input, filters=64):
        """
        Following specks of the residual block described in paper
        """
        d = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(layer_input)
        d = layers.BatchNormalization()(d)
        d = layers.PReLU(shared_axes=[1, 2])(d)
        d = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(d)
        d = layers.BatchNormalization()(d)
        return layers.Add()([d, layer_input])

    def upsampling_block(layer_input):
        # u = layers.UpSampling2D(size=2)(layer_input)
        # u = layers.Conv2D(filters=256, kernel_size=3,
        #                 strides=1, padding="same")(u)
        u = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(layer_input)
        u = layers.Lambda(pixel_shuffle(scale=2))(u)
        u = layers.PReLU(shared_axes=[1, 2])(u)
        return u
    
    # Low resolution image input
    img_lr = Input(shape=lr_shape)
    # may add rescalling to the network itself, will think about it later. Keep things as simple as possible.
    # img_lr = layers.rescalling(scale=1/255.0)(img_lr) # take care of recaling in network itself

    # Pre-residual block
    c1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(img_lr)
    c1 = layers.PReLU(shared_axes=[1, 2])(c1)

    # 16 residual blocks
    r = residual_block(c1) # first residual block
    for i in range(b_residual_blocks-1): # add residual block one after the other
        r = residual_block(r)

    # Post residual block
    c2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(r)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Add()([c2, c1]) # skip connection


    # upsampling: the multiplier is controlled here
    u1 = upsampling_block(c2) # x2
    u2 = upsampling_block(u1) # x2

    # Last conv layer
    # gen_hr = layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same", activation="tanh")(u1)
    gen_hr = layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same", activation="tanh")(u2)

    
    return Model(inputs=[img_lr], outputs=[gen_hr], name="Generator")


# generator = build_generator()
# generator.summary()

# img = tf.keras.utils.load_img("/Users/raunavghosh/Documents/Research for Vehant/SRGAN-tf/BSR/BSDS500/data/images/train/65019.jpg")
# img_arr = tf.keras.utils.img_to_array(img)
# print(img_arr.shape)
# out = generator.predict(tf.expand_dims(img_arr, axis=0))
# print(out.shape)
# img.save("shit.png")
# tf.keras.utils.array_to_img(tf.squeeze(out, axis=0)).save("sr_shit.png")

def build_discriminator(hr_shape=(256,256,3)):
    """
    Builds a discriminator network according to specs given by SRGAN.
    The network takes in a High resolution image and classifies it as real or fake.
    """
    
    def d_block(layer_input, filters=3, strides=1, batchnormalise=True):
        d = layers.Conv2D(filters, kernel_size=3,
                   strides=strides, padding="same")(layer_input)
        if batchnormalise == True:
            d = layers.BatchNormalization()(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        return d
    
    # Input img
    d0 = Input(shape=hr_shape)
    d1 = d_block(d0, filters=64, strides=1, batchnormalise=False)
    d2 = d_block(d1, filters=64, strides=2, batchnormalise=True)
    d3 = d_block(d2, filters=128, strides=1, batchnormalise=True)
    d4 = d_block(d3, filters=128, strides=2, batchnormalise=True)
    d5 = d_block(d4, filters=256, strides=1, batchnormalise=True)
    d6 = d_block(d5, filters=256, strides=2, batchnormalise=True)
    d7 = d_block(d6, filters=512, strides=1, batchnormalise=True)
    d8 = d_block(d6, filters=512, strides=2, batchnormalise=True)

    d8_flat = layers.Flatten()(d8)
    d9 = layers.Dense(units=1024)(d8_flat)
    d10 = layers.LeakyReLU(alpha=0.2)(d9)
    score = layers.Dense(units=1, activation="sigmoid")(d10)
    return Model(inputs=[d0], outputs=[score], name="Discriminator")

# discriminator = build_discriminator((64,64,3))
# discriminator.summary()
def VGG_54():
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    vgg_54 = Model(inputs=[vgg.input], outputs=[vgg.get_layer('block5_conv4').output])
    return vgg_54

# print(vgg.get_layer('block5_conv4').name)
# vgg_54 = VGG_54()
# vgg_54.summary()

class MySRGAN(Model):
    def __init__(self, generator, discriminator, vgg):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator    
        self.vgg = vgg
        # self.vgg.trainable = False # just for safety

    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def content_loss(self, hr, sr):
        sr = (sr + 1)*127.5 # may need to remove this based on how i tackle the model later on, need to give input to preprocess_input in range [0, 255]
        hr = (hr + 1)*127.5
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)

        sr_features = self.vgg(sr)/12.75
        hr_features = self.vgg(hr)/12.75

        return tf.keras.losses.mean_squared_error(hr_features, sr_features)
    def train_step(self, batch):
        # latent vector
        # latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        lr = batch[0]
        hr = batch[1]

        # super-resolve and generate image
        sr = self.generator(lr)
        all_images = tf.concat([hr, sr], axis=0)

        #Assemble labels(reals are labeled 1, fakes are labeled 0)
        labels = tf.concat(
            [tf.ones((tf.shape(hr)[0], 1)),
             tf.zeros((tf.shape(sr)[0], 1)),], axis=0
        )
        # soft labels
        # labels += tf.random.uniform((tf.shape(labels)), minval=-0.05, maxval=0.05) # help the generator a little bit. Restricts the discriminator to learn too fast. Lets the generator to catch up

        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        
        # Train the discriminator
        with tf.GradientTape() as tape:
            score = self.discriminator(all_images)
            d_loss = self.discriminator_loss(labels, score)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Sample random points in the latent space
        # latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.ones((tf.shape(sr)[0], 1))
        
        # Train the generator (note that we should *not* update the weights of the discriminator)!
        with tf.GradientTape() as tape:
            sr = self.generator(lr)
            content_loss = self.content_loss(hr, sr)
            score = self.discriminator(sr)
            gen_loss = self.generator_loss(misleading_labels, score)
            perceptual_loss = content_loss + 0.001 * gen_loss # SRGAN speciality

        grads = tape.gradient(perceptual_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"Discriminator loss":d_loss, "Generator loss":gen_loss}
    
    def call(self, latent_vectors):
        x = self.generator(latent_vectors)
        # return self.discriminator(x)
        return x
    
    # def summary(self):
    #     super().summary()
    #     print("\n\n")
    #     self.generator.summary()
    #     print("\n\n")
    #     self.discriminator.summary()
    #     print("\n\n")
# def psnr_metrics(a, b):
#     return tf.image.psnr(a, b, max_val=)

def apply_random_crop(input, crop_size):
    h = input.shape[1]
    w = input.shape[2]

    x = tf.random.uniform(shape=(), maxval=(h-crop_size+1), dtype=tf.int32)
    y = tf.random.uniform(shape=(), maxval=(w-crop_size+1), dtype=tf.int32)

    crop = input[:, x:x+crop_size, y:y+crop_size]
    return crop