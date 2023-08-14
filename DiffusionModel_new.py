import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Conv1D, MaxPool1D, UpSampling1D
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import dataset
from keras import backend as K
import DG


def preprocess_fn(entry):
    image = float(entry['image']) / 255.
    image = tf.where(image > .5, 1.0, 0.0)
    image = tf.cast(image, tf.float32)
    return image

def corrupt(xs, amount):
    noise = tf.random.uniform(xs.shape, dtype=tf.float64)
    amount = tf.reshape(amount, (-1, 1)) # Sort shape so broadcasting works
    #xs = tf.cast(xs, tf.float64)
    amount = tf.cast(amount, tf.float64)

    return xs*(1-amount) + noise * amount


bs = 256
batch_size = 4
seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)

#ds = tfds.load('mnist')

#merge_ds = ds['train'].concatenate(ds['test'])


class BasicUNet(Model):
    def __init__(self, input_shape, batch_size=batch_size):
        super(BasicUNet, self).__init__(name='basic-unet')
        self.inputs = layers.Input(shape=input_shape)
        
        # Encoder
        self.s1, self.p1 = self.encoder_block(self.inputs, 32)
        self.s2, self.p2 = self.encoder_block(self.p1, 64)
        self.s3, self.p3 = self.encoder_block(self.p2, 128)
        self.b4 = self.conv_block(self.p3, 128)
        
        # Decoder
        self.d3 = self.decoder_block(self.b4, self.s3, 128)
        self.d2 = self.decoder_block(self.d3, self.s2, 64)
        self.d1 = self.decoder_block(self.d2, self.s1, 32)
        
        self.outputs = layers.Conv1D(1, 1, activation='relu')(self.d1)
        self.model = Model(self.inputs, self.outputs)
        return None

    def call(self, x):
        return self.model(x)
        
    def conv_block(self, inputs, num_filters):
        x = layers.Conv1D(num_filters, 1, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(num_filters, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def encoder_block(self, inputs, num_filters):
        x = self.conv_block(inputs, num_filters)
        p = layers.MaxPooling1D(1)(x)
        return x, p

    def decoder_block(self, inputs, skip_features, num_filters):
        x = layers.Conv1DTranspose(num_filters, 1, strides=1, padding='same')(inputs)
        x = layers.concatenate([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x


# Usage:
net = BasicUNet((dataset.WINDOW*dataset.SIZE, 2))
#model.summary()


#batch = merge_ds.map(lambda x: x['image']).take(8)
#xs = tf.convert_to_tensor(list(batch), dtype=tf.float32)
train_ds, test_ds = dataset.prepare_dataset("/home/leo/DiffusionModel/RNA_2/",batch_size=batch_size) #merge_ds.map(preprocess_fn).shuffle(1024).batch(bs).prefetch(tf.data.AUTOTUNE)
amount = tf.linspace(0.0, 1.0, batch_size) # Left to right -> more corruption
#noised_xs = corrupt(xs, amount)
#print(noised_xs.shape)

#net = BasicUNet()
#x = tf.random.uniform((8, 28, 28, 1))

epochs = 3000

# Define a loss finction
loss_fn = tf.keras.losses.MeanSquaredError()

# Define an optimizer
opt = tf.keras.optimizers.Adam(learning_rate=1e-4) 

# Record the losses
losses, avg_losses = [], []

# Iterate over epochs.
for epoch in tqdm(range(epochs)):

    # Iterate over the batches of the dataset.
    for step, xb in enumerate(train_ds):
        with tf.GradientTape() as tape:
            labels = xb[:,1,:]
            xb = xb[:,0,:]
#            print(xb)
#            print(xb.shape)
            # Create noisy version of the input
            noise_amount = tf.random.uniform((xb.shape[0],))
            noisy_xb = corrupt(xb, noise_amount)
            noised_tensor = tf.concat([noisy_xb,labels],axis=1)
#            print(noised_tensor.shape)
            noised_tensor = tf.reshape(noised_tensor, (batch_size, dataset.WINDOW*dataset.SIZE, 2)) #3000 - size of sequence, 2 - batch size
            # Get the model prediction
            pred = net(noised_tensor)
#            print(pred)
#            print(xb)
            # Calculate the loss to determine how close the output is to the input
            loss = loss_fn(pred, xb)

        grads = tape.gradient(loss, net.trainable_weights)
        opt.apply_gradients(zip(grads, net.trainable_weights))
        # Store the loss
        losses.append(loss.numpy())
    # Calculate the average loss for this epoch
    avg_loss = sum(losses[-len(xb):])/len(xb)
    avg_losses.append(avg_loss)

print(avg_losses[-1])
