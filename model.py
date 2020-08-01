import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib
#from viz import Visualize


class Res50Unet():

    def __init__(self, name, ResNet50_2D, num_classes=2):
        """
        """
        self.ResNet50_2D = ResNet50_2D
        self.model = None
    
    def load_weights(self):
      """
      Loads the weights from the pre-trained ResNet50 2D into the ResNet50 3D modified version.

      """    
      weights = self.ResNet50_2D.weights
      this_weights = self.model.weights
      non_expandable = ['bn', 'bias', 'relu', 'pad', 'pool']
          
      for layer in self.ResNet50_2D.layers:
        exp_weights = []
        for weight in layer.weights:
          if np.array([False for t in non_expandable if t in weight.name]).all():
            exp_weights.append(np.expand_dims(weight.numpy(),axis=0))
          else:
            exp_weights.append(weight.numpy())

        self.model.get_layer(layer.name).set_weights(exp_weights)
                          

    def create_model(self):
        """ Create the Res50Unet model using the functional API 
            (mainly to mimick how the pre-trained Keras model is made)

            Returns:
              model -- an instance of the tf.keras.Model class
        """

        ## STAGE 1 ##

        inputs = tf.keras.Input(shape=[1, 128, 128, 3], name="input_1", dtype=tf.float32)
        x = tf.keras.layers.ZeroPadding3D(
            name="conv1_pad", padding=((0, 0), (3, 3), (3, 3)))(inputs)
        x = tf.keras.layers.Conv3D(name="conv1_conv", filters=64, kernel_size=(
            1, 7, 7), strides=(1, 2, 2), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv1_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv1_relu").get_config())(x)
        x = tf.keras.layers.ZeroPadding3D(
            name="pool1_pad", padding=((0, 0), (1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPool3D(
            name="pool1_pool", pool_size=(1, 3, 3), strides=(1, 2, 2))(x)

        ### STAGE 2 (Conv2) ###

        ## Block 1 (Conv1) ##

        # Residual path
        x_skip = tf.keras.layers.Conv3D(
            name="conv2_block1_0_conv", filters=256, kernel_size=(1, 1, 1), activation="linear")(x)
        x_skip = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block1_0_bn")(x_skip)

        # Main path
        x = tf.keras.layers.Conv3D(name="conv2_block1_1_conv", filters=64, kernel_size=(
            1, 1, 1), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block1_1_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block1_1_relu").get_config())(x)

        x = tf.keras.layers.Conv3D(name="conv2_block1_2_conv", filters=64, kernel_size=(
            1, 3, 3), padding="same", activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block1_2_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block1_2_relu").get_config())(x)

        x = tf.keras.layers.Conv3D(name="conv2_block1_3_conv", filters=256, kernel_size=(
            1, 1, 1), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block1_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name="conv2_block1_add")([x, x_skip])
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block1_out").get_config())(x)

        ## Identity Block 2 ##

        # Residual path
        x_skip = x

        # Main path
        x = tf.keras.layers.Conv3D(name="conv2_block2_1_conv", filters=64, kernel_size=(
            1, 1, 1), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block2_1_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block2_1_relu").get_config())(x)

        x = tf.keras.layers.Conv3D(name="conv2_block2_2_conv", filters=64, kernel_size=(
            1, 3, 3), padding="same", activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block2_2_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block2_2_relu").get_config())(x)

        x = tf.keras.layers.Conv3D(name="conv2_block2_3_conv", filters=256, kernel_size=(
            1, 1, 1), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block2_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name="conv2_block2_add")([x, x_skip])
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block2_out").get_config())(x)

        ## Identity Block 3 ##

        # Residual path
        x_skip = x

        # Main path
        x = tf.keras.layers.Conv3D(name="conv2_block3_1_conv", filters=64, kernel_size=(
            1, 1, 1), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block3_1_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block3_1_relu").get_config())(x)

        x = tf.keras.layers.Conv3D(name="conv2_block3_2_conv", filters=64, kernel_size=(
            1, 3, 3), padding="same", activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block3_2_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block3_2_relu").get_config())(x)

        x = tf.keras.layers.Conv3D(name="conv2_block3_3_conv", filters=256, kernel_size=(
            1, 1, 1), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv2_block3_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name="conv2_block3_add")([x, x_skip])
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv2_block3_out").get_config())(x)

        ### STAGE 3 (Conv3) ###

        ## Block 1 ##

        # Residual path
        x_skip = tf.keras.layers.Conv3D(
            name="conv3_block1_0_conv", filters=512, kernel_size=(1, 1, 1), strides=(1, 2, 2), activation="linear")(x)
        x_skip = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block1_0_bn")(x_skip)

        # Main path
        x = tf.keras.layers.Conv3D(name="conv3_block1_1_conv", filters=128, kernel_size=(
            1, 1, 1), strides=(1, 2, 2), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block1_1_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv3_block1_1_relu").get_config())(x)

        x = tf.keras.layers.Conv3D(name="conv3_block1_2_conv", filters=128, kernel_size=(
            1, 3, 3), padding="same", activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block1_2_bn")(x)
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv3_block1_2_relu").get_config())(x)

        x = tf.keras.layers.Conv3D(name="conv3_block1_3_conv", filters=512, kernel_size=(
            1, 1, 1), activation="linear")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block1_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name="conv3_block1_add")([x, x_skip])
        x = tf.keras.layers.Activation(activation="relu").from_config(
            self.ResNet50_2D.get_layer("conv3_block1_out").get_config())(x)
        
        ## Block 2 ##
        
        # Residual path
        x_skip = x

        # Main path
        x = tf.keras.layers.Conv3D(name='conv3_block2_1_conv', filters=128, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block2_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv3_block2_1_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv3_block2_2_conv', filters=128, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block2_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv3_block2_2_relu').get_config())(x)      
        
        x = tf.keras.layers.Conv3D(name='conv3_block2_3_conv', filters=512, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block2_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name='conv3_block2_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv3_block2_out').get_config())(x)
        
        ## Block 3 ##
        
        # Residual path
        x_skip = x
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv3_block3_1_conv', filters=128, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block3_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv3_block3_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv3_block3_2_conv', filters=128, padding='same', kernel_size=(1,3,3), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block3_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv3_block3_2_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv3_block3_3_conv', filters=512, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block3_3_bn")(x)
        
        # Combine main + residual
        x = tf.keras.layers.Add(name='conv3_block3_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv3_block3_out').get_config())(x)
      
        ## Block 4 ##

        # Residual path
        x_skip = x
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv3_block4_1_conv', filters=128, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block4_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv3_block4_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv3_block4_2_conv', filters=128, padding='same', kernel_size=(1,3,3), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block4_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv3_block4_2_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv3_block4_3_conv', filters=512, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv3_block4_3_bn")(x)
        
        # Combine main + residual
        x = tf.keras.layers.Add(name='conv3_block4_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv3_block4_out').get_config())(x)

    
        ### STAGE 4 (Conv4) ###
      
        ## Block 1 ##

        # Residual path
        x_skip = tf.keras.layers.Conv3D(
            name="conv4_block1_0_conv", filters=1024, kernel_size=(1, 1, 1), strides=(1, 2, 2), activation="linear")(x)
        x_skip = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block1_0_bn")(x_skip)
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv4_block1_1_conv', filters=256, kernel_size=(1,1,1), strides=(1,2,2), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block1_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block1_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv4_block1_2_conv', filters=256, padding='same', kernel_size=(1,3,3), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block1_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block1_2_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv4_block1_3_conv', filters=1024, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block1_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name='conv4_block1_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv4_block1_out').get_config())(x)
        
        ## Block 2 ##

        # Residual path
        x_skip = x

        # Main path
        x = tf.keras.layers.Conv3D(name='conv4_block2_1_conv', filters=256, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block2_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block2_1_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv4_block2_2_conv', filters=256, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block2_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block2_2_relu').get_config())(x)      
        
        x = tf.keras.layers.Conv3D(name='conv4_block2_3_conv', filters=1024, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block2_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name='conv4_block2_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv4_block2_out').get_config())(x)
      
        
        ## Block 3 ## 
         
        # Residual path
        x_skip = x
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv4_block3_1_conv', filters=256, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block3_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block3_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv4_block3_2_conv', filters=256, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block3_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block3_2_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv4_block3_3_conv', filters=1024, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block3_3_bn")(x)
        
        # Combine main + residual
        x = tf.keras.layers.Add(name='conv4_block3_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv4_block3_out').get_config())(x)
      
        ## Block 4 ##
        
        # Residual path
        x_skip = x
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv4_block4_1_conv', filters=256, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block4_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block4_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv4_block4_2_conv', filters=256, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block4_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block4_2_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv4_block4_3_conv', filters=1024, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block4_3_bn")(x)
        
        # Combine main + residual
        x = tf.keras.layers.Add(name='conv4_block4_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv4_block4_out').get_config())(x)
      
        ## Block 5 ##

        # Residual path
        x_skip = x
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv4_block5_1_conv', filters=256, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block5_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block5_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv4_block5_2_conv', filters=256, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block5_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block5_2_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv4_block5_3_conv', filters=1024, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block5_3_bn")(x)
        
        # Combine main + residual
        x = tf.keras.layers.Add(name='conv4_block5_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv4_block5_out').get_config())(x)
        
        ## Block 6 ##

        # Residual path
        x_skip = x
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv4_block6_1_conv', filters=256, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block6_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block6_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv4_block6_2_conv', filters=256, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block6_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv4_block6_2_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv4_block6_3_conv', filters=1024, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv4_block6_3_bn")(x)
        
        # Combine main + residual
        x = tf.keras.layers.Add(name='conv4_block6_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv4_block6_out').get_config())(x)


        ### STAGE 5 (Conv5) ###
        
        ## Block 1 ##

        # Residual path
        x_skip = tf.keras.layers.Conv3D(
            name="conv5_block1_0_conv", filters=2048, kernel_size=(1, 1, 1), strides=(1, 2, 2), activation="linear")(x)
        x_skip = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block1_0_bn")(x_skip)
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv5_block1_1_conv', filters=512, kernel_size=(1,1,1), strides=(1,2,2), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block1_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv5_block1_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv5_block1_2_conv', filters=512, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block1_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv5_block1_2_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv5_block1_3_conv', filters=2048, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block1_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name='conv5_block1_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv5_block1_out').get_config())(x)
        
        ## Block 2 ##
         
        # Residual path
        x_skip = x

        # Main path
        x = tf.keras.layers.Conv3D(name='conv5_block2_1_conv', filters=512, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block2_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv5_block2_1_relu').get_config())(x)
      
        x = tf.keras.layers.Conv3D(name='conv5_block2_2_conv', filters=512, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block2_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv5_block2_2_relu').get_config())(x)      
        
        x = tf.keras.layers.Conv3D(name='conv5_block2_3_conv', filters=2048, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block2_3_bn")(x)

        # Combine main + residual
        x = tf.keras.layers.Add(name='conv5_block2_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv5_block2_out').get_config())(x)
      
        ## Block 3 ##

        # Residual path
        x_skip = x
        
        # Main path
        x = tf.keras.layers.Conv3D(name='conv5_block3_1_conv', filters=512, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block3_1_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv5_block3_1_relu').get_config())(x)

        x = tf.keras.layers.Conv3D(name='conv5_block3_2_conv', filters=512, kernel_size=(1,3,3), padding='same', activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block3_2_bn")(x)
        x = tf.keras.layers.Activation(activation='relu').from_config(self.ResNet50_2D.get_layer('conv5_block3_2_relu').get_config())(x)
      
      
        x = tf.keras.layers.Conv3D(name='conv5_block3_3_conv', filters=2048, kernel_size=(1,1,1), activation='linear')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-05, name="conv5_block3_3_bn")(x)
        
        # Combine main + residual
        x = tf.keras.layers.Add(name='conv5_block3_add')([x, x_skip])
        x = tf.keras.layers.Activation(activation='linear').from_config(self.ResNet50_2D.get_layer('conv5_block3_out').get_config())(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=x, name="test_model")  



def main():

    decoder = tf.keras.applications.ResNet50(
        include_top=False, input_shape=[128, 128, 3])

    model = Res50Unet("test", decoder)
    model.create_model()
    model.load_weights()
    
    model.model.summary()
    
    #tf.keras.utils.plot_model(decoder, show_shapes=True)

    # Copy weights from functional_model to subclassed_model.


    #for layer in model.model.layers:
    #    print(layer.get_config())
    #    print()


if __name__ == "__main__":
    main()
