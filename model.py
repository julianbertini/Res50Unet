import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib
#from viz import Visualize


class Res50Unet():

    def __init__(self, ResNet50_2D):
        """
        """
        self.ResNet50_2D = ResNet50_2D
        self.net = None
    
    def load_weights(self, model):
      """
      Loads the weights from the pre-trained ResNet50 2D into the ResNet50 3D modified version.

      """    
      weights = self.ResNet50_2D.weights
      this_weights = model.weights
      non_expandable = ['bn', 'bias', 'relu', 'pad', 'pool']
          
      for layer in self.ResNet50_2D.layers:
        exp_weights = []
        for weight in layer.weights:
          if np.array([False for t in non_expandable if t in weight.name]).all():
            exp_weights.append(np.expand_dims(weight.numpy(),axis=0))
          else:
            exp_weights.append(weight.numpy())

        model.get_layer(layer.name).set_weights(exp_weights)

    
    def upsample_block(self, x, nfeatures, name):
        """ Create the upsample block
        """
        x = self.intra_slice_block(x, nfeatures, name+'_upsample')
        x = tf.keras.layers.Conv3DTranspose(name=name+"_upsample_trans_conv",
                                     kernel_size=(1,1,1),
                                     filters=nfeatures,
                                     strides=(1,2,2))(x)
        x = tf.keras.layers.BatchNormalization(name=name+"_upsample_bn",
                                        axis=-1)(x)
        x = tf.keras.layers.Activation(name=name+'_upsample_relu', activation='relu')(x)
      
        return x
                                    
    
    def inter_slice_block(self, x, nfeatures, name):
        """ Create the 3x1x1 conv block (purple block)
        """
        x = tf.keras.layers.Conv3D(name=name+'_inter_conv',
                            kernel_size=(3,1,1),
                            padding='same',
                            filters=nfeatures)(x)
        x = tf.keras.layers.BatchNormalization(name=name+'_inter_bn',
                                        axis=-1)(x)
        x = tf.keras.layers.Activation(name=name+'_inter_relu', activation='relu')(x)
        
        return x
                          
    def intra_slice_block(self, x, nfeatures, name):
        """ Create the 1x3x3 conv block (blue block)
        """
        x = tf.keras.layers.Conv3D(name=name+'_intra_conv',
                            kernel_size=(1,3,3),
                            padding='same',
                            filters=nfeatures)(x)
        x = tf.keras.layers.BatchNormalization(name=name+'_intra_bn',
                                        axis=-1)(x)
        x = tf.keras.layers.Activation(name=name+'_intra_relu', activation='relu')(x)
      
        return x

    def prediction(self, x, nfeatures, name):
        """ Last prediction block
        """
        x = tf.keras.layers.Conv3D(name=name+'_conv',
                                   kernel_size=(1,1,1),
                                   filters=nfeatures)(x)
        x = tf.nn.softmax(x) 
        
        return x
    
    def decoder(self, x, x_skip1, x_skip2, x_skip3, name='decoder'):

        # Stage 6
        x = self.inter_slice_block(x, 2048, name + '_stage6')
        x = self.intra_slice_block(x, 2048, name + '_stage6')
        x = self.upsample_block(x, 1024, name + '_stage6') 
        # Skip connection
        x = tf.add(x, x_skip1)
        
        # Stage 7 
        x = self.upsample_block(x, 512, name + '_stage7')
        x = self.inter_slice_block(x, 512, name + '_stage7')
        # Skip connection
        x = tf.add(x, x_skip2)
        
        # Stage 8
        x = self.upsample_block(x, 256, name + '_stage8')
        # Skip connection
        x = tf.add(x, x_skip3)
        
        # Stage 9 
        x = self.upsample_block(x, 128, name + '_stage9a')
        x = self.inter_slice_block(x, 128, name + '_stage9a')

        x = self.intra_slice_block(x, 64, name + '_stage9b')
        x = self.inter_slice_block(x, 64, name + '_stage9b')
       
        # Stage 10 
        x = self.upsample_block(x, 64, name + '_stage10a')
        x = self.intra_slice_block(x, 64, name + '_stage10a')
        x = self.inter_slice_block(x, 64, name + '_stage10a')

        x = self.inter_slice_block(x, 16, name + '_stage10b')
        x = self.intra_slice_block(x, 16, name + '_stage10b')
      
        x = self.prediction(x, 4, name + '_pred')
        
        return x 
        
        
  
    def encoder(self, input_shape):
        """ Create the Res50Unet model using the functional API 
            (mainly to mimick how the pre-trained Keras model is made)

            Returns:
              model -- an instance of the tf.keras.Model class
        """

        ## STAGE 1 ##

        x_input = tf.keras.Input(shape=input_shape, name="input_1", dtype=tf.float32)
        x = tf.keras.layers.ZeroPadding3D(
            name="conv1_pad", padding=((0, 0), (3, 3), (3, 3)))(x_input)
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

        # Skip connection
        x_skip3 = x

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
        
        # Skip connection 
        x_skip2 = x

    
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
      
        # Skip connection
        x_skip1 = x


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
        
        return tf.keras.Model(inputs=x_input, outputs=[x, x_skip1, x_skip2, x_skip3], name="encoder")
        
    
    def create_model(self, input_shape):
        
        
        inputs = tf.keras.Input(shape=input_shape, name="main_model_input", dtype=tf.float32)
        
        ## Encoder ##
        # Create encoder model
        encoder_model = self.encoder(input_shape)
        # Load pre-trained ResNet50 weights 
        self.load_weights(encoder_model)
        # Freeze encoder weights
        encoder_model.trainable = False
        # Call encoder model
        x, x_skip1, x_skip2, x_skip3 = encoder_model(inputs, training=False) 
        
        ## Decoder ##
        output  = self.decoder(x, x_skip1, x_skip2, x_skip3)

        self.net = tf.keras.Model(inputs=inputs, outputs=output, name="main_model")  


def main():

    resnet = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=[128, 128, 3])

    model = Res50Unet(resnet)
    model.create_model([24, 128, 128, 3])
    
    model.net.summary()
    #tf.keras.utils.plot_model(decoder, show_shapes=True)




if __name__ == "__main__":
    main()
