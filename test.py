import tensorflow as tf
import numpy as np



def main():
    # Create a simple functional model
    inputs = tf.keras.Input(shape=(1, 128, 128, 3), name="digits")
    x = tf.keras.layers.MaxPool3D(name="conv1", pool_size=(1, 2, 2))(inputs)
    outputs = tf.keras.layers.Conv3D(name="conv1_1",filters=2, kernel_size=[1,2,2])(x)
    model1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

    
    # Create a simple functional model
    inputs = tf.keras.Input(shape=(128, 128, 3), name="digits")
    outputs = tf.keras.layers.Conv2D(name="conv2",filters=2, kernel_size=[2,2])(inputs)
    model2 = tf.keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")
    
    # create a layer with from_config
    l1 = tf.keras.layers.Conv2D(filters=2, kernel_size=[1,1])    
    l2 = tf.keras.layers.Conv2D(filters=4, kernel_size=[3,3]).from_config(l1.get_config())   
    print(l1.get_config())
    print(l2.get_config())
 
    conv2_weights = model2.get_layer("conv2").get_weights()
    conv1_weights = model1.get_layer("conv1").get_weights()
    weights = model1.weights
    #print(weights)
    # Expand the dimensions
    expanded_conv2_weights = np.expand_dims(conv2_weights[0], axis=0)
    # convert back to list of numpy arrays
    conv2_weights[0] = expanded_conv2_weights
    
    model1.set_weights(conv2_weights)    

if __name__ == "__main__":
    main()
