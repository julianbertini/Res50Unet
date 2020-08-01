import tensorflow as tf
import numpy as np



def main():
    # Create a simple functional model
    inputs = tf.keras.Input(shape=(1, 128, 128, 3), name="digits")
    x = tf.keras.layers.MaxPool3D(name="conv1", pool_size=(1, 2, 2))(inputs)
    x = tf.keras.layers.BatchNormalization(name="conv2_block2_3_bn")(x)
    outputs = tf.keras.layers.Conv3D(name="conv1_1",filters=2, kernel_size=[1,2,2])(x)
    model1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

    
    # Create a simple functional model
    inputs = tf.keras.Input(shape=(128, 128, 3), name="digits")
    x = tf.keras.layers.Dropout(0.5)(inputs)
    x = tf.keras.layers.MaxPool2D(name="conv1", pool_size=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(name="conv2_block2_3_bn")(x)

    outputs = tf.keras.layers.Conv2D(name="conv2",filters=2, kernel_size=[2,2])(x)
    model2 = tf.keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")
    
    # create a layer with from_config
    l1 = tf.keras.layers.Conv2D(filters=2, kernel_size=[1,1])    
    l2 = tf.keras.layers.Conv2D(filters=4, kernel_size=[3,3]).from_config(l1.get_config())   
    #print(l1.get_config())
    #print(l2.get_config())
 
    conv2_weights = model2.get_layer("conv2").get_weights()
    conv1_weights = model1.get_layer("conv1").get_weights()
    

    weights = model2.weights
    weights1 = model1.weights
    print(len(weights))
    print(weights)
    print(weights1)
    
    exp_weights = [] 
    for weight in weights:
      #print(weight)
      if "conv" in weight.name and 'bias' not in weight.name and 'bn' not in weight.name:
        exp_weight = np.expand_dims(weight.numpy(),axis=0)
        exp_weights.append(exp_weight)
      else:
        exp_weights.append(weight.numpy())

    #print(exp_weights)
    
    model1.set_weights(exp_weights)  
    
    # Expand the dimensions
    #expanded_conv2_weights = np.expand_dims(conv2_weights[0], axis=0)
    # convert back to list of numpy arrays
    #conv2_weights[0] = expanded_conv2_weights
    
    #model1.set_weights(conv2_weights)    

if __name__ == "__main__":
    main()
