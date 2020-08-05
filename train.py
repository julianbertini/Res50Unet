import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib

from viz import Visualize
from data_preprocessor import *
from model import *


class DisplayCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


class DiceScore(tf.keras.metrics.Metric):

    def __init__(self, name='dice_score', **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='score', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        gdl = GeneralizedDiceLoss()
        self.score = (gdl(y_true, y_pred) - 1) * -1

    def result(self):
        return self.score


class GeneralizedDiceLoss(tf.keras.losses.Loss):

    def labels_to_one_hot(self, ground_truth, num_classes=1):
        """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
        if isinstance(num_classes, tf.Tensor):
            num_classes_tf = tf.cast(num_classes, dtype=tf.int32)
        else:
            num_classes_tf = tf.constant(num_classes, tf.int32)
        input_shape = tf.shape(ground_truth)
        output_shape = tf.concat(
            [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

        #if num_classes == 1:
        #    # need a sparse representation?
        #    return tf.reshape(ground_truth, output_shape)

        # squeeze the spatial shape
        ground_truth = tf.reshape(ground_truth, (-1,))
        # shape of squeezed output
        dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

        # create a rank-2 sparse tensor
        ground_truth = tf.cast(ground_truth, dtype=tf.int64)
        ids = tf.range(tf.cast(dense_shape[0], dtype=tf.int64), dtype=tf.int64)
        ids = tf.stack([ids, ground_truth], axis=1)
        one_hot = tf.sparse.SparseTensor(
            indices=ids,
            values=tf.ones_like(ground_truth, dtype=tf.float32),
            dense_shape=tf.cast(dense_shape, dtype=tf.int64))

        # resume the spatial dims
        one_hot = tf.sparse.reshape(one_hot, output_shape)

        return one_hot

    def call(self, ground_truth, prediction, weight_map=None):
        """
                `weight_map` represents same thing as `loss_weight` in tf
                except that we apply it here directly instead of passing in
                through the model.fit attribute `loss_weight`
        """
        type_weight = "Square"

        prediction = tf.cast(prediction, tf.float32)
        # Calculate softmax for prediction. Default is to reduce based on last
        # dimension.
        prediction = tf.nn.softmax(prediction)

        if len(ground_truth.shape) == len(prediction.shape):
            ground_truth = ground_truth[..., -1]
        one_hot = self.labels_to_one_hot(
            ground_truth, num_classes=tf.shape(prediction)[-1])
        if weight_map is not None:
            num_classes = prediction.shape[1].value
            weight_map_nclasses = tf.tile(
                tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
            ref_vol = tf.sparse.reduce_sum(
                weight_map_nclasses * one_hot, axis=0)

            intersect = tf.sparse.reduce_sum(
                weight_map_nclasses * one_hot * prediction, axis=0)
            seg_vol = tf.math.reduce_sum(
                tf.multiply(weight_map_nclasses, prediction), 0)
        else:
            ref_vol = tf.sparse.reduce_sum(one_hot, axis=0)
            intersect = tf.sparse.reduce_sum(one_hot * prediction,
                                             axis=0)
            seg_vol = tf.math.reduce_sum(prediction, 0)
        if type_weight == 'Square':
            weights = tf.math.reciprocal(tf.square(ref_vol))
        elif type_weight == 'Simple':
            weights = tf.math.reciprocal(ref_vol)
        elif type_weight == 'Uniform':
            weights = tf.ones_like(ref_vol)
        else:
            raise ValueError("The variable type_weight \"{}\""
                             "is not defined.".format(type_weight))
        new_weights = tf.where(tf.math.is_inf(
            weights), tf.zeros_like(weights), weights)
        weights = tf.where(tf.math.is_inf(weights), tf.ones_like(weights) *
                           tf.math.reduce_max(new_weights), weights)
        generalised_dice_numerator = \
            2 * tf.math.reduce_sum(tf.math.multiply(weights, intersect))
        generalised_dice_denominator = tf.math.reduce_sum(
            tf.multiply(weights, tf.math.maximum(seg_vol + ref_vol, 1)))
        generalised_dice_score = \
            generalised_dice_numerator / generalised_dice_denominator
        generalised_dice_score = tf.where(tf.math.is_nan(generalised_dice_score), 1.0,
                                          generalised_dice_score)
        return 1 - generalised_dice_score


def main():

  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

  dp = DataPreprocessor()

  train_img_dir = pathlib.Path(dp.path_to_train_imgs)
  val_img_dir = pathlib.Path(dp.path_to_val_imgs)

  TRAIN_LENGTH = len(list(train_img_dir.glob("*.nii.gz")))
  VAL_LENGTH = len(list(val_img_dir.glob("*.nii.gz")))

    # The tf.data.Dataset API supports writing descriptive and efficient input pipelines.
    #   - Create a source dataset from your input data
    #   - Apply dataset transformation to preprocess the data
    #   - Iterate over the dataset and process the elements
    # Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory at once.
  dataset = tf.data.Dataset

    # Generates a dataset list of all files matching one or more glob patters.
    # It will return filenames in a non-deterministic random shuffle order.
  train_img_ds = dataset.list_files(str(train_img_dir/"*"))
  val_img_ds = dataset.list_files(str(val_img_dir/"*"))

    # Takes in the imgs paths and does all data preprocesesing, returning shuffled batches ready for training
  train_dataset = dp.prepare_for_training(train_img_ds, cache='/home/jbertini/scratch-midway2/Python/Res50Unet/model_cache')
  val_dataset = dp.prepare_for_testing(val_img_ds, purpose="val")
    # Setup training

  EPOCHS = 50
  STEPS_PER_EPOCH = 100 #TRAIN_LENGTH // BATCH_SIZE
  VAL_STEPS = VAL_LENGTH // BATCH_SIZE

  print("STEPS PER EPOCH")
  print(STEPS_PER_EPOCH)

  resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=[128, 128, 3])
  model = Res50Unet(resnet)
  model.create_model(IMG_PATCH_SIZE)

  model.net.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
          loss=GeneralizedDiceLoss(),
          metrics=[DiceScore()])

  model_history = model.net.fit(train_dataset,
                  epochs=EPOCHS,
                  steps_per_epoch=STEPS_PER_EPOCH,
                  validation_data=val_dataset,
                  validation_steps=VAL_STEPS,
                  validation_freq=1)


  tf.saved_model.save(model.net, str(os.path.basename(__file__)))

  loss = model_history.history['loss']
  val_loss = model_history.history['val_loss']

  epochs = range(EPOCHS)

  plt.figure()
  plt.plot(epochs, loss, 'r', label='Training loss')
  plt.plot(epochs, val_loss, 'bo', label='Validation loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss Value')
  plt.ylim([0, 1])
  plt.legend()
  plt.savefig(str(os.path.basename(__file__)) + ".png")


    # Visualize a patch
    # viz = Visualize()

    # for image, label in train_dataset.take(1):
    #    print("Image shape: ", image.shape)
    #    print("Label: ", label.shape)
    #    print(np.nanmax(label.numpy()))
    #
    #    viz.multi_slice_viewer([image.numpy()[0,:,:,:,0], label.numpy()[0,:,:,:]])
    #    plt.show()


if __name__ == "__main__":
    main()
