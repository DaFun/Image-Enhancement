import shutil

import tensorflow as tf

import hdrnet.models as models
import hdrnet.utils as utils
import os
import numpy as np


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/checkpoint_dir/faces',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/hdrnet_output',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', 256,
                            """Needs to provide same value as in training.""")

FLAGS = tf.app.flags.FLAGS


def preprocess_image(image_buffer):
    '''
    Preprocess JPEG encoded bytes to 3D float Tensor and rescales
    it so that pixels are in a range of [-1, 1]
    :param image_buffer: Buffer that contains JPEG image
    :return: 4D image tensor (1, width, height,channels) with pixels scaled
             to [-1, 1]. First dimension is a batch size (1 is our case)
    '''

    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3, dct_method='INTEGER_ACCURATE')

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    #image = tf.expand_dims(image, 0)

    return image


def preprocess_low_image(image_buffer):
    """Preprocess JPEG encoded bytes to 3D float Tensor."""

    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3, dct_method='INTEGER_ACCURATE')
    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    # image = tf.image.central_crop(image, central_fraction=0.875)
    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_nearest_neighbor(image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
    image = tf.squeeze(image, [0])
    # Finally, rescale to [-1,1] instead of [0, 1)
    #image = tf.subtract(image, 0.5)
    #image = tf.multiply(image, 2.0)
    return image

# def cv_preprocess_low_image(image_buffer):
#     record_defaults = [['']] * (256 * 256 * 3)
#     flat = tf.decode_csv(image_buffer, record_defaults=record_defaults)
#     flat = tf.string_to_number(flat, out_type=tf.float32)
#     return tf.expand_dims(tf.reshape(flat, [256, 256, 3]), 0)
#
#
# def cv_preprocess_image(image_buffer):
#     #array = np.load(image_buffer)
#
#     record_defaults = [['']] * 1920
#     flat = tf.stack(tf.decode_csv(image_buffer, record_defaults=record_defaults))
#     flat = tf.string_to_number(flat, out_type=tf.float32)
#
#     #array = tf.convert_to_tensor(array, dtype=tf.float32)
#     return tf.expand_dims(tf.reshape(flat, [1920, 1080, 3]), 0)


def main(_):
    with tf.Graph().as_default():
        # Inject placeholder into the graph
        serialized_tf_example = tf.placeholder(tf.string, name='input_image')
        serialized_low_example = tf.placeholder(tf.string, name='low_image')
        #serialized_shape = tf.placeholder(tf.string, name='shape_image')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string)
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        tf_low_example = tf.parse_example(serialized_low_example, feature_configs)
        #tf_low_shape = tf.parse_example(serialized_shape, feature_configs)

        jpegs = tf_example['image/encoded']
        low_jpegs = tf_low_example['image/encoded']
        #shape_jpegs = tf_low_shape['image/encoded']

        full_images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
        low_images = tf.map_fn(preprocess_low_image, low_jpegs, dtype=tf.float32)
        #full_images = tf.squeeze(full_images, [0])
        #low_images = tf.squeeze(low_images, [0])

        # now the image shape is (1, ?, ?, 3)

        # Create model
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        metapath = ".".join([checkpoint_path, "meta"])
        tf.train.import_meta_graph(metapath)
        with tf.Session() as sess:
            model_params = utils.get_model_params(sess)
        mdl = getattr(models, model_params['model_name'])

        with tf.variable_scope('inference'):
            prediction = mdl.inference(low_images, full_images, model_params, is_training=False)
        output = tf.cast(255.0 * tf.squeeze(tf.clip_by_value(prediction, 0, 1)), tf.uint8)
        #output_img = tf.image.encode_png(tf.image.convert_image_dtype(output[0], dtype=tf.uint8))


        # Create saver to restore from checkpoints
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Restore the model from last checkpoints
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

            # (re-)create export directory
            export_path = os.path.join(
                tf.compat.as_bytes(FLAGS.output_dir),
                tf.compat.as_bytes(str(FLAGS.model_version)))
            if os.path.exists(export_path):
                shutil.rmtree(export_path)

            # create model builder
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # create tensors info
            predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(jpegs)
            predict_tensor_low_info = tf.saved_model.utils.build_tensor_info(low_jpegs)
            #predict_tensor_shape_info = tf.saved_model.utils.build_tensor_info(shape_jpegs)
            predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(output)

            # build prediction signature
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': predict_tensor_inputs_info,
                            'low': predict_tensor_low_info},
                            #'shape': predict_tensor_shape_info},
                    outputs={'result': predict_tensor_scores_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

            # save the model
            #legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images': prediction_signature
                })
                #legacy_init_op=legacy_init_op)

            builder.save()

    print("Successfully exported hdr model version '{}' into '{}'".format(
        FLAGS.model_version, FLAGS.output_dir))

if __name__ == '__main__':
    tf.app.run()