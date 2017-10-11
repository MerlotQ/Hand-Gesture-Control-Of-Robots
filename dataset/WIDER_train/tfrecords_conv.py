import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
from object_detection.utils import dataset_util
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto.
  """
  if not isinstance(value, list):
      value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = example[0] # Image height
  width = example[1] # Image width
  filename = example[2] # Filename of the image. Empty if image is not from file
  encoded_image_data = ((example[3]).tobytes()) # Encoded image bytes
  image_format = example[4] # b'jpeg' or b'png'

  xmins = [example[5]/width] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [example[6]/width] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [example[7]/height] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [example[8]/height] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [example[9]] # List of string class name of bounding box (1 per box)
  classes = [example[10]] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def bounding_box_scaler(bbox, img):
    h, w, _ = img.shape
    bbox[0] = int(float(bbox[0])*w)
    bbox[1] = int(float(bbox[1])*h)
    bbox[2] = int(float(bbox[2])*w)
    bbox[3] = int(float(bbox[3])*h)
    return (bbox, h, w)

#output_path = "output_tfrecords"
#print 
writer = tf.python_io.TFRecordWriter("training_tfrecord.record")
writer1 = tf.python_io.TFRecordWriter("eval_tfrecord.record")
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
def main(_):
  i = 0
  with open("wider_face_train.txt", "r+") as f:
      lines = f.readlines()
      np.random.shuffle(lines)
      idx = 0
      for line in tqdm(lines):
          if idx <200:
            words = line.strip().split("    ")
            image_name = words[0]
            img = cv2.imread("/home/tlokeshkumar/Desktop/gesture_recognition/finget_tip_detection/darkflow/dataset/WIDER_train/images/"+image_name)
            bbox, h, w = bounding_box_scaler([words[1], words[2], words[3], words[4]], img)
            xmin = bbox[0]
            xmax = bbox[2]
            ymin = bbox[1]
            ymax = bbox[3]
            example = [h, w,image_name,img,b'jpeg', xmin, xmax, ymin, ymax, 'hands', 1]
            tf_example = create_tf_example(example)
            writer1.write(tf_example.SerializeToString())
            idx = idx + 1
          else:
            words = line.strip().split("    ")
            image_name = words[0]
            img = cv2.imread("/home/tlokeshkumar/Desktop/gesture_recognition/finget_tip_detection/darkflow/dataset/WIDER_train/images/"+image_name)
            bbox, h, w = bounding_box_scaler([words[1], words[2], words[3], words[4]], img)
            xmin = bbox[0]
            xmax = bbox[2]
            ymin = bbox[1]
            ymax = bbox[3]
            example = [h, w,image_name,img,b'png', xmin, xmax, ymin, ymax, 'hands', 1]
            tf_example = create_tf_example(example)
            writer.write(tf_example.SerializeToString())
            idx = idx + 1

  writer.close()
  writer1.close()
  

if __name__ == '__main__':
  tf.app.run()
        
        


# TODO(user): Write code to read in your dataset to examples variable
