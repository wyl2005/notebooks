import tensorflow as tf


tf.app.flags.DEFINE_integer("max_step", 100, "模型训练的步数")
tf.app.flags.DEFINE_string("model_dir", "./test ", "模型的加载路径")
tf.app.flags.DEFINE_boolean("is_bool", True, "is_bool")
tf.app.flags.DEFINE_float("float_var", 3.14, "float var")

FLAGS = tf.app.flags.FLAGS


print(FLAGS.max_step)
print(FLAGS.model_dir)
print(FLAGS.is_bool)
print(FLAGS.float_var)
