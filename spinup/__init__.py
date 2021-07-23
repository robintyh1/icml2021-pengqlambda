# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__
