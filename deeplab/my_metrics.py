# -*- coding:utf-8 -*-
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import confusion_matrix
 
 
def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""
    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[
            ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
        ],
        validate_shape=validate_shape,
        name=name)
 
 
def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.
    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.
    Args:
      labels: A `Tensor` of ground truth labels with shape [batch size] and of
        type `int32` or `int64`. The tensor will be flattened if its rank > 1.
      predictions: A `Tensor` of prediction results for semantic labels, whose
        shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
      num_classes: The possible number of labels the prediction task can
        have. This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
    Returns:
      total_cm: A `Tensor` representing the confusion matrix.
      update_op: An operation that increments the confusion matrix.
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = metric_variable(
        [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')
 
    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = math_ops.to_int64(predictions)
    labels = math_ops.to_int64(labels)
    num_classes = math_ops.to_int64(num_classes)
 
    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])
 
    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])
 
    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = array_ops.reshape(weights, [-1])
 
    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
        labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return total_cm, update_op
 
 
def _safe_div(numerator, denominator, name):
    """Divides two tensors element-wise, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    t = math_ops.truediv(numerator, denominator)
    zero = array_ops.zeros_like(t, dtype=denominator.dtype)
    condition = math_ops.greater(denominator, zero)
    zero = math_ops.cast(zero, t.dtype)
    return array_ops.where(condition, t, zero, name=name)
 
 
 
def iou(labels,
             predictions,
             num_classes,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_iou is not supported when '
                           'eager execution is enabled.')
 
    with variable_scope.variable_scope(name, 'iou',
                                       (predictions, labels, weights)):
        # Check if shape is compatible.
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
 
        total_cm, update_op = _streaming_confusion_matrix(labels, predictions,
                                                          num_classes, weights)
 
        def compute_iou(name):
            """Compute the mean intersection-over-union via the confusion matrix."""
            # 列向量求和
            sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
            # 行向量求和
            sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
            # 交集-对角线向量
            cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
            # 并集-即混淆矩阵求和：列向量和+行向量和-对角线向量和
            denominator = sum_over_row + sum_over_col - cm_diag
            # The mean is only computed over classes that appear in the
            # label or prediction tensor. If the denominator is 0, we need to
            # ignore the class.
            num_valid_entries = math_ops.reduce_sum(
                math_ops.cast(
                    math_ops.not_equal(denominator, 0), dtype=dtypes.float32))
 
            # If the value of the denominator is 0, set it to 1 to avoid
            # zero division.
            denominator = array_ops.where(
                math_ops.greater(denominator, 0), denominator,
                array_ops.ones_like(denominator))
            # iou即交并比： 交集/并集
            iou = math_ops.div(cm_diag, denominator)
            return iou
 
        iou_v = compute_iou('iou')
 
        if metrics_collections:
            ops.add_to_collections(metrics_collections, iou_v)
 
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return iou_v, update_op
