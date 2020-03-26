import numpy as np


def dice_score_binary(y_pred: np.ndarray,
                      y_true: np.ndarray
                      ) -> float:
    """
    Function to calculate Sørensen–Dice coefficient for binary segmentation.

    Args:
        y_pred: Model output (after arg_max application).
            Shape (N, H, W) or (H, W) - must be equal with y_true.
            Dtype: np.bool, np.(u)int8, np.(u)int16, np.(u)int32, np.(u)int64
            Values: [0, 1]
        y_true: Ground-truth segmentation mask.
            Shape (N, H, W) or (H, W)
            Dtype: np.bool, np.(u)int8, np.(u)int16, np.(u)int32, np.(u)int64
            Values: [0, 1]
    """
    return dice_score(
        y_pred=y_pred,
        y_true=y_true,
        num_classes=2
    )


def dice_score(y_pred: np.ndarray,
               y_true: np.ndarray,
               num_classes: int,
               stability_coef: float = 1e-8
               ) -> float:
    """
    Function to calculate Sørensen–Dice coefficient.
    Details: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient


    Args:
        y_pred: Model output (after arg_max application).
            Shape (N, H, W) or (H, W) -
            must be equal with y_true.
            Dtype: np.bool, np.(u)int8, np.(u)int16, np.(u)int32, np.(u)int64
            Values: [0, num_classes]
        y_true: Ground-truth segmentation mask.
            Shape (N, H, W) or (H, W)
            Dtype: np.bool, np.(u)int8, np.(u)int16, np.(u)int32, np.(u)int64
            Values: [0, num_classes-1]
        num_classes: Number of classes.
        stability_coef: Value used to avoid 0 division error and errors when only TNs.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape.")
    y_pred = standardise_matrix_nhw_one_hot_cls(
        matrix=y_pred,
        num_classes=num_classes
    )
    y_true = standardise_matrix_nhw_one_hot_cls(
        matrix=y_true,
        num_classes=num_classes
    )
    per_class_intersection = np.sum(y_pred * y_true, axis=(1, 2))
    denom = np.sum(y_pred, axis=(1, 2)) + np.sum(y_true, axis=(1, 2))
    samples_score_per_class = \
        (2 * per_class_intersection + stability_coef) / (denom + stability_coef)
    return np.average(samples_score_per_class)


def standardise_matrix_nhw_one_hot_cls(matrix: np.ndarray,
                                       num_classes: int) -> np.ndarray:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2.")
    if len(matrix.shape) not in {2, 3}:
        raise ValueError(
            "Matrix should be in format (H, W) or (N, H, W)."
        )
    if np.min(matrix) < 0 or np.max(matrix) >= num_classes:
        raise ValueError(
            "Values of the matrix must be in range [0, num_classes)."
        )
    matrix = np.eye(num_classes, dtype=matrix.dtype)[matrix]
    return np.expand_dims(matrix, axis=0) if len(matrix.shape) == 3 else matrix

