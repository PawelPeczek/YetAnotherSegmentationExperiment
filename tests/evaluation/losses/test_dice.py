import numpy as np

from src.evaluation.losses.dice import dice_score, dice_score_binary


def test_multi_class_dice_score_when_all_predictions_correct_and_one_class_occupied() -> None:
    # given
    y_pred = np.zeros((10, 10), dtype=np.uint32)
    y_true = np.zeros((10, 10), dtype=np.uint32)

    # when
    result = dice_score(y_pred=y_pred, y_true=y_true, num_classes=4)

    # then
    assert abs(result - 1.0) < 1e-7


def test_multi_class_dice_score_when_half_predictions_correct() -> None:
    # given
    y_pred = np.zeros((10, 10), dtype=np.uint32)
    y_true = np.ones((10, 10), dtype=np.uint32)

    # when
    result = dice_score(y_pred=y_pred, y_true=y_true, num_classes=4)

    # then
    assert abs(result - 0.5) < 1e-7


def test_multi_class_dice_score_when_all_predictions_correct_and_multiple_classes_occupied() -> None:
    # given
    y_pred = np.zeros((10, 10), dtype=np.uint32)
    y_true = np.zeros((10, 10), dtype=np.uint32)
    y_pred[0:5, 0:5] = 1
    y_true[0:5, 0:5] = 1

    # when
    result = dice_score(y_pred=y_pred, y_true=y_true, num_classes=4)

    # then
    assert abs(result - 1.0) < 1e-7


def test_multi_class_dice_score_with_complex_set_up_1() -> None:
    """
    PER CLASS DICE:
    0 => 0,5 TP = 25, 2*TP + FP + FN = 100
    1 => 1.0 TP = 25, 2*TP + FP + FN = 50
    2 => 1.0 (no occurrences)
    3 => 0.0 TP = 0, 2*TP + FP + FN = 50
    2.4 / 4 = 0.625
    """
    # given
    y_pred = np.zeros((10, 10), dtype=np.uint32)
    y_true = np.zeros((10, 10), dtype=np.uint32)
    y_pred[0:5, 0:5] = 1
    y_true[0:5, 0:5] = 1
    y_true[0:10, 5:10] = 3

    # when
    result = dice_score(y_pred=y_pred, y_true=y_true, num_classes=4)

    # then
    assert abs(result - 0.625) < 1e-7


def test_multi_class_dice_score_with_complex_set_up_2() -> None:
    """
    PER CLASS DICE:
    0 => 0,88888888 TP = 20, 2*TP + FP + FN = 45
    1 => 0,9090909  TP = 25, 2*TP + FP + FN = 55
    2 => 0,9090909  TP = 25, 2*TP + FP + FN = 55
    3 => 0,88888888 TP = 20, 2*TP + FP + FN = 45
    """
    # given
    y_pred = np.zeros((10, 10), dtype=np.uint32)
    y_true = np.zeros((10, 10), dtype=np.uint32)
    y_pred[0:5, 0:5] = 0
    y_true[0:4, 0:5] = 0
    y_pred[5:10, 0:5] = 1
    y_true[4:10, 0:5] = 1
    y_pred[0:5, 5:10] = 2
    y_true[0:6, 5:10] = 2
    y_pred[5:10, 5:10] = 3
    y_true[6:10, 5:10] = 3

    # when
    result = dice_score(y_pred=y_pred, y_true=y_true, num_classes=4)

    # then
    assert abs(result - 0.898989894) < 1e-7


def test_multi_class_dice_score_when_all_predictions_wrong() -> None:
    # given
    y_pred = np.zeros((3, 3), dtype=np.uint32)
    y_true = np.ones((3, 3), dtype=np.uint32) * 2
    y_pred[0, :] = 2
    y_pred[2, :] = 1
    y_true[0, :] = 0
    y_true[1, :] = 1

    # when
    result = dice_score(y_pred=y_pred, y_true=y_true, num_classes=3)

    # then
    assert abs(result - 0.0) < 1e-7


def test_binary_dice_score_when_quarter_of_prediction_correct() -> None:
    # given
    y_pred = np.zeros((10, 10), dtype=np.uint32)
    y_true = np.zeros((10, 10), dtype=np.uint32)
    y_pred[0:10, 0:5] = 1
    y_true[5:10, 0:10] = 1

    # when
    result = dice_score_binary(y_pred=y_pred, y_true=y_true)

    # then
    assert abs(result - 0.5) < 1e-7


def test_binary_dice_score_when_all_predictions_correct_and_mix_predictions_occur() -> None:
    # given
    y_pred = np.ones((10, 10), dtype=np.uint32)
    y_true = np.ones((10, 10), dtype=np.uint32)
    y_pred[0:5, 0:5] = 0
    y_true[0:5, 0:5] = 0

    # when
    result = dice_score_binary(y_pred=y_pred, y_true=y_true)

    # then
    assert abs(result - 1.0) < 1e-7


def test_binary_dice_score_when_all_predictions_correct_and_single_class_prediction_occurs() -> None:
    # given
    y_pred = np.ones((10, 10), dtype=np.uint32)
    y_true = np.ones((10, 10), dtype=np.uint32)

    # when
    result = dice_score_binary(y_pred=y_pred, y_true=y_true)

    # then
    assert abs(result - 1.0) < 1e-7


def test_binary_dice_score_when_all_predictions_wrong() -> None:
    # given
    y_pred = np.zeros((10, 10), dtype=np.uint32)
    y_true = np.ones((10, 10), dtype=np.uint32)

    # when
    result = dice_score_binary(y_pred=y_pred, y_true=y_true)

    # then
    assert abs(result - 0.0) < 1e-7
