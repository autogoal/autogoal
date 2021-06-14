from autogoal.experimental.hyperopt_search import format_hyperopt_args


def test_format_hyperopt_args():
    args = {
        "Test": "X",
        "X_y": {"X_y_z": 12, "__CATEGORICAL__": {"X_y_cat": "cat"}},
        "__CATEGORICAL__": {"X_cat": "cat2"},
    }
    expected = {
        "Test": "X",
        "X_y": {"X_y_z": 12, "X_y_cat": "cat"},
        "X_cat": "cat2",
    }
    formatted_args = format_hyperopt_args(args)
    assert expected == formatted_args
