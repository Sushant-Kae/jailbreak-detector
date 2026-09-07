import pytest

from src.data import load_dataset, make_splits

DATA_PATH = "data/dataset.csv"


def test_load_dataset_has_expected_columns():
    df = load_dataset(DATA_PATH)
    assert {"text", "label"}.issubset(df.columns)


def test_load_dataset_labels_are_valid():
    df = load_dataset(DATA_PATH)
    assert set(df["label"].unique()) <= {"normal", "suspicious"}


def test_load_dataset_rejects_bad_labels(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("text,label\nhello,not_a_real_label\n")
    with pytest.raises(ValueError):
        load_dataset(str(bad_csv))


def test_make_splits_no_overlap():
    df = load_dataset(DATA_PATH)
    splits = make_splits(df)

    train_set = set(splits.x_train)
    val_set = set(splits.x_val)
    test_set = set(splits.x_test)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_make_splits_preserves_total_count():
    df = load_dataset(DATA_PATH)
    splits = make_splits(df)
    total = len(splits.x_train) + len(splits.x_val) + len(splits.x_test)
    assert total == len(df)
