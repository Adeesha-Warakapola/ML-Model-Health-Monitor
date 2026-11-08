"""Tests for the batch simulator.

Two properties matter most: batches must come only from data the model never
trained on, and the same seed must always produce the same batches.
"""

import pandas as pd
import pytest

from monitor import config, schema
from monitor.data import load_adult_data
from monitor.simulation import (
    BatchSimulator,
    combine,
    get_reference_features,
    overwrite_category,
    scale_numeric,
    shift_numeric,
    split_reference_and_holdout,
)


@pytest.fixture(scope="module")
def adult():
    return load_adult_data()


@pytest.fixture(scope="module")
def holdout(adult):
    _, held = split_reference_and_holdout(adult)
    return held


# --- reference and holdout --------------------------------------------------


def test_reference_and_holdout_do_not_overlap(adult):
    reference, held = split_reference_and_holdout(adult)

    # This is the guarantee that batches contain no training rows.
    assert set(reference.index).isdisjoint(set(held.index))
    assert len(reference) + len(held) == len(adult)


def test_holdout_matches_the_training_test_size(adult):
    _, held = split_reference_and_holdout(adult)
    assert len(held) == pytest.approx(len(adult) * config.TEST_SIZE, rel=0.01)


def test_split_is_the_same_one_used_for_training(adult):
    # Must reproduce Phase 3's split exactly, or the "unseen" claim is false.
    from monitor.preprocessing import split_train_test

    reference, held = split_reference_and_holdout(adult)
    train_df, test_df = split_train_test(adult, test_size=config.TEST_SIZE)

    assert list(reference.index) == list(train_df.index)
    assert list(held.index) == list(test_df.index)


def test_reference_features_exclude_the_label(adult):
    features = get_reference_features(adult)

    assert list(features.columns) == schema.FEATURE_COLUMNS
    assert config.TARGET_COLUMN not in features.columns


# --- batch structure and label separation -----------------------------------


def test_batch_features_contain_no_label_column(holdout):
    batch = BatchSimulator(holdout, batch_size=100, seed=1).next_batch()

    assert config.TARGET_COLUMN not in batch.features.columns
    assert list(batch.features.columns) == schema.FEATURE_COLUMNS


def test_labels_are_kept_separately_and_stay_aligned(holdout):
    batch = BatchSimulator(holdout, batch_size=100, seed=1).next_batch()

    assert isinstance(batch.labels, pd.Series)
    assert len(batch.labels) == len(batch.features)
    assert list(batch.labels.index) == list(batch.features.index)
    assert set(batch.labels).issubset(set(config.TARGET_LABELS))


def test_labels_match_the_source_rows(holdout):
    batch = BatchSimulator(holdout, batch_size=50, seed=3).next_batch()

    # Each label must still be the label of its own row in the holdout data.
    expected = holdout.loc[batch.features.index, config.TARGET_COLUMN]
    pd.testing.assert_series_equal(batch.labels, expected)


def test_batch_rows_all_come_from_the_holdout_set(holdout):
    batch = BatchSimulator(holdout, batch_size=200, seed=5).next_batch()
    assert set(batch.features.index).issubset(set(holdout.index))


# --- batch metadata ---------------------------------------------------------


def test_batch_carries_identity_and_provenance(holdout):
    simulator = BatchSimulator(holdout, batch_size=100, seed=11)
    first = simulator.next_batch()
    second = simulator.next_batch()

    assert first.batch_id != second.batch_id
    assert first.batch_index == 1 and second.batch_index == 2
    assert first.seed == 11
    assert first.created_at
    assert first.modifications == []


def test_actual_class_distribution_is_reported(holdout):
    batch = BatchSimulator(holdout, batch_size=200, seed=2).next_batch()
    distribution = batch.actual_class_distribution()

    assert sum(stats["count"] for stats in distribution.values()) == 200
    assert "rows" in batch.summary()


# --- repeatability ----------------------------------------------------------


def test_same_seed_produces_the_same_batches(holdout):
    first = BatchSimulator(holdout, batch_size=100, seed=42).next_batch()
    second = BatchSimulator(holdout, batch_size=100, seed=42).next_batch()

    assert list(first.features.index) == list(second.features.index)
    pd.testing.assert_frame_equal(first.features, second.features)


def test_different_seeds_produce_different_batches(holdout):
    first = BatchSimulator(holdout, batch_size=100, seed=1).next_batch()
    second = BatchSimulator(holdout, batch_size=100, seed=2).next_batch()

    assert list(first.features.index) != list(second.features.index)


def test_whole_sequence_of_batches_is_reproducible(holdout):
    def sequence(seed):
        simulator = BatchSimulator(holdout, batch_size=50, seed=seed)
        return [list(b.features.index) for b in simulator.generate_batches(4)]

    assert sequence(7) == sequence(7)


def test_reset_replays_the_same_sequence(holdout):
    simulator = BatchSimulator(holdout, batch_size=50, seed=9)
    first = list(simulator.next_batch().features.index)
    simulator.reset()
    again = list(simulator.next_batch().features.index)

    assert first == again
    assert simulator.rows_remaining == len(holdout) - 50


# --- disjoint batches and pool management -----------------------------------


def test_successive_batches_do_not_repeat_rows(holdout):
    simulator = BatchSimulator(holdout, batch_size=100, seed=4)
    batches = simulator.generate_batches(5)

    seen = [index for batch in batches for index in batch.features.index]
    assert len(seen) == len(set(seen))  # every record arrives at most once


def test_rows_and_batches_remaining_count_down(holdout):
    simulator = BatchSimulator(holdout, batch_size=100, seed=4)
    before = simulator.rows_remaining

    simulator.next_batch()

    assert simulator.rows_remaining == before - 100
    assert simulator.batches_remaining == (before - 100) // 100


def test_exhausting_the_pool_raises_a_clear_error(holdout):
    small = holdout.head(120)
    simulator = BatchSimulator(small, batch_size=100, seed=1)
    simulator.next_batch()

    with pytest.raises(ValueError, match="Only 20 rows remain"):
        simulator.next_batch()


def test_requesting_too_many_batches_is_rejected(holdout):
    simulator = BatchSimulator(holdout.head(250), batch_size=100, seed=1)
    with pytest.raises(ValueError, match="only 2"):
        simulator.generate_batches(3)


# --- configurable size ------------------------------------------------------


def test_batch_size_is_configurable(holdout):
    for size in (1, 10, 377):
        batch = BatchSimulator(holdout, batch_size=size, seed=1).next_batch()
        assert batch.n_rows == size


def test_size_can_be_overridden_per_batch(holdout):
    simulator = BatchSimulator(holdout, batch_size=100, seed=1)
    assert simulator.next_batch(size=25).n_rows == 25


# --- natural distribution ---------------------------------------------------


def test_batches_preserve_the_natural_distribution_on_average(holdout):
    # Sampling is random rather than stratified, so individual batches vary.
    # Across many batches the average must sit near the population rate -
    # that is what "not artificially balanced" means in practice.
    simulator = BatchSimulator(holdout, batch_size=500, seed=42)
    batches = simulator.generate_batches(15)

    population = (holdout[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean()
    rates = [
        (b.labels == config.POSITIVE_LABEL).mean() for b in batches
    ]
    average = sum(rates) / len(rates)

    assert abs(average - population) < 0.02
    # And no batch is forced to exactly the population rate.
    assert len(set(round(r, 4) for r in rates)) > 1


def test_batches_are_not_balanced_to_fifty_fifty(holdout):
    batch = BatchSimulator(holdout, batch_size=1000, seed=42).next_batch()
    positive_rate = (batch.labels == config.POSITIVE_LABEL).mean()

    # Must stay near the natural ~24%, nowhere near a balanced 50%.
    assert 0.15 < positive_rate < 0.35


# --- simulator input validation ---------------------------------------------


def test_empty_holdout_is_rejected():
    with pytest.raises(ValueError, match="empty holdout"):
        BatchSimulator(pd.DataFrame(), batch_size=10)


def test_batch_size_larger_than_the_pool_is_rejected(holdout):
    with pytest.raises(ValueError, match="exceeds"):
        BatchSimulator(holdout.head(10), batch_size=50)


def test_zero_or_negative_batch_size_is_rejected(holdout):
    for bad in (0, -5):
        with pytest.raises(ValueError, match="at least 1"):
            BatchSimulator(holdout, batch_size=bad)


def test_holdout_without_labels_is_rejected(holdout):
    with pytest.raises(ValueError, match="must include the label column"):
        BatchSimulator(holdout.drop(columns=[config.TARGET_COLUMN]), batch_size=10)


def test_holdout_missing_features_is_rejected(holdout):
    with pytest.raises(ValueError, match="missing feature columns"):
        BatchSimulator(holdout.drop(columns=["age"]), batch_size=10)


# --- modifiers (the hook for later drift experiments) -----------------------


def test_shift_numeric_moves_a_column_by_a_constant(holdout):
    simulator = BatchSimulator(holdout, batch_size=100, seed=1)
    plain = simulator.next_batch()
    simulator.reset()
    shifted = simulator.next_batch(modifier=shift_numeric("age", 10))

    assert shifted.features["age"].mean() == pytest.approx(
        plain.features["age"].mean() + 10
    )


def test_scale_numeric_multiplies_a_column(holdout):
    simulator = BatchSimulator(holdout, batch_size=100, seed=1)
    plain = simulator.next_batch()
    simulator.reset()
    scaled = simulator.next_batch(modifier=scale_numeric("hours_per_week", 2.0))

    assert scaled.features["hours_per_week"].mean() == pytest.approx(
        plain.features["hours_per_week"].mean() * 2
    )


def test_overwrite_category_replaces_the_requested_fraction(holdout):
    batch = BatchSimulator(holdout, batch_size=100, seed=1).next_batch(
        modifier=overwrite_category("occupation", "Prompt-Engineer", fraction=0.25)
    )
    assert (batch.features["occupation"] == "Prompt-Engineer").sum() == 25


def test_overwrite_category_can_introduce_an_unseen_value(holdout):
    batch = BatchSimulator(holdout, batch_size=50, seed=1).next_batch(
        modifier=overwrite_category("workclass", "Gig-economy")
    )
    allowed = schema.FEATURE_SPECS["workclass"].allowed_values

    assert (batch.features["workclass"] == "Gig-economy").all()
    assert "Gig-economy" not in allowed


def test_combine_applies_modifiers_in_order(holdout):
    simulator = BatchSimulator(holdout, batch_size=100, seed=1)
    plain = simulator.next_batch()
    simulator.reset()
    both = simulator.next_batch(
        modifier=combine(
            shift_numeric("age", 5), overwrite_category("sex", "Unknown", 0.5)
        )
    )

    assert both.features["age"].mean() == pytest.approx(plain.features["age"].mean() + 5)
    assert (both.features["sex"] == "Unknown").sum() == 50


def test_modifier_does_not_touch_the_labels(holdout):
    simulator = BatchSimulator(holdout, batch_size=200, seed=1)
    plain = simulator.next_batch()
    simulator.reset()
    modified = simulator.next_batch(
        modifier=combine(shift_numeric("age", 20), scale_numeric("capital_gain", 10))
    )

    # The whole point: inputs change, ground truth does not.
    pd.testing.assert_series_equal(plain.labels, modified.labels)
    assert plain.actual_class_distribution() == modified.actual_class_distribution()


def test_modification_is_recorded_on_the_batch(holdout):
    batch = BatchSimulator(holdout, batch_size=50, seed=1).next_batch(
        modifier=shift_numeric("age", 5), modification_label="age +5"
    )

    assert batch.modifications == ["age +5"]
    assert "age +5" in batch.summary()


def test_unmodified_batch_records_no_modifications(holdout):
    batch = BatchSimulator(holdout, batch_size=50, seed=1).next_batch()
    assert batch.modifications == []


def test_modifiers_reject_a_column_of_the_wrong_kind():
    with pytest.raises(ValueError, match="not a numeric feature"):
        shift_numeric("occupation", 1)
    with pytest.raises(ValueError, match="not a numeric feature"):
        scale_numeric("sex", 2)
    with pytest.raises(ValueError, match="not a categorical feature"):
        overwrite_category("age", "x")


def test_overwrite_category_rejects_an_invalid_fraction():
    for bad in (0, -0.5, 1.5):
        with pytest.raises(ValueError, match="fraction must be"):
            overwrite_category("sex", "Unknown", fraction=bad)


def test_modifier_that_drops_a_column_is_caught(holdout):
    simulator = BatchSimulator(holdout, batch_size=50, seed=1)

    with pytest.raises(ValueError, match="dropped required feature columns"):
        simulator.next_batch(modifier=lambda f: f.drop(columns=["age"]))


def test_modifier_that_changes_the_row_count_is_caught(holdout):
    simulator = BatchSimulator(holdout, batch_size=50, seed=1)

    with pytest.raises(ValueError, match="number of rows"):
        simulator.next_batch(modifier=lambda f: f.head(10))
