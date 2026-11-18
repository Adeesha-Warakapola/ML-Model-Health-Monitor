"""Tests for drift detection.

Several tests use small synthetic frames so the expected answer can be
reasoned about directly. The ones that use the real reference data check the
properties that matter most: unchanged data must not be flagged, deliberately
changed data must be, and no feature may ever vanish from the report.
"""

import json

import numpy as np
import pandas as pd
import pytest

from monitor import config, schema
from monitor.data import load_adult_data
from monitor.drift import (
    DRIFT,
    MODERATE,
    NOT_COMPUTED,
    OTHER_CATEGORY,
    STABLE,
    DriftDetector,
    DriftThresholds,
    categorical_psi,
    cramers_v,
    frequent_categories,
    population_stability_index,
    quantile_bin_edges,
)
from monitor.simulation import (
    BatchSimulator,
    combine,
    get_reference_features,
    overwrite_category,
    scale_numeric,
    shift_numeric,
    split_reference_and_holdout,
)
from tests.test_validation import make_valid_batch


@pytest.fixture(scope="module")
def adult():
    return load_adult_data()


@pytest.fixture(scope="module")
def reference(adult):
    return get_reference_features(adult)


@pytest.fixture(scope="module")
def holdout(adult):
    _, held = split_reference_and_holdout(adult)
    return held


@pytest.fixture(scope="module")
def detector(reference):
    return DriftDetector(reference)


def batch_of(holdout, size=500, seed=42, modifier=None):
    simulator = BatchSimulator(holdout, batch_size=size, seed=seed)
    return simulator.next_batch(modifier=modifier)


# --- PSI mechanics ----------------------------------------------------------


def test_psi_is_zero_for_identical_distributions():
    values = pd.Series(np.arange(1000, dtype=float))
    edges = quantile_bin_edges(values, 10)

    assert population_stability_index(values, values, edges) == pytest.approx(0.0)


def test_psi_grows_as_the_shift_grows():
    reference = pd.Series(np.random.default_rng(0).normal(0, 1, 5000))
    edges = quantile_bin_edges(reference, 10)

    scores = [
        population_stability_index(reference, reference + shift, edges)
        for shift in (0.1, 0.5, 2.0)
    ]
    assert scores[0] < scores[1] < scores[2]


def test_psi_does_not_grow_merely_because_the_batch_is_larger():
    # The key property that makes PSI a better decision basis than a p-value.
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(0, 1, 5000))
    edges = quantile_bin_edges(reference, 10)

    small = population_stability_index(reference, pd.Series(rng.normal(0, 1, 200)), edges)
    large = population_stability_index(reference, pd.Series(rng.normal(0, 1, 20000)), edges)

    assert small < 0.3 and large < 0.05  # both small; size alone does not inflate it


def test_psi_is_finite_when_a_bin_is_empty():
    # ln(0) is undefined, so empty bins must be handled, not crash.
    reference = pd.Series(np.arange(100, dtype=float))
    edges = quantile_bin_edges(reference, 10)
    batch = pd.Series([1.0] * 50)  # occupies one bin only

    value = population_stability_index(reference, batch, edges)

    assert np.isfinite(value)
    assert value > 0


def test_psi_rejects_an_empty_sample():
    reference = pd.Series(np.arange(100, dtype=float))
    edges = quantile_bin_edges(reference, 10)

    with pytest.raises(ValueError, match="empty"):
        population_stability_index(reference, pd.Series([], dtype=float), edges)


# --- binning edge cases -----------------------------------------------------


def test_bin_edges_are_unbounded_so_shifted_values_still_count():
    # With finite outer edges, drifted rows would fall outside every bin and
    # be dropped - hiding the drift. This is the guard against that.
    values = pd.Series(np.arange(100, dtype=float))
    edges = quantile_bin_edges(values, 10)

    assert edges[0] == -np.inf
    assert edges[-1] == np.inf

    far_away = pd.Series([1e6] * 20)
    assert population_stability_index(values, far_away, edges) > 0


def test_zero_inflated_feature_still_gets_multiple_bins(reference):
    # capital_gain is ~92% zeros. Plain quantile binning collapses to one
    # bin, which would make PSI always 0 and silently report "no drift".
    edges = quantile_bin_edges(reference["capital_gain"], 10)

    assert edges is not None
    assert len(edges) - 1 > 1


def test_zero_inflated_feature_reacts_to_a_real_change(reference):
    edges = quantile_bin_edges(reference["capital_gain"], 10)
    original = reference["capital_gain"].head(2000)
    tripled = original.where(original == 0, original * 3)

    assert population_stability_index(reference["capital_gain"], tripled, edges) > 0.05


def test_constant_reference_has_no_bin_edges():
    assert quantile_bin_edges(pd.Series([5.0] * 100), 10) is None


def test_two_valued_feature_is_binnable():
    edges = quantile_bin_edges(pd.Series([0.0] * 90 + [1.0] * 10), 10)
    assert edges is not None and len(edges) - 1 >= 1


# --- categorical mechanics --------------------------------------------------


def test_categorical_psi_is_zero_for_identical_shares():
    values = pd.Series(["a"] * 70 + ["b"] * 30)
    assert categorical_psi(values, values) == pytest.approx(0.0)


def test_categorical_psi_detects_a_changed_mix():
    reference = pd.Series(["a"] * 70 + ["b"] * 30)
    batch = pd.Series(["a"] * 30 + ["b"] * 70)

    assert categorical_psi(reference, batch) > 0.25


def test_frequent_categories_drops_only_rare_ones():
    values = pd.Series(["common"] * 990 + ["rare"] * 10)
    keep = frequent_categories(values, min_share=0.02)

    assert keep == {"common"}


def test_pooling_moves_rare_and_unseen_values_into_one_group():
    reference = pd.Series(["a"] * 950 + ["rare"] * 50)
    keep = frequent_categories(reference, min_share=0.10)
    batch = pd.Series(["a"] * 50 + ["brand_new"] * 50)

    # "brand_new" is unseen and "rare" is below the floor; both pool together,
    # and a batch that is half pooled values is still clearly different.
    assert keep == {"a"}
    assert categorical_psi(reference, batch, keep=keep) > 0.25


def test_cramers_v_is_zero_when_there_is_no_association():
    assert cramers_v(0.0, 100, (2, 5)) == 0.0


def test_cramers_v_ignores_a_degenerate_table():
    assert cramers_v(10.0, 100, (1, 1)) == 0.0


# --- unchanged data must not be flagged -------------------------------------


def test_reference_compared_against_itself_is_stable(detector, reference):
    report = detector.detect(reference)

    assert report.status == STABLE
    assert report.max_psi == pytest.approx(0.0, abs=1e-9)
    assert report.drifted_features == []


def test_unmodified_holdout_batch_is_stable(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)

    assert report.status == STABLE, report.explain()
    assert report.share_drifted == 0.0


def test_several_unmodified_batches_are_all_stable(detector, holdout):
    # Unmodified data flagged as drifted is by definition a false alarm.
    simulator = BatchSimulator(holdout, batch_size=500, seed=11)
    for batch in simulator.generate_batches(6):
        report = detector.detect(batch.features)
        assert report.status == STABLE, f"{batch.batch_id}\n{report.explain()}"


def test_high_cardinality_feature_is_quiet_on_unchanged_data(detector, holdout):
    # native_country has 41 categories, most of them rare - the feature that
    # produced false alarms before rare categories were pooled.
    report = detector.detect(batch_of(holdout, size=500).features)
    result = next(r for r in report.feature_results if r.feature == "native_country")

    assert result.status == STABLE
    assert result.psi < 0.05


# --- numerical drift --------------------------------------------------------


def test_shifted_numeric_feature_is_detected(detector, holdout):
    batch = batch_of(holdout, modifier=shift_numeric("age", 15))
    report = detector.detect(batch.features)

    assert report.status == DRIFT
    age = next(r for r in report.feature_results if r.feature == "age")
    assert age.status == DRIFT
    assert age.psi > 1.0
    assert age.p_value < 0.05


def test_scaled_numeric_feature_is_detected(detector, holdout):
    batch = batch_of(holdout, modifier=scale_numeric("hours_per_week", 1.5))
    report = detector.detect(batch.features)

    hours = next(r for r in report.feature_results if r.feature == "hours_per_week")
    assert hours.status == DRIFT


def test_only_the_changed_feature_is_flagged(detector, holdout):
    batch = batch_of(holdout, modifier=shift_numeric("age", 15))
    report = detector.detect(batch.features)

    flagged = {r.feature for r in report.drifted_features + report.moderate_features}
    assert flagged == {"age"}


def test_bigger_shifts_give_bigger_psi(detector, holdout):
    scores = []
    for shift in (2, 8, 20):
        batch = batch_of(holdout, modifier=shift_numeric("age", shift))
        report = detector.detect(batch.features)
        scores.append(next(r.psi for r in report.feature_results if r.feature == "age"))

    assert scores[0] < scores[1] < scores[2]


def test_numeric_result_reports_the_method_and_sample_sizes(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)
    age = next(r for r in report.feature_results if r.feature == "age")

    assert age.method == "kolmogorov_smirnov + psi"
    assert age.statistic_name == "ks_statistic"
    assert age.n_reference == 39073
    assert age.n_batch == 500
    assert age.n_bins is not None


# --- categorical drift ------------------------------------------------------


def test_changed_categorical_distribution_is_detected(detector, holdout):
    batch = batch_of(
        holdout, modifier=overwrite_category("education", "Bachelors", fraction=0.6)
    )
    report = detector.detect(batch.features)

    education = next(r for r in report.feature_results if r.feature == "education")
    assert education.status == DRIFT
    assert report.status == DRIFT


def test_unseen_category_is_detected_and_named(detector, holdout):
    batch = batch_of(
        holdout, modifier=overwrite_category("occupation", "Prompt-Engineer", 0.4)
    )
    report = detector.detect(batch.features)
    occupation = next(r for r in report.feature_results if r.feature == "occupation")

    assert occupation.unseen_categories == ["Prompt-Engineer"]
    assert occupation.status == DRIFT
    assert any("absent from the reference" in note for note in occupation.notes)


def test_unseen_category_also_surfaces_as_a_data_quality_finding(detector, holdout):
    # The same event is both: a new category is a distribution change *and*
    # something the encoder cannot represent. Both views are reported.
    batch = batch_of(
        holdout, modifier=overwrite_category("workclass", "Gig-economy", 0.3)
    )
    report = detector.detect(batch.features)

    assert any(
        issue.check == "unknown_categories" for issue in report.validation.issues
    )


def test_categorical_result_reports_the_method(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)
    race = next(r for r in report.feature_results if r.feature == "race")

    assert race.method == "chi_square + psi"
    assert race.statistic_name == "chi2_statistic"
    assert race.effect_size_name == "cramers_v"
    assert race.effect_size is not None


def test_multiple_drifting_features_are_all_reported(detector, holdout):
    batch = batch_of(
        holdout,
        modifier=combine(
            shift_numeric("age", 12),
            scale_numeric("hours_per_week", 1.5),
            overwrite_category("education", "Bachelors", 0.6),
        ),
    )
    report = detector.detect(batch.features)
    flagged = {r.feature for r in report.drifted_features}

    assert {"age", "hours_per_week", "education"} <= flagged
    assert report.share_drifted > 0.1


# --- no feature may ever be silently dropped --------------------------------


def test_every_feature_appears_in_the_report(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)

    assert len(report.feature_results) == len(schema.FEATURE_COLUMNS)
    assert {r.feature for r in report.feature_results} == set(schema.FEATURE_COLUMNS)


def test_features_that_cannot_be_computed_are_reported_with_a_reason(detector, holdout):
    batch = batch_of(holdout, size=500).features.copy()
    batch["age"] = np.nan  # nothing left to compare

    report = detector.detect(batch)
    age = next(r for r in report.feature_results if r.feature == "age")

    assert age.status == NOT_COMPUTED
    assert age.notes  # the reason is stated, not left blank
    assert len(report.feature_results) == len(schema.FEATURE_COLUMNS)
    assert age in report.not_computed_features


def test_constant_reference_feature_is_reported_not_dropped():
    reference = make_valid_batch(300)  # every row identical, so every feature constant
    detector = DriftDetector(reference)
    report = detector.detect(make_valid_batch(60))

    age = next(r for r in report.feature_results if r.feature == "age")
    assert age.status == NOT_COMPUTED
    assert any("constant" in note for note in age.notes)
    assert len(report.feature_results) == len(schema.FEATURE_COLUMNS)


# --- small samples and other edge cases -------------------------------------


def test_small_batch_is_computed_but_flagged(detector, holdout):
    report = detector.detect(batch_of(holdout, size=100).features)

    assert any("SMALL SAMPLE" in note for note in report.notes)
    assert report.computed_features  # still computed, not refused


def test_large_batch_carries_no_small_sample_note(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)
    assert not any("SMALL SAMPLE" in note for note in report.notes)


def test_single_row_batch_does_not_crash(detector, holdout):
    report = detector.detect(batch_of(holdout, size=1).features)

    assert report.status in (STABLE, MODERATE, DRIFT, NOT_COMPUTED)
    assert len(report.feature_results) == len(schema.FEATURE_COLUMNS)


def test_batch_where_a_feature_is_constant_is_noted(detector, holdout):
    batch = batch_of(holdout, size=500).features.copy()
    batch["age"] = 40

    report = detector.detect(batch)
    age = next(r for r in report.feature_results if r.feature == "age")

    assert any("same value" in note for note in age.notes)


def test_increased_missingness_is_reported(detector, holdout):
    batch = batch_of(holdout, size=500).features.copy()
    batch.iloc[:250, batch.columns.get_loc("education")] = None

    report = detector.detect(batch)
    education = next(r for r in report.feature_results if r.feature == "education")

    assert education.missing_rate_batch == pytest.approx(0.5)
    assert any("Missing-value rate moved" in note for note in education.notes)


def test_missing_values_do_not_break_the_statistics(detector, holdout):
    batch = batch_of(holdout, size=500).features.copy()
    batch.iloc[:100, batch.columns.get_loc("age")] = np.nan

    report = detector.detect(batch)
    age = next(r for r in report.feature_results if r.feature == "age")

    assert age.n_batch == 400  # NaNs excluded from the comparison, but counted
    assert age.missing_rate_batch == pytest.approx(0.2)
    assert np.isfinite(age.psi)


def test_empty_batch_is_rejected(detector):
    with pytest.raises(ValueError, match="empty batch"):
        detector.detect(pd.DataFrame(columns=schema.FEATURE_COLUMNS))


def test_batch_missing_a_feature_column_is_rejected(detector, holdout):
    batch = batch_of(holdout, size=100).features.drop(columns=["age"])

    with pytest.raises(ValueError, match="missing feature columns"):
        detector.detect(batch)


def test_detector_rejects_an_empty_reference():
    with pytest.raises(ValueError, match="empty"):
        DriftDetector(pd.DataFrame(columns=schema.FEATURE_COLUMNS))


def test_detector_rejects_a_reference_missing_columns(reference):
    with pytest.raises(ValueError, match="missing feature columns"):
        DriftDetector(reference.drop(columns=["sex"]))


# --- data quality is not the same as drift ----------------------------------


def test_impossible_values_are_a_quality_issue_not_drift(detector, holdout):
    batch = batch_of(holdout, size=500).features.copy()
    batch.iloc[:50, batch.columns.get_loc("age")] = -5

    report = detector.detect(batch)

    # Validation catches it; the distribution has barely moved.
    assert any(
        issue.check == "value_range" and issue.column == "age"
        for issue in report.validation.issues
    )
    assert report.status == STABLE


def test_broken_schema_is_an_error_while_drift_is_still_attempted(detector, holdout):
    batch = batch_of(holdout, size=500).features.copy()
    batch["sex"] = 1  # a number where text belongs: a pipeline bug

    report = detector.detect(batch)

    assert report.has_data_quality_errors
    assert not report.validation.is_valid
    # Drift results are still produced rather than everything being refused.
    assert len(report.feature_results) == len(schema.FEATURE_COLUMNS)


def test_clean_batch_has_no_data_quality_errors(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)
    assert not report.has_data_quality_errors


# --- thresholds -------------------------------------------------------------


def test_thresholds_are_configurable(reference, holdout):
    strict = DriftDetector(reference, DriftThresholds(psi_moderate=0.001, psi_major=0.002))
    lenient = DriftDetector(reference, DriftThresholds(psi_moderate=5.0, psi_major=10.0))
    batch = batch_of(holdout, size=500).features

    assert strict.detect(batch).status == DRIFT
    assert lenient.detect(batch).status == STABLE


def test_thresholds_are_recorded_in_the_report(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)

    assert report.thresholds.psi_moderate == 0.10
    assert report.to_dict()["thresholds"]["psi_major"] == 0.25


def test_nonsensical_thresholds_are_rejected():
    with pytest.raises(ValueError, match="psi_moderate < psi_major"):
        DriftThresholds(psi_moderate=0.5, psi_major=0.1)
    with pytest.raises(ValueError, match="p_value"):
        DriftThresholds(p_value=1.5)
    with pytest.raises(ValueError, match="n_bins"):
        DriftThresholds(n_bins=1)


def test_measured_small_sample_floor_is_the_default():
    # Documented as measured from false-alarm rates, not chosen arbitrarily.
    assert DriftThresholds().min_batch_rows == 400


# --- effect size versus p-value (requirement: do not trust p alone) ---------


def test_significant_p_with_negligible_effect_is_explained(detector, reference):
    # A huge batch that differs trivially: p collapses towards zero while PSI
    # stays tiny. The report must say so rather than cry drift.
    batch = reference.copy()
    batch["age"] = batch["age"] + 1

    report = detector.detect(batch)
    age = next(r for r in report.feature_results if r.feature == "age")

    assert age.p_value < 0.05
    assert age.psi < 0.10
    assert age.status == STABLE
    assert any("statistically significant but PSI" in note for note in age.notes)


def test_chi_square_warns_when_expected_counts_are_too_small(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)
    country = next(r for r in report.feature_results if r.feature == "native_country")

    assert any("Smallest expected count" in note for note in country.notes)


# --- report shape and reuse outside any UI ----------------------------------


def test_to_frame_has_one_row_per_feature_sorted_by_psi(detector, holdout):
    frame = detector.detect(batch_of(holdout, size=500).features).to_frame()

    assert len(frame) == len(schema.FEATURE_COLUMNS)
    psi_values = frame["psi"].dropna().tolist()
    assert psi_values == sorted(psi_values, reverse=True)
    for column in ("feature", "method", "status", "psi", "p_value", "n_reference", "n_batch"):
        assert column in frame.columns


def test_to_dict_is_json_serialisable(detector, holdout):
    batch = batch_of(holdout, size=500)
    payload = detector.detect(batch.features, batch_id=batch.batch_id).to_dict()

    text = json.dumps(payload, default=str)
    restored = json.loads(text)

    assert restored["batch_id"] == batch.batch_id
    assert restored["status"] == STABLE
    assert len(restored["features"]) == len(schema.FEATURE_COLUMNS)
    assert "data_quality" in restored


def test_report_separates_quality_from_drift_in_its_payload(detector, holdout):
    payload = detector.detect(batch_of(holdout, size=500).features).to_dict()

    assert set(payload["data_quality"]) == {
        "is_valid", "n_errors", "n_warnings", "issues"
    }
    assert "status" in payload and "features" in payload


def test_batch_id_is_optional(detector, holdout):
    report = detector.detect(batch_of(holdout, size=500).features)
    assert report.batch_id is None
    assert "(unnamed)" in report.summary()


def test_summary_and_explain_are_plain_text(detector, holdout):
    batch = batch_of(holdout, modifier=shift_numeric("age", 15))
    report = detector.detect(batch.features, batch_id="batch_test")

    summary = report.summary()
    assert "batch_test" in summary
    assert "DRIFT" in summary
    assert "age" in report.explain()


def test_drift_module_does_not_depend_on_streamlit():
    import sys

    import monitor.drift  # noqa: F401

    assert "streamlit" not in sys.modules


# --- reference stays fixed --------------------------------------------------


def test_detector_does_not_learn_from_batches(detector, holdout):
    # A reference that absorbed each batch would follow the drift and stop
    # detecting it. Scoring a drifted batch must not change later verdicts.
    drifted = batch_of(holdout, modifier=shift_numeric("age", 15))
    detector.detect(drifted.features)

    report = detector.detect(batch_of(holdout, size=500).features)
    assert report.status == STABLE


def test_same_batch_gives_the_same_report_twice(detector, holdout):
    batch = batch_of(holdout, size=500).features

    first = detector.detect(batch).to_frame()
    second = detector.detect(batch).to_frame()

    pd.testing.assert_frame_equal(first, second)


# --- integration with the Phase 4 simulator ---------------------------------


def test_simulator_batches_keep_their_natural_class_balance(holdout):
    # Drift scenarios must not quietly rebalance the labels.
    plain = batch_of(holdout, size=500)
    drifted = batch_of(holdout, size=500, modifier=shift_numeric("age", 15))

    assert plain.actual_class_distribution() == drifted.actual_class_distribution()
    positive_rate = (plain.labels == config.POSITIVE_LABEL).mean()
    assert 0.15 < positive_rate < 0.35


def test_detection_works_on_a_batch_straight_from_the_simulator(detector, holdout):
    batch = BatchSimulator(holdout, batch_size=500, seed=3).next_batch()
    report = detector.detect(batch.features, batch_id=batch.batch_id)

    assert report.batch_id == batch.batch_id
    assert report.n_batch == 500
    assert report.n_reference == 39073
