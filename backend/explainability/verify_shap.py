"""
F1.4 Verification Script — SHAP Explainability
Runs comprehensive checks on the SHAP explainer and plain English converter.
"""
import os
import sys
import json
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(BASE)
DATA_DIR = os.path.join(BACKEND, "data")

sys.path.insert(0, BACKEND)

passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = ""):
    """Run a single check and track pass/fail."""
    global passed, failed, total
    total += 1
    status = "✅ PASS" if condition else "❌ FAIL"
    if not condition:
        failed += 1
    else:
        passed += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{total:02d}] {status}  {name}{suffix}")


def main():
    print("=" * 65)
    print("  F1.4 VERIFICATION — SHAP Explainability")
    print("=" * 65)

    # ── Section 1: Module Imports ─────────────────────────────────────────
    print("\n  § 1. Module Imports")

    try:
        import shap
        shap_imported = True
    except ImportError:
        shap_imported = False
    check("SHAP library imports", shap_imported)

    shap_version = getattr(shap, "__version__", "unknown") if shap_imported else "N/A"
    check("SHAP version is 0.45.x", shap_version.startswith("0.45"), f"v{shap_version}")

    try:
        from explainability.shap_explainer import ShapExplainer
        explainer_imported = True
    except ImportError as e:
        explainer_imported = False
        print(f"    [ERROR] {e}")
    check("ShapExplainer imports", explainer_imported)

    try:
        from explainability.plain_english import PlainEnglishConverter
        converter_imported = True
    except ImportError as e:
        converter_imported = False
        print(f"    [ERROR] {e}")
    check("PlainEnglishConverter imports", converter_imported)

    # ── Section 2: ShapExplainer Initialization ───────────────────────────
    print("\n  § 2. ShapExplainer Initialization")

    try:
        explainer = ShapExplainer()
        explainer_init = True
    except Exception as e:
        explainer_init = False
        print(f"    [ERROR] {e}")
    check("ShapExplainer initializes without error", explainer_init)

    if explainer_init:
        check("Has 4 TreeExplainers (one per target)", len(explainer.explainers) == 4,
              f"found {len(explainer.explainers)}")

        targets = ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]
        all_targets = all(t in explainer.explainers for t in targets)
        check("All 4 targets have explainers", all_targets)

        check("Has 4 baselines", len(explainer.baselines) == 4)

        # Baselines should be positive and reasonable
        for t in targets:
            b = explainer.baselines.get(t, 0)
            check(f"Baseline {t} > 0", b > 0, f"{b:.2f}")

    # ── Section 3: Single Explanation ─────────────────────────────────────
    print("\n  § 3. Single Batch Explanation")

    # Optimal batch — should have small delta from baseline
    optimal_params = {
        "temperature": 183.0,
        "conveyor_speed": 75.0,
        "hold_time": 18.0,
        "batch_size": 500.0,
        "material_type": 0,
        "hour_of_day": 10,
        "operator_exp": 2,
    }

    if explainer_init:
        try:
            result = explainer.explain_single(optimal_params, target="energy_kwh")
            single_ok = True
        except Exception as e:
            single_ok = False
            result = {}
            print(f"    [ERROR] {e}")
        check("explain_single() returns without error", single_ok)

        if single_ok:
            check("Result has 'target' key", "target" in result)
            check("Result has 'baseline_prediction' key", "baseline_prediction" in result)
            check("Result has 'final_prediction' key", "final_prediction" in result)
            check("Result has 'unit' key", "unit" in result)
            check("Result has 'feature_contributions' list", "feature_contributions" in result)

            contribs = result.get("feature_contributions", [])
            check("Has 13 feature contributions", len(contribs) == 13, f"found {len(contribs)}")

            # Each contribution should have required keys
            required_keys = {"feature", "display_name", "value", "contribution", "direction"}
            if contribs:
                first = contribs[0]
                has_keys = all(k in first for k in required_keys)
                check("Contributions have all required keys", has_keys,
                      f"keys: {set(first.keys())}")

            # SHAP additivity: sum(contributions) + baseline ≈ prediction
            total_shap = sum(c["contribution"] for c in contribs)
            baseline = result.get("baseline_prediction", 0)
            prediction = result.get("final_prediction", 0)
            expected = baseline + total_shap
            diff = abs(expected - prediction)
            check("SHAP additivity holds (sum + baseline ≈ prediction)",
                  diff < 0.5, f"|{expected:.2f} - {prediction:.2f}| = {diff:.4f}")

            # Sorted by absolute contribution (largest first)
            abs_sorted = all(
                abs(contribs[i]["contribution"]) >= abs(contribs[i + 1]["contribution"])
                for i in range(len(contribs) - 1)
            )
            check("Contributions sorted by |value| descending", abs_sorted)

    # ── Section 4: Multi-target Explanation ────────────────────────────────
    print("\n  § 4. Multi-target Explanation (all 4 targets)")

    if explainer_init:
        try:
            all_result = explainer.explain_single(optimal_params, target=None)
            multi_ok = True
        except Exception as e:
            multi_ok = False
            all_result = {}
            print(f"    [ERROR] {e}")
        check("explain_single(target=None) returns all 4 targets", multi_ok)

        if multi_ok:
            check("Result contains all 4 targets", len(all_result) == 4,
                  f"found {len(all_result)} targets")
            for t in targets:
                check(f"Contains {t}", t in all_result)

    # ── Section 5: Batch Explanation ──────────────────────────────────────
    print("\n  § 5. Batch Explanation (multiple samples)")

    if explainer_init:
        import pandas as pd
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))

        from models.multi_target_predictor import FEATURE_COLS
        X_test = test_df[FEATURE_COLS].values[:10]  # First 10 samples

        try:
            shap_vals = explainer.explain_batch(X_test, target="energy_kwh")
            batch_ok = True
        except Exception as e:
            batch_ok = False
            shap_vals = None
            print(f"    [ERROR] {e}")
        check("explain_batch() returns without error", batch_ok)

        if batch_ok and shap_vals is not None:
            check("SHAP values shape is (10, 13)", shap_vals.shape == (10, 13),
                  f"shape: {shap_vals.shape}")
            check("No NaN in SHAP values", not np.any(np.isnan(shap_vals)))

    # ── Section 6: Global Importance ──────────────────────────────────────
    print("\n  § 6. Global Feature Importance")

    if explainer_init:
        try:
            gi = explainer.global_importance("energy_kwh", n_samples=100)
            gi_ok = True
        except Exception as e:
            gi_ok = False
            gi = {}
            print(f"    [ERROR] {e}")
        check("global_importance() returns without error", gi_ok)

        if gi_ok:
            check("Returns 13 features", len(gi) == 13, f"found {len(gi)}")
            check("All values > 0", all(v > 0 for v in gi.values()))

            # Sorted descending
            vals = list(gi.values())
            is_sorted = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
            check("Sorted by importance descending", is_sorted)

            # Top energy driver should be hold_time or material_type per domain
            top_feat = list(gi.keys())[0]
            check("Top energy feature is hold_time or material_type",
                  top_feat in ("hold_time", "material_type"), f"found '{top_feat}'")

    # ── Section 7: Plain English Converter ────────────────────────────────
    print("\n  § 7. Plain English Converter")

    if converter_imported and explainer_init:
        converter = PlainEnglishConverter()

        # Suboptimal batch for plain English
        suboptimal = {
            "temperature": 190.0,
            "conveyor_speed": 85.0,
            "hold_time": 25.0,
            "batch_size": 600.0,
            "material_type": 2,
            "hour_of_day": 15,
            "operator_exp": 0,
        }

        explanation = explainer.explain_single(suboptimal, target="energy_kwh")

        try:
            converted = converter.convert(explanation)
            convert_ok = True
        except Exception as e:
            convert_ok = False
            converted = {}
            print(f"    [ERROR] {e}")
        check("convert() returns without error", convert_ok)

        if convert_ok:
            check("Result has 'summary' key", "summary" in converted)
            check("Summary is non-empty string", isinstance(converted.get("summary"), str) and len(converted["summary"]) > 10,
                  f"length: {len(converted.get('summary', ''))}")

            # Each contribution should now have plain_english
            contribs = converted.get("feature_contributions", [])
            has_pe = all("plain_english" in c for c in contribs)
            check("All contributions have 'plain_english' key", has_pe)

            if has_pe and contribs:
                first_pe = contribs[0]["plain_english"]
                check("plain_english is non-empty string", isinstance(first_pe, str) and len(first_pe) > 10,
                      f"'{first_pe[:60]}...'")

                # Should mention units for the top contributor
                check("Plain English mentions unit (kWh)", "kWh" in first_pe, f"text: '{first_pe[:80]}'")

        # Test feature_sentence directly
        sentence = converter.feature_sentence(
            feature="hold_time",
            value=25.0,
            contribution=3.8,
            target="energy_kwh",
            unit="kWh",
        )
        check("feature_sentence() for hold_time mentions 'optimal'", "optimal" in sentence.lower(),
              f"'{sentence[:80]}'")
        check("feature_sentence() mentions actual value (25)", "25" in sentence, f"'{sentence[:80]}'")

        # Test feature_sentence for material_type
        mat_sentence = converter.feature_sentence(
            feature="material_type",
            value=2.0,
            contribution=1.9,
            target="energy_kwh",
            unit="kWh",
        )
        check("material_type sentence mentions Type-C", "Type-C" in mat_sentence,
              f"'{mat_sentence[:80]}'")

    # ── Section 8: Edge Cases ─────────────────────────────────────────────
    print("\n  § 8. Edge Cases")

    if explainer_init:
        # Invalid target should raise ValueError
        try:
            explainer.explain_single(optimal_params, target="invalid_target")
            invalid_caught = False
        except ValueError:
            invalid_caught = True
        except Exception:
            invalid_caught = False
        check("Invalid target raises ValueError", invalid_caught)

        # get_baseline for all targets
        for t in targets:
            b = explainer.get_baseline(t)
            check(f"get_baseline('{t}') returns float > 0", isinstance(b, float) and b > 0,
                  f"{b:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  🎉 ALL CHECKS PASSED — F1.4 SHAP Explainability verified!")
    else:
        print(f"  ⚠️  {failed} check(s) failed — review above.")
    print("=" * 65)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
