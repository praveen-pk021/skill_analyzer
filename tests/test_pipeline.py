import pandas as pd

from src.placement_pipeline import build_gap_report, prepare_features


def test_prepare_features_keeps_expected_columns():
    df = pd.DataFrame(
        {
            "gender": ["M", "F"],
            "ssc_p": [80, 65],
            "hsc_p": [70, 60],
            "degree_p": [75, 68],
            "etest_p": [70, 72],
            "mba_p": [80, 69],
            "specialisation": ["Mkt&HR", "Mkt&Fin"],
            "workex": ["Yes", "No"],
            "status": ["Placed", "Not Placed"],
        }
    )

    features, target = prepare_features(df)

    assert "ssc_p" in features.columns
    assert "status" not in features.columns
    assert target.tolist() == [1, 0]


def test_build_gap_report_flags_weakest_features():
    placed_profile = {
        "ssc_p": 75.0,
        "degree_p": 78.0,
        "mba_p": 70.0,
        "etest_p": 72.0,
    }
    user_profile = {"ssc_p": 60.0, "degree_p": 70.0, "mba_p": 68.0, "etest_p": 75.0}

    report = build_gap_report(user_profile, placed_profile, top_n=2)

    assert len(report) == 2
    assert report[0]["feature"] == "ssc_p"
    assert report[0]["gap"] > 0
