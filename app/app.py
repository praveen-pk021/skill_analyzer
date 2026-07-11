import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.placement_pipeline import (
    build_gap_report,
    build_model,
    load_model,
    load_stats,
    predict_probability,
)


def main() -> None:
    st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")

    st.title("Campus Placement Predictor + Skill Gap Analyzer")
    st.caption("Estimate placement probability and see which profile factors are holding the score back.")

    try:
        model = load_model()
        stats = load_stats()
    except (FileNotFoundError, ModuleNotFoundError, ValueError):
        st.info("Training model artifacts on first launch...")
        try:
            model = build_model()
            stats = load_stats()
        except Exception as exc:
            st.error(f"Model initialization failed: {exc}")
            st.stop()

    with st.sidebar:
        st.header("Profile Input")
        gender = st.selectbox("Gender", ["M", "F"])
        ssc_p = st.slider("10th %", 40, 100, 75)
        hsc_p = st.slider("12th %", 40, 100, 75)
        degree_p = st.slider("Degree %", 40, 100, 75)
        etest_p = st.slider("MBA Entrance Test %", 40, 100, 75)
        mba_p = st.slider("MBA %", 40, 100, 75)
        specialisation = st.selectbox("Specialisation", ["Mkt&Fin", "Mkt&HR"])
        workex = st.selectbox("Work Experience", ["Yes", "No"])

    profile = {
        "gender": gender,
        "ssc_p": ssc_p,
        "hsc_p": hsc_p,
        "degree_p": degree_p,
        "etest_p": etest_p,
        "mba_p": mba_p,
        "specialisation": specialisation,
        "workex": workex,
    }

    prediction_tab, gap_tab = st.tabs(["Predict my placement chances", "Gap report"])

    with prediction_tab:
        if st.button("Predict my placement chances", width="stretch"):
            probability = predict_probability(model, profile)
            st.metric("Placement probability", f"{probability * 100:.1f}%")
            st.progress(probability)

            if probability >= 0.75:
                st.success("Strong likelihood of placement based on this profile.")
            elif probability >= 0.5:
                st.info("Moderate outlook. A few profile areas could improve the score.")
            else:
                st.warning("Lower-than-average outlook. The gap report below highlights the main factors to improve.")

    with gap_tab:
        gap_report = build_gap_report(profile, stats, top_n=4)
        st.subheader("Weakest factors")
        if gap_report:
            report_df = pd.DataFrame(gap_report)
            report_df["target"] = report_df["placed_mean"].round(1)
            st.dataframe(report_df[["feature", "user_value", "placed_mean", "gap", "target"]], width="stretch")
            for item in gap_report:
                st.write(
                    f"- {item['feature']}: your score {item['user_value']:.1f} vs placed cohort mean {item['placed_mean']:.1f} (gap {item['gap']:.1f})"
                )
        else:
            st.write("No gaps detected.")


if __name__ == "__main__":
    main()
