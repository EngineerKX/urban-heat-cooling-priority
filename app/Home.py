"""Streamlit entry point (S7). Registers every page with st.navigation,
grouped by who it is for, and runs the selected one. The landing page itself
(findings, top 5, pipeline status) is app/overview.py.

Page paths are relative to this file, so st.switch_page("pages/...") and
st.page_link("pages/...") calls in the pages keep working.

Run with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Urban Heat & Cooling-Priority Mapping", page_icon="🌡️", layout="wide")

navigation = st.navigation({
    "": [st.Page("overview.py", title="Overview", icon=":material/home:", default=True)],
    "Explore": [
        st.Page("pages/2_Island_Map.py", title="Island Map", icon=":material/map:"),
        st.Page("pages/3_Subzone_Breakdown.py", title="Subzone Breakdown", icon=":material/bar_chart:"),
        st.Page("pages/4_Counterfactual_Greening.py", title="Counterfactual Greening", icon=":material/park:"),
    ],
    "Evidence": [
        st.Page("pages/5_Validation_Dashboard.py", title="Validation Dashboard", icon=":material/verified:"),
    ],
    "Team tools": [
        st.Page("pages/1_Label_Validation_Points.py", title="Label Validation Points", icon=":material/edit_location:"),
    ],
})
navigation.run()
