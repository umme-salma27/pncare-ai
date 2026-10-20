"""PNCare-AI: Delayed Postnatal Care Risk Predictor.

Streamlit front end for the final soft-voting model (LR + LightGBM + XGBoost + RF) trained on pooled
BDHS 2017-18 and 2022 data. The inputs are the 22 features selected by RFE, entered as 12 fields (several
features are computed from the same field, e.g. place of delivery gives 8 of them). Survey year is fixed at 2022.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
import json
import math
import os
import sys
import warnings

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import pnc_core  # noqa: E402,F401  (the saved pipeline is built from these classes)

warnings.filterwarnings("ignore")
MODEL_PATH = os.path.join(APP_DIR, "pnc_app_model.joblib")
REF_PATH = os.path.join(APP_DIR, "pnc_app_reference.json")
MIN_PLACE_SHARE = 0.02

st.set_page_config(page_title="PNCare-AI", page_icon=":material/child_care:", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------------------------
# Inputs (display text -> value the model was trained on)
# ----------------------------------------------------------------------------------------------
PLACE = {
    "Home": "home",
    "Upazila Health Complex": "upazila health complex",
    "District hospital": "district hospital",
    "Medical college hospital": "medical college hospital",
    "Specialized government hospital": "specialized govt hospital",
    "Maternal and Child Welfare Centre (MCWC)": "mcwc",
    "Union Health and Family Welfare Centre": "uh & family welfare center",
    "Community clinic": "community clinic",
    "Other public facility": "other public sector",
    "Private hospital": "private hospital",
    "Private clinic": "private clinic",
    "Private medical college hospital": "private medical college hospital",
    "Other private medical facility": "other private medical",
    "NGO static clinic": "ngo static clinic",
    "Delivery hut": "delivery hut",
    "Other place": "other",
}
EDUCATION = {"No education": "no education", "Primary": "primary", "Secondary or higher": "secondary"}
DIVISION = {d.title(): d for d in ["barishal", "chattogram", "dhaka", "khulna", "mymensingh", "rajshahi", "rangpur", "sylhet"]}
RESIDENCE = {"Rural": "rural", "Urban": "urban"}
MEDIA = {"Not at all": "not at all", "Less than weekly": "less than once a week", "At least weekly": "at least once a week"}
MEDIA_SCORE = {"not at all": 0, "less than once a week": 1, "at least once a week": 2}
YES_NO = {"Yes": True, "No": False}

INPUTS = [
    dict(key="place", csv="place_of_delivery", label="Place of delivery", kind="cat", options=PLACE),
    dict(key="anc", csv="anc_visits", label="ANC visits", kind="num", min=0, max=20, unit="visits"),
    dict(key="age", csv="mother_age_at_birth", label="Mother's age at birth", kind="num", min=13, max=49, unit="years"),
    dict(key="edu", csv="education", label="Education", kind="cat", options=EDUCATION),
    dict(key="division", csv="division", label="Division", kind="cat", options=DIVISION),
    dict(key="residence", csv="residence", label="Residence", kind="cat", options=RESIDENCE),
    dict(key="newspaper", csv="newspaper", label="Newspaper", kind="cat", options=MEDIA),
    dict(key="radio", csv="radio", label="Radio", kind="cat", options=MEDIA),
    dict(key="tv", csv="tv", label="Television", kind="cat", options=MEDIA),
    dict(key="poor", csv="poor_household", label="Poor household", kind="bool"),
    dict(key="parity", csv="birth_order_4_plus", label="4th or later birth", kind="bool"),
    dict(key="distance", csv="distance_big_problem", label="Distance a big problem", kind="bool"),
]
SPEC = {s["key"]: s for s in INPUTS}
for s in INPUTS:
    if s["kind"] == "cat":
        s["display"] = {v: k for k, v in s["options"].items()}
        s["parse"] = {**{k.lower(): v for k, v in s["options"].items()}, **{v: v for v in s["options"].values()}}
    elif s["kind"] == "bool":
        s["parse"] = {"yes": True, "no": False, "1": True, "0": False, "true": True, "false": False}

# Exact Shapley values are computed over these 10 players (the three media fields form one feature group)
PLAYERS = [("Place of delivery", ["place"]), ("ANC visits", ["anc"]), ("Mother's age at birth", ["age"]),
           ("Education", ["edu"]), ("Division", ["division"]), ("Residence", ["residence"]),
           ("Media exposure", ["newspaper", "radio", "tv"]), ("Poor household", ["poor"]),
           ("4th or later birth", ["parity"]), ("Distance a big problem", ["distance"])]

# The 22 RFE-selected model features, grouped by the input that produces them
FEATURE_NAMES = {
    "fe_place_group__home": "Delivery place: home",
    "fe_place_group__public": "Delivery place: public facility",
    "fe_place_group__private": "Delivery place: private facility",
    "m15_1__upazila_health_complex": "Upazila Health Complex",
    "m15_1__district_hospital": "District hospital",
    "m15_1__medical_college_hospital": "Medical college hospital",
    "m15_1__private_hospital": "Private hospital",
    "m15_1__private_clinic": "Private clinic",
    "fe_facility_x_2022": "Facility delivery x 2022",
    "fe_home_rural": "Home birth in a rural area",
    "m14_1": "ANC visits",
    "fe_no_anc": "No ANC visit",
    "fe_age_at_birth": "Mother's age at birth",
    "fe_teen_birth": "Teenage birth (<20 years)",
    "fe_secondary_plus": "Secondary or higher education",
    "v024__chattogram": "Division: Chattogram",
    "v024__barishal": "Division: Barishal",
    "v024__khulna": "Division: Khulna",
    "fe_media_index": "Media exposure index (0-6)",
    "fe_media_any": "Any media exposure",
    "fe_vulnerability": "Vulnerability score (0-7)",
    "survey_year": "Survey year 2022 (fixed)",
}
FEATURE_GROUPS = [
    ("Place of delivery", ["fe_place_group__home", "fe_place_group__public", "fe_place_group__private",
                           "m15_1__upazila_health_complex", "m15_1__district_hospital", "m15_1__medical_college_hospital",
                           "m15_1__private_hospital", "m15_1__private_clinic", "fe_facility_x_2022", "fe_home_rural"]),
    ("Antenatal care", ["m14_1", "fe_no_anc"]),
    ("Mother", ["fe_age_at_birth", "fe_teen_birth", "fe_secondary_plus"]),
    ("Location", ["v024__chattogram", "v024__barishal", "v024__khulna"]),
    ("Media exposure", ["fe_media_index", "fe_media_any"]),
    ("Vulnerability", ["fe_vulnerability"]),
    ("Survey", ["survey_year"]),
]
NUMERIC_FEATURES = {"m14_1", "fe_age_at_birth", "fe_media_index", "fe_vulnerability"}

TEST_METRICS = [("ROC-AUC", 0.9405), ("Accuracy", 0.8760), ("Precision", 0.8534), ("Recall", 0.8778),
                ("Specificity", 0.8745), ("F1-score", 0.8654)]
TEAL, RED = "#0f766e", "#dc2626"

WHO_URL = "https://www.who.int/publications/i/item/9789240045989"
REFERENCES = [
    "World Health Organization. <i>WHO recommendations on maternal and newborn care for a positive postnatal "
    f"experience.</i> Geneva: WHO; 2022. ISBN 978-92-4-004598-9. <a href='{WHO_URL}'>who.int</a>",
    "Croft TN, Marshall AMJ, Allen CK, et al. <i>Guide to DHS Statistics.</i> Rockville, Maryland, USA: ICF; 2018 "
    "(postnatal check during the first 2 days after birth).",
    "NIPORT, ICF. <i>Bangladesh Demographic and Health Survey 2017-18.</i> Dhaka and Rockville: NIPORT and ICF; 2020.",
    "NIPORT, ICF. <i>Bangladesh Demographic and Health Survey 2022: Final Report.</i> Dhaka and Rockville: "
    "NIPORT and ICF; 2024.",
]
ACTIONS_HIGH = [
    ("Ensure a postnatal check of the mother and newborn by skilled health personnel within 24 hours of birth: "
     "a facility stay of at least 24 hours after birth, or, after a home birth, the first contact as early as "
     "possible within 24 hours.", "Rec. 44"),
    ("Arrange a home visit in the first week after birth by skilled health personnel or a trained community "
     "health worker; if home visits are not feasible, book an outpatient postnatal contact.", "Rec. 48"),
    ("Assess the mother at each contact (bleeding, uterine tone, temperature, pulse and blood pressure in the "
     "first 24 hours; general well-being afterwards).", "Rec. 1"),
    ("Check the newborn for danger signs at each contact and refer if any is present: not feeding well, "
     "convulsions, fast breathing (>60/min), severe chest in-drawing, no spontaneous movement, fever (>37.5 °C), "
     "low temperature (<35.5 °C), jaundice in the first 24 hours or yellow palms and soles. Ask the family to seek "
     "care early if they see any of these signs.", "Rec. 25"),
    ("Schedule the remaining postnatal contacts: 48-72 hours, 7-14 days and week six after birth.", "Rec. 44"),
]
ACTIONS_LOW = [
    ("Follow the routine schedule of at least four postnatal contacts: within 24 hours, 48-72 hours, 7-14 days "
     "and week six after birth.", "Rec. 44"),
    ("Explain the newborn danger signs to the family and encourage them to seek care early if any appears.", "Rec. 25"),
]


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    import xgboost
    b = joblib.load(MODEL_PATH)
    # XGBoost is stored as model JSON (portable across operating systems), not as a pickled booster
    for est in b["model"].named_steps["model"].estimators_:
        if type(est).__name__ == "XGBClassifier":
            booster = xgboost.Booster()
            booster.load_model(bytearray(b["xgb_json"]))
            est._Booster = booster
    return b


bundle = load_model()
MODEL = bundle["model"]
TO_FEATURES = MODEL[:4]
THRESHOLD = bundle["threshold"]
YEAR = bundle["survey_year"]
_levels = {q["column"]: set(q.get("levels", [])) for q in bundle["questions"]}
_bad = {c: sorted(set(o.values()) - _levels[c]) for c, o in
        [("m15_1", PLACE), ("v106", EDUCATION), ("v024", DIVISION), ("v025", RESIDENCE), ("v157", MEDIA)]}
if any(_bad.values()):
    st.error(f"Input options do not match the model: { {k: v for k, v in _bad.items() if v} }")
    st.stop()
_missing = [c for c in bundle["model_columns"] if c not in FEATURE_NAMES]
if _missing or len(bundle["model_columns"]) != len(FEATURE_NAMES):
    st.error(f"Selected features do not match the model: {_missing}")
    st.stop()


def to_raw(df):
    """App inputs -> the raw columns the pipeline expects. Every mapping reproduces the selected features exactly:
    age at birth = mother's age with child age 0; wealth and birth order only enter through the vulnerability score
    (poor = poorest/poorer quintile; 4th or later birth)."""
    return pd.DataFrame({
        "m15_1": df["place"].values,
        "m14_1": df["anc"].astype(float).values,
        "v012": df["age"].astype(float).values,
        "b19_01": 0.0,
        "bord_01": np.where(df["parity"].astype(bool), 4.0, 2.0),
        "v106": df["edu"].values,
        "v190": np.where(df["poor"].astype(bool), "poorer", "middle"),
        "v467d": np.where(df["distance"].astype(bool), "big problem", "not a big problem"),
        "v024": df["division"].values,
        "v025": df["residence"].values,
        "v157": df["newspaper"].values,
        "v158": df["radio"].values,
        "v159": df["tv"].values,
        "survey_year": float(YEAR),
    })


def predict(df):
    return MODEL.predict_proba(to_raw(df))[:, 1]


def media_index(x):
    return sum(MEDIA_SCORE[x[k]] for k in ("newspaper", "radio", "tv"))


def vulnerability(x):
    return int(sum([x["edu"] == "no education", bool(x["poor"]), x["residence"] == "rural", x["age"] < 20,
                    bool(x["parity"]), bool(x["distance"]), media_index(x) == 0]))


# Reference mother for the explanations: most common answer / median of every input; place of delivery as a mix of
# the common places weighted by their share. pnc_app_reference.json holds these aggregates for 2022 training mothers.
_r = bundle["reference"]
REFERENCE = {"place": _r["m15_1"], "anc": float(_r["m14_1"]), "age": float(round(_r["v012"] - _r["b19_01"] / 12)),
             "edu": {"higher": "secondary"}.get(_r["v106"], _r["v106"]), "division": _r["v024"], "residence": _r["v025"],
             "newspaper": _r["v157"], "radio": _r["v158"], "tv": _r["v159"], "poor": _r["v190"] in ("poorest", "poorer"),
             "parity": float(_r["bord_01"]) >= 4, "distance": _r["v467d"] == "big problem"}
PLACE_MIX = [(REFERENCE["place"], 1.0)]
REF_TEXT = "a reference mother with the most common answer to every input"
if os.path.exists(REF_PATH):
    with open(REF_PATH, encoding="utf-8") as f:
        _ref = json.load(f)
    for k, d in _ref["inputs"].items():
        REFERENCE[k] = d["median"] if "median" in d else d["mode"]
    _shares = {k: v for k, v in _ref["inputs"]["place"]["shares"].items() if v >= MIN_PLACE_SHARE and k in PLACE.values()}
    PLACE_MIX = [(k, v / sum(_shares.values())) for k, v in _shares.items()]
    REF_TEXT = (f"a typical {YEAR} mother from the training data (most common answer or median of every input, with "
                f"place of delivery weighted by how often each place was used)")
REF_KEY = json.dumps([REFERENCE, PLACE_MIX], sort_keys=True, default=str)

N_P = len(PLAYERS)
PLACE_IDX = 0
_MASKS = ((np.arange(2 ** N_P)[:, None] >> np.arange(N_P)) & 1).astype(bool)
_SIZES = _MASKS.sum(axis=1)
_WEIGHTS = np.array([math.factorial(k) * math.factorial(N_P - k - 1) / math.factorial(N_P) for k in range(N_P)])


@st.cache_data(show_spinner=False)
def explain(input_items, ref_key):
    """Exact Shapley values over the 10 input groups, relative to the reference mother.
    Every subset of groups takes the mother's inputs, the rest the reference inputs (2^10 subsets); when place of
    delivery comes from the reference, the prediction is averaged over the weighted places in PLACE_MIX."""
    x = dict(input_items)
    X = pd.DataFrame({k: np.where(_MASKS[:, i], x[k], REFERENCE[k]) for i, (_, keys) in enumerate(PLAYERS) for k in keys})
    own = _MASKS[:, PLACE_IDX]
    p = predict(pd.concat([X[own]] + [X[~own].assign(place=pl) for pl, _ in PLACE_MIX], ignore_index=True))
    v = np.empty(2 ** N_P)
    v[own] = p[:own.sum()]
    v[~own] = np.array([w for _, w in PLACE_MIX]) @ p[own.sum():].reshape(len(PLACE_MIX), -1)
    idx = np.arange(2 ** N_P)
    phi = np.empty(N_P)
    for i in range(N_P):
        s = idx[~_MASKS[:, i]]
        phi[i] = np.sum(_WEIGHTS[_SIZES[s]] * (v[s | (1 << i)] - v[s]))
    return float(v[-1]), float(v[0]), phi


def selected_features(x):
    z = TO_FEATURES.transform(to_raw(pd.DataFrame([x]))).iloc[0]
    rows = []
    for group, cols in FEATURE_GROUPS:
        for c in cols:
            val = z[c]
            if c == "survey_year":
                shown = str(YEAR)
            elif c in NUMERIC_FEATURES:
                shown = f"{val:g}"
            else:
                shown = "Yes" if val >= 0.5 else "No"
            rows.append({"Group": group, "Selected feature": FEATURE_NAMES[c], "Value": shown})
    return pd.DataFrame(rows)


def player_text(name, keys, x):
    if name == "Media exposure":
        return f"Media exposure: index {media_index(x)} of 6"
    k = keys[0]
    s = SPEC[k]
    if s["kind"] == "cat":
        return f"{name}: {s['display'][x[k]]}"
    if s["kind"] == "bool":
        return f"{name}: {'Yes' if x[k] else 'No'}"
    return f"{name}: {x[k]:g} {s['unit']}"


# ----------------------------------------------------------------------------------------------
# Look
# ----------------------------------------------------------------------------------------------
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 1.5rem; max-width: 1320px;}}
header[data-testid="stHeader"] {{background: transparent;}}
[data-testid="stSidebar"] {{background: #f0fdfa; border-right: 1px solid #ccfbf1;}}
[data-testid="stSidebar"] .block-container {{padding-top: 1rem;}}
.pnc-banner {{background: linear-gradient(100deg, {TEAL} 0%, #14b8a6 62%, #5eead4 100%); color: #fff;
  padding: 1.05rem 1.5rem; border-radius: 14px; margin-bottom: 1rem; display: flex; align-items: center; gap: 1.1rem;}}
.pnc-banner .logo {{font-size: 2.3rem; background: rgba(255,255,255,.18); border-radius: 12px; width: 3.6rem; height: 3.6rem;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;}}
.pnc-banner h1 {{color: #fff; margin: 0; padding: 0; font-size: 1.85rem; line-height: 1.15;}}
.pnc-banner .sub {{font-size: 1.05rem; font-weight: 600; opacity: .96;}}
.pnc-banner p {{margin: .3rem 0 0; opacity: .93; font-size: .9rem;}}
.sec {{color: {TEAL}; font-weight: 700; font-size: .98rem; margin: .5rem 0 0; border-bottom: 2px solid #ccfbf1;
  padding-bottom: .1rem;}}
.feat {{color: #64748b; font-size: .76rem; margin: .1rem 0 .35rem;}}
.risk-card {{background: #fff; border-radius: 14px; padding: 1rem 1.25rem; box-shadow: 0 2px 10px rgba(15,23,42,.08);
  border: 1px solid #e2e8f0;}}
.risk-label {{font-weight: 800; letter-spacing: .03em; font-size: 1rem;}}
.risk-value {{font-size: 3rem; font-weight: 800; line-height: 1.05; margin: .15rem 0;}}
.risk-sub {{color: #475569; font-size: .86rem;}}
.bar {{position: relative; height: 14px; background: #e2e8f0; border-radius: 7px; margin: .85rem 0 .3rem;}}
.bar .fill {{position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px;}}
.bar .thr {{position: absolute; top: -5px; bottom: -5px; width: 3px; background: #0f172a; border-radius: 2px;}}
.bar-note {{color: #475569; font-size: .78rem;}}
.act {{border-radius: 12px; padding: .7rem 1rem .5rem; margin-top: .2rem;}}
.act ol {{margin: .2rem 0 .3rem 1.1rem; padding: 0;}}
.act li {{margin: .25rem 0; font-size: .88rem; line-height: 1.4;}}
.act .rec {{display: inline-block; font-size: .7rem; font-weight: 700; padding: 0 .4rem; border-radius: 6px;
  background: rgba(15,23,42,.08); margin-left: .25rem; white-space: nowrap;}}
.act .src {{font-size: .74rem; color: #475569; margin-top: .25rem;}}
.ref li {{font-size: .78rem; margin: .2rem 0;}}
.footer {{color: #64748b; font-size: .76rem; text-align: center; margin-top: 1.2rem; border-top: 1px solid #e2e8f0;
  padding-top: .6rem;}}
</style>
<div class="pnc-banner">
  <div class="logo">&#x1F931;</div>
  <div>
    <h1>PNCare-AI</h1>
    <div class="sub">Delayed Postnatal Care Risk Predictor</div>
    <p>Estimates the risk that a mother in Bangladesh will <b>not</b> receive a postnatal check from a medically
    trained provider within 48 hours of birth, from the 22 features selected by the model. Explainable machine learning
    on BDHS 2017-18 and 2022.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------
# Sidebar: guide, about, performance, features, limitations, references
# ----------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### How to use")
    st.markdown("1. Enter the mother's details.\n2. Click **Assess risk**.\n3. Read the risk, the reasons and the "
                "WHO-based suggested actions.")
    st.markdown("### Decision threshold")
    st.markdown(f"A mother is flagged **high risk** when her predicted risk is **{THRESHOLD:.0%} or more** "
                "(chosen on validation data to favour sensitivity).")
    with st.expander("About the model", expanded=True, icon=":material/info:"):
        st.markdown("**Outcome:** no postnatal check from a medically trained provider (doctor, nurse/midwife, FWV, "
                    "SACMO) within 48 hours of birth.\n\n"
                    "**Data:** BDHS 2017-18 and 2022, 9,733 mothers (45.4% delayed); train/validation/test 70/15/15.\n\n"
                    "**Model:** soft-voting ensemble of logistic regression, LightGBM, XGBoost and random forest "
                    "(Optuna-tuned), using 22 features selected by RFE.")
    with st.expander("Performance on the test set", icon=":material/monitoring:"):
        st.dataframe(pd.DataFrame(TEST_METRICS, columns=["Metric", f"Threshold {THRESHOLD:.2f}"]),
                     hide_index=True, width="stretch")
        st.caption("Held-out test set, 1,460 mothers. 5-fold CV ROC-AUC 0.937; nested CV 0.937.")
    with st.expander("The 22 selected features", icon=":material/checklist:"):
        st.markdown("\n".join(f"- **{g}:** " + "; ".join(FEATURE_NAMES[c] for c in cols) for g, cols in FEATURE_GROUPS))
        st.caption("The vulnerability score counts 7 factors: no education, poor household, rural residence, teenage "
                   "birth, 4th or later birth, distance a big problem and no media exposure.")
    with st.expander("Limitations", icon=":material/warning:"):
        st.markdown("- Cross-sectional survey data: associations, not causes.\n"
                    "- Lower accuracy within 2022 alone (ROC-AUC 0.83 vs 0.98 for 2017); the tool uses the 2022 setting.\n"
                    "- Not yet validated prospectively in health facilities.\n"
                    "- Decision support only; it does not replace clinical judgement.")
    with st.expander("References", icon=":material/menu_book:"):
        st.markdown('<ol class="ref">' + "".join(f"<li>{r}</li>" for r in REFERENCES) + "</ol>", unsafe_allow_html=True)
    st.caption(f"App version {bundle['app_version']}. Inputs are processed in memory and not stored.")

tab_one, tab_batch = st.tabs([":material/person: Risk assessment", ":material/groups: Batch screening"])

# ----------------------------------------------------------------------------------------------
# Tab 1: one mother
# ----------------------------------------------------------------------------------------------
with tab_one:
    x = {}

    def pick(key, label, help_text=None):
        s = SPEC[key]
        if s["kind"] == "cat":
            labels = list(s["options"])
            default = s["display"].get(REFERENCE[key], labels[0])
            x[key] = s["options"][st.selectbox(label, labels, index=labels.index(default), key=key, help=help_text)]
        elif s["kind"] == "num":
            x[key] = float(st.number_input(label, min_value=s["min"], max_value=s["max"], value=int(REFERENCE[key]),
                                           step=1, key=key, help=help_text))
        else:
            x[key] = YES_NO[st.radio(label, list(YES_NO), index=0 if REFERENCE[key] else 1, horizontal=True, key=key,
                                     help=help_text)]

    # ---------------- input form: two equal halves, button across the full width ----------------
    with st.container(border=True):
        form_left, form_right = st.columns(2, gap="large")
        with form_left:
            st.markdown('<div class="sec">Place of delivery and antenatal care</div>'
                        '<div class="feat">Features: delivery place group (home / public / private), 5 facility types, '
                        'facility delivery x 2022, home birth in a rural area, ANC visits, no ANC visit</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns([1.6, 1])
            with c1:
                pick("place", "Place of delivery")
            with c2:
                pick("anc", "ANC visits (number)")

            st.markdown('<div class="sec">Mother and location</div>'
                        '<div class="feat">Features: mother\'s age at birth, teenage birth, secondary or higher education, '
                        'division (Chattogram / Barishal / Khulna)</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                pick("age", "Mother's age at birth (years)")
                pick("division", "Division")
            with c2:
                pick("edu", "Education")
                pick("residence", "Residence")

        with form_right:
            st.markdown('<div class="sec">Media exposure</div>'
                        '<div class="feat">Features: media exposure index (0-6), any media exposure</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                pick("newspaper", "Newspaper")
            with c2:
                pick("radio", "Radio")
            with c3:
                pick("tv", "Television")

            st.markdown('<div class="sec">Vulnerability score</div>'
                        '<div class="feat">Feature: vulnerability score (0-7). Education, rural residence, teenage birth and '
                        'media exposure are taken from the left; add the other three factors here.</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                pick("poor", "Poor household", "Poorest or poorer wealth quintile.")
            with c2:
                pick("parity", "4th or later birth", "This baby is the mother's fourth or later child.")
            with c3:
                pick("distance", "Distance a big problem", "Distance to a health facility is a big problem for her.")
            st.markdown(f"**Vulnerability score: {vulnerability(x)} of 7** &nbsp;|&nbsp; "
                        f"**Media exposure index: {media_index(x)} of 6**")

        submitted = st.button("Assess risk", type="primary", width="stretch", icon=":material/monitor_heart:")

    if submitted:
        with st.spinner("Calculating risk and explanation..."):
            risk, ref_risk, phi = explain(tuple(sorted(x.items())), REF_KEY)
        st.session_state["result"] = {"inputs": dict(x), "risk": risk, "ref_risk": ref_risk, "phi": phi}

    # ---------------- output: below the form, full width ----------------
    res = st.session_state.get("result")
    if res is not None:
        if res["inputs"] != x:
            st.warning("The inputs have changed. Click **Assess risk** to update the result.", icon=":material/refresh:")
        risk, ref_risk, phi, xin = res["risk"], res["ref_risk"], res["phi"], res["inputs"]
        high = risk >= THRESHOLD
        color = RED if high else TEAL

        out_left, out_right = st.columns([1, 1.25], gap="large")
        with out_left:
            st.markdown(f"""
<div class="risk-card" style="border-left: 8px solid {color};">
  <div class="risk-label" style="color:{color};">{'HIGH RISK' if high else 'LOW RISK'} OF DELAYED POSTNATAL CARE</div>
  <div class="risk-value" style="color:{color};">{risk:.0%}</div>
  <div class="risk-sub">Estimated probability of no postnatal check by a medically trained provider within 48 hours of birth</div>
  <div class="bar"><div class="fill" style="width:{risk * 100:.1f}%; background:{color};"></div>
    <div class="thr" style="left:calc({THRESHOLD * 100:.1f}% - 1px);"></div></div>
  <div class="bar-note">Black line = decision threshold ({THRESHOLD:.0%}). At this threshold the model identified 87.8% of
  mothers with delayed care in the test data (specificity 87.5%).</div>
</div>""", unsafe_allow_html=True)

        with out_right:
            st.markdown("#### Suggested actions")
            actions = ACTIONS_HIGH if high else ACTIONS_LOW
            bg, border = ("#fef2f2", "#fecaca") if high else ("#f0fdfa", "#99f6e4")
            items = "".join(f'<li>{t} <span class="rec">WHO {r}</span></li>' for t, r in actions)
            st.markdown(f"""
<div class="act" style="background:{bg}; border:1px solid {border};">
  <ol>{items}</ol>
  <div class="src">Source: World Health Organization (2022). <i>WHO recommendations on maternal and newborn care for a
  positive postnatal experience</i>, recommendation numbers as shown. <a href="{WHO_URL}">who.int</a></div>
</div>""", unsafe_allow_html=True)

        st.markdown("#### Why this result?")
        contrib = pd.DataFrame({"factor": [player_text(n, k, xin) for n, k in PLAYERS], "points": phi * 100})
        contrib["effect"] = np.where(contrib["points"] >= 0, "Raises risk", "Lowers risk")
        contrib = contrib.reindex(contrib["points"].abs().sort_values(ascending=False).index)
        top = contrib[contrib["points"].abs() >= 0.5].head(3)
        if len(top):
            st.markdown("Main reasons: " + "; ".join(
                f"**{r.factor}** ({'+' if r.points > 0 else ''}{r.points:.1f} points)" for r in top.itertuples()) + ".")
        chart = alt.Chart(contrib).mark_bar(cornerRadius=3).encode(
            x=alt.X("points:Q", title="Contribution to risk (percentage points)"),
            y=alt.Y("factor:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=320, labelOverlap=False, labelFontSize=12)),
            color=alt.Color("effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RED, TEAL]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("factor:N", title="Input"), alt.Tooltip("points:Q", title="Points", format="+.1f")],
        ).properties(height=30 * len(contrib))
        st.altair_chart(chart, width="stretch")
        st.caption(f"SHAP (exact Shapley) values over the input groups, compared with {REF_TEXT}; reference risk "
                   f"{ref_risk:.0%}. The bars add up to this mother's risk minus the reference risk "
                   f"({(risk - ref_risk) * 100:+.1f} points). They show how the model uses the inputs, not proven causes.")

        with st.expander("The 22 selected features computed for this mother", icon=":material/table_view:"):
            st.dataframe(selected_features(xin), hide_index=True, width="stretch", height=35 * 23 + 3)
            st.caption("Values as the model sees them (ANC visits are capped at the training range).")

# ----------------------------------------------------------------------------------------------
# Tab 2: many mothers
# ----------------------------------------------------------------------------------------------
with tab_batch:
    st.markdown("Screen many mothers at once: upload a CSV file with one row per mother and the 12 columns below, then "
                "download the results with a risk score and risk level for each mother.")
    guide = pd.DataFrame([{"column": s["csv"], "input": s["label"],
                           "allowed values": (", ".join(s["options"]) if s["kind"] == "cat" else
                                              "Yes, No" if s["kind"] == "bool" else f"number {s['min']}-{s['max']}")}
                          for s in INPUTS])
    with st.expander("Required columns and allowed values", expanded=True, icon=":material/view_column:"):
        st.dataframe(guide, hide_index=True, width="stretch", height=35 * (len(guide) + 1) + 3)
    template = pd.DataFrame([
        {"place_of_delivery": "Home", "anc_visits": 1, "mother_age_at_birth": 19, "education": "Primary",
         "division": "Sylhet", "residence": "Rural", "newspaper": "Not at all", "radio": "Not at all", "tv": "Not at all",
         "poor_household": "Yes", "birth_order_4_plus": "No", "distance_big_problem": "Yes"},
        {"place_of_delivery": "Upazila Health Complex", "anc_visits": 4, "mother_age_at_birth": 27,
         "education": "Secondary or higher", "division": "Dhaka", "residence": "Urban", "newspaper": "Less than weekly",
         "radio": "Not at all", "tv": "At least weekly", "poor_household": "No", "birth_order_4_plus": "No",
         "distance_big_problem": "No"},
    ])
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                       file_name="pncare_template.csv", mime="text/csv", icon=":material/download:")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        raw.columns = [c.strip().lower() for c in raw.columns]
        missing = [s["csv"] for s in INPUTS if s["csv"] not in raw.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        elif len(raw) > 20000:
            st.error("Please upload at most 20,000 rows at a time.")
        else:
            X = pd.DataFrame(index=raw.index)
            problems = pd.Series("", index=raw.index)
            for s in INPUTS:
                if s["kind"] == "num":
                    v = pd.to_numeric(raw[s["csv"]], errors="coerce")
                    bad = v.isna() | (v < s["min"]) | (v > s["max"])
                else:
                    v = raw[s["csv"]].astype(str).str.strip().str.lower().map(s["parse"])
                    bad = v.isna()
                X[s["key"]] = v
                problems[bad] += s["csv"] + "; "
            ok = problems == ""
            out = raw.copy()
            out["risk_percent"] = np.nan
            out["risk_level"] = "not scored"
            if ok.any():
                p = predict(X.loc[ok])
                out.loc[ok, "risk_percent"] = np.round(p * 100, 1)
                out.loc[ok, "risk_level"] = np.where(p >= THRESHOLD, "High", "Low")
            out["invalid_columns"] = problems.str.rstrip("; ")
            c1, c2, c3 = st.columns(3)
            c1.metric("Mothers scored", f"{int(ok.sum()):,}", border=True)
            c2.metric("High risk", f"{int((out['risk_level'] == 'High').sum()):,}",
                      f"{(out['risk_level'] == 'High').sum() / max(ok.sum(), 1):.0%} of scored", delta_color="off", border=True)
            c3.metric("Rows with invalid values", f"{int((~ok).sum()):,}", border=True)
            if ok.any():
                hist = alt.Chart(out[ok]).mark_bar().encode(
                    x=alt.X("risk_percent:Q", bin=alt.Bin(step=5, extent=[0, 100]), title="Predicted risk (%)"),
                    y=alt.Y("count()", title="Mothers"),
                    color=alt.Color("risk_level:N", scale=alt.Scale(domain=["High", "Low"], range=[RED, TEAL]),
                                    legend=alt.Legend(title="Risk level", orient="bottom")))
                rule = alt.Chart(pd.DataFrame({"t": [THRESHOLD * 100]})).mark_rule(color="#0f172a", strokeDash=[5, 4]).encode(x="t:Q")
                st.altair_chart(hist + rule, width="stretch")
            st.dataframe(out, hide_index=True, width="stretch")
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="pncare_results.csv", mime="text/csv", type="primary", icon=":material/download:")

st.markdown('<div class="footer">PNCare-AI | Source study: PNCare-AI: An Explainable Machine Learning Framework for '
            'Predicting Delayed Postnatal Care in Bangladesh Using BDHS 2017-18 and 2022 Data | Suggested actions follow '
            'WHO (2022) postnatal care recommendations | Decision support only.</div>', unsafe_allow_html=True)
