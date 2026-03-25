import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VeriTaScan – Malayalam Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Noto+Sans+Malayalam:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Courier New', monospace; }

/* Dark background */
.stApp { background-color: #080c14; color: #e8e0d0; }

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #f0c060, #e06020);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle {
    font-size: 0.75rem;
    letter-spacing: 4px;
    color: rgba(200,180,140,0.5);
    margin-top: 4px;
}

/* Result boxes */
.result-fake {
    background: rgba(255,60,60,0.08);
    border: 1px solid rgba(255,80,80,0.4);
    border-left: 4px solid #ff5050;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.result-true {
    background: rgba(60,255,160,0.08);
    border: 1px solid rgba(60,255,160,0.4);
    border-left: 4px solid #40ffb0;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.verdict-fake {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    letter-spacing: 3px;
    color: #ff6060;
}
.verdict-true {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    letter-spacing: 3px;
    color: #40ffb0;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin: 1rem 0; }
.metric-card {
    flex: 1;
    background: rgba(13,18,32,0.9);
    border: 1px solid rgba(240,192,96,0.15);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: #f0c060;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: rgba(200,180,140,0.4);
    margin-top: 2px;
}

/* Section labels */
.section-label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: rgba(200,180,140,0.45);
    margin-bottom: 6px;
    font-family: 'Syne', sans-serif;
}

/* Divider */
hr { border-color: rgba(240,192,96,0.1); }

/* Streamlit overrides */
div[data-testid="stTextArea"] textarea {
    background-color: rgba(5,8,16,0.8) !important;
    border: 1px solid rgba(240,192,96,0.2) !important;
    color: #e8e0d0 !important;
    font-family: 'Noto Sans Malayalam', monospace !important;
    font-size: 15px !important;
    border-radius: 8px !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #f0c060, #e06020) !important;
    color: #080c14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 6px !important;
    width: 100%;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(240,192,96,0.35) !important;
}
.stSelectbox > div > div {
    background-color: rgba(13,18,32,0.9) !important;
    border: 1px solid rgba(240,192,96,0.2) !important;
    color: #e8e0d0 !important;
}
.stProgress > div > div { background-color: #f0c060 !important; }
[data-testid="stSidebar"] { background-color: #0d1220 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    true_df = pd.read_csv("ma_true.csv")
    fake_df = pd.read_csv("ma_fake.csv")

    true_df["label"] = 1
    fake_df["label"] = 0

    data = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    X, y = data["text"].astype(str), data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # character n-grams work well for Malayalam
        ngram_range=(2, 4),
        max_features=50_000,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Fake", "True"], output_dict=True)

    return vectorizer, model, acc, report, data


def predict(text, vectorizer, model):
    vec  = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return pred, prob


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">🔍 VERITASCAN</div>
  <div class="app-subtitle">MALAYALAM FAKE NEWS DETECTOR · TF-IDF + NAIVE BAYES</div>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("⚙️ Training model on dataset…"):
    vectorizer, model, acc, report, data = load_model()

st.markdown("---")

# ── Model metrics banner ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Accuracy",  f"{acc*100:.1f}%")
c2.metric("Fake Precision", f"{report['Fake']['precision']*100:.1f}%")
c3.metric("True Precision", f"{report['True']['precision']*100:.1f}%")
c4.metric("Dataset Size",   f"{len(data):,}")

st.markdown("---")

# ── Precision / Recall / F1 section ───────────────────────────────────────────
st.markdown('<div class="section-label">PRECISION · RECALL · F1 SCORE</div>', unsafe_allow_html=True)

# Per-class table
prf_data = {
    "Class":     ["⚠ Fake News", "✅ True News", "Macro Avg", "Weighted Avg"],
    "Precision": [
        f"{report['Fake']['precision']*100:.1f}%",
        f"{report['True']['precision']*100:.1f}%",
        f"{report['macro avg']['precision']*100:.1f}%",
        f"{report['weighted avg']['precision']*100:.1f}%",
    ],
    "Recall": [
        f"{report['Fake']['recall']*100:.1f}%",
        f"{report['True']['recall']*100:.1f}%",
        f"{report['macro avg']['recall']*100:.1f}%",
        f"{report['weighted avg']['recall']*100:.1f}%",
    ],
    "F1 Score": [
        f"{report['Fake']['f1-score']*100:.1f}%",
        f"{report['True']['f1-score']*100:.1f}%",
        f"{report['macro avg']['f1-score']*100:.1f}%",
        f"{report['weighted avg']['f1-score']*100:.1f}%",
    ],
    "Support": [
        int(report['Fake']['support']),
        int(report['True']['support']),
        int(report['macro avg']['support']),
        int(report['weighted avg']['support']),
    ],
}
prf_df = pd.DataFrame(prf_data).set_index("Class")
st.dataframe(prf_df, use_container_width=True)

# Visual bar breakdown
st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
col_f, col_t = st.columns(2)

with col_f:
    st.markdown('<div class="section-label">⚠ FAKE NEWS</div>', unsafe_allow_html=True)
    for metric, val in [
        ("Precision", report['Fake']['precision']),
        ("Recall",    report['Fake']['recall']),
        ("F1 Score",  report['Fake']['f1-score']),
    ]:
        st.markdown(f"<span style='font-size:12px;color:rgba(200,180,140,0.5);letter-spacing:1px;'>{metric}</span> "
                    f"<strong style='color:#ff6060'>{val*100:.1f}%</strong>", unsafe_allow_html=True)
        st.progress(int(val * 100))

with col_t:
    st.markdown('<div class="section-label">✅ TRUE NEWS</div>', unsafe_allow_html=True)
    for metric, val in [
        ("Precision", report['True']['precision']),
        ("Recall",    report['True']['recall']),
        ("F1 Score",  report['True']['f1-score']),
    ]:
        st.markdown(f"<span style='font-size:12px;color:rgba(200,180,140,0.5);letter-spacing:1px;'>{metric}</span> "
                    f"<strong style='color:#40ffb0'>{val*100:.1f}%</strong>", unsafe_allow_html=True)
        st.progress(int(val * 100))

# ── F-beta Score ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">F-BETA SCORE (Fβ)</div>', unsafe_allow_html=True)
st.markdown(
    "<span style='font-size:12px;color:rgba(200,180,140,0.45);'>"
    "Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall) &nbsp;·&nbsp; "
    "β &lt; 1 favours Precision &nbsp;·&nbsp; β = 1 is F1 &nbsp;·&nbsp; β &gt; 1 favours Recall"
    "</span>",
    unsafe_allow_html=True
)

beta = st.slider("Select β value", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

def fbeta(precision, recall, b):
    if precision + recall == 0:
        return 0.0
    return (1 + b**2) * precision * recall / (b**2 * precision + recall)

fb_fake  = fbeta(report['Fake']['precision'], report['Fake']['recall'], beta)
fb_true  = fbeta(report['True']['precision'], report['True']['recall'], beta)
fb_macro = (fb_fake + fb_true) / 2

col1, col2, col3 = st.columns(3)
col1.metric(f"F{beta:.1f} — Fake",     f"{fb_fake*100:.2f}%")
col2.metric(f"F{beta:.1f} — True",     f"{fb_true*100:.2f}%")
col3.metric(f"F{beta:.1f} — Macro Avg",f"{fb_macro*100:.2f}%")

# Bar chart comparing F1 vs Fbeta
fb_chart = pd.DataFrame({
    "F1 Score":      [report['Fake']['f1-score']*100, report['True']['f1-score']*100],
    f"F{beta:.1f} Score": [fb_fake*100, fb_true*100],
}, index=["Fake", "True"])
st.bar_chart(fb_chart)

st.markdown(
    f"<div style='font-size:11px;color:rgba(200,180,140,0.35);margin-top:4px;'>"
    f"With β={beta:.1f}: {'recall is weighted higher than precision' if beta > 1 else 'precision is weighted higher than recall' if beta < 1 else 'precision and recall are equally weighted (standard F1)'}"
    f"</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_detect, tab_batch, tab_explore = st.tabs(["🔍 Detect", "📋 Batch", "📊 Explore"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – SINGLE DETECTION
# ════════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.markdown('<div class="section-label">PASTE MALAYALAM NEWS TEXT</div>', unsafe_allow_html=True)

    # Sample selector
    samples = {
        "— choose a sample —": "",
        "⚠ Fake: Modi NYT claim":       "മോദി ലോകത്തിൻ്റെ പ്രതീക്ഷ  പ്രധാനമന്ത്രിയെ പുകഴ്ത്തിയ ന്യൂ യോർക്ക് ടൈംസ് വാർത്ത വ്യാജമോ?",
        "⚠ Fake: Amarinder BJP claim":  "അമരീന്ദര്‍ സിങ്ങ് ബിജെപിയിൽ ചേര്‍ന്നോ? അമിത് ഷായ്ക്കൊപ്പമുള്ള ചിത്രത്തിന്റെ സത്യമെന്ത്?",
        "✓ True: Kochi fire news":      "കൊച്ചി ചെരുപ്പ് വിതരണ കേന്ദ്രത്തിലെ തീപിടിത്തം; അന്വേഷണം തുടങ്ങി, അട്ടിമറി സാധ്യതയും പരിശോധിക്കുന്നു",
        "✓ True: Sreesanth ban lifted":  "ശ്രീശാന്തിന്റെ ആജീവനാന്ത വിലക്ക് പിന്‍വലിച്ചു",
    }
    choice = st.selectbox("Quick sample", list(samples.keys()))
    default_text = samples[choice]

    user_text = st.text_area(
        "News text",
        value=default_text,
        height=160,
        placeholder="Type or paste Malayalam news here…",
        label_visibility="collapsed",
    )

    if st.button("ANALYZE →", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Scanning…"):
                time.sleep(0.4)   # slight pause for UX
                pred, prob = predict(user_text, vectorizer, model)

            label      = "FAKE NEWS" if pred == 0 else "TRUE NEWS"
            css_class  = "result-fake" if pred == 0 else "result-true"
            verdict_cls = "verdict-fake" if pred == 0 else "verdict-true"
            icon       = "⚠️" if pred == 0 else "✅"
            conf_fake  = prob[0] * 100
            conf_true  = prob[1] * 100

            st.markdown(f"""
            <div class="{css_class}">
              <div class="{verdict_cls}">{icon} {label}</div>
              <div style="margin-top:10px;font-size:13px;color:rgba(200,180,140,0.65);">
                Fake confidence: <strong style="color:#ff8080">{conf_fake:.1f}%</strong>
                &nbsp;|&nbsp;
                True confidence: <strong style="color:#40ffb0">{conf_true:.1f}%</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">FAKE PROBABILITY</div>', unsafe_allow_html=True)
            st.progress(int(conf_fake))

            # Top TF-IDF features (explainability)
            st.markdown("---")
            st.markdown('<div class="section-label">TOP CONTRIBUTING CHARACTER N-GRAMS</div>', unsafe_allow_html=True)
            vec_out   = vectorizer.transform([user_text])
            feature_names = vectorizer.get_feature_names_out()
            nonzero_idx   = vec_out.nonzero()[1]
            scores        = np.array(vec_out[0, nonzero_idx]).flatten()
            top_idx       = nonzero_idx[np.argsort(scores)[::-1][:10]]
            top_features  = [(feature_names[i], float(vec_out[0, i])) for i in top_idx]

            cols = st.columns(2)
            for i, (feat, score) in enumerate(top_features):
                cols[i % 2].code(f"{feat}  ({score:.3f})")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – BATCH DETECTION
# ════════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("Upload a CSV with a **`text`** column to classify multiple headlines at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        if "text" not in df_in.columns:
            st.error("CSV must have a `text` column.")
        else:
            with st.spinner(f"Classifying {len(df_in)} rows…"):
                df_in["prediction"] = df_in["text"].astype(str).apply(
                    lambda t: "True" if predict(t, vectorizer, model)[0] == 1 else "Fake"
                )
                df_in["fake_prob"] = df_in["text"].astype(str).apply(
                    lambda t: f"{predict(t, vectorizer, model)[1][0]*100:.1f}%"
                )

            st.success(f"Done! Classified {len(df_in)} articles.")
            fake_count = (df_in["prediction"] == "Fake").sum()
            true_count = (df_in["prediction"] == "True").sum()

            col1, col2 = st.columns(2)
            col1.metric("⚠ Fake", fake_count)
            col2.metric("✅ True", true_count)

            st.dataframe(df_in[["text", "prediction", "fake_prob"]], use_container_width=True)

            csv_out = df_in.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Results CSV", csv_out, "results.csv", "text/csv")

    else:
        st.info("No file uploaded yet. You can also try the sample CSVs `ma_true.csv` / `ma_fake.csv`.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown('<div class="section-label">DATASET OVERVIEW</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", len(data))
    col2.metric("Fake Articles",  int((data["label"] == 0).sum()))
    col3.metric("True Articles",  int((data["label"] == 1).sum()))

    st.markdown("---")
    st.markdown('<div class="section-label">F-SCORE SUMMARY</div>', unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:12px;color:rgba(200,180,140,0.45);'>"
        "The harmonic mean of precision and recall. F1 equally weights both metrics."
        "</span>",
        unsafe_allow_html=True
    )

    fscore_data = {
        "Class":   ["⚠ Fake News", "✅ True News", "Macro Average", "Weighted Average"],
        "F1 Score": [
            f"{report['Fake']['f1-score']*100:.2f}%",
            f"{report['True']['f1-score']*100:.2f}%",
            f"{report['macro avg']['f1-score']*100:.2f}%",
            f"{report['weighted avg']['f1-score']*100:.2f}%",
        ],
    }
    fscore_df = pd.DataFrame(fscore_data)
    st.dataframe(fscore_df, use_container_width=True, hide_index=True)

    # Metrics cards for F-scores
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    fcol1.metric("F1 — Fake", f"{report['Fake']['f1-score']*100:.2f}%")
    fcol2.metric("F1 — True", f"{report['True']['f1-score']*100:.2f}%")
    fcol3.metric("F1 — Macro", f"{report['macro avg']['f1-score']*100:.2f}%")
    fcol4.metric("F1 — Weighted", f"{report['weighted avg']['f1-score']*100:.2f}%")

    # F-score visualization
    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
    fscore_chart = pd.DataFrame({
        "F1 Score": [report['Fake']['f1-score']*100, report['True']['f1-score']*100],
    }, index=["Fake", "True"])
    st.bar_chart(fscore_chart)

    st.markdown("---")
    st.markdown('<div class="section-label">SAMPLE ARTICLES</div>', unsafe_allow_html=True)

    label_filter = st.radio("Filter by", ["All", "Fake", "True"], horizontal=True)
    n_show = st.slider("Rows to show", 5, 50, 10)

    filtered = data.copy()
    filtered["label_name"] = filtered["label"].map({0: "⚠ Fake", 1: "✅ True"})
    if label_filter == "Fake":
        filtered = filtered[filtered["label"] == 0]
    elif label_filter == "True":
        filtered = filtered[filtered["label"] == 1]

    st.dataframe(
        filtered[["text", "label_name"]].head(n_show).rename(columns={"label_name": "Label"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown('<div class="section-label">TEXT LENGTH DISTRIBUTION</div>', unsafe_allow_html=True)
    data["char_count"] = data["text"].astype(str).str.len()
    chart_data = pd.DataFrame({
        "Fake": data[data["label"] == 0]["char_count"].value_counts().sort_index(),
        "True": data[data["label"] == 1]["char_count"].value_counts().sort_index(),
    }).fillna(0)
    st.bar_chart(chart_data.clip(upper=200).head(80))


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:11px;letter-spacing:2px;color:rgba(200,180,140,0.25);'>"
    "VERITASCAN · MALAYALAM FAKE NEWS DETECTION · TF-IDF + MULTINOMIAL NAIVE BAYES"
    "</div>",
    unsafe_allow_html=True,
)
