import hashlib
import io
from html import escape

import pandas as pd
import altair as alt
import pydeck as pdk
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from data_preprocessing import CLASSES, FEATURES, MODEL_PATH, POLLUTANTS
from predict import load_model, predict_data

st.set_page_config(page_title="AQI Prediction", page_icon="🌍", layout="wide")
LABELS = {"pm25": "PM2.5 AQI subindex", "pm10": "PM10 AQI subindex",
          "no2": "NO2 AQI subindex", "co": "CO AQI subindex", "temp_c": "Temperature (°C)",
          "lat": "Latitude", "lon": "Longitude"}
RANGES = dict(zip(CLASSES, ["0-50", "51-100", "101-150", "151-200", "201-300", "301+"]))
TEMPLATE = pd.DataFrame([{ "pm25": 35.0, "pm10": 20.0, "lon": 79.8, "lat": 6.9,
                           "no2": 8.0, "co": 2.0, "temp_c": 25.0}], columns=FEATURES)
CATEGORY_COLORS = dict(zip(CLASSES, ["#21844b", "#b88200", "#d66b17", "#d13743", "#854ac0", "#872b49"]))
SHORT_CATEGORIES = ["Good", "Moderate", "Sensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"]


def apply_style():
    """Keep presentation here; model loading and prediction stay separate."""
    st.markdown("""
    <style>
    .stApp {background: #f5f7fb; color: #192b42;}
    .block-container {max-width: 1480px; padding-top: 2.5rem; padding-bottom: 2rem;}
    [data-testid="stHeader"] {background: #f5f7fb;}
    h1, h2, h3 {color: #192b42;}
    [data-testid="stSidebar"] {background: #252d3d;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {color: #edf3fb;}
    [data-testid="stSidebar"] [data-testid="stForm"] {border: 0; padding: 0;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #c5d0df;}
    [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {background: white;}
    .stTabs [data-baseweb="tab-list"] {background: #e7edf5; padding: 6px; border-radius: 10px; gap: 6px;}
    .stTabs [data-baseweb="tab"] {height: 44px; border-radius: 7px; padding: 0 22px; color: #32465f;}
    .stTabs [aria-selected="true"] {background: #2463ae; color: white;}
    .stTabs [aria-selected="true"] p {color: white;}
    .stTabs [data-baseweb="tab-highlight"] {background: transparent;}
    .stTabs [data-baseweb="tab-border"] {display: none;}
    .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
        background: #2463ae; border-color: #2463ae; color: white; border-radius: 8px;}
    .stDownloadButton button {border-color: #c8d8ec; color: #205c9e; background: #edf4fc;}
    [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius: 12px;}
    .st-key-score_card, .st-key-pollutant_card, .st-key-location_card {
        background: #ffffff; border-radius: 14px;}
    [data-testid="stVegaLiteChart"], [data-testid="stDeckGlJsonChart"] {
        border-radius: 10px; overflow: hidden;}
    .dashboard-heading {font-size: clamp(1.8rem, 3vw, 2.65rem); font-weight: 750;
        line-height: 1.2; margin: 0 0 10px; letter-spacing: -.03em;}
    .dashboard-subtitle {color: #607188; margin-bottom: 24px;}
    .eyebrow {font-size: .75rem; text-transform: uppercase; letter-spacing: .13em;
        color: #2463ae; font-weight: 700; margin-bottom: 10px;}
    .result-panel {background: white; border: 1px solid #dfe6ef; border-left: 6px solid;
        border-radius: 12px; padding: 24px; margin: 18px 0 22px;}
    .result-label {font-size: .85rem; color: #607188; margin-bottom: 8px;}
    .result-category {font-size: clamp(1.5rem, 2.5vw, 2.2rem); font-weight: 750; margin-bottom: 10px;}
    .result-detail {font-size: .95rem; color: #41546b;}
    .empty-panel {background: white; border: 1px solid #dfe6ef; border-radius: 12px;
        padding: 36px; margin: 20px 0;}
    .empty-panel h3 {margin: 0 0 10px;}
    .empty-panel p {color: #607188; margin: 0;}
    @media (max-width: 700px) {
        .block-container {padding: 1.5rem 1rem;}
        .stTabs [data-baseweb="tab"] {padding: 0 12px;}
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def cached_model(modified_time):
    return load_model()


def pdf_report(result):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    rows = [["Row", "Estimated category", "Class score"]]
    for index, row in result.head(500).reset_index(drop=True).iterrows():
        rows.append([str(index + 1), Paragraph(escape(row["Predicted category"]), styles["BodyText"]),
                     f"{row['Class score']:.3f}"])
    table = Table(rows, colWidths=[45, 285, 100], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d7eced")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    document.build([Paragraph("AQI Category Estimates", styles["Title"]),
                    Paragraph("Class scores are uncalibrated model outputs. "
                              "This report contains at most the first 500 input rows.", styles["BodyText"]),
                    Spacer(1, 12), table])
    return buffer.getvalue()


def light_chart(chart):
    """Explicit colours keep charts readable even with a browser dark theme."""
    return (chart.properties(background="#ffffff")
            .configure_view(stroke=None)
            .configure_axis(labelColor="#52647a", titleColor="#52647a",
                            labelFontSize=12, titleFontSize=12,
                            gridColor="#edf1f6", domain=False, tickColor="#dce4ee")
            .configure_text(color="#192b42", fontSize=12)
            .configure_legend(labelColor="#52647a", titleColor="#192b42"))


def prediction_visuals(result, scores):
    """Two readable charts followed by a full-width, light location map."""
    row = result.iloc[0]
    category = row["Predicted category"]
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        with st.container(border=True, key="score_card"):
            st.markdown("#### Category scores")
            st.caption("How the model distributes its score across the six categories.")
            chart_data = pd.DataFrame({"Category": CLASSES, "Label": SHORT_CATEGORIES,
                                       "Score": scores.iloc[0].to_numpy()})
            base = alt.Chart(chart_data).encode(
                x=alt.X("Score:Q", title=None, scale=alt.Scale(domain=[0, 1.15], nice=False),
                        axis=alt.Axis(values=[0, .25, .5, .75, 1], format="%")),
                y=alt.Y("Label:N", sort=SHORT_CATEGORIES, title=None,
                        axis=alt.Axis(labelLimit=140, labelPadding=10, ticks=False)),
                tooltip=[alt.Tooltip("Category:N", title="Category"),
                         alt.Tooltip("Score:Q", title="Model score", format=".1%")])
            bars = base.mark_bar(size=22, cornerRadiusEnd=5).encode(
                color=alt.Color("Category:N", scale=alt.Scale(domain=CLASSES,
                                range=list(CATEGORY_COLORS.values())), legend=None))
            labels = base.mark_text(align="left", dx=7, color="#192b42").encode(
                text=alt.Text("Score:Q", format=".1%"))
            st.altair_chart(light_chart((bars + labels).properties(height=280)),
                            width="stretch", theme=None)
            st.caption("Sensitive groups = Unhealthy for Sensitive Groups. Scores are uncalibrated.")
    with right:
        with st.container(border=True, key="pollutant_card"):
            st.markdown("#### Supplied pollutant indices")
            st.caption("Values you entered, displayed in AQI subindex units.")
            indices = pd.to_numeric(result[POLLUTANTS].iloc[0], errors="coerce").dropna()
            chart_data = indices.rename_axis("Pollutant").reset_index(name="Value")
            chart_data["Pollutant"] = chart_data["Pollutant"].map(
                {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂", "co": "CO"})
            maximum = max(1.0, float(indices.max()) * 1.22)
            base = alt.Chart(chart_data).encode(
                x=alt.X("Pollutant:N", sort=["PM2.5", "PM10", "NO₂", "CO"], title=None,
                        axis=alt.Axis(labelAngle=0, labelPadding=10, ticks=False)),
                y=alt.Y("Value:Q", title="AQI subindex",
                        scale=alt.Scale(domain=[0, maximum], nice=True)),
                tooltip=["Pollutant:N", alt.Tooltip("Value:Q", title="AQI subindex", format=".2f")])
            bars = base.mark_bar(color="#3174bc", size=44,
                                 cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            labels = base.mark_text(dy=-10, color="#192b42").encode(
                text=alt.Text("Value:Q", format=".1f"))
            st.altair_chart(light_chart((bars + labels).properties(height=280)),
                            width="stretch", theme=None)
            st.caption("Missing pollutants are omitted from this chart. These are not concentration values.")

    with st.container(border=True, key="location_card"):
        st.markdown("#### Input location")
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            lat, lon = float(row["lat"]), float(row["lon"])
            st.caption(f"Latitude {lat:.4f} · Longitude {lon:.4f} · Marker shows your supplied coordinates.")
            color = CATEGORY_COLORS[category].lstrip("#")
            rgb = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
            point = pd.DataFrame({"lat": [lat], "lon": [lon], "category": [category]})
            marker = pdk.Layer("ScatterplotLayer", data=point, get_position="[lon, lat]",
                               get_fill_color=rgb + [220], get_line_color=[255, 255, 255],
                               get_radius=200, radius_min_pixels=9, radius_max_pixels=18,
                               stroked=True, line_width_min_pixels=2, pickable=True)
            deck = pdk.Deck(
                layers=[marker], map_provider="carto",
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=9, pitch=0),
                tooltip={"text": "Input location\nLatitude: {lat}\nLongitude: {lon}\nEstimated category: {category}"})
            st.pydeck_chart(deck, height=350, width="stretch", key="input_location_map")
            st.caption("The marker represents one entered observation, not an area-wide pollution measurement. "
                       "Map tiles require an internet connection. Map data: © OpenStreetMap contributors · CARTO.")
        else:
            st.info("Enter latitude and longitude in the sidebar to show a location.")


def single_prediction(bundle):
    with st.sidebar.form("single_form"):
        st.markdown("### Set your parameters")
        st.caption("Pollutant values are AQI subindices, not concentrations. "
                   "Provide at least one pollutant. Empty fields use training medians.")
        values = {}
        for name in ["pm25", "pm10", "no2", "co", "temp_c", "lon", "lat"]:
            options = {"value": None, "format": "%.2f", "placeholder": "Enter value"}
            if name in POLLUTANTS:
                options["min_value"] = 0.0
            elif name == "lat":
                options.update(min_value=-90.0, max_value=90.0)
            elif name == "lon":
                options.update(min_value=-180.0, max_value=180.0)
            values[name] = st.number_input(LABELS[name], key=name, **options)
        submitted = st.form_submit_button("Predict AQI category", type="primary", width="stretch")
    if submitted:
        try:
            result, scores = predict_data(pd.DataFrame([values], columns=FEATURES), bundle)
            st.session_state["single"] = (result, scores)
        except (ValueError, TypeError, RuntimeError) as exc:
            st.session_state.pop("single", None)
            st.error(str(exc))
    if "single" not in st.session_state:
        st.markdown("""
        <div class="empty-panel"><div class="eyebrow">Single prediction</div>
        <h3>Explore the air quality at a location</h3>
        <p>Enter pollutant indices in the left sidebar, then select <b>Predict AQI category</b>.
        Your category, input charts and location will appear here.</p></div>
        """, unsafe_allow_html=True)
        return
    result, scores = st.session_state["single"]
    row = result.iloc[0]
    category = row["Predicted category"]
    color = CATEGORY_COLORS[category]
    st.markdown(f"""
    <div class="result-panel" style="border-left-color: {color};">
    <div class="result-label">ESTIMATED AQI CATEGORY</div>
    <div class="result-category" style="color: {color};">{escape(category)}</div>
    <div class="result-detail">Category AQI range: <b>{RANGES[category]}</b>
    &nbsp;&middot;&nbsp; Model score: <b>{row['Class score']:.1%}</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("The range describes the predicted category. The score is an uncalibrated model output.")
    if row["Missing inputs"]:
        st.caption("Filled from training medians: " + row["Missing inputs"])
    prediction_visuals(result, scores)
    st.download_button("Download prediction", result.to_csv(index=False),
                       "aqi_prediction.csv", mime="text/csv")


def batch_prediction(bundle):
    st.subheader("Predict from a file")
    st.write("Upload observations to estimate an AQI category for each row.")
    st.download_button("Download input template", TEMPLATE.to_csv(index=False),
                       "input_template.csv", mime="text/csv")
    st.caption("The template contains synthetic example inputs.")
    uploaded = st.file_uploader("Upload CSV or Excel observations", type=["csv", "xlsx"])
    if uploaded is None:
        return
    signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
    try:
        content = io.BytesIO(uploaded.getvalue())
        frame = pd.read_excel(content) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(content)
        if len(frame) > 50000:
            raise ValueError("Upload at most 50,000 rows at a time.")
        st.dataframe(frame.head(10), width="stretch")
        if st.button("Predict uploaded rows", type="primary"):
            result, _ = predict_data(frame, bundle)
            st.session_state["batch"] = (signature, result)
    except (ValueError, TypeError, RuntimeError, OSError, UnicodeError, ImportError) as exc:
        st.error(str(exc))
        return
    stored = st.session_state.get("batch")
    if stored is None or stored[0] != signature:
        return
    result = stored[1]
    st.success(f"Completed {len(result):,} predictions.")
    counts = result["Predicted category"].value_counts().reindex(CLASSES, fill_value=0)
    chart_data = pd.DataFrame({"Category": CLASSES, "Label": SHORT_CATEGORIES,
                               "Observations": counts.to_numpy()})
    chart = alt.Chart(chart_data).mark_bar(color="#3174bc", cornerRadiusEnd=4).encode(
        x=alt.X("Observations:Q", title="Number of observations", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Label:N", title=None, sort=SHORT_CATEGORIES),
        tooltip=["Category:N", "Observations:Q"]).properties(height=240)
    st.altair_chart(light_chart(chart), width="stretch", theme=None)
    st.dataframe(result, width="stretch")
    st.download_button("Download CSV", result.to_csv(index=False),
                       "aqi_batch_predictions.csv", mime="text/csv")
    st.download_button("Download PDF (first 500 rows)", pdf_report(result),
                       "aqi_report.pdf", mime="application/pdf")


def main():
    apply_style()
    with st.sidebar:
        st.markdown("## 🌍 AQI Prediction")
        st.caption("Pollutant inputs · category estimates")
        st.divider()
    st.markdown("""
    <div class="eyebrow">Air quality dashboard</div>
    <h1 class="dashboard-heading">Air Quality Index (AQI) Prediction</h1>
    <p class="dashboard-subtitle">Explore air quality categories from pollutant indices and weather observations.</p>
    """, unsafe_allow_html=True)
    try:
        bundle = cached_model(MODEL_PATH.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as exc:
        st.error("The saved predictor is unavailable. " + str(exc))
        st.stop()
    single, batch = st.tabs(["⚡ Single prediction", "📂 Batch prediction"])
    with single:
        single_prediction(bundle)
    with batch:
        batch_prediction(bundle)
    st.divider()
    st.caption("Data source: World Air Quality Index and originating monitoring agencies. "
               "Predictions use the observations you enter or upload.")


if __name__ == "__main__":
    main()
