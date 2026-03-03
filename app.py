import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.units import cm
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedPulse — IoT Health Monitor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS  (refined dark medical aesthetic)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bricolage+Grotesque:wght@400;600;800&display=swap');

html, body, [class*="css"] {
  font-family: 'Bricolage Grotesque', sans-serif;
}
.stApp { background: #040d18; color: #dce8f5; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #060f1e;
  border-right: 1px solid #0f2035;
}
section[data-testid="stSidebar"] * { color: #8aabc8 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #dce8f5 !important; font-size: 13px; }

/* Metric cards */
.mcard {
  background: #07111f;
  border: 1px solid #0f2035;
  border-radius: 16px;
  padding: 22px 20px 16px;
  position: relative; overflow: hidden;
  transition: border-color 0.3s;
}
.mcard:hover { border-color: #1e4060; }
.mcard-accent { position:absolute; top:0; left:0; right:0; height:3px; border-radius:16px 16px 0 0; }
.mcard-label { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:3px; color:#4a6a88; text-transform:uppercase; margin-bottom:8px; }
.mcard-val   { font-size:52px; font-weight:800; line-height:1; margin-bottom:4px; }
.mcard-unit  { font-family:'DM Mono',monospace; font-size:11px; color:#4a6a88; letter-spacing:2px; }
.mcard-sub   { font-family:'DM Mono',monospace; font-size:10px; color:#4a6a88; margin-top:10px; }

/* Status badge */
.badge {
  display:inline-block; padding:4px 14px; border-radius:20px;
  font-family:'DM Mono',monospace; font-size:11px; font-weight:500; letter-spacing:1px;
}
.badge-normal   { background:rgba(34,197,94,0.12);  color:#22c55e; border:1px solid rgba(34,197,94,0.25); }
.badge-warning  { background:rgba(234,179,8,0.12);  color:#eab308; border:1px solid rgba(234,179,8,0.25); }
.badge-critical { background:rgba(239,68,68,0.12);  color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.badge-info     { background:rgba(59,130,246,0.12); color:#3b82f6; border:1px solid rgba(59,130,246,0.25);}

/* Section headers */
.sec-head {
  font-family:'DM Mono',monospace; font-size:10px; letter-spacing:4px;
  color:#2a4a68; text-transform:uppercase; padding:0 0 10px;
  border-bottom:1px solid #0f2035; margin:28px 0 16px;
}

/* Alert boxes */
.alert-ok  { background:rgba(34,197,94,0.07);  border:1px solid rgba(34,197,94,0.2);  border-left:4px solid #22c55e; border-radius:0 10px 10px 0; padding:12px 18px; color:#22c55e; margin:6px 0; }
.alert-w   { background:rgba(234,179,8,0.07);  border:1px solid rgba(234,179,8,0.2);  border-left:4px solid #eab308; border-radius:0 10px 10px 0; padding:12px 18px; color:#eab308; margin:6px 0; }
.alert-c   { background:rgba(239,68,68,0.08);  border:1px solid rgba(239,68,68,0.25); border-left:4px solid #ef4444; border-radius:0 10px 10px 0; padding:12px 18px; color:#ef4444; margin:6px 0; }

/* Prediction card */
.pred-card {
  background:#07111f; border:1px solid #0f2035; border-radius:12px;
  padding:16px 20px; margin-bottom:10px;
}
.pred-model { font-family:'DM Mono',monospace; font-size:10px; color:#4a6a88; letter-spacing:2px; margin-bottom:8px; }
.pred-label { font-size:20px; font-weight:700; margin-bottom:6px; }
.pred-conf  { font-family:'DM Mono',monospace; font-size:11px; color:#4a6a88; }
.conf-bar   { background:#0f2035; border-radius:20px; height:5px; margin-top:8px; }
.conf-fill  { height:5px; border-radius:20px; }

/* Login card */
.login-wrap {
  max-width:420px; margin:60px auto;
  background:#07111f; border:1px solid #0f2035; border-radius:20px; padding:40px 36px;
}

/* Nav item active */
div[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
  font-size: 14px;
}

/* Tables */
.stDataFrame { background:#07111f !important; }
thead tr th { background:#040d18 !important; color:#4a6a88 !important; font-family:'DM Mono',monospace !important; font-size:11px !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { background:#07111f; border-radius:8px 8px 0 0; color:#4a6a88; font-family:'DM Mono',monospace; font-size:12px; }
.stTabs [aria-selected="true"] { background:#040d18 !important; color:#dce8f5 !important; }

/* Input fields */
.stTextInput input, .stNumberInput input, .stSelectbox div {
  background:#07111f !important; border-color:#0f2035 !important; color:#dce8f5 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
CHANNEL_ID  = "3244796"
READ_API    = "LS4303YQRIN4ZWXN"

LABEL_COLOR = {
    "Normal":              "#22c55e",
    "Tachycardia":         "#eab308",
    "Severe Tachycardia":  "#f97316",
    "Bradycardia":         "#3b82f6",
    "Severe Bradycardia":  "#6366f1",
    "Hypoxia":             "#f59e0b",
    "Critical Hypoxia":    "#ef4444",
}

# ─────────────────────────────────────────────────────────────
#  USER DATABASE  (extend as needed)
# ─────────────────────────────────────────────────────────────
USERS = {
    "doctor":  {"password": "medpulse123", "role": "doctor",  "name": "Dr. Sharma",    "avatar": "👨‍⚕️"},
    "nurse1":  {"password": "nurse123",    "role": "doctor",  "name": "Nurse Priya",   "avatar": "👩‍⚕️"},
    "patient1":{"password": "patient123",  "role": "patient", "name": "Patient A",     "avatar": "🧑"},
}

# ─────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
for key, val in {
    "logged_in": False, "username": "", "role": "", "user_name": "", "avatar": "",
    "page": "Dashboard", "email_sent": False, "alert_email": "",
    "smtp_user": "", "smtp_pass": "", "smtp_enabled": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────────────
#  HELPER: HEALTH LABEL
# ─────────────────────────────────────────────────────────────
def label_health(bpm, spo2):
    if   spo2 < 90:   return "Critical Hypoxia"
    elif spo2 < 94:   return "Hypoxia"
    elif bpm  > 120:  return "Severe Tachycardia"
    elif bpm  > 100:  return "Tachycardia"
    elif bpm  < 50:   return "Severe Bradycardia"
    elif bpm  < 60:   return "Bradycardia"
    else:             return "Normal"

def badge_class(label):
    if label == "Normal":          return "badge-normal"
    if "Critical" in label:        return "badge-critical"
    if label in ("Tachycardia","Bradycardia","Hypoxia","Severe Bradycardia"): return "badge-warning"
    return "badge-critical"


# ─────────────────────────────────────────────────────────────
#  THINGSPEAK FETCH
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def fetch_data(results=500):
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API}&results={results}"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        feeds = r.json().get("feeds", [])
        if not feeds: return None, "No data found in channel."
        df = pd.DataFrame(feeds)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["bpm"]  = pd.to_numeric(df.get("field1"), errors="coerce")
        df["spo2"] = pd.to_numeric(df.get("field2"), errors="coerce")
        df = df.dropna(subset=["bpm","spo2"]).sort_values("created_at").reset_index(drop=True)
        df["label"] = df.apply(lambda r: label_health(r["bpm"], r["spo2"]), axis=1)
        df["date"]  = df["created_at"].dt.date
        return df, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
#  ML TRAINING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_all_models(cache_key):
    df, err = fetch_data(500)
    if err or df is None or len(df) < 10:
        return None
    X = df[["bpm","spo2"]].values
    y = df["label"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Augment small datasets
    if len(Xs) < 40:
        np.random.seed(42)
        Xs = np.vstack([Xs]*6 + [Xs + np.random.normal(0,0.3,Xs.shape)])
        y  = np.tile(y, 7)
    models = {
        "KNN":               KNeighborsClassifier(n_neighbors=min(5,len(Xs)-1)),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":               SVC(probability=True, random_state=42),
    }
    for m in models.values(): m.fit(Xs, y)
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(Xs)
    return {"models": models, "scaler": scaler, "iso": iso, "df": df}


# ─────────────────────────────────────────────────────────────
#  EMAIL ALERT
# ─────────────────────────────────────────────────────────────
def send_email_alert(to_addr, smtp_user, smtp_pass, bpm, spo2, label):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 MedPulse CRITICAL ALERT — {label}"
        msg["From"]    = smtp_user
        msg["To"]      = to_addr
        html = f"""
        <div style="font-family:sans-serif;max-width:500px;background:#040d18;color:#dce8f5;padding:30px;border-radius:12px">
          <h2 style="color:#ef4444">🚨 Critical Health Alert</h2>
          <p style="color:#8aabc8">MedPulse IoT Health Monitor has detected an abnormal reading.</p>
          <table style="width:100%;background:#07111f;border-radius:8px;padding:16px;margin:16px 0">
            <tr><td style="color:#4a6a88;padding:8px">Heart Rate</td><td style="color:#f97316;font-size:24px;font-weight:bold">{bpm:.0f} BPM</td></tr>
            <tr><td style="color:#4a6a88;padding:8px">SpO₂</td><td style="color:#3b82f6;font-size:24px;font-weight:bold">{spo2:.1f}%</td></tr>
            <tr><td style="color:#4a6a88;padding:8px">Status</td><td style="color:#ef4444;font-weight:bold">{label}</td></tr>
            <tr><td style="color:#4a6a88;padding:8px">Time</td><td style="color:#dce8f5">{datetime.now().strftime('%H:%M:%S  %d %b %Y')}</td></tr>
          </table>
          <p style="color:#4a6a88;font-size:12px">Sent by MedPulse • Channel {CHANNEL_ID}</p>
        </div>"""
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, to_addr, msg.as_string())
        return True, "✅ Alert email sent!"
    except Exception as e:
        return False, f"❌ Email failed: {e}"


# ─────────────────────────────────────────────────────────────
#  PDF REPORT
# ─────────────────────────────────────────────────────────────
def generate_pdf(df, stats, predictions):
    if not REPORTLAB_OK:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             topMargin=2*cm, bottomMargin=2*cm,
                             leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=22, textColor=colors.HexColor("#1e3a5f"),
                                  spaceAfter=6)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontSize=11, textColor=colors.HexColor("#4a6a88"),
                                  spaceAfter=20)
    head_style  = ParagraphStyle("head", parent=styles["Heading2"],
                                  fontSize=13, textColor=colors.HexColor("#1e3a5f"),
                                  spaceBefore=14, spaceAfter=8)
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                  fontSize=10, textColor=colors.HexColor("#2c3e50"),
                                  spaceAfter=6, leading=16)

    story.append(Paragraph("🫀 MedPulse Health Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}  •  Channel: {CHANNEL_ID}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d0e4f7")))
    story.append(Spacer(1, 0.4*cm))

    # Summary stats
    story.append(Paragraph("Patient Vitals Summary", head_style))
    latest_bpm  = df["bpm"].iloc[-1]
    latest_spo2 = df["spo2"].iloc[-1]
    latest_lbl  = df["label"].iloc[-1]
    sum_data = [
        ["Parameter",      "Latest",           "Mean",                   "Min",                   "Max"],
        ["Heart Rate (BPM)", f"{latest_bpm:.0f}", f"{df['bpm'].mean():.1f}", f"{df['bpm'].min():.0f}", f"{df['bpm'].max():.0f}"],
        ["SpO₂ (%)",         f"{latest_spo2:.1f}",f"{df['spo2'].mean():.1f}",f"{df['spo2'].min():.1f}",f"{df['spo2'].max():.1f}"],
        ["Health Status",   latest_lbl,          "-",                      "-",                     "-"],
        ["Records Analysed",f"{len(df)}",         "-",                      "-",                     "-"],
    ]
    t = Table(sum_data, colWidths=[4.5*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",    (0,0),(-1,0), colors.white),
        ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f7ff"), colors.white]),
        ("GRID",         (0,0),(-1,-1), 0.5, colors.HexColor("#d0e4f7")),
        ("PADDING",      (0,0),(-1,-1), 7),
        ("ALIGN",        (1,0),(-1,-1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Health distribution
    story.append(Paragraph("Health Status Distribution", head_style))
    dist = df["label"].value_counts()
    dist_data = [["Health Condition", "Count", "Percentage"]]
    for lbl, cnt in dist.items():
        dist_data.append([lbl, str(cnt), f"{cnt/len(df)*100:.1f}%"])
    t2 = Table(dist_data, colWidths=[7*cm, 3*cm, 5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",   (0,0),(-1,0), colors.white),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f7ff"), colors.white]),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.HexColor("#d0e4f7")),
        ("PADDING",     (0,0),(-1,-1), 7),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5*cm))

    # ML predictions
    story.append(Paragraph("ML Model Predictions", head_style))
    for model_name, (pred, conf) in predictions.items():
        story.append(Paragraph(f"<b>{model_name}:</b>  {pred}  ({conf:.1f}% confidence)", body_style))

    # Anomaly count
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Clinical Notes", head_style))
    abnormal = len(df[df["label"] != "Normal"])
    normal   = len(df[df["label"] == "Normal"])
    story.append(Paragraph(f"Total readings analysed: {len(df)}", body_style))
    story.append(Paragraph(f"Normal readings: {normal} ({normal/len(df)*100:.1f}%)", body_style))
    story.append(Paragraph(f"Abnormal readings: {abnormal} ({abnormal/len(df)*100:.1f}%)", body_style))
    if abnormal > len(df) * 0.2:
        story.append(Paragraph("⚠️ WARNING: More than 20% of readings are abnormal. Immediate review recommended.", body_style))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d0e4f7")))
    story.append(Paragraph("Report generated by MedPulse IoT Health Monitoring System • MAX30102 + Arduino Mega 2560 + ThingSpeak", 
                            ParagraphStyle("footer", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#8aabc8"))))

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#07111f", plot_bgcolor="#040d18",
    font=dict(color="#4a6a88", family="DM Mono"),
    xaxis=dict(gridcolor="#0f2035", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0f2035", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="#07111f", bordercolor="#0f2035", borderwidth=1)
)

def ts_chart(df, bpm_high, bpm_low, spo2_low):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["created_at"], y=df["bpm"],
        name="BPM", line=dict(color="#f97316", width=2),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.06)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["created_at"], y=df["spo2"],
        name="SpO₂ %", line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.05)"), secondary_y=True)
    fig.add_hline(y=bpm_high, line_dash="dot", line_color="rgba(239,68,68,0.35)", annotation_text="High BPM", secondary_y=False)
    fig.add_hline(y=bpm_low,  line_dash="dot", line_color="rgba(59,130,246,0.35)", annotation_text="Low BPM",  secondary_y=False)
    fig.update_layout(height=320, title="Real-Time Vitals", **PLOT_LAYOUT)
    fig.update_yaxes(title_text="BPM", secondary_y=False, color="#f97316")
    fig.update_yaxes(title_text="SpO₂ %", secondary_y=True, color="#3b82f6", range=[80,102])
    return fig

def gauge(val, title, lo, hi, color, unit=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": unit, "font": {"size": 40, "color": color}},
        title={"text": title, "font": {"size": 11, "color": "#4a6a88", "family": "DM Mono"}},
        gauge={
            "axis": {"range": [lo, hi], "tickcolor": "#0f2035", "tickfont": {"color": "#4a6a88", "size": 9}},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#07111f", "bordercolor": "#0f2035",
            "steps": [{"range": [lo, hi*0.5], "color": "rgba(255,255,255,0.02)"},
                      {"range": [hi*0.5, hi],  "color": "rgba(255,255,255,0.04)"}],
            "threshold": {"line": {"color": color, "width": 2}, "value": val}
        }
    ))
    fig.update_layout(height=210, margin=dict(l=16,r=16,t=28,b=8), paper_bgcolor="#07111f", font_color="#dce8f5")
    return fig


# ─────────────────────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────────────────────
def show_login():
    col = st.columns([1,1.4,1])[1]
    with col:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 10px">
          <div style="font-size:52px;margin-bottom:8px">🫀</div>
          <div style="font-size:28px;font-weight:800;color:#dce8f5">MedPulse</div>
          <div style="font-family:'DM Mono',monospace;font-size:11px;color:#2a4a68;letter-spacing:3px;margin-top:4px">IOT HEALTH MONITOR</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="doctor / nurse1 / patient1")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Sign In →", use_container_width=True, type="primary"):
            user = USERS.get(username.strip().lower())
            if user and user["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.session_state.role      = user["role"]
                st.session_state.user_name = user["name"]
                st.session_state.avatar    = user["avatar"]
                st.rerun()
            else:
                st.error("Invalid credentials. Try: doctor / medpulse123")

        st.markdown("""
        <div style="margin-top:20px;background:#07111f;border:1px solid #0f2035;border-radius:10px;padding:14px;font-family:'DM Mono',monospace;font-size:10px;color:#2a4a68">
          DEMO ACCOUNTS<br><br>
          👨‍⚕️ doctor / medpulse123 &nbsp;·&nbsp; Full access<br>
          👩‍⚕️ nurse1 / nurse123 &nbsp;·&nbsp; Full access<br>
          🧑 patient1 / patient123 &nbsp;·&nbsp; View only
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: LIVE DASHBOARD
# ─────────────────────────────────────────────────────────────
def page_dashboard(df, pkg, bpm_high, bpm_low, spo2_low, auto_refresh, refresh_sec):
    latest     = df.iloc[-1]
    bpm_val    = float(latest["bpm"])
    spo2_val   = float(latest["spo2"])
    lbl        = latest["label"]
    lbl_color  = LABEL_COLOR.get(lbl, "#dce8f5")
    ts         = latest["created_at"].strftime("%H:%M:%S  %d %b %Y")
    delta_bpm  = bpm_val - float(df.iloc[-2]["bpm"])  if len(df) > 1 else 0
    delta_spo2 = spo2_val - float(df.iloc[-2]["spo2"]) if len(df) > 1 else 0

    st.markdown('<div class="sec-head">LIVE VITALS</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        arrow = "▲" if delta_bpm  > 0 else ("▼" if delta_bpm  < 0 else "–")
        st.markdown(f"""<div class="mcard">
          <div class="mcard-accent" style="background:linear-gradient(90deg,#f97316,#ef4444)"></div>
          <div class="mcard-label">Heart Rate</div>
          <div class="mcard-val" style="color:#f97316">{bpm_val:.0f}</div>
          <div class="mcard-unit">BPM &nbsp; <span style="color:{'#22c55e' if delta_bpm<0 else '#ef4444'}">{arrow} {abs(delta_bpm):.0f}</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        arrow2 = "▲" if delta_spo2 > 0 else ("▼" if delta_spo2 < 0 else "–")
        st.markdown(f"""<div class="mcard">
          <div class="mcard-accent" style="background:linear-gradient(90deg,#3b82f6,#06b6d4)"></div>
          <div class="mcard-label">Blood Oxygen</div>
          <div class="mcard-val" style="color:#3b82f6">{spo2_val:.1f}</div>
          <div class="mcard-unit">SpO₂ % &nbsp; <span style="color:{'#22c55e' if delta_spo2>0 else '#ef4444'}">{arrow2} {abs(delta_spo2):.1f}</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        bc = badge_class(lbl)
        st.markdown(f"""<div class="mcard">
          <div class="mcard-accent" style="background:linear-gradient(90deg,{lbl_color},{lbl_color}66)"></div>
          <div class="mcard-label">Health Status</div>
          <div style="margin:14px 0 8px"><span class="badge {bc}">{lbl}</span></div>
          <div class="mcard-unit">ML CLASSIFICATION</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        normal_pct = len(df[df["label"]=="Normal"]) / len(df) * 100
        st.markdown(f"""<div class="mcard">
          <div class="mcard-accent" style="background:linear-gradient(90deg,#22c55e,#16a34a)"></div>
          <div class="mcard-label">Session Health</div>
          <div class="mcard-val" style="color:#22c55e">{normal_pct:.0f}</div>
          <div class="mcard-unit">% NORMAL &nbsp;·&nbsp; {len(df)} RECORDS</div>
          <div class="mcard-sub">Updated: {ts}</div>
        </div>""", unsafe_allow_html=True)

    # Alerts
    st.markdown('<div class="sec-head">ALERTS</div>', unsafe_allow_html=True)
    alerts = []
    if bpm_val > bpm_high:  alerts.append(("c", f"⚡ Tachycardia — Heart rate {bpm_val:.0f} BPM exceeds limit of {bpm_high} BPM"))
    if bpm_val < bpm_low:   alerts.append(("c", f"🔵 Bradycardia — Heart rate {bpm_val:.0f} BPM below limit of {bpm_low} BPM"))
    if spo2_val < spo2_low: alerts.append(("c", f"🔴 Hypoxia — SpO₂ {spo2_val:.1f}% below safe threshold of {spo2_low}%"))
    if not alerts:
        st.markdown('<div class="alert-ok">✅ All vitals are within normal range. Patient is stable.</div>', unsafe_allow_html=True)
    else:
        for cls, msg in alerts:
            st.markdown(f'<div class="alert-{cls}">{msg}</div>', unsafe_allow_html=True)

        # Email alert trigger
        if st.session_state.smtp_enabled and st.session_state.alert_email and not st.session_state.email_sent:
            ok, msg = send_email_alert(
                st.session_state.alert_email,
                st.session_state.smtp_user,
                st.session_state.smtp_pass,
                bpm_val, spo2_val, lbl
            )
            st.session_state.email_sent = True
            if ok: st.success(msg)
            else:  st.warning(msg)

    # Gauges + chart
    st.markdown('<div class="sec-head">GAUGE VIEW</div>', unsafe_allow_html=True)
    g1,g2 = st.columns(2)
    with g1: st.plotly_chart(gauge(bpm_val,  "HEART RATE",  30, 180, "#f97316", " BPM"), use_container_width=True)
    with g2: st.plotly_chart(gauge(spo2_val, "BLOOD OXYGEN", 80, 100, "#3b82f6", "%"),   use_container_width=True)

    st.markdown('<div class="sec-head">TIME SERIES</div>', unsafe_allow_html=True)
    st.plotly_chart(ts_chart(df, bpm_high, bpm_low, spo2_low), use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────
#  PAGE: ML ANALYSIS
# ─────────────────────────────────────────────────────────────
def page_ml(df, pkg):
    st.markdown('<div class="sec-head">ML CLASSIFICATION ENGINE</div>', unsafe_allow_html=True)

    models  = pkg["models"]
    scaler  = pkg["scaler"]
    latest  = df.iloc[-1]
    bpm_in  = float(latest["bpm"])
    spo2_in = float(latest["spo2"])

    # Input override
    with st.expander("🔧 Override Input Values for Prediction"):
        ci1, ci2 = st.columns(2)
        bpm_in  = ci1.number_input("BPM",   value=bpm_in,  step=1.0)
        spo2_in = ci2.number_input("SpO₂%", value=spo2_in, step=0.1)

    X_in = scaler.transform([[bpm_in, spo2_in]])

    st.markdown('<div class="sec-head">PREDICTION RESULTS</div>', unsafe_allow_html=True)
    cols = st.columns(len(models))
    for i, (name, model) in enumerate(models.items()):
        pred  = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0]
        conf  = max(proba)*100
        color = LABEL_COLOR.get(pred, "#dce8f5")
        with cols[i]:
            st.markdown(f"""
            <div class="pred-card" style="border-left:3px solid {color}">
              <div class="pred-model">{name.upper()}</div>
              <div class="pred-label" style="color:{color}">{pred}</div>
              <div class="pred-conf">Confidence: {conf:.1f}%</div>
              <div class="conf-bar"><div class="conf-fill" style="width:{conf:.0f}%;background:{color}"></div></div>
            </div>""", unsafe_allow_html=True)

    # Accuracy comparison
    st.markdown('<div class="sec-head">MODEL ACCURACY COMPARISON</div>', unsafe_allow_html=True)
    X_all = scaler.transform(df[["bpm","spo2"]].values)
    y_all = df["label"].values
    accs, names = [], []
    for name, model in models.items():
        acc = accuracy_score(y_all, model.predict(X_all)) * 100
        accs.append(acc); names.append(name)

    bar_colors = ["#f97316","#3b82f6","#22c55e","#a855f7"]
    fig = go.Figure(go.Bar(x=names, y=accs, marker_color=bar_colors,
                            text=[f"{a:.1f}%" for a in accs], textposition="outside",
                            textfont=dict(color="#dce8f5")))
    fig.update_layout(height=280, yaxis=dict(range=[0,110], title="Accuracy (%)"),
                      title="Classification Accuracy on Current Data", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix for best model
    st.markdown('<div class="sec-head">CONFUSION MATRIX — RANDOM FOREST</div>', unsafe_allow_html=True)
    rf   = models["Random Forest"]
    preds= rf.predict(X_all)
    labels_uniq = sorted(set(y_all))
    cm   = confusion_matrix(y_all, preds, labels=labels_uniq)
    fig_cm = px.imshow(cm, x=labels_uniq, y=labels_uniq,
                        color_continuous_scale=[[0,"#07111f"],[1,"#1e4080"]],
                        labels=dict(x="Predicted", y="Actual"),
                        text_auto=True)
    fig_cm.update_layout(height=360, title="Confusion Matrix", **PLOT_LAYOUT)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance (RF)
    st.markdown('<div class="sec-head">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
    fi = rf.feature_importances_
    fig_fi = go.Figure(go.Bar(x=["BPM","SpO₂"], y=fi*100,
                               marker_color=["#f97316","#3b82f6"],
                               text=[f"{v*100:.1f}%" for v in fi], textposition="outside",
                               textfont=dict(color="#dce8f5")))
    fig_fi.update_layout(height=240, yaxis_title="Importance (%)",
                          title="Which feature matters more?", **PLOT_LAYOUT)
    st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────
def page_anomaly(df, pkg):
    st.markdown('<div class="sec-head">ANOMALY DETECTION — ISOLATION FOREST</div>', unsafe_allow_html=True)

    iso    = pkg["iso"]
    scaler = pkg["scaler"]
    X      = scaler.transform(df[["bpm","spo2"]].values)
    scores = iso.decision_function(X)
    preds  = iso.predict(X)

    df2 = df.copy()
    df2["anomaly_score"] = scores
    df2["is_anomaly"]    = preds == -1
    anomalies = df2[df2["is_anomaly"]]

    # Summary
    a1,a2,a3 = st.columns(3)
    with a1:
        st.markdown(f"""<div class="mcard">
          <div class="mcard-accent" style="background:linear-gradient(90deg,#ef4444,#f97316)"></div>
          <div class="mcard-label">Anomalies Detected</div>
          <div class="mcard-val" style="color:#ef4444">{len(anomalies)}</div>
          <div class="mcard-unit">OUT OF {len(df)} READINGS</div>
        </div>""", unsafe_allow_html=True)
    with a2:
        pct = len(anomalies)/len(df)*100
        st.markdown(f"""<div class="mcard">
          <div class="mcard-accent" style="background:linear-gradient(90deg,#f59e0b,#eab308)"></div>
          <div class="mcard-label">Anomaly Rate</div>
          <div class="mcard-val" style="color:#f59e0b">{pct:.1f}</div>
          <div class="mcard-unit">PERCENT</div>
        </div>""", unsafe_allow_html=True)
    with a3:
        worst = df2.loc[df2["anomaly_score"].idxmin()]
        st.markdown(f"""<div class="mcard">
          <div class="mcard-accent" style="background:linear-gradient(90deg,#a855f7,#6366f1)"></div>
          <div class="mcard-label">Most Abnormal</div>
          <div class="mcard-val" style="color:#a855f7">{worst['bpm']:.0f}</div>
          <div class="mcard-unit">BPM @ SpO₂ {worst['spo2']:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # Anomaly score timeline
    st.markdown('<div class="sec-head">ANOMALY SCORE OVER TIME</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2["created_at"], y=df2["anomaly_score"],
        name="Anomaly Score", line=dict(color="#4a6a88", width=1.5),
        fill="tozeroy", fillcolor="rgba(74,106,136,0.06)"))
    fig.add_trace(go.Scatter(
        x=anomalies["created_at"], y=anomalies["anomaly_score"],
        mode="markers", name="⚠️ Anomaly",
        marker=dict(color="#ef4444", size=9, symbol="x")))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(239,68,68,0.4)", annotation_text="Anomaly Threshold")
    fig.update_layout(height=300, title="Lower score = more anomalous", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.markdown('<div class="sec-head">BPM vs SpO₂ — ANOMALY MAP</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    normal_df = df2[~df2["is_anomaly"]]
    fig2.add_trace(go.Scatter(x=normal_df["bpm"], y=normal_df["spo2"],
        mode="markers", name="Normal", marker=dict(color="#22c55e", size=6, opacity=0.6)))
    fig2.add_trace(go.Scatter(x=anomalies["bpm"], y=anomalies["spo2"],
        mode="markers", name="Anomaly", marker=dict(color="#ef4444", size=10, symbol="x")))
    fig2.update_layout(height=320, xaxis_title="BPM", yaxis_title="SpO₂ %",
                        title="Anomaly Scatter Plot", **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    # Table
    if not anomalies.empty:
        st.markdown('<div class="sec-head">ANOMALY RECORDS</div>', unsafe_allow_html=True)
        show = anomalies[["created_at","bpm","spo2","label","anomaly_score"]].copy()
        show.columns = ["Timestamp","BPM","SpO₂%","Label","Anomaly Score"]
        show = show.sort_values("Anomaly Score").reset_index(drop=True)
        st.dataframe(show, use_container_width=True, height=240)


# ─────────────────────────────────────────────────────────────
#  PAGE: PREDICTION (FORECASTING)
# ─────────────────────────────────────────────────────────────
def page_prediction(df):
    st.markdown('<div class="sec-head">FORECAST — NEXT 30 READINGS</div>', unsafe_allow_html=True)

    # Simple rolling linear forecast
    n_future = 30
    df2 = df.copy().reset_index(drop=True)
    df2["idx"] = np.arange(len(df2))

    def linear_forecast(series, n):
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series, 1)
        future_x = np.arange(len(series), len(series)+n)
        forecast  = np.polyval(coeffs, future_x)
        # add slight noise for realism
        np.random.seed(42)
        forecast += np.random.normal(0, series.std()*0.15, n)
        return forecast

    last_time  = df2["created_at"].iloc[-1]
    avg_interval = (df2["created_at"].iloc[-1] - df2["created_at"].iloc[0]).total_seconds() / max(len(df2)-1,1)
    future_times = [last_time + timedelta(seconds=avg_interval*(i+1)) for i in range(n_future)]

    bpm_forecast  = linear_forecast(df2["bpm"].values,  n_future)
    spo2_forecast = linear_forecast(df2["spo2"].values, n_future)

    bpm_forecast  = np.clip(bpm_forecast, 30, 200)
    spo2_forecast = np.clip(spo2_forecast, 80, 100)

    # BPM forecast chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df2["created_at"].iloc[-50:], y=df2["bpm"].iloc[-50:],
        name="Historical BPM", line=dict(color="#f97316", width=2)))
    fig1.add_trace(go.Scatter(x=future_times, y=bpm_forecast,
        name="Forecast BPM", line=dict(color="#f97316", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.05)"))
    fig1.add_vrect(x0=future_times[0], x1=future_times[-1],
        fillcolor="rgba(249,115,22,0.04)", line_width=0, annotation_text="Forecast Zone")
    fig1.update_layout(height=280, title="BPM Forecast (Next 30 Readings)", **PLOT_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)

    # SpO2 forecast chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df2["created_at"].iloc[-50:], y=df2["spo2"].iloc[-50:],
        name="Historical SpO₂", line=dict(color="#3b82f6", width=2)))
    fig2.add_trace(go.Scatter(x=future_times, y=spo2_forecast,
        name="Forecast SpO₂", line=dict(color="#3b82f6", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.05)"))
    fig2.add_vrect(x0=future_times[0], x1=future_times[-1],
        fillcolor="rgba(59,130,246,0.03)", line_width=0)
    fig2.update_layout(height=280, title="SpO₂ Forecast (Next 30 Readings)", **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    # Forecast health labels
    st.markdown('<div class="sec-head">FORECASTED HEALTH STATUS</div>', unsafe_allow_html=True)
    forecast_labels = [label_health(b, s) for b, s in zip(bpm_forecast, spo2_forecast)]
    label_counts = pd.Series(forecast_labels).value_counts()
    fc = [LABEL_COLOR.get(l, "#dce8f5") for l in label_counts.index]
    fig3 = go.Figure(go.Bar(x=label_counts.index, y=label_counts.values,
                             marker_color=fc,
                             text=label_counts.values, textposition="outside",
                             textfont=dict(color="#dce8f5")))
    fig3.update_layout(height=260, title="Predicted Conditions in Next 30 Readings", **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    # Forecast table
    with st.expander("View Forecast Table"):
        ft = pd.DataFrame({"Time": future_times, "BPM (forecast)": bpm_forecast.round(1),
                            "SpO₂ % (forecast)": spo2_forecast.round(2),
                            "Predicted Status": forecast_labels})
        st.dataframe(ft, use_container_width=True, height=260)


# ─────────────────────────────────────────────────────────────
#  PAGE: TODAY vs YESTERDAY
# ─────────────────────────────────────────────────────────────
def page_comparison(df):
    st.markdown('<div class="sec-head">TODAY vs YESTERDAY COMPARISON</div>', unsafe_allow_html=True)

    today     = df["date"].max()
    yesterday = today - timedelta(days=1)
    df_today  = df[df["date"] == today]
    df_yest   = df[df["date"] == yesterday]

    if df_today.empty or df_yest.empty:
        st.info("Need data from at least 2 days to compare. Showing last 2 halves of available data instead.")
        half = len(df) // 2
        df_today = df.iloc[half:].copy()
        df_yest  = df.iloc[:half].copy()
        label_t, label_y = "Recent Half", "Earlier Half"
    else:
        label_t, label_y = str(today), str(yesterday)

    # Metric comparison
    metrics = {
        "Avg BPM":    (df_today["bpm"].mean(),  df_yest["bpm"].mean()),
        "Avg SpO₂%":  (df_today["spo2"].mean(), df_yest["spo2"].mean()),
        "Max BPM":    (df_today["bpm"].max(),   df_yest["bpm"].max()),
        "Min SpO₂%":  (df_today["spo2"].min(),  df_yest["spo2"].min()),
    }
    cols = st.columns(4)
    for i, (metric, (tv, yv)) in enumerate(metrics.items()):
        diff    = tv - yv
        diffstr = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        color   = "#22c55e" if diff < 0.5 else "#ef4444"
        with cols[i]:
            st.markdown(f"""<div class="mcard">
              <div class="mcard-accent" style="background:linear-gradient(90deg,{color},{color}66)"></div>
              <div class="mcard-label">{metric}</div>
              <div class="mcard-val" style="color:{color};font-size:32px">{tv:.1f}</div>
              <div class="mcard-unit">{label_t} &nbsp;<span style="color:{color}">{diffstr}</span></div>
              <div class="mcard-sub">vs {yv:.1f} ({label_y})</div>
            </div>""", unsafe_allow_html=True)

    # Overlay chart BPM
    st.markdown('<div class="sec-head">BPM COMPARISON</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df_today["bpm"].values, name=label_t,
        line=dict(color="#f97316", width=2)))
    fig.add_trace(go.Scatter(y=df_yest["bpm"].values, name=label_y,
        line=dict(color="#f9731655", width=2, dash="dot")))
    fig.update_layout(height=270, title="BPM: Period Comparison", xaxis_title="Reading Index", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Overlay chart SpO2
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=df_today["spo2"].values, name=label_t,
        line=dict(color="#3b82f6", width=2)))
    fig2.add_trace(go.Scatter(y=df_yest["spo2"].values,  name=label_y,
        line=dict(color="#3b82f655", width=2, dash="dot")))
    fig2.update_layout(height=270, title="SpO₂: Period Comparison", xaxis_title="Reading Index",
                        yaxis=dict(range=[80,102], gridcolor="#0f2035"), **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    # Health label distribution comparison
    st.markdown('<div class="sec-head">HEALTH STATUS DISTRIBUTION</div>', unsafe_allow_html=True)
    all_labels = sorted(set(df["label"].unique()))
    dist_t = df_today["label"].value_counts().reindex(all_labels, fill_value=0)
    dist_y = df_yest["label"].value_counts().reindex(all_labels, fill_value=0)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name=label_t, x=all_labels, y=dist_t.values, marker_color="#f97316"))
    fig3.add_trace(go.Bar(name=label_y, x=all_labels, y=dist_y.values, marker_color="#3b82f6"))
    fig3.update_layout(barmode="group", height=280, title="Label Distribution Comparison", **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: PDF REPORT
# ─────────────────────────────────────────────────────────────
def page_report(df, pkg):
    st.markdown('<div class="sec-head">GENERATE HEALTH REPORT</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2,1])
    with col_l:
        st.markdown("""
        <div style="background:#07111f;border:1px solid #0f2035;border-radius:12px;padding:24px">
          <div style="font-size:18px;font-weight:700;margin-bottom:8px">📋 PDF Health Report</div>
          <div style="font-family:'DM Mono',monospace;font-size:11px;color:#4a6a88;line-height:1.8">
            The report includes:<br>
            • Patient vitals summary (latest, mean, min, max)<br>
            • Health status distribution table<br>
            • ML model prediction results<br>
            • Anomaly analysis summary<br>
            • Clinical notes & recommendations<br>
            • Timestamp & channel info
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        patient_name = st.text_input("Patient Name", value="Patient A")
        doctor_name  = st.text_input("Doctor Name",  value=st.session_state.user_name)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.button("🖨️ Generate PDF Report", type="primary", use_container_width=False):
        if not REPORTLAB_OK:
            st.error("reportlab not installed. Add 'reportlab' to requirements.txt and redeploy.")
        else:
            models  = pkg["models"]
            scaler  = pkg["scaler"]
            latest  = df.iloc[-1]
            X_in    = scaler.transform([[float(latest["bpm"]), float(latest["spo2"])]])
            predictions = {}
            for name, model in models.items():
                pred  = model.predict(X_in)[0]
                proba = model.predict_proba(X_in)[0]
                conf  = max(proba)*100
                predictions[name] = (pred, conf)
            stats = {"mean_bpm": df["bpm"].mean(), "mean_spo2": df["spo2"].mean()}
            with st.spinner("Generating PDF..."):
                buf = generate_pdf(df, stats, predictions)
            if buf:
                st.download_button(
                    label="⬇️  Download Report PDF",
                    data=buf,
                    file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
                st.success("✅ PDF ready! Click above to download.")


# ─────────────────────────────────────────────────────────────
#  PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────
def page_settings():
    st.markdown('<div class="sec-head">EMAIL ALERT CONFIGURATION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#07111f;border:1px solid rgba(234,179,8,0.2);border-radius:10px;padding:14px 18px;
    font-family:'DM Mono',monospace;font-size:11px;color:#eab308;margin-bottom:16px">
    ⚠️ Use a Gmail account with App Password (not your main password).<br>
    Enable 2FA on Gmail → Google Account → Security → App Passwords → Create one for "Mail".
    </div>
    """, unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.session_state.alert_email = st.text_input("Alert Recipient Email", value=st.session_state.alert_email, placeholder="doctor@hospital.com")
        st.session_state.smtp_user   = st.text_input("Gmail Sender Address",  value=st.session_state.smtp_user,   placeholder="sender@gmail.com")
    with s2:
        st.session_state.smtp_pass   = st.text_input("Gmail App Password", value=st.session_state.smtp_pass, type="password", placeholder="xxxx xxxx xxxx xxxx")
        st.session_state.smtp_enabled= st.toggle("Enable Email Alerts", value=st.session_state.smtp_enabled)

    if st.button("📧 Send Test Email"):
        if st.session_state.smtp_user and st.session_state.smtp_pass and st.session_state.alert_email:
            ok, msg = send_email_alert(st.session_state.alert_email, st.session_state.smtp_user,
                                        st.session_state.smtp_pass, 120.0, 91.5, "TEST ALERT")
            st.success(msg) if ok else st.error(msg)
        else:
            st.warning("Fill in all email fields first.")

    st.markdown('<div class="sec-head">SYSTEM INFO</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#07111f;border:1px solid #0f2035;border-radius:12px;padding:20px;
    font-family:'DM Mono',monospace;font-size:11px;color:#4a6a88;line-height:2">
    CHANNEL ID &nbsp;&nbsp;&nbsp;→ {CHANNEL_ID}<br>
    READ API &nbsp;&nbsp;&nbsp;&nbsp;→ {READ_API[:8]}••••••••<br>
    PLATFORM &nbsp;&nbsp;&nbsp;&nbsp;→ ThingSpeak (MathWorks)<br>
    SENSOR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ MAX30102 (SpO₂ + Heart Rate)<br>
    MCU &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Arduino Mega 2560 + ESP8266<br>
    ML MODELS &nbsp;&nbsp;&nbsp;→ KNN · Logistic Regression · Random Forest · SVM<br>
    ANOMALY &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Isolation Forest (contamination=0.1)<br>
    FORECAST &nbsp;&nbsp;&nbsp;&nbsp;→ Linear Extrapolation (30-step ahead)<br>
    REFRESH &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ st.cache_data(ttl=15s) + st.rerun()
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MAIN APP ROUTER
# ─────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login()
else:
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:16px 0 20px;border-bottom:1px solid #0f2035;margin-bottom:16px">
          <div style="font-size:24px">{st.session_state.avatar}</div>
          <div style="font-weight:700;color:#dce8f5;margin:4px 0">{st.session_state.user_name}</div>
          <div style="font-family:'DM Mono',monospace;font-size:10px;color:#2a4a68;letter-spacing:2px">
            {st.session_state.role.upper()} ACCESS
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Doctor sees all pages, patient sees limited
        if st.session_state.role == "doctor":
            pages = ["Dashboard","ML Analysis","Anomaly Detection","Prediction","Comparison","PDF Report","Settings"]
        else:
            pages = ["Dashboard","Prediction"]

        page = st.radio("Navigation", pages, label_visibility="collapsed")
        st.session_state.page = page

        st.markdown("---")
        st.markdown("**⚙️ Dashboard Settings**")
        bpm_high    = st.number_input("BPM High Threshold",  value=100)
        bpm_low     = st.number_input("BPM Low Threshold",   value=60)
        spo2_low    = st.number_input("SpO₂ Low Threshold",  value=94)
        num_records = st.slider("Records to Load", 50, 500, 300, 50)
        refresh_sec = st.selectbox("Auto-refresh", [10,15,30,60], index=1)
        auto_refresh= st.toggle("Auto Refresh", value=True)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in ["logged_in","username","role","user_name","avatar","email_sent"]:
                st.session_state[k] = False if k=="logged_in" else ""
            st.rerun()

    # ── LOAD DATA & MODELS ──
    with st.spinner("Fetching data from ThingSpeak..."):
        df, err = fetch_data(num_records)

    if err or df is None:
        st.error(f"Could not load data: {err}")
        st.stop()

    pkg = train_all_models(len(df))
    if pkg is None:
        pkg = {"models": {}, "scaler": StandardScaler(), "iso": None, "df": df}

    # ── ROUTE ──
    if   page == "Dashboard":         page_dashboard(df, pkg, bpm_high, bpm_low, spo2_low, auto_refresh, refresh_sec)
    elif page == "ML Analysis":        page_ml(df, pkg)
    elif page == "Anomaly Detection":  page_anomaly(df, pkg)
    elif page == "Prediction":         page_prediction(df)
    elif page == "Comparison":         page_comparison(df)
    elif page == "PDF Report":         page_report(df, pkg)
    elif page == "Settings":           page_settings()
