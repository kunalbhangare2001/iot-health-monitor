import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings("ignore")

from supabase import create_client, Client
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.units import cm
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ═════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MedPulse — IoT Health Monitor",
    page_icon="🫀", layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════
#  CSS
# ═════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bricolage+Grotesque:wght@400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Bricolage Grotesque',sans-serif;}
.stApp{background:#040d18;color:#dce8f5;}
section[data-testid="stSidebar"]{background:#060f1e;border-right:1px solid #0f2035;}
section[data-testid="stSidebar"] *{color:#8aabc8 !important;}
.mcard{background:#07111f;border:1px solid #0f2035;border-radius:16px;padding:22px 20px 16px;position:relative;overflow:hidden;}
.mcard-accent{position:absolute;top:0;left:0;right:0;height:3px;border-radius:16px 16px 0 0;}
.mcard-label{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;color:#4a6a88;text-transform:uppercase;margin-bottom:8px;}
.mcard-val{font-size:48px;font-weight:800;line-height:1;margin-bottom:4px;}
.mcard-unit{font-family:'DM Mono',monospace;font-size:11px;color:#4a6a88;letter-spacing:2px;}
.mcard-sub{font-family:'DM Mono',monospace;font-size:10px;color:#4a6a88;margin-top:10px;}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-family:'DM Mono',monospace;font-size:11px;font-weight:500;letter-spacing:1px;}
.badge-normal{background:rgba(34,197,94,.12);color:#22c55e;border:1px solid rgba(34,197,94,.25);}
.badge-warning{background:rgba(234,179,8,.12);color:#eab308;border:1px solid rgba(234,179,8,.25);}
.badge-critical{background:rgba(239,68,68,.12);color:#ef4444;border:1px solid rgba(239,68,68,.3);}
.sec-head{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:4px;color:#2a4a68;text-transform:uppercase;padding:0 0 10px;border-bottom:1px solid #0f2035;margin:28px 0 16px;}
.alert-ok{background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.2);border-left:4px solid #22c55e;border-radius:0 10px 10px 0;padding:12px 18px;color:#22c55e;margin:6px 0;}
.alert-w{background:rgba(234,179,8,.07);border:1px solid rgba(234,179,8,.2);border-left:4px solid #eab308;border-radius:0 10px 10px 0;padding:12px 18px;color:#eab308;margin:6px 0;}
.alert-c{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-left:4px solid #ef4444;border-radius:0 10px 10px 0;padding:12px 18px;color:#ef4444;margin:6px 0;}
.pred-card{background:#07111f;border:1px solid #0f2035;border-radius:12px;padding:16px 20px;margin-bottom:10px;}
.pred-model{font-family:'DM Mono',monospace;font-size:10px;color:#4a6a88;letter-spacing:2px;margin-bottom:8px;}
.pred-label{font-size:20px;font-weight:700;margin-bottom:6px;}
.pred-conf{font-family:'DM Mono',monospace;font-size:11px;color:#4a6a88;}
.conf-bar{background:#0f2035;border-radius:20px;height:5px;margin-top:8px;}
.conf-fill{height:5px;border-radius:20px;}
.stTextInput input,.stNumberInput input{background:#07111f !important;border-color:#0f2035 !important;color:#dce8f5 !important;}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════
LABEL_COLOR = {
    "Normal":             "#22c55e",
    "Tachycardia":        "#eab308",
    "Severe Tachycardia": "#f97316",
    "Bradycardia":        "#3b82f6",
    "Severe Bradycardia": "#6366f1",
    "Hypoxia":            "#f59e0b",
    "Critical Hypoxia":   "#ef4444",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="#07111f", plot_bgcolor="#040d18",
    font=dict(color="#4a6a88", family="DM Mono"),
    xaxis=dict(gridcolor="#0f2035", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0f2035", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="#07111f", bordercolor="#0f2035", borderwidth=1)
)

# ═════════════════════════════════════════════════════════
#  LOGIN USERS
# ═════════════════════════════════════════════════════════
USERS = {
    "doctor":  {"password": "medpulse123", "role": "doctor",  "name": "Dr. Sharma",  "avatar": "👨‍⚕️"},
    "nurse1":  {"password": "nurse123",    "role": "doctor",  "name": "Nurse Priya", "avatar": "👩‍⚕️"},
    "patient1":{"password": "patient123",  "role": "patient", "name": "Patient View","avatar": "🧑"},
}

# ═════════════════════════════════════════════════════════
#  SESSION STATE
# ═════════════════════════════════════════════════════════
for k, v in {
    "logged_in": False, "username": "", "role": "",
    "user_name": "", "avatar": "", "email_sent": False,
    "alert_email": "", "smtp_user": "", "smtp_pass": "", "smtp_enabled": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═════════════════════════════════════════════════════════
#  SUPABASE CONNECTION
#  Add these to Streamlit secrets (.streamlit/secrets.toml):
#  [supabase]
#  url = "https://fevnqqwowsbshtakrpuv.supabase.co"
#  key = "your_anon_public_key"
# ═════════════════════════════════════════════════════════
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

# ═════════════════════════════════════════════════════════
#  HEALTH LABEL
# ═════════════════════════════════════════════════════════
def label_health(bpm, spo2):
    if   spo2 < 90:  return "Critical Hypoxia"
    elif spo2 < 94:  return "Hypoxia"
    elif bpm  > 120: return "Severe Tachycardia"
    elif bpm  > 100: return "Tachycardia"
    elif bpm  < 50:  return "Severe Bradycardia"
    elif bpm  < 60:  return "Bradycardia"
    else:             return "Normal"

def badge_class(label):
    if label == "Normal":   return "badge-normal"
    if "Critical" in label: return "badge-critical"
    return "badge-warning"

# ═════════════════════════════════════════════════════════
#  LOAD ALL PATIENTS FROM SUPABASE
# ═════════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def load_patient_db():
    """Fetch all patients from Supabase — refreshes every 30s."""
    try:
        supabase = get_supabase()
        res = supabase.table("patients").select("*").order("id").execute()
        if res.data:
            return res.data
    except Exception as e:
        st.sidebar.warning(f"Could not load patients: {e}")

    # Fallback demo patient
    return [{
        "id": "P001", "name": "Demo Patient", "age": 30,
        "doctor": "Dr. Unknown", "admitted": "2026-03-01",
    }]

# ═════════════════════════════════════════════════════════
#  FETCH READINGS FOR A PATIENT FROM SUPABASE
# ═════════════════════════════════════════════════════════
@st.cache_data(ttl=15)
def fetch_data(patient_id: str, results: int = 300):
    """Load readings for a patient from Supabase readings table."""
    try:
        supabase = get_supabase()
        res = supabase.table("readings") \
                      .select("*") \
                      .eq("patient_id", patient_id) \
                      .order("recorded_at", desc=True) \
                      .limit(results) \
                      .execute()

        if not res.data:
            return None, "No readings yet. Run medpulse_supabase.py to send data."

        df = pd.DataFrame(res.data)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df = df.sort_values("recorded_at").reset_index(drop=True)

        df["bpm"]  = pd.to_numeric(df["bpm"],  errors="coerce")
        df["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")
        df = df.dropna(subset=["bpm", "spo2"])

        if df.empty:
            return None, "No valid readings after filtering."

        df["bpm"]        = df["bpm"].clip(20, 300)
        df["spo2"]       = df["spo2"].clip(50, 100)
        df["label"]      = df.apply(lambda r: label_health(r["bpm"], r["spo2"]), axis=1)
        df["date"]       = df["recorded_at"].dt.date
        df["created_at"] = df["recorded_at"]   # keep alias for chart compatibility
        return df, None

    except Exception as e:
        return None, str(e)

# ═════════════════════════════════════════════════════════
#  TRAIN ML MODELS
# ═════════════════════════════════════════════════════════
@st.cache_resource
def train_models(patient_id, data_hash):
    df, err = fetch_data(patient_id, 300)
    if err or df is None or len(df) < 5:
        return None

    X  = df[["bpm","spo2"]].values
    y  = df["label"].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    if len(Xs) < 40:
        np.random.seed(42)
        Xs = np.vstack([Xs]*6 + [Xs + np.random.normal(0,.3,Xs.shape)])
        y  = np.tile(y, 7)

    mdls = {
        "KNN":                 KNeighborsClassifier(n_neighbors=min(5,len(Xs)-1)),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
    }
    for m in mdls.values(): m.fit(Xs, y)
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(Xs)
    return {"models": mdls, "scaler": sc, "iso": iso}

# ═════════════════════════════════════════════════════════
#  CHART HELPERS
# ═════════════════════════════════════════════════════════
def make_gauge(val, title, lo, hi, color, unit=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={"suffix": unit, "font": {"size": 38, "color": color}},
        title={"text": title, "font": {"size":11,"color":"#4a6a88","family":"DM Mono"}},
        gauge={
            "axis":{"range":[lo,hi],"tickcolor":"#0f2035","tickfont":{"color":"#4a6a88","size":9}},
            "bar":{"color":color,"thickness":0.22},
            "bgcolor":"#07111f","bordercolor":"#0f2035",
            "steps":[{"range":[lo,hi*.5],"color":"rgba(255,255,255,.02)"},
                     {"range":[hi*.5,hi], "color":"rgba(255,255,255,.04)"}],
            "threshold":{"line":{"color":color,"width":2},"value":val}
        }
    ))
    fig.update_layout(height=210,margin=dict(l=16,r=16,t=28,b=8),
                      paper_bgcolor="#07111f",font_color="#dce8f5")
    return fig

# ═════════════════════════════════════════════════════════
#  EMAIL ALERT
# ═════════════════════════════════════════════════════════
def send_email(to, smtp_user, smtp_pass, bpm, spo2, label):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 MedPulse ALERT — {label}"
        msg["From"]    = smtp_user
        msg["To"]      = to
        html = f"""<div style="font-family:sans-serif;background:#040d18;color:#dce8f5;padding:30px;border-radius:12px">
          <h2 style="color:#ef4444">🚨 Critical Health Alert</h2>
          <p>MedPulse detected an abnormal reading:</p>
          <table style="width:100%;background:#07111f;padding:16px;border-radius:8px;margin:16px 0">
            <tr><td style="color:#4a6a88;padding:8px">Heart Rate</td><td style="color:#f97316;font-size:22px;font-weight:bold">{bpm:.0f} BPM</td></tr>
            <tr><td style="color:#4a6a88;padding:8px">SpO₂</td><td style="color:#3b82f6;font-size:22px;font-weight:bold">{spo2:.1f}%</td></tr>
            <tr><td style="color:#4a6a88;padding:8px">Status</td><td style="color:#ef4444;font-weight:bold">{label}</td></tr>
            <tr><td style="color:#4a6a88;padding:8px">Time</td><td>{datetime.now().strftime('%H:%M:%S  %d %b %Y')}</td></tr>
          </table>
        </div>"""
        msg.attach(MIMEText(html,"html"))
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, to, msg.as_string())
        return True, "✅ Alert email sent!"
    except Exception as e:
        return False, f"❌ Email failed: {e}"

# ═════════════════════════════════════════════════════════
#  PDF REPORT
# ═════════════════════════════════════════════════════════
def generate_pdf(patient, df, predictions):
    if not REPORTLAB_OK:
        return None
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              topMargin=2*cm, bottomMargin=2*cm,
                              leftMargin=2*cm, rightMargin=2*cm)
    st_  = getSampleStyleSheet()
    T    = ParagraphStyle("T", parent=st_["Title"],   fontSize=22, textColor=colors.HexColor("#1e3a5f"), spaceAfter=6)
    S    = ParagraphStyle("S", parent=st_["Normal"],  fontSize=11, textColor=colors.HexColor("#4a6a88"), spaceAfter=20)
    H    = ParagraphStyle("H", parent=st_["Heading2"],fontSize=13, textColor=colors.HexColor("#1e3a5f"), spaceBefore=14, spaceAfter=8)
    B    = ParagraphStyle("B", parent=st_["Normal"],  fontSize=10, textColor=colors.HexColor("#2c3e50"), spaceAfter=6, leading=16)
    story = []
    story.append(Paragraph("🫀 MedPulse Health Report", T))
    story.append(Paragraph(f"Patient: {patient['name']}  |  ID: {patient['id']}  |  {datetime.now().strftime('%d %B %Y  %H:%M')}", S))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d0e4f7")))
    story.append(Spacer(1,.4*cm))

    story.append(Paragraph("Patient Information", H))
    info_data = [["Field","Value"],
                 ["Name",     patient["name"]],
                 ["Age",      str(patient.get("age","—"))],
                 ["Doctor",   patient.get("doctor","—")],
                 ["Admitted", patient.get("admitted","—")]]
    t = Table(info_data, colWidths=[5*cm,11.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f7ff"),colors.white]),
        ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#d0e4f7")),
        ("PADDING",(0,0),(-1,-1),7),
    ]))
    story.append(t)
    story.append(Spacer(1,.4*cm))

    story.append(Paragraph("Vitals Summary", H))
    latest = df.iloc[-1]
    vs = [["Parameter","Latest","Mean","Min","Max"],
          ["Heart Rate (BPM)", f"{float(latest['bpm']):.0f}",
           f"{df['bpm'].mean():.1f}", f"{df['bpm'].min():.0f}", f"{df['bpm'].max():.0f}"],
          ["SpO₂ (%)", f"{float(latest['spo2']):.1f}",
           f"{df['spo2'].mean():.1f}", f"{df['spo2'].min():.1f}", f"{df['spo2'].max():.1f}"],
          ["Records", str(len(df)), "─","─","─"]]
    t2 = Table(vs, colWidths=[5*cm,3.1*cm,3.1*cm,3.1*cm,3.1*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f7ff"),colors.white]),
        ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#d0e4f7")),
        ("PADDING",(0,0),(-1,-1),7),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
    ]))
    story.append(t2)
    story.append(Spacer(1,.4*cm))

    story.append(Paragraph("ML Model Predictions", H))
    for mname,(pred,conf) in predictions.items():
        story.append(Paragraph(f"<b>{mname}:</b>  {pred}  ({conf:.1f}% confidence)", B))

    story.append(Paragraph("Health Status Distribution", H))
    dist = df["label"].value_counts()
    for lbl,cnt in dist.items():
        story.append(Paragraph(f"  {lbl}: {cnt} readings ({cnt/len(df)*100:.1f}%)", B))

    story.append(Spacer(1,.5*cm))
    story.append(HRFlowable(width="100%",thickness=.5,color=colors.HexColor("#d0e4f7")))
    story.append(Paragraph("Generated by MedPulse IoT Health Monitoring System",
                            ParagraphStyle("f",parent=st_["Normal"],fontSize=8,textColor=colors.HexColor("#8aabc8"))))
    doc.build(story)
    buf.seek(0)
    return buf

# ═════════════════════════════════════════════════════════
#  LOGIN PAGE
# ═════════════════════════════════════════════════════════
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
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="doctor / nurse1 / patient1")
        password = st.text_input("Password", type="password")
        if st.button("Sign In →", use_container_width=True, type="primary"):
            u = USERS.get(username.strip().lower())
            if u and u["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.session_state.role      = u["role"]
                st.session_state.user_name = u["name"]
                st.session_state.avatar    = u["avatar"]
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown("""
        <div style="margin-top:16px;background:#07111f;border:1px solid #0f2035;border-radius:10px;
             padding:14px;font-family:'DM Mono',monospace;font-size:10px;color:#2a4a68;line-height:2">
          DEMO ACCOUNTS<br><br>
          👨‍⚕️ doctor / medpulse123 · Full access<br>
          👩‍⚕️ nurse1 / nurse123 · Full access<br>
          🧑 patient1 / patient123 · View only
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════
def page_dashboard(df, pkg, patient, bpm_high, bpm_low, spo2_low, auto_refresh, refresh_sec):
    if df is None or df.empty or len(df) < 2:
        st.warning("Not enough data yet. Run medpulse_supabase.py to send readings.")
        return

    latest    = df.iloc[-1]
    bpm_val   = float(latest["bpm"])
    spo2_val  = float(latest["spo2"])
    lbl       = latest["label"]
    lbl_color = LABEL_COLOR.get(lbl, "#dce8f5")
    ts        = latest["recorded_at"].strftime("%H:%M:%S  %d %b %Y")
    d_bpm     = bpm_val  - float(df.iloc[-2]["bpm"])
    d_spo2    = spo2_val - float(df.iloc[-2]["spo2"])
    normal_pct= len(df[df["label"]=="Normal"]) / len(df) * 100

    st.markdown('<div class="sec-head">LIVE VITALS</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        arr = "▲" if d_bpm > 0 else ("▼" if d_bpm < 0 else "–")
        clr = "#ef4444" if d_bpm > 0 else "#22c55e"
        st.markdown(f"""<div class="mcard"><div class="mcard-accent" style="background:linear-gradient(90deg,#f97316,#ef4444)"></div>
          <div class="mcard-label">Heart Rate</div><div class="mcard-val" style="color:#f97316">{bpm_val:.0f}</div>
          <div class="mcard-unit">BPM &nbsp;<span style="color:{clr}">{arr} {abs(d_bpm):.0f}</span></div></div>""", unsafe_allow_html=True)
    with c2:
        arr2 = "▲" if d_spo2 > 0 else ("▼" if d_spo2 < 0 else "–")
        clr2 = "#22c55e" if d_spo2 > 0 else "#ef4444"
        st.markdown(f"""<div class="mcard"><div class="mcard-accent" style="background:linear-gradient(90deg,#3b82f6,#06b6d4)"></div>
          <div class="mcard-label">Blood Oxygen</div><div class="mcard-val" style="color:#3b82f6">{spo2_val:.1f}</div>
          <div class="mcard-unit">SpO₂ % &nbsp;<span style="color:{clr2}">{arr2} {abs(d_spo2):.1f}</span></div></div>""", unsafe_allow_html=True)
    with c3:
        bc = badge_class(lbl)
        st.markdown(f"""<div class="mcard"><div class="mcard-accent" style="background:linear-gradient(90deg,{lbl_color},{lbl_color}66)"></div>
          <div class="mcard-label">Health Status</div>
          <div style="margin:14px 0 8px"><span class="badge {bc}">{lbl}</span></div>
          <div class="mcard-unit">ML CLASSIFICATION</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="mcard"><div class="mcard-accent" style="background:linear-gradient(90deg,#22c55e,#16a34a)"></div>
          <div class="mcard-label">Session Health</div><div class="mcard-val" style="color:#22c55e">{normal_pct:.0f}</div>
          <div class="mcard-unit">% NORMAL · {len(df)} RECORDS</div>
          <div class="mcard-sub">Updated: {ts}</div></div>""", unsafe_allow_html=True)

    # Alerts
    st.markdown('<div class="sec-head">ALERTS</div>', unsafe_allow_html=True)
    alerts = []
    if bpm_val  > bpm_high:  alerts.append(("c", f"⚡ High Heart Rate: {bpm_val:.0f} BPM > {bpm_high} BPM limit"))
    if bpm_val  < bpm_low:   alerts.append(("c", f"🔵 Low Heart Rate: {bpm_val:.0f} BPM < {bpm_low} BPM limit"))
    if spo2_val < spo2_low:  alerts.append(("c", f"🔴 Low SpO₂: {spo2_val:.1f}% < {spo2_low}% threshold"))
    if not alerts:
        st.markdown('<div class="alert-ok">✅ All vitals are within normal range.</div>', unsafe_allow_html=True)
    else:
        for cls, msg in alerts:
            st.markdown(f'<div class="alert-{cls}">{msg}</div>', unsafe_allow_html=True)
        if st.session_state.smtp_enabled and st.session_state.alert_email and not st.session_state.email_sent:
            ok, emsg = send_email(st.session_state.alert_email, st.session_state.smtp_user,
                                   st.session_state.smtp_pass, bpm_val, spo2_val, lbl)
            st.session_state.email_sent = True
            st.success(emsg) if ok else st.warning(emsg)

    # Gauges
    st.markdown('<div class="sec-head">GAUGE VIEW</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1: st.plotly_chart(make_gauge(bpm_val,  "HEART RATE",   30, 180, "#f97316", " BPM"), use_container_width=True)
    with g2: st.plotly_chart(make_gauge(spo2_val, "BLOOD OXYGEN", 80, 100, "#3b82f6", "%"),    use_container_width=True)

    # Time series
    st.markdown('<div class="sec-head">TIME SERIES</div>', unsafe_allow_html=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["recorded_at"], y=df["bpm"],  name="BPM",
                              line=dict(color="#f97316", width=2),
                              fill="tozeroy", fillcolor="rgba(249,115,22,.06)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["recorded_at"], y=df["spo2"], name="SpO₂%",
                              line=dict(color="#3b82f6", width=2),
                              fill="tozeroy", fillcolor="rgba(59,130,246,.05)"), secondary_y=True)
    fig.add_hline(y=bpm_high, line_dash="dot", line_color="rgba(239,68,68,.35)", annotation_text="High BPM",  secondary_y=False)
    fig.add_hline(y=bpm_low,  line_dash="dot", line_color="rgba(59,130,246,.35)", annotation_text="Low BPM",   secondary_y=False)
    fig.update_layout(height=320, title="Real-Time Vitals", **PLOT_LAYOUT)
    fig.update_yaxes(title_text="BPM",   secondary_y=False, color="#f97316")
    fig.update_yaxes(title_text="SpO₂%", secondary_y=True,  color="#3b82f6", range=[80,102])
    st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.cache_data.clear()
        st.rerun()

# ═════════════════════════════════════════════════════════
#  PAGE: ML ANALYSIS
# ═════════════════════════════════════════════════════════
def page_ml(df, pkg, patient):
    if pkg is None or not pkg.get("models"):
        st.warning("Need more data to train models. Send at least 10 readings.")
        return

    st.markdown('<div class="sec-head">ML PREDICTION</div>', unsafe_allow_html=True)
    models = pkg["models"]; scaler = pkg["scaler"]
    latest = df.iloc[-1]
    c1, c2 = st.columns([1,2])
    with c1:
        bpm_in  = st.number_input("BPM",   value=float(latest["bpm"]),  step=1.0)
        spo2_in = st.number_input("SpO₂%", value=float(latest["spo2"]), step=0.1)
    X_in  = scaler.transform([[bpm_in, spo2_in]])
    with c2:
        cols = st.columns(len(models))
        for i,(name,model) in enumerate(models.items()):
            pred  = model.predict(X_in)[0]
            proba = model.predict_proba(X_in)[0]
            conf  = max(proba)*100
            color = LABEL_COLOR.get(pred,"#dce8f5")
            with cols[i]:
                st.markdown(f"""<div class="pred-card" style="border-left:3px solid {color}">
                  <div class="pred-model">{name.upper()}</div>
                  <div class="pred-label" style="color:{color}">{pred}</div>
                  <div class="pred-conf">Confidence: {conf:.1f}%</div>
                  <div class="conf-bar"><div class="conf-fill" style="width:{conf:.0f}%;background:{color}"></div></div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">MODEL ACCURACY</div>', unsafe_allow_html=True)
    X_all = scaler.transform(df[["bpm","spo2"]].values); y_all = df["label"].values
    accs  = [accuracy_score(y_all, m.predict(X_all))*100 for m in models.values()]
    names = list(models.keys())
    fig = go.Figure(go.Bar(x=names, y=accs,
                            marker_color=["#f97316","#3b82f6","#22c55e","#a855f7"],
                            text=[f"{a:.1f}%" for a in accs], textposition="outside",
                            textfont=dict(color="#dce8f5")))
    fig.update_layout(height=260, yaxis=dict(range=[0,110]), title="Accuracy on Patient Data", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-head">CONFUSION MATRIX — RANDOM FOREST</div>', unsafe_allow_html=True)
    rf  = models["Random Forest"]
    pds = rf.predict(X_all)
    lbs = sorted(set(y_all))
    cm_ = confusion_matrix(y_all, pds, labels=lbs)
    fig2= px.imshow(cm_, x=lbs, y=lbs,
                     color_continuous_scale=[[0,"#07111f"],[1,"#1e4080"]],
                     labels=dict(x="Predicted",y="Actual"), text_auto=True)
    fig2.update_layout(height=340, title="Confusion Matrix", **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-head">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
    fi  = rf.feature_importances_
    fig3= go.Figure(go.Bar(x=["BPM","SpO₂"], y=fi*100,
                            marker_color=["#f97316","#3b82f6"],
                            text=[f"{v*100:.1f}%" for v in fi], textposition="outside",
                            textfont=dict(color="#dce8f5")))
    fig3.update_layout(height=230, yaxis_title="Importance (%)", title="Which feature matters more?", **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

# ═════════════════════════════════════════════════════════
#  PAGE: ANOMALY DETECTION
# ═════════════════════════════════════════════════════════
def page_anomaly(df, pkg):
    if pkg is None or pkg.get("iso") is None:
        st.warning("Need more data for anomaly detection.")
        return

    st.markdown('<div class="sec-head">ANOMALY DETECTION — ISOLATION FOREST</div>', unsafe_allow_html=True)
    iso = pkg["iso"]; sc = pkg["scaler"]
    X   = sc.transform(df[["bpm","spo2"]].values)
    df2 = df.copy()
    df2["anomaly_score"] = iso.decision_function(X)
    df2["is_anomaly"]    = iso.predict(X) == -1
    anom = df2[df2["is_anomaly"]]

    a1,a2,a3 = st.columns(3)
    with a1:
        st.markdown(f"""<div class="mcard"><div class="mcard-accent" style="background:linear-gradient(90deg,#ef4444,#f97316)"></div>
          <div class="mcard-label">Anomalies Found</div><div class="mcard-val" style="color:#ef4444">{len(anom)}</div>
          <div class="mcard-unit">OUT OF {len(df)} READINGS</div></div>""", unsafe_allow_html=True)
    with a2:
        pct = len(anom)/len(df)*100
        st.markdown(f"""<div class="mcard"><div class="mcard-accent" style="background:linear-gradient(90deg,#f59e0b,#eab308)"></div>
          <div class="mcard-label">Anomaly Rate</div><div class="mcard-val" style="color:#f59e0b">{pct:.1f}</div>
          <div class="mcard-unit">PERCENT</div></div>""", unsafe_allow_html=True)
    with a3:
        worst = df2.loc[df2["anomaly_score"].idxmin()]
        st.markdown(f"""<div class="mcard"><div class="mcard-accent" style="background:linear-gradient(90deg,#a855f7,#6366f1)"></div>
          <div class="mcard-label">Most Abnormal</div><div class="mcard-val" style="color:#a855f7">{worst['bpm']:.0f}</div>
          <div class="mcard-unit">BPM @ SpO₂ {worst['spo2']:.1f}%</div></div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2["recorded_at"], y=df2["anomaly_score"], name="Score",
                              line=dict(color="#4a6a88",width=1.5), fill="tozeroy",
                              fillcolor="rgba(74,106,136,.06)"))
    fig.add_trace(go.Scatter(x=anom["recorded_at"], y=anom["anomaly_score"],
                              mode="markers", name="⚠️ Anomaly",
                              marker=dict(color="#ef4444",size=9,symbol="x")))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(239,68,68,.4)", annotation_text="Threshold")
    fig.update_layout(height=290, title="Anomaly Score Timeline", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    norm = df2[~df2["is_anomaly"]]
    fig2.add_trace(go.Scatter(x=norm["bpm"], y=norm["spo2"], mode="markers", name="Normal",
                               marker=dict(color="#22c55e",size=6,opacity=0.6)))
    fig2.add_trace(go.Scatter(x=anom["bpm"], y=anom["spo2"], mode="markers", name="Anomaly",
                               marker=dict(color="#ef4444",size=10,symbol="x")))
    fig2.update_layout(height=300, xaxis_title="BPM", yaxis_title="SpO₂%",
                        title="Anomaly Scatter Map", **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

# ═════════════════════════════════════════════════════════
#  PAGE: FORECAST
# ═════════════════════════════════════════════════════════
def page_prediction(df):
    st.markdown('<div class="sec-head">FORECAST — NEXT 30 READINGS</div>', unsafe_allow_html=True)
    n = 30
    last_t    = df["recorded_at"].iloc[-1]
    avg_int   = (df["recorded_at"].iloc[-1] - df["recorded_at"].iloc[0]).total_seconds() / max(len(df)-1,1)
    fut_times = [last_t + timedelta(seconds=avg_int*(i+1)) for i in range(n)]

    def forecast(series, n):
        x = np.arange(len(series))
        c = np.polyfit(x, series, 1)
        f = np.polyval(c, np.arange(len(series), len(series)+n))
        np.random.seed(42)
        return f + np.random.normal(0, series.std()*0.15, n)

    bpm_f  = np.clip(forecast(df["bpm"].values,  n), 30, 200)
    spo2_f = np.clip(forecast(df["spo2"].values, n), 80, 100)

    for vals_hist, vals_fore, title, color in [
        (df["bpm"].iloc[-50:],  bpm_f,  "BPM Forecast",  "#f97316"),
        (df["spo2"].iloc[-50:], spo2_f, "SpO₂ Forecast", "#3b82f6"),
    ]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=vals_hist.values, name="Historical", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=list(range(len(vals_hist), len(vals_hist)+n)),
                                  y=vals_fore, name="Forecast",
                                  line=dict(color=color, width=2, dash="dot"),
                                  fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},.05)"))
        fig.update_layout(height=260, title=title, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    forecast_labels = [label_health(b, s) for b, s in zip(bpm_f, spo2_f)]
    lc = pd.Series(forecast_labels).value_counts()
    fig3 = go.Figure(go.Bar(x=lc.index, y=lc.values,
                             marker_color=[LABEL_COLOR.get(l,"#dce8f5") for l in lc.index],
                             text=lc.values, textposition="outside",
                             textfont=dict(color="#dce8f5")))
    fig3.update_layout(height=240, title="Predicted Conditions (Next 30 Readings)", **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

# ═════════════════════════════════════════════════════════
#  PAGE: COMPARISON
# ═════════════════════════════════════════════════════════
def page_comparison(df):
    st.markdown('<div class="sec-head">TODAY vs YESTERDAY</div>', unsafe_allow_html=True)
    today = df["date"].max()
    yest  = today - timedelta(days=1)
    df_t  = df[df["date"] == today]
    df_y  = df[df["date"] == yest]
    lt, ly = str(today), str(yest)

    if df_t.empty or df_y.empty:
        half = len(df)//2
        df_t, df_y = df.iloc[half:], df.iloc[:half]
        lt, ly = "Recent Half", "Earlier Half"

    metrics = {
        "Avg BPM":   (df_t["bpm"].mean(),  df_y["bpm"].mean()),
        "Avg SpO₂%": (df_t["spo2"].mean(), df_y["spo2"].mean()),
        "Max BPM":   (df_t["bpm"].max(),   df_y["bpm"].max()),
        "Min SpO₂%": (df_t["spo2"].min(),  df_y["spo2"].min()),
    }
    cols = st.columns(4)
    for i,(metric,(tv,yv)) in enumerate(metrics.items()):
        diff = tv-yv
        dstr = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        clr  = "#22c55e" if abs(diff) < 2 else "#ef4444"
        with cols[i]:
            st.markdown(f"""<div class="mcard">
              <div class="mcard-accent" style="background:linear-gradient(90deg,{clr},{clr}66)"></div>
              <div class="mcard-label">{metric}</div>
              <div class="mcard-val" style="color:{clr};font-size:30px">{tv:.1f}</div>
              <div class="mcard-unit">{lt} &nbsp;<span style="color:{clr}">{dstr}</span></div>
              <div class="mcard-sub">vs {yv:.1f} ({ly})</div>
            </div>""", unsafe_allow_html=True)

    for vals_t, vals_y, title in [
        (df_t["bpm"].values,  df_y["bpm"].values,  "BPM Comparison"),
        (df_t["spo2"].values, df_y["spo2"].values, "SpO₂ Comparison"),
    ]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=vals_t, name=lt, line=dict(color="#f97316", width=2)))
        fig.add_trace(go.Scatter(y=vals_y, name=ly, line=dict(color="#f9731655", width=2, dash="dot")))
        fig.update_layout(height=250, title=title, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════
#  PAGE: PDF REPORT
# ═════════════════════════════════════════════════════════
def page_report(df, pkg, patient):
    st.markdown('<div class="sec-head">PDF HEALTH REPORT</div>', unsafe_allow_html=True)
    if not REPORTLAB_OK:
        st.error("Add 'reportlab' to requirements.txt and redeploy.")
        return

    if st.button("🖨️ Generate PDF", type="primary"):
        predictions = {}
        if pkg and pkg.get("models"):
            sc   = pkg["scaler"]
            X_in = sc.transform([[float(df.iloc[-1]["bpm"]), float(df.iloc[-1]["spo2"])]])
            for name, model in pkg["models"].items():
                pred = model.predict(X_in)[0]
                conf = max(model.predict_proba(X_in)[0]) * 100
                predictions[name] = (pred, conf)
        with st.spinner("Generating PDF..."):
            buf = generate_pdf(patient, df, predictions)
        if buf:
            st.download_button("⬇️ Download Report",
                                buf,
                                f"report_{patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                "application/pdf")
            st.success("✅ Ready to download!")

# ═════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ═════════════════════════════════════════════════════════
def page_settings():
    st.markdown('<div class="sec-head">EMAIL ALERT SETTINGS</div>', unsafe_allow_html=True)
    st.info("Use Gmail with App Password. Enable 2FA → Google Account → Security → App Passwords")
    c1,c2 = st.columns(2)
    with c1:
        st.session_state.alert_email  = st.text_input("Recipient Email",   value=st.session_state.alert_email)
        st.session_state.smtp_user    = st.text_input("Gmail Address",     value=st.session_state.smtp_user)
    with c2:
        st.session_state.smtp_pass    = st.text_input("Gmail App Password",value=st.session_state.smtp_pass, type="password")
        st.session_state.smtp_enabled = st.toggle("Enable Alerts",         value=st.session_state.smtp_enabled)
    if st.button("📧 Send Test Email"):
        if st.session_state.smtp_user and st.session_state.smtp_pass and st.session_state.alert_email:
            ok, msg = send_email(st.session_state.alert_email, st.session_state.smtp_user,
                                  st.session_state.smtp_pass, 120, 91.5, "TEST ALERT")
            st.success(msg) if ok else st.error(msg)
        else:
            st.warning("Fill all email fields first.")

    st.markdown('<div class="sec-head">SYSTEM INFO</div>', unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#07111f;border:1px solid #0f2035;border-radius:12px;padding:20px;
    font-family:'DM Mono',monospace;font-size:11px;color:#4a6a88;line-height:2.2">
    DATABASE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → Supabase (PostgreSQL)<br>
    PROJECT URL &nbsp;&nbsp;&nbsp; → {st.secrets.get('supabase',{}).get('url','Not configured')}<br>
    SENSOR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → MAX30102 (SpO₂ + Heart Rate)<br>
    MCU &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → Arduino Mega 2560<br>
    ML MODELS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → KNN · Logistic Regression · Random Forest · SVM<br>
    ANOMALY &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → Isolation Forest<br>
    FORECAST &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → Linear Extrapolation (30-step)<br>
    REFRESH &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → 15s data cache + auto rerun
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    PATIENTS    = load_patient_db()
    PATIENT_MAP = {p["id"]: p for p in PATIENTS}

    with st.sidebar:
        st.markdown(f"""<div style="padding:16px 0 20px;border-bottom:1px solid #0f2035;margin-bottom:16px">
          <div style="font-size:24px">{st.session_state.avatar}</div>
          <div style="font-weight:700;color:#dce8f5;margin:4px 0">{st.session_state.user_name}</div>
          <div style="font-family:'DM Mono',monospace;font-size:10px;color:#2a4a68;letter-spacing:2px">{st.session_state.role.upper()} ACCESS</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**🧑‍⚕️ Select Patient**")
        if not PATIENTS:
            st.error("No patients found. Run medpulse_supabase.py first.")
            st.stop()

        pat_labels  = [f"{p['id']} — {p['name']}" for p in PATIENTS]
        pat_choice  = st.selectbox("Patient", pat_labels, label_visibility="collapsed")
        current_pat = PATIENTS[pat_labels.index(pat_choice)]

        st.markdown(f"""<div style="background:#040d18;border:1px solid #0f2035;border-radius:10px;
            padding:14px;margin:10px 0 16px;font-family:'DM Mono',monospace;font-size:10px;color:#4a6a88;line-height:2.1">
          <span style="color:#dce8f5;font-size:13px;font-weight:700">🧑 {current_pat['name']}</span><br>
          ID &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → {current_pat['id']}<br>
          Age &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → {current_pat.get('age','—')}<br>
          Doctor &nbsp;&nbsp; → {current_pat.get('doctor','—')}<br>
          Admitted → {current_pat.get('admitted','—')}
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        pages = (["Dashboard","ML Analysis","Anomaly Detection","Prediction","Comparison","PDF Report","Settings"]
                 if st.session_state.role == "doctor"
                 else ["Dashboard","Prediction"])
        page = st.radio("Navigation", pages, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("**⚙️ Settings**")
        bpm_high    = st.number_input("BPM High Threshold",  value=100)
        bpm_low     = st.number_input("BPM Low Threshold",   value=60)
        spo2_low    = st.number_input("SpO₂ Low Threshold",  value=94)
        num_records = st.slider("Records to Load", 50, 500, 300, 50)
        refresh_sec = st.selectbox("Auto-refresh (sec)", [10,15,30,60], index=1)
        auto_refresh= st.toggle("Auto Refresh", value=True)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in ["logged_in","username","role","user_name","avatar","email_sent"]:
                st.session_state[k] = False if k == "logged_in" else ""
            st.rerun()

    # Fetch data from Supabase
    with st.spinner(f"Loading data for {current_pat['name']}..."):
        df, err = fetch_data(current_pat["id"], num_records)

    if err or df is None:
        st.error(f"No data for **{current_pat['name']}**: {err}")
        st.info("Run `python medpulse_supabase.py` and enter this patient's name to start sending readings.")
        st.stop()

    # Train models
    pkg = train_models(current_pat["id"], f"{current_pat['id']}_{len(df)}")
    if pkg is None:
        pkg = {"models": {}, "scaler": StandardScaler(), "iso": None}

    # Route pages
    if   page == "Dashboard":         page_dashboard(df, pkg, current_pat, bpm_high, bpm_low, spo2_low, auto_refresh, refresh_sec)
    elif page == "ML Analysis":        page_ml(df, pkg, current_pat)
    elif page == "Anomaly Detection":  page_anomaly(df, pkg)
    elif page == "Prediction":         page_prediction(df)
    elif page == "Comparison":         page_comparison(df)
    elif page == "PDF Report":         page_report(df, pkg, current_pat)
    elif page == "Settings":           page_settings()
