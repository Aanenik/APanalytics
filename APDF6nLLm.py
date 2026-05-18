# =========================================================
# AI ENERGY ANALYTICS DASHBOARD
# FASTAPI + MYSQL + OPENAI LLM + SMART ALERTS
# =========================================================

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
#from openai import OpenAI
import os
import traceback
from datetime import datetime 
from sklearn.ensemble import IsolationForest

# =========================================================
# FASTAPI APP

app = FastAPI()

# =========================================================
# DATABASE
# =========================================================

password = quote_plus(
    os.getenv("DB_PASSWORD", "Node@2025")
)

engine = create_engine(
    f"mysql+pymysql://root:{password}@194.61.31.18:3369/Technode"
)


# =========================================================
# GET MTIDS
# =========================================================

def get_all_mtids():

    df = pd.read_sql(
        "SELECT DISTINCT MTID FROM todayslive LIMIT 100",
        engine
    )

    return df["MTID"].dropna().tolist()

# =========================================================
# FETCH DATA
# =========================================================

def fetch_data(table, mtid):

    query = f"""
    SELECT
        GatewayRT,
        KWH,
        KVAH,
        KVArH,
        KW,
        Avg_PF,
        AVG_VLL AS Voltage,
        Avg_I AS Current
    FROM {table}
    WHERE MTID=%s
    ORDER BY GatewayRT
    LIMIT 5000
    """

    df = pd.read_sql(query, engine, params=(mtid,))

    if df.empty:
        return df

    df["GatewayRT"] = pd.to_datetime(
        df["GatewayRT"],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["GatewayRT"])
        .set_index("GatewayRT")
        .sort_index()
    )

    cols = [
        "KWH",
        "KVAH",
        "KVArH",
        "KW",
        "Avg_PF",
        "Voltage",
        "Current"
    ]

    for col in cols:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace("NA", "")
            .str.replace("-", "")
            .str.strip()
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df.dropna(how="all")

# =========================================================
# PROCESS DATA
# =========================================================

def process(table, mtid):

    df = fetch_data(table, mtid)

    if df.empty:
        return df

    # ENERGY DIFFERENCE

    for col in ["KWH", "KVAH", "KVArH"]:

        diff = df[col].diff()

        df[col + "_diff"] = diff.where(
            (diff >= 0) & (diff < 10000)
        )

    # 15 MIN RESAMPLE

    df = df.resample("15min").agg({

        "KWH": "max",

        "KWH_diff": "sum",

        "KVAH_diff": "sum",

        "KVArH_diff": "sum",

        "KW": "mean",

        "Avg_PF": "mean",

        "Voltage": "mean",

        "Current": "mean"
    })

    # MAXIMUM DEMAND

    # MAXIMUM DEMAND

    df["Maximum_Demand_kW"] = (
        df["KWH_diff"] * 4
    )
    
    df["Maximum_Demand_kVA"] = (
        df["Maximum_Demand_kW"] / df["Avg_PF"].replace(0, pd.NA)
    )

    return df.dropna(how="all")
# =========================================================
# ANALYSIS
# =========================================================

def analyze(df, col):

    if df.empty or col not in df.columns:

        return {
            "avg": 0,
            "max": 0,
            "min": 0,
            "max_time": "-",
            "min_time": "-"
        }

    s = df[col].dropna()

    if s.empty:

        return {
            "avg": 0,
            "max": 0,
            "min": 0,
            "max_time": "-",
            "min_time": "-"
        }

    return {

        "avg": round(s.mean(), 2),

        "max": round(s.max(), 2),

        "min": round(s.min(), 2),

        "max_time": s.idxmax().strftime("%I:%M %p"),

        "min_time": s.idxmin().strftime("%I:%M %p")
    }

# =========================================================
# ENERGY ANALYSIS
# =========================================================

def analyze_energy(df, col):

    if df.empty or col not in df.columns:

        return {
            "total": 0,
            "avg": 0,
            "max": 0
        }

    s = df[col].dropna()

    return {

        "total": round(s.sum(), 2),

        "avg": round(s.mean(), 2),

        "max": round(s.max(), 2)
    }

# =========================================================
# TOD ANALYSIS
# =========================================================

def calculate_tod(df):

    if df.empty:
        return {}

    def zone(t):

        if t.hour < 6:
            return "A"

        elif t.hour < 9:
            return "B"

        elif t.hour < 17:
            return "C"

        else:
            return "D"

    df = df.copy()

    df["zone"] = df.index.map(zone)

    energy = (
        df.groupby("zone")["KWH_diff"]
        .sum()
        .to_dict()
    )

    for z in ["A", "B", "C", "D"]:
        energy.setdefault(z, 0)

    total = sum(energy.values())

    return {

        **{
            k: round(v, 2)
            for k, v in energy.items()
        },

        "total": round(total, 2),

        "max_zone": (
            max(energy, key=energy.get)
            if total else "-"
        )
    }

# =========================================================
# SMART ALERTS
# =========================================================

def smart_alerts(df):

    alerts = []

    if df.empty:
        return alerts

    avg_pf = df["Avg_PF"].mean()
    avg_voltage = df["Voltage"].mean()
    max_demand = df["Maximum_Demand_kW"].max()
    current = df["Current"].mean()

    # PF ALERT
    if avg_pf < 0.90:
        alerts.append(("❌ Critical: Power Factor too low", "critical"))
    elif avg_pf < 0.95:
        alerts.append(("⚠ Warning: Power Factor below optimal", "warning"))

    # VOLTAGE ALERT
    if avg_voltage > 450:
        alerts.append(("❌ Critical: Voltage too high", "critical"))
    elif avg_voltage > 440:
        alerts.append(("⚠ Warning: Voltage slightly high", "warning"))

    elif avg_voltage < 380:
        alerts.append(("❌ Critical: Voltage too low", "critical"))
    elif avg_voltage < 390:
        alerts.append(("⚠ Warning: Voltage slightly low", "warning"))

    # DEMAND ALERT
    if max_demand > 120:
        alerts.append(("❌ Critical: Demand heavily exceeded", "critical"))
    elif max_demand > 100:
        alerts.append(("⚠ Warning: Demand exceeded threshold", "warning"))

    # CURRENT ALERT
    if current > 350:
        alerts.append(("❌ Critical: Very high current", "critical"))
    elif current > 300:
        alerts.append(("⚠ Warning: High current", "warning"))

    # NORMAL
    if not alerts:
        alerts.append(("✅ System operating normally", "normal"))

    return alerts

# ========================================================
# INTERPRETATION LAYER
# ========================================================
def interpret_feature(col, direction):

    if col == "Current":
        return "High current draw indicates increased load consumption"

    elif col == "Voltage":
        if direction == "up":
            return "Voltage rise may be contributing to increased power usage"
        else:
            return "Voltage drop may indicate system inefficiency or load stress"

    elif col == "Avg_PF":
        return "Low power factor indicates inefficient energy usage"

    elif col == "KW":
        return "Active power (kW) increase directly raises demand"

    else:
        return f"{col} is influencing demand"

# ========================================================
# DEMAND ROOT CAUSE 
# ========================================================
def demand_root_cause(df):
    if df.empty or len(df) < 20:
        return ["Not enough data"]

    recent = df.tail(50)

    causes = []

    # Trend
    demand_trend = recent["Maximum_Demand_kW"].diff().mean()

    if demand_trend <= 0:
        return ["Demand is stable"]

    # Correlation check
    corr = recent.corr(numeric_only=True)["Maximum_Demand_kW"]

    # Remove self
    #corr = corr.drop("Maximum_Demand_kW")
    corr = corr.drop([
    "Maximum_Demand_kW",
    "KWH_diff",
    "KVAH_diff"
], errors="ignore")

    top = corr.abs().sort_values(ascending=False).head(3)

    for col in top.index:
        val = corr[col]
        
        direction = "up" if val > 0 else "down"

        interpretation = interpret_feature(col, direction)
        
        causes.append(interpretation)

    return causes

# =========================================================
# PF DROP
# =========================================================
def pf_root_cause(df):

    if df.empty:
        return ["No data"]

    low_pf = df[df["Avg_PF"] < 0.95]
    normal_pf = df[df["Avg_PF"] >= 0.95]

    if low_pf.empty or normal_pf.empty:
        return ["Not enough PF variation"]

    causes = []

    for col in ["Voltage", "Current", "KW"]:
        
        diff = low_pf[col].mean() - normal_pf[col].mean()

        if abs(diff) > 5:
            direction = "higher" if diff > 0 else "lower"
            causes.append(f"{col} is {direction} during low PF")

    return causes if causes else ["No strong PF pattern found"]

 # =========================================================
# DETECT ANOMALIES
# =========================================================   
def detect_anomalies(df):

    if df.empty or len(df) < 50:
        return ["Not enough data"]

    features = df[[
        "Voltage",
        "Current",
        "KW",
        "Avg_PF",
        "Maximum_Demand_kW"
    ]].dropna()

    if len(features) < 20:
        return ["Insufficient clean data"]

    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(features)

    anomalies = features[preds == -1]

    if anomalies.empty:
        return ["No major anomalies detected"]

    return [
        f"Detected {len(anomalies)} abnormal behavior points"
    ]

# =========================================================
# AI INSIGHT CARD
# ========================================================= 
def ai_root_cause_card(df):
    demand = demand_root_cause(df)
    pf = pf_root_cause(df)
    anomaly = detect_anomalies(df)

    return f"""
    <div class="card">

        <h2>🧠 AI Root Cause Analysis</h2>

        <b>📈 Demand Increase:</b><br>
        {'<br>'.join(demand)}<br><br>

        <b>⚡ Power Factor Drop:</b><br>
        {'<br>'.join(pf)}<br><br>

        <b>🚨 Hidden Anomalies:</b><br>
        {'<br>'.join(anomaly)}

    </div>
    """
# =========================================================
# UI CARDS
# =========================================================

def card(title, d, unit=""):

    return f"""

    <div class="card">

        <div class="card-top">

            <div class="card-title">
                {title}
            </div>

            <div class="unit">
                {unit}
            </div>

        </div>

        <div class="main-value">
            {d['avg']}
        </div>

        <div class="stats">

            <div class="stat-box">

                <span>MAX</span>

                <b>{d['max']}</b>

                <small>{d['max_time']}</small>

            </div>

            <div class="stat-box">

                <span>MIN</span>

                <b>{d['min']}</b>

                <small>{d['min_time']}</small>

            </div>

        </div>

    </div>
    """

# =========================================================
# ENERGY CARD
# =========================================================

def energy_card(title, d,unit):

    return f"""

    <div class="card">

        <div class="card-top">

            <div class="card-title">
                {title}
            </div>

            <div class="unit">
                {unit}
            </div>

        </div>

        <div class="main-value">
            {d['total']}
        </div>

        <div class="stats">

            <div class="stat-box">

                <span>AVG</span>

                <b>{d['avg']}</b>

            </div>

            <div class="stat-box">

                <span>MAX</span>

                <b>{d['max']}</b>

            </div>

        </div>

    </div>
    """

# =========================================================
# ALERT CARD
# =========================================================

def alert_card(alerts):

    items = ""

    for text, severity in alerts:

        if severity == "critical":
            color = "#fee2e2"
            border = "#dc2626"

        elif severity == "warning":
            color = "#fff7ed"
            border = "#f97316"

        else:
            color = "#ecfdf5"
            border = "#10b981"

        items += f"""
        <div style="
            background:{color};
            border-left:5px solid {border};
            padding:14px;
            margin-bottom:12px;
            border-radius:10px;
            font-weight:500;
        ">
            {text}
        </div>
        """

    return f"""
    <div class="card">
        <h2 style="margin-bottom:15px;">🚨 Smart Alerts</h2>
        {items}
    </div>
    """



# =========================================================
# BUILD SECTION
# =========================================================

def build_section(title, df):

    alerts = smart_alerts(df)

    return f"""

    <div class="section-title">
        {title}
    </div>

    <div class="grid">

        {card("Power Factor", analyze(df,"Avg_PF"), "PF")}
        {card("Voltage", analyze(df,"Voltage"), "V")}
        {card("Current", analyze(df,"Current"), "A")}
        {card("Maximum Demand kVA", analyze(df,"Maximum_Demand_kVA"), "kVA")}

        {energy_card("KWH", analyze_energy(df,"KWH_diff"), "kWh")}
        {energy_card("KVAH", analyze_energy(df,"KVAH_diff"), "kVAh")}
        {energy_card("KVArH", analyze_energy(df,"KVArH_diff"), "kVArh")}

        {alert_card(alerts)}

        {ai_root_cause_card(df)}

    </div>
    """

# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    options = "".join([

        f'<option value="{m}">{m}</option>'

        for m in get_all_mtids()
    ])

    return f"""

    <html>

    <head>

    <style>

    body{{
        font-family:Segoe UI;
        background:linear-gradient(135deg,#4f46e5,#7c3aed);
        height:100vh;
        display:flex;
        justify-content:center;
        align-items:center;
        margin:0;
    }}

    .container{{
        background:white;
        width:420px;
        padding:40px;
        border-radius:24px;
        text-align:center;
        box-shadow:0 10px 40px rgba(0,0,0,0.2);
    }}

    h1{{
        font-size:34px;
        color:#4f46e5;
    }}

    p{{
        color:#6b7280;
        margin-bottom:30px;
    }}

    select{{
        width:100%;
        padding:16px;
        border-radius:14px;
        border:1px solid #ddd;
        margin-bottom:20px;
    }}

    button{{
        width:100%;
        padding:16px;
        border:none;
        border-radius:14px;
        background:#4f46e5;
        color:white;
        font-size:16px;
        font-weight:700;
        cursor:pointer;
    }}

    </style>

    </head>

    <body>

        <div class="container">

        <div style="
            display:flex;
            align-items:center;
            justify-content:center;
            gap:12px;
            margin-bottom:15px;
        ">
        
        <img
            src="https://bayasense.com/IMG/bayasense.png"
            style="
                height:45px;
                width:auto;
                object-fit:contain;
            "
        >
       <!-- 
        <h1 style="
            margin:0;
            font-size:36px;
            font-weight:800;
        ">
            AI Energy Dashboard
        </h1>
        -->
        </div>

            <p>
                AI Powered Electrical Monitoring
            </p>

            <form action="/dashboard">

                <select name="mtid">

                    {options}

                </select>

                <button>
                    Open Dashboard
                </button>

            </form>

        </div>

    </body>

    </html>
    """

# =========================================================
# DASHBOARD
# =========================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(mtid: str):

    try:

        df_today = process("todayslive", mtid)
        df_15 = process("dashbordlive", mtid)
        df_month = process("onemonthlive", mtid)

        now = datetime.now().strftime(
            "%d %b %Y %I:%M %p"
        )

        return f"""

        <html>

        <head>

        <style>

        body{{
            font-family:Segoe UI;
            background:#f3f6fb;
            margin:0;
            padding:30px;
        }}

        .header{{
            background:linear-gradient(135deg,#4f46e5,#7c3aed);
            color:white;
            padding:30px;
            border-radius:24px;
            margin-bottom:30px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            position: relative;
        }}

        .header h1{{
            margin:0;
            font-size:38px;
        }}

        .header-box{{
            background:rgba(255,255,255,0.15);
            padding:16px 22px;
            border-radius:16px;
        }}

                /* TABS */
        
        .tabs{{
            display:flex;
            gap:12px;
            margin-bottom:30px;
        }}
        
        .tab-btn{{
            padding:12px 22px;
            border:none;
            border-radius:12px;
            background:#e5e7eb;
            cursor:pointer;
            font-weight:600;
            transition:0.2s;
        }}
        
        .tab-btn:hover{{
            background:#d1d5db;
        }}
        
        .tab-btn.active{{
            background:#4f46e5;
            color:white;
        }}
        
        .tab-content{{
            display:none;
        }}
        
        .tab-content.active{{
            display:block;
        }}

        .section-title{{
            font-size:30px;
            margin-bottom:25px;
            font-weight:700;
        }}

        .grid{{
            display:grid;
            grid-template-columns:
            repeat(auto-fit,minmax(320px,1fr));
            gap:24px;
        }}

        .card{{
            background:white;
            padding:26px;
            border-radius:22px;
            box-shadow:0 6px 20px rgba(0,0,0,0.06);
        }}

        .card-top{{
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:20px;
        }}

        .card-title{{
            font-size:18px;
            font-weight:600;
        }}

        .unit{{
            background:#eef2ff;
            color:#4f46e5;
            padding:6px 12px;
            border-radius:999px;
            font-size:12px;
        }}

        .main-value{{
            font-size:54px;
            font-weight:800;
            color:#111827;
            margin-bottom:20px;
        }}

        .stats{{
            display:flex;
            gap:14px;
        }}

        .stat-box{{
            flex:1;
            background:#f9fafb;
            padding:18px;
            border-radius:14px;
        }}

        .stat-box span{{
            color:#6b7280;
            font-size:12px;
        }}

        .stat-box b{{
            display:block;
            margin:8px 0;
            font-size:26px;
        }}

        .stat-box small{{
            font-size:15px;
            color:#6b7280;
        }}

        .alert-item{{
            background:#fff7ed;
            border-left:5px solid #f97316;
            padding:14px;
            margin-bottom:12px;
            border-radius:10px;
        }}

        .llm-box{{
            background:#f8fafc;
            padding:18px;
            border-radius:14px;
            line-height:1.8;
            white-space:pre-wrap;
        }}

        </style>

        <script>

        function showTab(tab){{

            document
                .querySelectorAll(".tab-content")
                .forEach(t => t.classList.remove("active"));
        
            document
                .querySelectorAll(".tab-btn")
                .forEach(b => b.classList.remove("active"));
        
            document
                .getElementById(tab)
                .classList.add("active");
        
            document
                .getElementById(tab + "-btn")
                .classList.add("active");
        }}

</script>

        </head>

        <body>

        <div class="header">

        <!-- LEFT LOGO -->
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
            ">
                <img
                    src="https://bayasense.com/IMG/bayasense.png"
                    style="
                        height:45px;
                        width:auto;
                        object-fit:contain;
                    "
                >
               
            </div>
        
        
            <!-- CENTER TITLE -->
            <div style="
                position:absolute;
                left:50%;
                transform:translateX(-50%);
                text-align:center;
            ">
                <h1 style="
                    margin:0;
                    font-size:36px;
                    font-weight:800;
                ">
                    AI Energy Dashboard
                </h1>
        
                <p style="
                    margin:6px 0 0 0;
                    font-size:18px;
                    opacity:0.9;
                ">
                    Smart Electrical Monitoring System
                </p>
            </div>
        
        
            <!-- RIGHT BOX -->
            <div class="header-box">
        
                <b>MTID</b><br>
                {mtid}
        
                <br><br>
        
                <b>Updated</b><br>
                {now}
        
            </div>
        
        </div>

        <!-- TABS -->
        <div class="tabs">
        
            <button
                id="today-btn"
                class="tab-btn active"
                onclick="showTab('today')"
            >
                Today
            </button>
        
            <button
                id="fifteen-btn"
                class="tab-btn"
                onclick="showTab('fifteen')"
            >
                15 Days
            </button>
        
            <button
                id="monthly-btn"
                class="tab-btn"
                onclick="showTab('monthly')"
            >
                Monthly
            </button>
        
        </div>
        
        <!-- TODAY -->
        <div
            id="today"
            class="tab-content active"
        >
            {build_section("Today's Analytics", df_today)}
        </div>
        
        <!-- 15 DAYS -->
        <div
            id="fifteen"
            class="tab-content"
        >
            {build_section("15 Days Analytics", df_15)}
        </div>
        
        <!-- MONTHLY -->
        <div
            id="monthly"
            class="tab-content"
        >
            {build_section("Monthly Analytics", df_month)}
</div>

</body>

</html>

       
        """

    except Exception as e:

        print(traceback.format_exc())

        return f"<pre>{str(e)}</pre>"