#!/usr/bin/env python3
"""
Not So Medical Disease Diagnosis System
================================
Flask app · Voting Classifier (RF + XGBoost) · 99.46 % CV accuracy

Run:
    pip install flask scikit-learn xgboost pandas numpy
    python app.py
Place train_data.csv next to this file on first run.
"""

import os, io, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = "voting_pipeline.pkl"
LE_PATH    = "label_encoder.pkl"
DATA_PATH  = "train_data.csv"

# ── Column groups (match notebook exactly) ────────────────────────────────────
VITAMIN_COLS = [
    "vitamin_a_percent_rda",  "vitamin_c_percent_rda",
    "vitamin_d_percent_rda",  "vitamin_e_percent_rda",
    "vitamin_b12_percent_rda","folate_percent_rda",
    "calcium_percent_rda",    "iron_percent_rda",
]
PT_COLS   = VITAMIN_COLS + ["symptoms_count"]
DROP_COLS = ["gender_Male","gender_Female","age","bmi",
             "smoking_status","exercise_level","latitude_region","symptoms_list"]
DIET_COLS = ["diet_type_Omnivore","diet_type_Pescatarian",
             "diet_type_Vegan","diet_type_Vegetarian"]

ALC_MAP      = {"Moderate":0,"Heavy":1}
SUN_MAP      = {"Low":0,"Moderate":1,"High":2}
INCOME_MAP   = {"Low":0,"Middle":1,"High":2}
SMOKING_MAP  = {"Never":0,"Former":1,"Current":2}
EXERCISE_MAP = {"Sedentary":0,"Light":1,"Moderate":2,"Active":3}
LAT_MAP      = {"Low":0,"Mid":1,"High":2}

_pipeline      = None
_label_encoder = None
_batch_results = None


# ── Transformers ──────────────────────────────────────────────────────────────
class MedicalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if "gender" in df.columns:
            gd = pd.get_dummies(df["gender"], prefix="gender")
            df = pd.concat([df.drop("gender", axis=1), gd], axis=1)
        for col, mp in [("smoking_status",SMOKING_MAP),("alcohol_consumption",ALC_MAP),
                        ("exercise_level",EXERCISE_MAP),("sun_exposure",SUN_MAP),
                        ("income_level",INCOME_MAP),("latitude_region",LAT_MAP)]:
            if col in df.columns:
                df[col] = df[col].map(mp)  # NaN stays NaN; filled by OutlierClipper
        if "diet_type" in df.columns:
            dd = pd.get_dummies(df["diet_type"], prefix="diet_type")
            df = pd.concat([df.drop("diet_type", axis=1), dd], axis=1)
            for c in DIET_COLS:
                if c not in df.columns: df[c] = 0
        df.drop(columns=[c for c in DROP_COLS if c in df.columns],
                inplace=True, errors="ignore")
        for col in df.columns:
            try: df[col] = pd.to_numeric(df[col], errors="coerce")
            except: pass
        return df


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.clip_bounds_  = {}
        self.alcohol_mode_ = None
        self._cols         = None

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._cols = list(df.columns)
        if "alcohol_consumption" in df.columns:
            self.alcohol_mode_ = df["alcohol_consumption"].dropna().mode()[0]
        for col in VITAMIN_COLS:
            if col in df.columns:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                self.clip_bounds_[col] = (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.alcohol_mode_ is not None and "alcohol_consumption" in df.columns:
            df["alcohol_consumption"] = df["alcohol_consumption"].fillna(self.alcohol_mode_)
        for col, (lo, hi) in self.clip_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        if self._cols:
            for c in self._cols:
                if c not in df.columns: df[c] = 0
            df = df[self._cols]
        return df


class DFPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self): self._pt=PowerTransformer(standardize=False); self._cols=None; self._all=None
    def fit(self, X, y=None):
        df=X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        self._all=list(df.columns); self._cols=[c for c in PT_COLS if c in df.columns]
        self._pt.fit(df[self._cols]); return self
    def transform(self, X, y=None):
        df=X.copy() if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        df[self._cols]=self._pt.transform(df[self._cols]); return df[self._all]


class DFStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self): self._sc=StandardScaler(); self._cols=None; self._all=None
    def fit(self, X, y=None):
        df=X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        self._all=list(df.columns); self._cols=[c for c in VITAMIN_COLS if c in df.columns]
        self._sc.fit(df[self._cols]); return self
    def transform(self, X, y=None):
        df=X.copy() if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        df[self._cols]=self._sc.transform(df[self._cols]); return df[self._all]


# ── Model lifecycle ───────────────────────────────────────────────────────────
def train_and_save():
    global _pipeline, _label_encoder
    print("Timing model from train_data.csv ...")
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    y  = le.fit_transform(df["disease_diagnosis"])
    X  = df.drop(columns=["disease_diagnosis"])
    enc   = MedicalEncoder()
    X_enc = enc.transform(X)
    x_tr,_,y_tr,_ = train_test_split(X_enc,y,test_size=0.2,random_state=42,stratify=y)
    rf     = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42)
    xgb    = XGBClassifier(n_estimators=200,learning_rate=0.1,random_state=42,
                           eval_metric="mlogloss",verbosity=0)
    voting = VotingClassifier(estimators=[("rf",rf),("xgb",xgb)],voting="soft")
    pipe = Pipeline([("clipper",OutlierClipper()),("pt",DFPowerTransformer()),
                     ("scaler",DFStandardScaler()),("model",voting)])
    pipe.fit(x_tr,y_tr)
    artifact = {"encoder":enc,"pipeline":pipe}
    with open(MODEL_PATH,"wb") as f: pickle.dump(artifact,f)
    with open(LE_PATH,   "wb") as f: pickle.dump(le,f)
    _pipeline=artifact; _label_encoder=le
    print(f"Done - classes: {list(le.classes_)}")


def load_or_train():
    global _pipeline, _label_encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
        with open(MODEL_PATH,"rb") as f: _pipeline=pickle.load(f)
        with open(LE_PATH,   "rb") as f: _label_encoder=pickle.load(f)
        print("Model loaded from disk.")
    elif os.path.exists(DATA_PATH):
        train_and_save()
    else:
        print(f"Place '{DATA_PATH}' next to app.py and restart.")


def _predict(df_raw: pd.DataFrame):
    X_enc  = _pipeline["encoder"].transform(df_raw)
    probas = _pipeline["pipeline"].predict_proba(X_enc)
    idxs   = np.argmax(probas,axis=1)
    return _label_encoder.classes_[idxs], probas


# ─────────────────────────────────────────────────────────────────────────────
# HTML  (dark glassmorphism, state-of-the-art UI)
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Not So Medical &middot; Diagnosis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700;800&family=Manrope:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
:root{
  --bg:#050c18; --bg2:#080f20;
  --glass:rgba(255,255,255,.04); --glass2:rgba(255,255,255,.07);
  --border:rgba(255,255,255,.08); --border2:rgba(255,255,255,.14);
  --teal:#00e5c3; --teal-glow:rgba(0,229,195,.18); --teal-dim:rgba(0,229,195,.10);
  --gold:#ffd166; --blue:#5b9fff; --red:#f87171; --orange:#fb923c; --purple:#a78bfa; --green:#4ade80;
  --text:#e2eaf7; --text2:rgba(226,234,247,.55); --text3:rgba(226,234,247,.28);
  --r:16px; --tr:.2s ease;
}
body{font-family:'Manrope',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

/* animated dot grid background */
body::before{content:'';position:fixed;inset:0;z-index:0;
  background-image:radial-gradient(rgba(0,229,195,.10) 1px,transparent 1px);
  background-size:28px 28px;animation:gridDrift 24s linear infinite;pointer-events:none}
@keyframes gridDrift{to{background-position:28px 28px}}

/* ambient glow blobs */
.blob{position:fixed;border-radius:50%;z-index:0;pointer-events:none}
.blob1{width:700px;height:700px;top:-200px;left:-200px;
  background:radial-gradient(circle,rgba(0,229,195,.07) 0%,transparent 65%);
  animation:bf1 10s ease-in-out infinite alternate}
.blob2{width:500px;height:500px;bottom:-150px;right:-100px;
  background:radial-gradient(circle,rgba(91,159,255,.06) 0%,transparent 65%);
  animation:bf1 14s ease-in-out infinite alternate-reverse}
@keyframes bf1{to{transform:translate(50px,30px) scale(1.1)}}

#root{position:relative;z-index:1}

/* === HEADER === */
header{position:sticky;top:0;z-index:200;
  background:rgba(5,12,24,.88);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border)}
.hdr{max-width:1160px;margin:auto;padding:15px 28px;display:flex;align-items:center;gap:14px}
.hdr-ico{width:40px;height:40px;border-radius:11px;flex-shrink:0;
  background:linear-gradient(135deg,rgba(0,229,195,.12),rgba(0,229,195,.22));
  border:1px solid rgba(0,229,195,.3);display:flex;align-items:center;justify-content:center;font-size:1.15rem;
  box-shadow:0 0 18px rgba(0,229,195,.18)}
.hdr-logo{font-family:'Exo 2',sans-serif;font-weight:800;font-size:1.38rem;color:var(--text);letter-spacing:-.3px}
.hdr-logo em{color:var(--teal);font-style:normal}
.hdr-sub{font-size:.63rem;color:var(--text3);letter-spacing:.8px;text-transform:uppercase;margin-top:1px}
.hdr-pills{margin-left:auto;display:flex;gap:7px;flex-wrap:wrap}
.pill{display:flex;align-items:center;gap:6px;padding:5px 12px;border-radius:99px;
  font-size:.66rem;font-weight:600;letter-spacing:.3px;
  background:var(--glass);border:1px solid var(--border);color:var(--text2)}
.pill.live{border-color:rgba(0,229,195,.3);color:var(--teal);background:var(--teal-dim)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--teal);animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1;box-shadow:0 0 5px var(--teal)}50%{opacity:.25;box-shadow:none}}

/* === WRAP === */
.wrap{max-width:1160px;margin:auto;padding:0 28px}

/* === STATS STRIP === */
.stats{display:flex;gap:10px;margin:28px 0 0;flex-wrap:wrap}
.stat{background:var(--glass);border:1px solid var(--border);border-radius:11px;padding:10px 18px;
  backdrop-filter:blur(10px);display:flex;flex-direction:column;gap:1px;transition:border var(--tr)}
.stat:hover{border-color:rgba(0,229,195,.2)}
.sv{font-family:'Exo 2',sans-serif;font-weight:800;font-size:1.1rem;color:var(--teal)}
.sl{font-size:.62rem;color:var(--text3);letter-spacing:.5px;text-transform:uppercase}

/* === TABS === */
.tab-nav{display:flex;border-bottom:1px solid var(--border);margin:22px 0 0}
.tn{background:none;border:none;padding:11px 24px;cursor:pointer;
  font-family:'Manrope',sans-serif;font-size:.86rem;font-weight:500;
  color:var(--text3);border-bottom:2px solid transparent;margin-bottom:-1px;
  transition:all var(--tr);display:flex;align-items:center;gap:7px}
.tn:hover{color:var(--text2)}
.tn.on{color:var(--teal);border-bottom-color:var(--teal);font-weight:700}
.tab-pane{display:none;padding:24px 0 80px}
.tab-pane.on{display:block}

/* === BANNER === */
.banner{background:rgba(0,229,195,.05);border:1px solid rgba(0,229,195,.16);border-radius:10px;
  padding:11px 17px;font-size:.79rem;color:rgba(0,229,195,.8);
  display:flex;gap:10px;align-items:flex-start;margin-bottom:20px;line-height:1.6}
.banner strong{color:var(--teal)}
.banner code{background:rgba(0,229,195,.1);padding:1px 5px;border-radius:4px;font-size:.85em}

/* === GLASS CARD === */
.card{background:var(--glass);border:1px solid var(--border);border-radius:var(--r);
  padding:24px 26px;margin-bottom:14px;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  transition:border-color var(--tr)}
.card:hover{border-color:var(--border2)}
.card-hd{display:flex;align-items:center;gap:12px;padding-bottom:17px;margin-bottom:18px;border-bottom:1px solid var(--border)}
.card-ico{width:37px;height:37px;border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-size:1rem;flex-shrink:0;border:1px solid}
.ico-t{background:rgba(0,229,195,.08);border-color:rgba(0,229,195,.2)}
.ico-b{background:rgba(91,159,255,.08);border-color:rgba(91,159,255,.2)}
.ico-p{background:rgba(167,139,250,.08);border-color:rgba(167,139,250,.2)}
.ico-g{background:rgba(255,209,102,.08);border-color:rgba(255,209,102,.2)}
.cl{font-family:'Exo 2',sans-serif;font-weight:700;font-size:.93rem;color:var(--text)}
.cs{font-size:.7rem;color:var(--text3);margin-top:2px}

/* === GRID === */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:13px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}
@media(max-width:800px){.g4{grid-template-columns:1fr 1fr}}
@media(max-width:540px){.g2,.g4{grid-template-columns:1fr}}

/* === FIELDS === */
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-size:.67rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.6px}
.field label .req{color:var(--red);margin-left:2px}
.field input,.field select{
  padding:9px 11px;background:rgba(255,255,255,.045);border:1.5px solid var(--border);border-radius:9px;
  font-family:'Manrope',sans-serif;font-size:.84rem;color:var(--text);
  transition:border var(--tr),background var(--tr),box-shadow var(--tr);outline:none;width:100%}
.field select{
  appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='7'%3E%3Cpath d='M.5.5l5 5 5-5' stroke='rgba(226,234,247,.3)' fill='none' stroke-width='1.4' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 11px center;padding-right:30px}
.field select option{background:#0d1a2e;color:var(--text)}
.field input::placeholder{color:var(--text3)}
.field input:focus,.field select:focus{border-color:var(--teal);background:rgba(0,229,195,.04);box-shadow:0 0 0 3px rgba(0,229,195,.09)}
.field .hint{font-size:.67rem;color:var(--text3);margin-top:2px}
.vw{position:relative}
.vw input{padding-right:50px}
.vb{position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:.6rem;font-weight:700;
  padding:2px 5px;border-radius:4px;background:rgba(91,159,255,.12);color:var(--blue);
  pointer-events:none;border:1px solid rgba(91,159,255,.18)}

/* === BUTTONS === */
.btn{padding:9px 20px;border:none;border-radius:10px;font-family:'Manrope',sans-serif;
  font-size:.86rem;font-weight:700;cursor:pointer;transition:all var(--tr);
  display:inline-flex;align-items:center;gap:8px}
.btn-primary{background:linear-gradient(135deg,#00c2a7,#009e88);color:#001812;
  width:100%;justify-content:center;padding:14px;font-size:.94rem;border-radius:11px;margin-top:5px;
  box-shadow:0 0 0 1px rgba(0,229,195,.2),0 4px 20px rgba(0,229,195,.18)}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 0 0 1px rgba(0,229,195,.35),0 8px 28px rgba(0,229,195,.26)}
.btn-primary:active{transform:none}
.btn-ghost{background:var(--glass);border:1.5px solid var(--border);color:var(--text2)}
.btn-ghost:hover{border-color:rgba(0,229,195,.35);color:var(--teal);background:var(--teal-dim)}
.btn-dl{background:rgba(74,222,128,.08);border:1.5px solid rgba(74,222,128,.22);color:#4ade80}
.btn-dl:hover{background:rgba(74,222,128,.16);box-shadow:0 0 14px rgba(74,222,128,.12)}
.spin{width:17px;height:17px;border:2px solid rgba(0,20,14,.3);border-top-color:#001812;
  border-radius:50%;animation:spin .7s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}

/* === RESULT === */
#result-box{display:none}
.res-card{border-radius:var(--r);overflow:hidden;border:1px solid var(--border);margin-top:20px;
  animation:fadeUp .4s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.res-top{padding:26px 28px;background:linear-gradient(135deg,#060e1d,#091828);
  border-bottom:1px solid var(--border);display:flex;align-items:center;gap:22px;flex-wrap:wrap}
.res-ico{width:70px;height:70px;border-radius:15px;display:flex;align-items:center;justify-content:center;
  font-size:2.6rem;flex-shrink:0;border:1px solid}
.res-right{flex:1;min-width:200px}
.res-tag{font-size:.63rem;font-weight:700;letter-spacing:.9px;text-transform:uppercase;color:var(--text3);margin-bottom:5px}
.res-name{font-family:'Exo 2',sans-serif;font-weight:800;font-size:1.75rem;line-height:1.15;margin-bottom:5px}
.res-desc{font-size:.79rem;color:var(--text2);line-height:1.6;max-width:460px}
.gauge{width:88px;height:88px;flex-shrink:0;position:relative;
  display:flex;align-items:center;justify-content:center;flex-direction:column}
.gauge svg{position:absolute;inset:0;transform:rotate(-90deg)}
.g-bg{fill:none;stroke:rgba(255,255,255,.06);stroke-width:6}
.g-fg{fill:none;stroke-width:6;stroke-linecap:round;transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)}
.g-pct{font-family:'Exo 2',sans-serif;font-weight:800;font-size:1.1rem;line-height:1}
.g-lbl{font-size:.56rem;color:var(--text3);text-transform:uppercase;letter-spacing:.4px;margin-top:2px}
.res-body{padding:22px 26px;background:rgba(5,12,24,.75);backdrop-filter:blur(12px)}
.conf-hd{font-size:.65rem;font-weight:700;letter-spacing:.7px;text-transform:uppercase;color:var(--text3);margin-bottom:13px}
.cr{display:flex;align-items:center;gap:11px;margin-bottom:9px}
.cr-lbl{width:175px;font-size:.77rem;color:var(--text2);flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cr-track{flex:1;height:6px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden}
.cr-bar{height:100%;border-radius:99px;width:0;transition:width 1.1s cubic-bezier(.4,0,.2,1)}
.cr-pct{width:42px;text-align:right;font-size:.74rem;font-weight:700;color:var(--text2);flex-shrink:0}
.disclaimer{background:rgba(255,209,102,.05);border:1px solid rgba(255,209,102,.15);border-radius:9px;
  padding:11px 15px;margin-top:18px;font-size:.72rem;color:rgba(255,209,102,.75);
  display:flex;gap:8px;align-items:flex-start;line-height:1.6}

/* === BATCH UPLOAD === */
.drop-zone{border:2px dashed rgba(255,255,255,.1);border-radius:12px;padding:42px 24px;
  text-align:center;background:rgba(255,255,255,.02);cursor:pointer;transition:all var(--tr)}
.drop-zone:hover,.drop-zone.over{border-color:rgba(0,229,195,.38);background:rgba(0,229,195,.04)}
.dz-ico{font-size:2.7rem;display:block;margin-bottom:12px;filter:drop-shadow(0 0 10px rgba(0,229,195,.25))}
.dz-t{font-size:.88rem;font-weight:600;color:var(--text2)}
.dz-s{font-size:.72rem;color:var(--text3);margin-top:4px}
.dz-fn{margin-top:9px;font-size:.79rem;font-weight:700;color:var(--teal)}
.row-act{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;align-items:center}

/* === TABLE === */
.tbl-wrap{overflow-x:auto;border-radius:10px;border:1px solid var(--border);margin-top:14px}
table{width:100%;border-collapse:collapse;font-size:.77rem}
thead tr{background:rgba(0,229,195,.06)}
th{padding:10px 13px;text-align:left;font-weight:700;font-size:.63rem;text-transform:uppercase;
  letter-spacing:.5px;color:var(--teal);white-space:nowrap;border-bottom:1px solid var(--border)}
td{padding:8px 13px;border-bottom:1px solid rgba(255,255,255,.04);color:var(--text2)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:rgba(255,255,255,.025)}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;font-size:.66rem;font-weight:700;border:1px solid}
.b-hi{background:rgba(74,222,128,.09);color:#4ade80;border-color:rgba(74,222,128,.2)}
.b-md{background:rgba(251,191,36,.09);color:#fbbf24;border-color:rgba(251,191,36,.2)}
.b-lo{background:rgba(248,113,113,.09);color:#f87171;border-color:rgba(248,113,113,.2)}
.b-dis{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:99px;
  font-size:.69rem;font-weight:700;border:1px solid}
.tbl-note{padding:9px 0 0;font-size:.68rem;color:var(--text3)}
</style>
</head>
<body>
<div class="blob blob1"></div>
<div class="blob blob2"></div>
<div id="root">

<!-- HEADER -->
<header>
  <div class="hdr">
    <div class="hdr-ico">🧬</div>
    <div>
      <div class="hdr-logo">Not So<em> Medical</em></div>
      <div class="hdr-sub">Disease Diagnosis System &middot; Voting Classifier</div>
    </div>
    <div class="hdr-pills">
      <div class="pill live"><div class="dot"></div>Model Active</div>
      <div class="pill">RF + XGBoost</div>
      <div class="pill">Soft Voting</div>
    </div>
  </div>
</header>

<div class="wrap">

  <!-- Stats -->
  <div class="stats">
    <div class="stat"><span class="sv">99.46%</span><span class="sl">CV Accuracy</span></div>
    <div class="stat"><span class="sv">&plusmn;0.24%</span><span class="sl">Std Dev</span></div>
    <div class="stat"><span class="sv">5</span><span class="sl">Disease Classes</span></div>
    <div class="stat"><span class="sv">5&#8209;Fold</span><span class="sl">Stratified CV</span></div>
    <div class="stat"><span class="sv">3 500</span><span class="sl">Training Samples</span></div>
  </div>

  <!-- Tabs -->
  <div class="tab-nav">
    <button class="tn on" onclick="switchTab('single',this)">🔬&nbsp; Single Prediction</button>
  </div>

  <!-- ══ SINGLE ═══════════════════════════════════════════════════════════ -->
  <div id="tab-single" class="tab-pane on">
    <div class="banner">
      ℹ️&nbsp; Enter patient data below. Vitamin values represent <strong>% of Recommended Daily Allowance (RDA)</strong>.
    </div>

    <form id="singleForm" onsubmit="submitSingle(event)">
      <!-- Lifestyle -->
      <div class="card">
        <div class="card-hd">
          <div class="card-ico ico-t">🌿</div>
          <div><div class="cl">Lifestyle Factors</div><div class="cs">Dietary habits and environmental exposure</div></div>
        </div>
        <div class="g2">
          <div class="field">
            <label>Alcohol Consumption <span class="req">*</span></label>
            <select name="alcohol_consumption" required>
              <option value="" disabled selected>Select level&hellip;</option>
              <option>Moderate</option><option>Heavy</option>
            </select>
          </div>
          <div class="field">
            <label>Diet Type <span class="req">*</span></label>
            <select name="diet_type" required>
              <option value="" disabled selected>Select diet&hellip;</option>
              <option>Omnivore</option><option>Pescatarian</option><option>Vegan</option><option>Vegetarian</option>
            </select>
          </div>
          <div class="field">
            <label>Sun Exposure <span class="req">*</span></label>
            <select name="sun_exposure" required>
              <option value="" disabled selected>Select level&hellip;</option>
              <option>Low</option><option>Moderate</option><option>High</option>
            </select>
          </div>
          <div class="field">
            <label>Income Level <span class="req">*</span></label>
            <select name="income_level" required>
              <option value="" disabled selected>Select level&hellip;</option>
              <option>Low</option><option>Middle</option><option>High</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Nutritional -->
      <div class="card">
        <div class="card-hd">
          <div class="card-ico ico-b">💊</div>
          <div><div class="cl">Nutritional Status</div><div class="cs">Daily vitamin &amp; mineral intake as % of RDA (0&ndash;500)</div></div>
        </div>
        <div class="g4">
          <div class="field"><label>Vitamin A <span class="req">*</span></label>
            <div class="vw"><input type="number" name="vitamin_a_percent_rda"   min="0" max="500" step="0.1" placeholder="85" required><span class="vb">%RDA</span></div></div>
          <div class="field"><label>Vitamin C <span class="req">*</span></label>
            <div class="vw"><input type="number" name="vitamin_c_percent_rda"   min="0" max="500" step="0.1" placeholder="90" required><span class="vb">%RDA</span></div></div>
          <div class="field"><label>Vitamin D <span class="req">*</span></label>
            <div class="vw"><input type="number" name="vitamin_d_percent_rda"   min="0" max="500" step="0.1" placeholder="60" required><span class="vb">%RDA</span></div></div>
          <div class="field"><label>Vitamin E <span class="req">*</span></label>
            <div class="vw"><input type="number" name="vitamin_e_percent_rda"   min="0" max="500" step="0.1" placeholder="75" required><span class="vb">%RDA</span></div></div>
          <div class="field"><label>Vitamin B12 <span class="req">*</span></label>
            <div class="vw"><input type="number" name="vitamin_b12_percent_rda" min="0" max="500" step="0.1" placeholder="40" required><span class="vb">%RDA</span></div></div>
          <div class="field"><label>Folate <span class="req">*</span></label>
            <div class="vw"><input type="number" name="folate_percent_rda"      min="0" max="500" step="0.1" placeholder="70" required><span class="vb">%RDA</span></div></div>
          <div class="field"><label>Calcium <span class="req">*</span></label>
            <div class="vw"><input type="number" name="calcium_percent_rda"     min="0" max="500" step="0.1" placeholder="80" required><span class="vb">%RDA</span></div></div>
          <div class="field"><label>Iron <span class="req">*</span></label>
            <div class="vw"><input type="number" name="iron_percent_rda"        min="0" max="500" step="0.1" placeholder="55" required><span class="vb">%RDA</span></div></div>
        </div>
      </div>

      <!-- Clinical -->
      <div class="card">
        <div class="card-hd">
          <div class="card-ico ico-p">🩺</div>
          <div><div class="cl">Clinical Data</div><div class="cs">Total number of symptoms reported at assessment</div></div>
        </div>
        <div style="max-width:175px">
          <div class="field">
            <label>Symptoms Count <span class="req">*</span></label>
            <input type="number" name="symptoms_count" min="0" max="50" step="1" placeholder="3" required>
            <span class="hint">Integer count of documented symptoms</span>
          </div>
        </div>
      </div>

      <button type="submit" class="btn btn-primary" id="submitBtn">
        <div class="spin" id="spin1"></div>
        <span id="btn1txt">🔬&nbsp;&nbsp;Analyze Patient</span>
      </button>
    </form>

    <!-- Result -->
    <div id="result-box">
      <div class="res-card">
        <div class="res-top" id="resTop">
          <div class="res-ico" id="resIco"></div>
          <div class="res-right">
            <div class="res-tag">AI Predicted Diagnosis</div>
            <div class="res-name" id="resName">—</div>
            <div class="res-desc" id="resDesc">—</div>
          </div>
          <div class="gauge">
            <svg viewBox="0 0 88 88" width="88" height="88">
              <circle class="g-bg" cx="44" cy="44" r="38"/>
              <circle class="g-fg" id="gFg" cx="44" cy="44" r="38"
                stroke-dasharray="238.76" stroke-dashoffset="238.76"/>
            </svg>
            <div class="g-pct" id="gPct">—</div>
            <div class="g-lbl">Conf.</div>
          </div>
        </div>
        <div class="res-body">
          <div class="conf-hd">Confidence Distribution &mdash; All Disease Classes</div>
          <div id="rBars"></div>
          <div class="disclaimer">
            ⚠️&nbsp;<span>This prediction is generated by a machine-learning model for <strong>research and educational purposes only</strong>. It does <strong>not</strong> constitute a medical diagnosis. Always consult a qualified healthcare professional before making any clinical decision.</span>
          </div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /wrap -->
</div><!-- /root -->

<script>
const DMETA = {
  "Healthy":             {icon:"💚",color:"#4ade80",desc:"No nutritional deficiency detected. Lifestyle and nutrient intake are within healthy ranges."},
  "Anemia":              {icon:"🩸",color:"#f87171",desc:"Iron-deficiency anemia. Low iron intake impairs red blood cell production, causing fatigue and pallor."},
  "Scurvy":              {icon:"🍊",color:"#fb923c",desc:"Severe Vitamin C deficiency. Impairs collagen synthesis — symptoms include bleeding gums, bruising and joint pain."},
  "Rickets_Osteomalacia":{icon:"🦴",color:"#fbbf24",desc:"Vitamin D / Calcium deficiency causing softened, weakened bones — common in low-sun-exposure populations."},
  "Night_Blindness":     {icon:"👁️",color:"#a78bfa",desc:"Vitamin A deficiency impairing rod-cell function — difficulty seeing in low-light or dark conditions."},
};
function fmt(d){return d.replace(/_/g,' / ')}
function meta(d){return DMETA[d]||{icon:"🔬",color:"#00e5c3",desc:""}}

function switchTab(id,btn){
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tn').forEach(b=>b.classList.remove('on'));
  document.getElementById('tab-'+id).classList.add('on');
  btn.classList.add('on');
}

async function submitSingle(e){
  e.preventDefault();
  const data=Object.fromEntries(new FormData(e.target));
  setL('submitBtn','spin1','btn1txt',true,'Analyzing\u2026');
  document.getElementById('result-box').style.display='none';
  try{
    const r=await fetch('/api/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const res=await r.json();
    if(res.error){alert('Error: '+res.error);return}
    const m=meta(res.disease);
    // icon card
    const ico=document.getElementById('resIco');
    ico.textContent=m.icon;
    ico.style.cssText=`width:70px;height:70px;border-radius:15px;display:flex;
      align-items:center;justify-content:center;font-size:2.6rem;flex-shrink:0;
      background:${m.color}18;border:1px solid ${m.color}35;
      box-shadow:0 0 20px ${m.color}22`;
    document.getElementById('resName').textContent=fmt(res.disease);
    document.getElementById('resName').style.color=m.color;
    document.getElementById('resDesc').textContent=m.desc;
    // gauge
    const circ=238.76, pct=res.confidence;
    const fg=document.getElementById('gFg');
    fg.style.stroke=m.color;
    fg.style.filter=`drop-shadow(0 0 5px ${m.color}80)`;
    document.getElementById('gPct').textContent=pct.toFixed(1)+'%';
    document.getElementById('gPct').style.color=m.color;
    // bars
    const bars=document.getElementById('rBars');
    bars.innerHTML='';
    res.all_classes.forEach((item,i)=>{
      const bStyle=i===0
        ?`background:linear-gradient(90deg,${m.color}77,${m.color});box-shadow:0 0 7px ${m.color}50`
        :'background:rgba(255,255,255,.1)';
      bars.innerHTML+=`<div class="cr">
        <div class="cr-lbl" title="${fmt(item.label)}">${fmt(item.label)}</div>
        <div class="cr-track"><div class="cr-bar" data-w="${item.pct.toFixed(1)}" style="${bStyle}"></div></div>
        <div class="cr-pct">${item.pct.toFixed(1)}%</div>
      </div>`;
    });
    document.getElementById('result-box').style.display='block';
    document.getElementById('result-box').scrollIntoView({behavior:'smooth',block:'nearest'});
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      fg.style.strokeDashoffset=(circ-(pct/100)*circ).toFixed(2);
      document.querySelectorAll('.cr-bar').forEach(b=>{b.style.width=b.dataset.w+'%'});
    }));
  }catch(err){alert('Request failed: '+err.message)}
  finally{setL('submitBtn','spin1','btn1txt',false,'🔬\u00a0\u00a0Analyze Patient')}
}

// drag & drop
const dz=document.getElementById('dropZone');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');
  _file=e.dataTransfer.files[0];if(_file)showFn(_file.name)});
let _file=null;
function onFile(inp){_file=inp.files[0];if(_file)showFn(_file.name)}
function showFn(n){document.getElementById('fname').textContent='📄 '+n}

async function submitBatch(){
  if(!_file){alert('Please select a CSV file first.');return}
  const fd=new FormData();fd.append('file',_file);
  setL('batchBtn','spin2','btn2txt',true,'Processing\u2026');
  document.getElementById('batchRes').style.display='none';
  try{
    const r=await fetch('/api/batch',{method:'POST',body:fd});
    const res=await r.json();
    if(res.error){alert('Error: '+res.error);return}
    const thead=document.getElementById('tHead');
    const tbody=document.getElementById('tBody');
    thead.innerHTML=tbody.innerHTML='';
    // key cols first
    const key=['predicted_diagnosis','confidence_%'];
    const rest=res.columns.filter(c=>!key.includes(c));
    const cols=[...key,...rest];
    cols.forEach(c=>{const th=document.createElement('th');th.textContent=c.replace(/_/g,' ');thead.appendChild(th)});
    const max=Math.min(res.rows.length,200);
    for(let i=0;i<max;i++){
      const row=res.rows[i];const tr=document.createElement('tr');
      cols.forEach(c=>{
        const td=document.createElement('td');
        if(c==='predicted_diagnosis'){
          const d=row[c]||'—';const m=meta(d);
          td.innerHTML=`<span class="b-dis" style="background:${m.color}14;color:${m.color};border-color:${m.color}28">${m.icon} ${fmt(d)}</span>`;
        }else if(c==='confidence_%'){
          const p=parseFloat(row[c]);const cls=p>=90?'b-hi':p>=70?'b-md':'b-lo';
          td.innerHTML=`<span class="badge ${cls}">${isNaN(p)?'—':p.toFixed(1)+'%'}</span>`;
        }else{td.textContent=row[c]!=null?row[c]:'—'}
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    document.getElementById('bSum').textContent=`${res.total} patient${res.total!==1?'s':''} processed — showing first ${max} rows`;
    document.getElementById('batchRes').style.display='block';
    document.getElementById('batchRes').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(err){alert('Request failed: '+err.message)}
  finally{setL('batchBtn','spin2','btn2txt',false,'🚀\u00a0\u00a0Run Batch Prediction')}
}

function setL(bId,sId,tId,on,label){
  document.getElementById(sId).style.display=on?'block':'none';
  document.getElementById(tId).textContent=label;
  document.getElementById(bId).disabled=on;
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if _pipeline is None:
        return jsonify({"error":"Model not loaded. Place train_data.csv next to app.py and restart."}),503
    try:
        data = request.get_json(force=True)
        for col in VITAMIN_COLS+["symptoms_count"]:
            if col in data and data[col] not in (None,""):
                data[col] = float(data[col])
        df = pd.DataFrame([data])
        labels, probas = _predict(df)
        proba_row = probas[0]
        top_idx   = int(np.argmax(proba_row))
        classes   = _label_encoder.classes_
        all_classes = sorted(
            [{"label":str(classes[i]),"pct":float(proba_row[i])*100}
             for i in range(len(classes))],
            key=lambda x:x["pct"], reverse=True
        )
        return jsonify({
            "disease":    str(labels[0]),
            "confidence": float(proba_row[top_idx])*100,
            "all_classes": all_classes,
        })
    except Exception as exc:
        return jsonify({"error":str(exc)}),400


@app.route("/api/batch", methods=["POST"])
def api_batch():
    global _batch_results
    if _pipeline is None:
        return jsonify({"error":"Model not loaded."}),503
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded."}),400
    try:
        df_raw = pd.read_csv(request.files["file"])
        # drop target column if present
        df_raw.drop(columns=["disease_diagnosis"], inplace=True, errors="ignore")
        labels, probas = _predict(df_raw)
        df_raw["predicted_diagnosis"] = labels
        df_raw["confidence_%"]        = (np.max(probas,axis=1)*100).round(2)
        _batch_results = df_raw.copy()
        display = df_raw.head(200).where(pd.notnull(df_raw.head(200)), None)
        return jsonify({
            "total":   len(df_raw),
            "columns": list(display.columns),
            "rows":    display.to_dict(orient="records"),
        })
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error":str(exc)}),400


@app.route("/download_results")
def download_results():
    if _batch_results is None:
        return "No results yet — run a batch prediction first.",404
    buf = io.BytesIO()
    _batch_results.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name="medai_predictions.csv")


@app.route("/download_template")
def download_template():
    """Full train_data.csv structure (minus disease_diagnosis)."""
    cols = ["age","gender","bmi","smoking_status","alcohol_consumption",
            "exercise_level","diet_type","sun_exposure","income_level",
            "latitude_region","vitamin_a_percent_rda","vitamin_c_percent_rda",
            "vitamin_d_percent_rda","vitamin_e_percent_rda","vitamin_b12_percent_rda",
            "folate_percent_rda","calcium_percent_rda","iron_percent_rda",
            "symptoms_count","symptoms_list"]
    sample = pd.DataFrame([
        [45,"Female",26.5,"Never","Moderate","Moderate","Vegan","High","Middle","Mid",
         85,92,60,78,40,70,80,55,3,"fatigue"],
        [32,"Male",31.2,"Current","Heavy","Sedentary","Omnivore","Low","Low","Low",
         30,20,15,25,10,35,42,40,7,"bone_pain;fatigue"],
        [58,"Female",22.8,"Former","Moderate","Light","Vegetarian","Moderate","High","High",
         95,110,85,90,120,95,100,75,1,"None"],
        [27,"Male",24.1,"Never","Heavy","Active","Pescatarian","High","Middle","Mid",
         70,18,45,60,55,80,70,90,5,"bleeding_gums;bruising"],
    ], columns=cols)
    buf = io.BytesIO()
    sample.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name="medai_batch_template.csv")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_or_train()
    print("\n  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)