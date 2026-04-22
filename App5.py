import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD DATA
# =========================
from utils import load_concrete_strength, get_bounds
data = load_concrete_strength()
data.bounds = get_bounds(data.X_columns)

cols = data.X_columns[:-1]

# =========================
# LOAD TRAINED MODEL
# =========================
model = torch.load("concrete_model.pt", weights_only=False)

# =========================
# CREATE BASE INPUT (IMPORTANT)
# =========================
X_base = data.gwp_data[0][[0]].clone()


st.title("Concrete Strength Predictor")
# =========================
# DENSITIES (kg/m³)
# =========================
DENSITY = {
    "cement": 3132.935,
    "fly_ash": 2594.15,
    "slag": 2893.475,
    "water": 997.75,
    "hrwr": 1100,
    "fine": 2650,
    "coarse": 2673.97
}
# =========================
# USER INPUTS & VOLUMES
# =========================
# =========================
# HEADER
# =========================
h1, gap, h2 = st.columns([3, 1, 2])
with h1:
    st.markdown("### Inputs (kg/m³)")
with h2:
    st.markdown("### Volumes (m³)")

# =========================
# ROW 1 — CEMENT
# =========================
c1, gap, c2 = st.columns([3, 1, 2])

with c1:
    cement = st.number_input("Cement (kg/m3) [Range: 0 – 400 kg/m³]",min_value=0,max_value=400, value=300, step=1)

vol_cement = cement / DENSITY["cement"]

with c2:
    st.markdown(
    f"""
    Cement Volume  
    <b style='font-size:16px;'>{vol_cement:.4f} </b>
    """,
    unsafe_allow_html=True)

# =========================
# ROW 2 — FLY ASH
# =========================
c1, gap, c2 = st.columns([3, 1, 2])

with c1:
    fly_ash = st.number_input("Fly Ash (kg/m3) [Range: 0 – 400 kg/m³]",min_value=0,max_value=400, value=50,step=1)

vol_flyash = fly_ash / DENSITY["fly_ash"]

with c2:
    st.markdown(
    f"""
    Fly Ash Volume  
    <b style='font-size:16px;'>{vol_flyash:.4f} </b>
    """,
    unsafe_allow_html=True
)
    

# =========================
# ROW 3 — SLAG
# =========================
c1, gap, c2 = st.columns([3, 1, 2])

with c1:
    slag = st.number_input("Slag (kg/m3) [Range: 0 – 400 kg/m³]",min_value=0,max_value=400, value=50, step=1)

vol_slag = slag / DENSITY["slag"]

with c2:
    st.markdown(
    f"""
    Slag Volume  
    <b style='font-size:16px;'>{vol_slag:.4f} </b>
    """,
    unsafe_allow_html=True
)

# =========================
# ROW 4 — WATER
# =========================
c1, gap, c2 = st.columns([3, 1, 2])

with c1:
    water = st.number_input("Water (kg/m3) [Range: 100 – 250 kg/m³]",min_value=100,max_value=250, value=180, step=1)

vol_water = water / DENSITY["water"]

with c2:
    st.markdown(
    f"""
    Water Volume  
    <b style='font-size:16px;'>{vol_water:.4f} </b>
    """,
    unsafe_allow_html=True
)
# =========================
# ROW 5 — COARSE AGG
# =========================
c1, gap, c2 = st.columns([3, 1, 2])

with c1:
    coarse = st.number_input("Coarse Aggregates (kg/m3) [Range: 500 – 2000 kg/m³]",min_value=500,max_value=2000, value=900, step=1)

vol_coarse = coarse / DENSITY["coarse"]

with c2:
    st.markdown(
    f"""
    Coarse Aggregate Volume  
    <b style='font-size:16px;'>{vol_coarse:.4f} </b>
    """,
    unsafe_allow_html=True
)

# =========================
# ROW 6 — FINE AGGREGATE 
# =========================
c1, gap, c2 = st.columns([3, 1, 2])

with c1:
    fine = st.number_input("Fine Aggregate (kg/m3) [Range: 500 – 2000 kg/m³]",min_value=500,max_value=2000, value=920, step=1)

vol_fine = fine/ DENSITY["fine"]

with c2:
    st.markdown(
    f"""
    Fine Aggregate Volume  
    <b style='font-size:16px;'>{vol_fine:.4f} </b>
    """,
    unsafe_allow_html=True
)
    
# =========================
# TOTAL VOLUME
# =========================

# calculate volumes
total_volume = (
    vol_cement + vol_flyash + vol_slag +
    vol_water + vol_coarse + vol_fine
)

# check if user input is valid

TOL = 0.01   # 1%

lower_limit = 1 - TOL
upper_limit = 1 + TOL

col1, gap, col2 = st.columns([3, 1, 2])

with col1:
    st.markdown(
        """
        <div style='font-size:22px; font-weight:600;'>
        🎯 Target Volume
        </div>
        <div style='font-size:28px; font-weight:bold;'>
        1.000 m³
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style='font-size:22px; font-weight:600;'>
        ➡️ Total Volume
        </div>
        <div style='font-size:28px; font-weight:bold;'>
        {total_volume:.4f} m³
        </div>
        """,
        unsafe_allow_html=True
    )
if total_volume < lower_limit:
    st.error("❌ Total volume is too low. Please adjust inputs to stay within ±1% of the target (1 m³).")

elif total_volume > upper_limit:
    st.error("❌ Total volume is too high. Please adjust inputs to stay within ±1% of the target (1 m³).")

else:
    st.success("✅ Total volume is within ±1% tolerance of the target (1 m³).")

# =========================
# w/cm RATIO
# =========================
cementitious = cement + fly_ash + slag


if cementitious > 0:
    w_cm = water / cementitious
    st.info(f"Water-to-Cementitious Ratio (w/cm) = {w_cm:.3f}")

st.markdown("---")
st.markdown("### 🎮 Ready? Let’s compute your mix performance...")

hrwr = 0

# =========================
# FUNCTION: BUILD INPUT
# =========================
def build_input(time_value):
    X = X_base.clone()

    # update values
    X[0, cols.index("Cement (kg/m3)")] = float(cement)
    X[0, cols.index("Fly Ash (kg/m3)")] = float(fly_ash)
    X[0, cols.index("Slag (kg/m3)")] = float(slag)
    X[0, cols.index("Water (kg/m3)")] = float(water)
    X[0, cols.index("HRWR (kg/m3)")] = float(hrwr)
    X[0, cols.index("Fine Aggregate (kg/m3)")] = float(fine)
    X[0, cols.index("Coarse Aggregates (kg/m3)")] = float(coarse)

    # add time
    Xt = torch.cat([X, torch.tensor([[float(time_value)]])], dim=1)

    return Xt

# =========================
# 28-DAY STRENGTH
# =========================
if st.button("🧱 Get 28-Day Strength"):

    Xt = build_input(28)

    post = model.strength_model.posterior(Xt)
    pred = post.mean.detach().squeeze()

    strength_28 = pred.item()

    st.success(f"🧱 28-Day Strength = {strength_28:.3f} psi")

# =========================
# GWP VALUE
# =========================
if st.button("🌍 Get GWP"):

    # GWP model DOES NOT use time → use X without time column
   X = X_base.clone()
   X[0, cols.index("Cement (kg/m3)")] = float(cement)
   X[0, cols.index("Fly Ash (kg/m3)")] = float(fly_ash)
   X[0, cols.index("Slag (kg/m3)")] = float(slag)
   X[0, cols.index("Water (kg/m3)")] = float(water)
   X[0, cols.index("HRWR (kg/m3)")] = float(hrwr)
   X[0, cols.index("Fine Aggregate (kg/m3)")] = float(fine)
   X[0, cols.index("Coarse Aggregates (kg/m3)")] = float(coarse)
   
   post = model.gwp_model.posterior(X)
   pred = post.mean.detach().squeeze()
   gwp_value = -pred.mean().item()
   st.success(f"🌍 GWP = {gwp_value:.3f} kg CO₂/m³")


# =========================
# STRENGTH CURVE
# =========================
if st.button(""📈 Generate Strength Curve"):

    days = np.linspace(1, 28, 50)   # 🔥 KEY FIX
    predictions = []

    for t in days:
        Xt = build_input(t)

        post = model.strength_model.posterior(Xt)
        pred = post.mean.detach().squeeze()

        predictions.append(pred.item())
    
    Xt_28 = build_input(28)
    post_28 = model.strength_model.posterior(Xt_28)
    pred_28 = post_28.mean.detach().squeeze()
    strength_28 = pred_28.item()
    
    fig, ax = plt.subplots()

    ax.plot(days, predictions,label="Strength Curve")

    # 🔥 Highlight point
    ax.scatter(28, strength_28, s=30,color="red", label="28-Day Strength")
    ymin = min(predictions)
    ax.vlines(28, ymin, strength_28, linestyles="--", color="red", linewidth=1.5)
    ax.hlines(strength_28, 0, 28, linestyles="--", color="red", linewidth=1.5)
    ax.annotate(
    f"28-day Strength = {strength_28:.2f} psi",
    xy=(28, strength_28),
    xytext=(20, strength_28 + 75),
    ha="center",
    arrowprops=dict(arrowstyle="->", color="red"),
    color="red")
    
    ax.set_xlabel("Days")
    ax.set_ylabel("Strength (psi)")
    ax.set_title("Predicted Concrete Strength Curve")
    


    st.pyplot(fig)

PRICE = {
    "cement": 0.13,     # $/kg
    "fly_ash": 0.04,
    "slag": 0.07,
    "water": 0.001,
    "fine": 0.02,
    "coarse": 0.02
}
if st.button("💰 Get Cost"):

    total_cost = (
        cement * PRICE["cement"] +
        fly_ash * PRICE["fly_ash"] +
        slag * PRICE["slag"] +
        water * PRICE["water"] +
        fine * PRICE["fine"] +
        coarse * PRICE["coarse"]
    )

    st.success(f"💰 Total Cost = ${total_cost:.3f} per m³")
