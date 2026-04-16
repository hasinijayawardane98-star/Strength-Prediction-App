import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD DATA (for structure only)
# =========================
from utils import load_concrete_strength, get_bounds
data = load_concrete_strength()
data.bounds = get_bounds(data.X_columns)

cols = data.X_columns[:-1]

# =========================
# LOAD TRAINED MODEL (NO RETRAINING)
# =========================
model = torch.load("concrete_model.pt", weights_only=False)

# =========================
# CREATE BASE INPUT (IMPORTANT)
# =========================
X_base = data.gwp_data[0][[0]].clone()

# =========================
# UI
# =========================
st.title("Concrete Strength Predictor")
st.write("Enter mix proportions to predict strength and GWP")
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Input Mix Proportions")
    cement = st.number_input("Cement (kg/m3) [Range: 0 – 400 kg/m³]",min_value=0,max_value=400, value=300, step=1, format="%d")
    fly_ash = st.number_input("Fly Ash (kg/m3) [Range: 0 – 400 kg/m³]",min_value=0,max_value=400, value=50,step=1, format="%d")
    slag = st.number_input("Slag (kg/m3) [Range: 0 – 400 kg/m³]",min_value=0,max_value=400, value=50, step=1, format="%d")
    water = st.number_input("Water (kg/m3) [Range: 100 – 250 kg/m³]",min_value=100,max_value=250, value=180, step=1, format="%d")
    hrwr = st.number_input("HRWR (kg/m3)", value=0, step=1, format="%d")
    coarse = st.number_input("Coarse Aggregates (kg/m3) [Range: 500 – 2000 kg/m³]",min_value=500,max_value=2000, value=900, step=1, format="%d")
    fine = st.number_input("Fine Aggregate (kg/m3) [Range: 500 – 1000 kg/m³]",min_value=500,max_value=1000, value=700, step=1, format="%d")

# =========================
# CONSTANT DENSITIES (kg/m³)
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
# w/cm RATIO
# =========================
cementitious = cement + fly_ash + slag

if cementitious > 0:
    w_cm = water / cementitious
    st.info(f"Water-to-Cementitious Ratio (w/cm) = {w_cm:.3f}")
else:
    st.warning("Enter cementitious materials to calculate w/cm ratio")

# =========================
# VOLUME CALCULATIONS (m³)
# =========================
vol_cement = cement / DENSITY["cement"]
vol_flyash = fly_ash / DENSITY["fly_ash"]
vol_slag = slag / DENSITY["slag"]
vol_water = water / DENSITY["water"]
vol_hrwr = hrwr / DENSITY["hrwr"]
vol_coarse = fine / DENSITY["coarse"]

# =========================
# AUTO CALCULATE FINE
# =========================
remaining_volume = 1 - (
    vol_cement + vol_flyash + vol_slag +
    vol_water + vol_hrwr + vol_coarse
)

if remaining_volume > 0:
    fine = remaining_volume * DENSITY["fine"]
    vol_fine = remaining_volume
    st.success(f"Auto Fine Aggregate = {fine:.2f} kg/m³")
else:
    fine = 0
    vol_fine = 0
    st.error("❌ Total volume exceeded before adding fine aggregate!")

# =========================
# DISPLAY VOLUMES
# 
with col2:
    st.subheader("Component Volumes (m³)")
    
    st.write(f"Cement: {vol_cement:.4f}")
    st.write(f"Fly Ash: {vol_flyash:.4f}")
    st.write(f"Slag: {vol_slag:.4f}")
    st.write(f"Water: {vol_water:.4f}")
    st.write(f"HRWR: {vol_hrwr:.4f}")
    st.write(f"Coarse Aggregate: {vol_coarse:.4f}")
    st.write(f"Fine Aggregate: {vol_fine:.4f}")
   
# =========================
# TOTAL VOLUME CHECK
# =========================
total_volume = (
    vol_cement + vol_flyash + vol_slag +
    vol_water + vol_hrwr + vol_fine + vol_coarse
)

st.write(f"Total Volume = {total_volume:.4f} m³")

if total_volume > 1:
    st.error("❌ Total volume exceeds 1 m³")
else:
    st.success("✅ Total volume within 1 m³")

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
if st.button("Get 28-Day Strength"):

    Xt = build_input(28)

    post = model.strength_model.posterior(Xt)
    pred = post.mean.detach().squeeze()

    strength_28 = pred.item()

    st.success(f"28-Day Strength = {strength_28:.3f} Psi")

# =========================
# GWP VALUE
# =========================
if st.button("Get GWP"):

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
   st.success(f"GWP = {gwp_value:.3f} kg CO₂/m³")


# =========================
# STRENGTH CURVE
# =========================
if st.button("Generate Strength Curve"):

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
    f"28-day Strength = {strength_28:.2f} Psi",
    xy=(28, strength_28),
    xytext=(20, strength_28 + 75),
    ha="center",
    arrowprops=dict(arrowstyle="->", color="red"),
    color="red")
    
    ax.set_xlabel("Days")
    ax.set_ylabel("Strength (Psi)")
    ax.set_title("Predicted Concrete Strength Curve")
    


    st.pyplot(fig)
