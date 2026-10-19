import io
import re
import math
import time
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.metrics import mean_absolute_error, r2_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
import xlsxwriter

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="🌎", layout="wide")
st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB · historical indicators + interactive function analysis")

COUNTRIES = {
    "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Chile": "CHL", "Peru": "PER", "Venezuela": "VEN", "Ecuador": "ECU", "Uruguay": "URY"
}
# Official World Bank indicator metadata. Definitions are concise classroom descriptions.
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Expected years a newborn would live if current age-specific mortality rates persisted."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "GDP divided by the midyear population, expressed in current US dollars; not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants during the period (immigrants minus emigrants); not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "deaths per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a share of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP.")
}
WB_ROOT = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code, indicator_code, start_year, end_year):
    """Retrieve official World Bank observations. Returns a tidy DataFrame and metadata."""
    url = WB_ROOT.format(country=country_code, indicator=indicator_code)
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame(columns=["year", "value"]), f"World Bank returned no observations (HTTP {response.status_code}).", response.url
        records = []
        for item in payload[1]:
            year = item.get("date")
            val = item.get("value")
            if year is not None:
                try:
                    records.append({"year": int(year), "value": float(val) if val is not None else np.nan})
                except (ValueError, TypeError):
                    continue
        df = pd.DataFrame(records, columns=["year", "value"])
        if df.empty or df["value"].notna().sum() == 0:
            msg = "The API responded, but this country/indicator/year window has no non-null observations."
        else:
            msg = f"Retrieved {int(df['value'].notna().sum())} non-null observations from the World Bank API."
        return df.sort_values("year").reset_index(drop=True), msg, response.url
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Retry later or check network access.", url
    except (requests.RequestException, ValueError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"API request failed: {type(exc).__name__}: {exc}", url

def constraint_message(indicator_name, value):
    if not np.isfinite(value):
        return "Values must be finite numbers."
    if indicator_name == "Life expectancy at birth" and not (0 <= value <= 125):
        return "Life expectancy must be between 0 and 125 years."
    if indicator_name == "Unemployment, national estimate" and not (0 <= value <= 100):
        return "Unemployment percentage must be between 0 and 100."
    if indicator_name == "GDP per capita" and value < 0:
        return "GDP per capita cannot be negative."
    if indicator_name == "Crude birth rate" and value < 0:
        return "Crude birth rate cannot be negative."
    if indicator_name == "Intentional homicide rate" and value < 0:
        return "Homicide rate cannot be negative."
    if indicator_name == "Central government debt" and value < 0:
        return "Central government debt percentage cannot be negative."
    if indicator_name == "Total reserves including gold" and value < 0:
        return "Reserves cannot be negative."
    return None

def validate_editable_table(table, indicator_name):
    errors = []
    if table.empty:
        return ["The working table is empty."], pd.DataFrame(columns=["year", "value"])
    clean = table.copy()
    if not {"year", "value"}.issubset(clean.columns):
        return ["Working table must have year and value columns."], pd.DataFrame(columns=["year", "value"])
    numeric_year = pd.to_numeric(clean["year"], errors="coerce")
    numeric_val = pd.to_numeric(clean["value"], errors="coerce")
    if numeric_year.isna().any() or ((numeric_year % 1) != 0).any():
        errors.append("Every year must be an integer.")
    if numeric_val.isna().any():
        errors.append("Every value must be numeric; remove blank/null rows before fitting.")
    clean["year"] = numeric_year
    clean["value"] = numeric_val
    clean = clean.dropna(subset=["year", "value"])
    if clean["year"].duplicated().any():
        errors.append("Duplicate years found. Keep only one row per year.")
    if not clean.empty and ((clean["year"] < 1800) | (clean["year"] > 2200)).any():
        errors.append("Years must be between 1800 and 2200.")
    for v in clean["value"].astype(float):
        problem = constraint_message(indicator_name, v)
        if problem:
            errors.append(problem)
            break
    return errors, clean.sort_values("year").reset_index(drop=True)

def fit_polynomial(data, degree):
    """Fit y = c0 + c1*t + ... where t=year-base_year."""
    data = data.sort_values("year")
    years = data["year"].to_numpy(dtype=float)
    values = data["value"].to_numpy(dtype=float)
    base = int(years.min())
    t = years - base
    if len(np.unique(years)) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; found {len(np.unique(years))}.")
    coef = np.polynomial.polynomial.polyfit(t, values, degree)
    predicted = np.polynomial.polynomial.polyval(t, coef)
    residuals = values - predicted
    r2 = float(r2_score(values, predicted)) if len(values) > 1 else float("nan")
    mae = float(mean_absolute_error(values, predicted))
    return {"base": base, "degree": degree, "coef": coef, "years": years, "values": values,
            "predicted": predicted, "residuals": residuals, "r2": r2, "mae": mae,
            "min_year": int(years.min()), "max_year": int(years.max()), "mean": float(np.mean(values))}

def poly_value(model, year):
    return float(np.polynomial.polynomial.polyval(float(year) - model["base"], model["coef"]))

def equation_text(model, variable="t"):
    terms = []
    for power, c in enumerate(model["coef"]):
        if abs(c) < 1e-12:
            continue
        if power == 0:
            terms.append(f"{c:.8g}")
        elif power == 1:
            terms.append(f"{c:+.8g}{variable}")
        else:
            terms.append(f"{c:+.8g}{variable}^{power}")
    return "y = " + (" ".join(terms) if terms else "0")

def safe_sympy_expression(raw, x):
    """Parse common math notation using a restricted symbol/function dictionary."""
    raw = raw.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not raw:
        raise ValueError("Enter a function, such as x**2 - 3*x + 2.")
    allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
               "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
               "pi": sp.pi, "E": sp.E, "e": sp.E}
    try:
        expr = sp.sympify(raw, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not parse that expression. Use x, +, -, *, /, **, parentheses, sin, cos, tan, exp, log, sqrt, abs, pi, and e.") from exc
    if expr.free_symbols - {x}:
        raise ValueError("Use x as the only variable.")
    if expr.has(sp.Function) and any(f.func not in {sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt, sp.Abs} for f in expr.atoms(sp.Function)):
        raise ValueError("That function is not supported.")
    return expr

def numeric_value(expr, x, val):
    try:
        result = complex(sp.N(expr.subs(x, float(val))))
        if abs(result.imag) > 1e-8 or not np.isfinite(result.real):
            return np.nan
        return float(result.real)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return np.nan

def safe_curve(expr, x, xs):
    ys = np.array([numeric_value(expr, x, v) for v in xs], dtype=float)
    ys[~np.isfinite(ys)] = np.nan
    return ys

def near_zero_roots(expr, x, lo, hi, samples=1800):
    """Numerically locate real roots in an interval, verify by residual and deduplicate."""
    try:
        roots = sp.solve(sp.Eq(expr, 0), x)
    except Exception:
        roots = []
    found = []
    for root in roots:
        try:
            z = complex(sp.N(root))
            if abs(z.imag) < 1e-7 and lo - 1e-8 <= z.real <= hi + 1e-8:
                val = numeric_value(expr, x, z.real)
                if np.isfinite(val) and abs(val) < 1e-5:
                    found.append(float(z.real))
        except Exception:
            pass
    if not found:
        grid = np.linspace(lo, hi, samples)
        vals = safe_curve(expr, x, grid)
        for i in range(len(grid)-1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i] * vals[i+1] < 0:
                a, b = grid[i], grid[i+1]
                for _ in range(45):
                    mid = (a+b)/2
                    fm = numeric_value(expr, x, mid)
                    if not np.isfinite(fm):
                        break
                    if vals[i] * fm <= 0:
                        b = mid
                    else:
                        a = mid
                found.append((a+b)/2)
    result = []
    for v in sorted(found):
        if not result or abs(v-result[-1]) > 1e-4:
            result.append(v)
    return result

def fmt(v, digits=6):
    if v is None or not np.isfinite(v):
        return "undefined"
    return f"{v:.{digits}g}"

# Sidebar: project choices and data source
with st.sidebar:
    st.header("Project setup")
    st.text_input("Research question", key="research_question", placeholder="e.g., How did life expectancy change over time?")
    country_name = st.selectbox("Featured country", list(COUNTRIES), index=0)
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS), index=0)
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_names = st.multiselect("Compare countries (same indicator)", [n for n in COUNTRIES if n != country_name], default=[])
    printer_mode = st.checkbox("Printer-friendly report view", value=False)

indicator_code, units, definition = INDICATORS[indicator_name]
current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
retrieved_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
st.markdown(f"**Research question:** {st.session_state.get('research_question') or 'Not entered yet'}")
st.info("Brazil is included as a Latin American comparison country; its language and history differ from Spanish-speaking countries, so it should not be described as Hispanic.")
st.markdown(f"**Source:** [World Bank indicator {indicator_code}](https://data.worldbank.org/indicator/{indicator_code}) · [API endpoint](https://api.worldbank.org/v2/country/{COUNTRIES[country_name]}/indicator/{indicator_code}?date={start_year}%3A{end_year}&format=json)  \n**Definition:** {definition}  \n**Units:** {units}  \n**Requested years:** {start_year}–{end_year} inclusive · **Retrieved:** {retrieved_at}")

with st.spinner("Retrieving official World Bank data…"):
    source_df, source_message, api_url = fetch_world_bank(COUNTRIES[country_name], indicator_code, start_year, end_year)
st.caption(source_message)
if source_df.empty or source_df["value"].notna().sum() == 0:
    st.error("No usable source observations are available for this selection. Check the API message above, change the country/indicator, or retry later. No substitute values are generated.")
    st.stop()

source_df = source_df.dropna(subset=["value"]).copy()
source_df["year"] = source_df["year"].astype(int)
source_df["value"] = source_df["value"].astype(float)
source_df = source_df.sort_values("year").reset_index(drop=True)
coverage = f"{int(source_df.year.min())}–{int(source_df.year.max())}" if not source_df.empty else "none"
st.write(f"**Actual coverage:** {coverage} · **Non-null observations:** {len(source_df)} (requested window does not guarantee 70 data points).")

# Separate editable working table from immutable-in-session source copy.
data_key = f"work_{COUNTRIES[country_name]}_{indicator_code}"
source_key = f"source_{COUNTRIES[country_name]}_{indicator_code}"
if source_key not in st.session_state:
    st.session_state[source_key] = source_df[["year", "value"]].copy()
if data_key not in st.session_state:
    st.session_state[data_key] = source_df[["year", "value"]].copy()
if st.session_state.get("reset_requested"):
    st.session_state[data_key] = st.session_state[source_key].copy()
    st.session_state["reset_requested"] = False
if st.session_state.get("editor_target") != data_key:
    st.session_state["editor_target"] = data_key
    st.session_state["edited_table"] = st.session_state[data_key].copy()

with st.expander("Edit working data (source data stays separate)", expanded=False):
    st.caption("Edit year/value cells directly. Removing rows is allowed. Years must be unique integers; values must be numeric and plausible for the selected indicator.")
    edited = st.data_editor(st.session_state["edited_table"], num_rows="dynamic", use_container_width=True, key=f"editor_{data_key}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apply edited data", key=f"apply_{data_key}"):
            st.session_state[data_key] = edited.copy()
            st.session_state["edited_table"] = edited.copy()
            st.session_state["last_calculation"] = None
            st.rerun()
    with c2:
        if st.button("Reset to source", key=f"reset_{data_key}"):
            st.session_state[data_key] = st.session_state[source_key].copy()
            st.session_state["edited_table"] = st.session_state[source_key].copy()
            st.session_state["last_calculation"] = None
            st.rerun()

working = st.session_state[data_key].copy()
errors, valid_df = validate_editable_table(working, indicator_name)
edited_status = not working.reset_index(drop=True).equals(st.session_state[source_key].reset_index(drop=True))
if edited_status:
    st.warning("USER-MODIFIED SCENARIO: outputs are based on edited working data, not solely on the original World Bank observations.")
if errors:
    for err in errors:
        st.error(err)
    st.stop()
if len(valid_df) < degree + 1:
    st.error(f"Not enough valid observations for degree {degree}. Need at least {degree+1} distinct years; have {len(valid_df)}.")
    st.stop()
if valid_df["year"].nunique() < degree + 1:
    st.error("Insufficient distinct years after validation.")
    st.stop()

try:
    model = fit_polynomial(valid_df, degree)
except (ValueError, np.linalg.LinAlgError) as exc:
    st.error(f"Could not fit model: {exc}")
    st.stop()

# Main historical-data/model analysis
tab_data, tab_function, tab_compare, tab_export = st.tabs(["Historical data & fit", "Function studio", "Country comparison", "Exports & verification"])
with tab_data:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valid observations", len(valid_df))
    k2.metric("Historical coverage", f"{model['min_year']}–{model['max_year']}")
    k3.metric("In-sample R²", fmt(model["r2"], 5))
    k4.metric("In-sample MAE", f"{fmt(model['mae'])} {units}")
    st.markdown(f"**Centered-time model:** `{equation_text(model, 't')}` where **t = year − {model['base']}**. Degree = {degree}.")
    st.caption("R² and MAE are in-sample fit metrics, not proof of causation or a guarantee of future accuracy. Polynomial extrapolation can become implausible rapidly.")
    # Chronological holdout: reserve last ~20%, only if training still supports degree.
    n = len(valid_df)
    holdout_n = max(1, int(math.ceil(0.2*n)))
    if n - holdout_n >= degree + 1 and holdout_n >= 2:
        train = valid_df.iloc[:-holdout_n]
        test = valid_df.iloc[-holdout_n:]
        try:
            hold = fit_polynomial(train, degree)
            hold_pred = np.array([poly_value(hold, y) for y in test["year"]])
            st.write(f"**Chronological holdout MAE:** {fmt(mean_absolute_error(test['value'], hold_pred))} {units} across {len(test)} later observations. This is separate from the in-sample score.")
        except Exception:
            st.caption("Chronological holdout score unavailable for this selection.")
    else:
        st.caption("Chronological holdout comparison unavailable: too few observations remain to train this degree and evaluate a useful later sample.")
    grid_years = np.arange(model["min_year"], model["max_year"] + 1, sample_step, dtype=int)
    if len(grid_years) == 0 or grid_years[-1] != model["max_year"]:
        grid_years = np.append(grid_years, model["max_year"])
    plot_end = model["max_year"] + horizon
    curve_x = np.linspace(model["min_year"], max(model["max_year"], plot_end), 700)
    curve_y = np.array([poly_value(model, y) for y in curve_x])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(valid_df["year"], valid_df["value"], label="Observed / working data", color="#1f77b4", zorder=4)
    hist_mask = curve_x <= model["max_year"]
    ax.plot(curve_x[hist_mask], curve_y[hist_mask], label="Fitted curve (historical range)", color="#ff7f0e")
    if horizon > 0:
        ax.plot(curve_x[~hist_mask], curve_y[~hist_mask], label="Polynomial extrapolation (caution)", color="#d62728", linestyle="--")
    ax.axvline(model["max_year"], color="gray", linestyle=":", label="Historical boundary")
    ax.scatter(grid_years, [poly_value(model, y) for y in grid_years], marker="x", label=f"Model sample every {sample_step} years")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({units})")
    ax.set_title(f"{country_name}: {indicator_name}")
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    if not printer_mode:
        st.pyplot(fig, use_container_width=True)
    else:
        st.pyplot(fig, use_container_width=True)
    st.dataframe(valid_df.assign(fitted= [poly_value(model, y) for y in valid_df["year"]],
                                 residual=model["residuals"]), use_container_width=True)
    with st.expander("Residuals and model details"):
        residual_df = valid_df[["year", "value"]].copy()
        residual_df["fitted"] = model["predicted"]
        residual_df["residual (observed − fitted)"] = model["residuals"]
        st.dataframe(residual_df, use_container_width=True)
        st.write(f"Mean observed value: {fmt(model['mean'])} {units}.")
        st.write(f"Residual MAE: {fmt(model['mae'])} {units}. Residuals summarize in-sample mismatch; they do not create a validated prediction interval.")
        st.write("No 95% prediction interval is shown. A defensible interval would require a documented error model and assumptions that may not hold for historical indicators.")

# Persistent calculation state: invalidate when model/data identity changes.
signature = (COUNTRIES[country_name], indicator_code, degree,
             tuple(valid_df["year"].tolist()), tuple(np.round(valid_df["value"].to_numpy(), 10).tolist()))
if st.session_state.get("model_signature") != signature:
    st.session_state["model_signature"] = signature
    st.session_state["last_calculation"] = None
with tab_data:
    st.subheader("Calculate model value")
    single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=int(min(max(model["max_year"], 1950), 2100)), step=1)
    if st.button("Calculate Value"):
        st.session_state["last_calculation"] = {"signature": signature, "single": int(single_year)}
    result = st.session_state.get("last_calculation")
    if result and result.get("signature") == signature:
        yy = result["single"]
        vv = poly_value(model, yy)
        cls = "within fitted historical range" if model["min_year"] <= yy <= model["max_year"] else "outside fitted historical range (extrapolation)"
        if not np.isfinite(vv) or abs(vv) > 1e15:
            st.warning("This polynomial result is numerically extreme or non-finite. Treat it as implausible; it has not been clipped.")
        st.success(f"Year {yy}: model estimate = {fmt(vv)} {units}. This is {cls}.")
        if yy in set(valid_df["year"].astype(int)):
            actual = float(valid_df.loc[valid_df["year"] == yy, "value"].iloc[0])
            st.write(f"At this observed year, actual working-data value = {fmt(actual)} {units}; model fitted value = {fmt(vv)} {units}. These are not the same thing.")
    st.subheader("Batch calculations")
    batch_mode = st.radio("Batch input", ["Year range", "Custom comma-separated years"], horizontal=True)
    if batch_mode == "Year range":
        bc1, bc2, bc3 = st.columns(3)
        with bc1: batch_start = st.number_input("Start year", min_value=1800, max_value=2200, value=max(1950, model["min_year"]), key="batch_start")
        with bc2: batch_end = st.number_input("End year", min_value=1800, max_value=2200, value=min(2100, model["max_year"] + 5), key="batch_end")
        with bc3: batch_step = st.number_input("Step (1–10 years)", min_value=1, max_value=10, value=1, key="batch_step")
        if batch_end < batch_start:
            st.error("End year must be greater than or equal to start year.")
            batch_years = []
        else:
            batch_years = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
    else:
        raw_years = st.text_input("Years, separated by commas", value=f"{model['min_year']},{model['max_year']}")
        batch_years, bad_years = [], []
        for part in raw_years.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if not re.fullmatch(r"[+-]?\d+", part):
                    raise ValueError
                year = int(part)
                if not 1800 <= year <= 2200:
                    raise ValueError
                batch_years.append(year)
            except ValueError:
                bad_years.append(part)
        if bad_years:
            st.warning("Ignored invalid year input(s): " + ", ".join(bad_years) + ". Enter integer years from 1800 to 2200.")
        batch_years = list(dict.fromkeys(batch_years))
    if batch_years:
        batch_df = pd.DataFrame({"year": batch_years})
        batch_df["estimate"] = [poly_value(model, y) for y in batch_years]
        batch_df["units"] = units
        batch_df["classification"] = ["within fitted historical range" if model["min_year"] <= y <= model["max_year"] else "outside fitted historical range (extrapolation)" for y in batch_years]
        st.dataframe(batch_df, use_container_width=True)
        st.session_state["batch_results"] = batch_df
    else:
        batch_df = pd.DataFrame(columns=["year", "estimate", "units", "classification"])
        st.caption("No valid batch years to calculate.")

# Function studio: generated best-fit function, templates, or user-entered expression.
with tab_function:
    st.subheader("Function studio")
    context = st.selectbox("Context", ["Business", "Engineering", "Medical", "Sports science", "Biology", "Bioengineering", "Economic forecasting", "Weather", "Custom"])
    context_defaults = {
        "Business": ("Revenue", "time", "years", "revenue", "USD", "x**2 - 4*x + 8", "Revenue model; values and rates are interpreted in the selected units."),
        "Engineering": ("Position", "time", "seconds", "position", "meters", "x**2 - 2*x", "For position versus time, the first derivative is velocity and the second derivative is acceleration."),
        "Medical": ("Concentration", "time", "hours", "concentration", "mg/L", "10*exp(-0.3*x)", "A simplified concentration curve; it is not clinical guidance or a dosing recommendation."),
        "Sports science": ("Distance", "time", "seconds", "distance", "meters", "2*x - 0.1*x**2", "A simplified motion/performance model; real measurements and constraints matter."),
        "Biology": ("Population", "time", "days", "population", "organisms", "100*exp(0.08*x)", "A simplified population model; unbounded exponential growth is rarely realistic forever."),
        "Bioengineering": ("Cell growth", "time", "hours", "cell count", "cells", "50*exp(0.12*x)", "A simplified cell-growth curve; real systems have resource and measurement limits."),
        "Economic forecasting": ("Price index", "time", "years", "index", "index points", "100*exp(0.03*x)", "A hypothetical trend model, not a forecast guarantee."),
        "Weather": ("Temperature", "time", "hours", "temperature", "°C", "15 + 8*sin(2*pi*x/24)", "A periodic illustrative temperature cycle; not a live weather forecast."),
        "Custom": ("Custom quantity", "input", "units", "output", "units", "x**2", "Describe what the variables represent and the limits of the model.")
    }
    title_default, indep_default, indep_unit_default, dep_default, dep_unit_default, suggested, explanation = context_defaults[context]
    function_source = st.radio("Function source", ["Generated polynomial fit", "Suggested function", "Enter my own function"], horizontal=True)
    x = sp.Symbol("x", real=True)
    custom_description = st.text_input("Context description", value=explanation, key="context_description")
    col_a, col_b = st.columns(2)
    with col_a:
        indep_name = st.text_input("Independent variable", value=indep_default, key="indep_name")
        indep_unit = st.text_input("Independent-variable units", value=indep_unit_default, key="indep_unit")
    with col_b:
        dep_name = st.text_input("Dependent variable", value=dep_default, key="dep_name")
        dep_unit = st.text_input("Dependent-variable units", value=dep_unit_default, key="dep_unit")
    if function_source == "Generated polynomial fit":
        # Convert centered year polynomial to x-based expression where x is year.
        expr = sum(sp.Float(float(c), 14) * (x - model["base"])**i for i, c in enumerate(model["coef"]))
        expr_label = f"World Bank fitted polynomial: {equation_text(model, 't')}, t = year − {model['base']}"
        default_interval = (float(model["min_year"]), float(model["max_year"]))
    elif function_source == "Suggested function":
        raw_function = st.text_input("Suggested function (editable)", value=suggested, key=f"suggested_{context}")
        try:
            expr = safe_sympy_expression(raw_function, x)
            expr_label = str(sp.simplify(expr))
        except ValueError as exc:
            st.error(str(exc))
            expr = None
        default_interval = (0.0, 10.0)
    else:
        raw_function = st.text_input("Enter f(x)", value="x**2 - 3*x + 2", help="Use x**2 for powers. Supported: polynomial, rational, exponential, logarithmic, trigonometric, sqrt and abs.")
        try:
            expr = safe_sympy_expression(raw_function, x)
            expr_label = str(sp.simplify(expr))
        except ValueError as exc:
            st.error(str(exc))
            expr = None
        default_interval = (0.0, 10.0)
    if expr is not None:
        st.latex("f(x) = " + sp.latex(expr))
        # interval limits depend on data if fit mode; editable otherwise
        interval_low = float(default_interval[0])
        interval_high = float(default_interval[1])
        if interval_high <= interval_low:
            interval_high = interval_low + 1
        lo_col, hi_col = st.columns(2)
        with lo_col:
            interval_low = st.number_input("Analysis interval start", value=float(interval_low), key=f"interval_low_{context}_{function_source}")
        with hi_col:
            interval_high = st.number_input("Analysis interval end", value=float(interval_high), key=f"interval_high_{context}_{function_source}")
        if interval_high <= interval_low:
            st.error("Analysis interval end must be greater than its start.")
        else:
            d1 = sp.diff(expr, x)
            d2 = sp.diff(d1, x)
            st.markdown(f"**First derivative:**")
            st.latex("f'(x) = " + sp.latex(d1))
            st.markdown("**Second derivative:**")
            st.latex("f''(x) = " + sp.latex(d2))
            if course == "Honors Algebra 2":
                st.info("Algebra 2 focus: domain, range, end behavior, graphical trends, and average rates of change.")
            elif course == "Precalculus":
                st.info("Precalculus adds degree/coefficient effects, turning points, and concavity inferred from changing average rates.")
            else:
                st.info("Calculus focus: derivative-based rates, critical points, extrema, concavity, and inflection points. Numerical classifications are approximate.")
            st.write(f"**Interpretation:** {custom_description}")
            st.write(f"f(x) is {dep_name} ({dep_unit}); f′(x) is change in {dep_name} per {indep_name} ({dep_unit}/{indep_unit}); f″(x) is change in that rate ({dep_unit}/{indep_unit}²).")
            c1, c2, c3 = st.columns(3)
            current_x = c1.slider("Current input x", min_value=float(interval_low), max_value=float(interval_high), value=float((interval_low+interval_high)/2), key=f"current_x_{context}_{function_source}")
            a = c2.slider("Secant interval start a", min_value=float(interval_low), max_value=float(interval_high), value=float(interval_low), key=f"sec_a_{context}_{function_source}")
            b = c3.slider("Secant interval end b", min_value=float(interval_low), max_value=float(interval_high), value=float(interval_high), key=f"sec_b_{context}_{function_source}")
            fx = numeric_value(expr, x, current_x)
            dfx = numeric_value(d1, x, current_x)
            d2fx = numeric_value(d2, x, current_x)
            st.metric("Function value f(x)", f"{fmt(fx)} {dep_unit}")
            st.metric("Instantaneous rate f′(x)", f"{fmt(dfx)} {dep_unit}/{indep_unit}")
            st.metric("Rate changing f″(x)", f"{fmt(d2fx)} {dep_unit}/{indep_unit}²")
            fa, fb = numeric_value(expr, x, a), numeric_value(expr, x, b)
            if a != b and np.isfinite(fa) and np.isfinite(fb):
                avg_rate = (fb-fa)/(b-a)
                st.write(f"**Average rate of change** on [{a:g}, {b:g}]: ({fmt(fb)} − {fmt(fa)}) / ({b:g} − {a:g}) = **{fmt(avg_rate)} {dep_unit}/{indep_unit}**.")
            else:
                avg_rate = np.nan
                st.warning("Average rate unavailable: choose different endpoints with defined function values.")
            st.write(f"At x = {current_x:g}, tangent slope f′(x) = {fmt(dfx)}. The tangent line is shown when the derivative is defined.")
            show_f = st.checkbox("Show f(x)", value=True, key=f"show_f_{context}_{function_source}")
            show_d1 = st.checkbox("Show f′(x)", value=(course == "Calculus"), key=f"show_d1_{context}_{function_source}")
            show_d2 = st.checkbox("Show f″(x)", value=False, key=f"show_d2_{context}_{function_source}")
            xx = np.linspace(interval_low, interval_high, 900)
            yy = safe_curve(expr, x, xx)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            if show_f:
                ax2.plot(xx, yy, label="f(x)")
            if show_d1:
                ax2.plot(xx, safe_curve(d1, x, xx), label="f′(x)")
            if show_d2:
                ax2.plot(xx, safe_curve(d2, x, xx), label="f″(x)")
            if np.isfinite(fx):
                ax2.scatter([current_x], [fx], label="Current point", zorder=5)
                if np.isfinite(dfx):
                    tangent_x = np.linspace(max(interval_low, current_x-(interval_high-interval_low)*.18), min(interval_high, current_x+(interval_high-interval_low)*.18), 100)
                    ax2.plot(tangent_x, fx + dfx*(tangent_x-current_x), linestyle="--", label="Tangent line")
            if a != b and np.isfinite(fa) and np.isfinite(fb):
                sec_x = np.linspace(a, b, 100)
                ax2.plot(sec_x, fa + avg_rate*(sec_x-a), linestyle=":", label="Secant line")
            ax2.axvline(current_x, color="gray", alpha=.25)
            ax2.set_xlabel(f"{indep_name} ({indep_unit})")
            ax2.set_ylabel(f"{dep_name} / rate curves (units differ by derivative)")
            ax2.set_title(f"{context}: interactive function analysis")
            ax2.grid(True, alpha=.25)
            ax2.legend()
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            # Contextual simulation driven by the current function value, with explicit normalized scale.
            visual = st.selectbox("Contextual visual simulation", ["Walking person / prosthetic-leg concept", "Moving object or projectile", "Tank / concentration indicator", "Population / cell colony", "Thermometer / weather", "Business / economic indicator", "Simple graph point"], key=f"visual_{context}")
            st.caption("Simulation scale is normalized to the displayed function values over the selected interval; it is illustrative, not a physical or medical simulation.")
            finite_ys = yy[np.isfinite(yy)]
            if len(finite_ys):
                ymin, ymax = float(np.min(finite_ys)), float(np.max(finite_ys))
                fraction = .5 if ymax == ymin or not np.isfinite(fx) else float(np.clip((fx-ymin)/(ymax-ymin), 0, 1))
            else:
                fraction = 0.0
            progress = int(round(100*fraction))
            if visual == "Walking person / prosthetic-leg concept":
                st.markdown(f"🚶 **Illustrative gait indicator** · current normalized signal: {progress}%")
                st.progress(progress)
                st.caption("The graph value is mapped to a 0–100% visual indicator only; it does not model gait biomechanics or evaluate a prosthesis.")
            elif visual == "Moving object or projectile":
                st.markdown(f"⚽ **Moving object** · normalized vertical position: {progress}%")
                st.progress(progress)
            elif visual == "Tank / concentration indicator":
                st.markdown(f"🧪 **Tank / concentration level** · normalized indicator: {progress}%")
                st.progress(progress)
            elif visual == "Population / cell colony":
                st.markdown(f"🦠 **Population / cell-colony indicator** · normalized signal: {progress}%")
                st.progress(progress)
            elif visual == "Thermometer / weather":
                st.markdown(f"🌡️ **Weather indicator** · normalized temperature signal: {progress}%")
                st.progress(progress)
            elif visual == "Business / economic indicator":
                st.markdown(f"📈 **Business / economic indicator** · normalized signal: {progress}%")
                st.progress(progress)
            else:
                st.markdown(f"● Current graph-point level: {progress}%")
                st.progress(progress)
            play_col1, play_col2, play_col3 = st.columns(3)
            if "sim_x" not in st.session_state:
                st.session_state["sim_x"] = float(current_x)
            with play_col1:
                if st.button("▶ Play / step", key=f"play_{context}_{function_source}"):
                    st.session_state["sim_x"] = min(interval_high, float(st.session_state["sim_x"]) + max((interval_high-interval_low)/100, .01))
                    st.rerun()
            with play_col2:
                if st.button("⏸ Pause", key="pause_sim"):
                    st.session_state["sim_paused"] = True
                    st.caption("Paused. Use Play / step to advance manually.")
            with play_col3:
                if st.button("↺ Reset", key="reset_sim"):
                    st.session_state["sim_x"] = interval_low
                    st.rerun()
            speed = st.select_slider("Playback step size", options=["Slow", "Normal", "Fast"], value="Normal", key="sim_speed")
            st.caption("Playback is user-driven step playback (not an unattended timer), so the graph and metrics stay synchronized with Streamlit reruns. Change the Current input x slider to inspect any point.")
            # Derivative-based numerical analysis
            if course == "Calculus":
                st.subheader("Critical points, extrema, concavity, and domain")
                crit = near_zero_roots(d1, x, interval_low, interval_high)
                infl_candidates = near_zero_roots(d2, x, interval_low, interval_high)
                crit_rows = []
                for root in crit:
                    yroot = numeric_value(expr, x, root)
                    left = numeric_value(d1, x, root-1e-4)
                    right = numeric_value(d1, x, root+1e-4)
                    if np.isfinite(left) and np.isfinite(right) and left > 0 and right < 0:
                        classification = "local maximum (numerical derivative sign test)"
                    elif np.isfinite(left) and np.isfinite(right) and left < 0 and right > 0:
                        classification = "local minimum (numerical derivative sign test)"
                    else:
                        classification = "critical candidate; no extremum confirmed by sign change"
                    crit_rows.append({"x (approx.)": root, "f(x)": yroot, "classification": classification})
                if crit_rows:
                    st.dataframe(pd.DataFrame(crit_rows), use_container_width=True)
                else:
                    st.write("No critical points found in the interval by the available numerical method.")
                test_points = [interval_low] + crit + [interval_high]
                vals_ext = [(v, numeric_value(expr, x, v)) for v in test_points]
                vals_ext = [(v, yv) for v, yv in vals_ext if np.isfinite(yv)]
                if vals_ext:
                    st.write(f"**Absolute extrema candidates on the closed interval:** minimum at x≈{min(vals_ext, key=lambda z:z[1])[0]:.6g}, maximum at x≈{max(vals_ext, key=lambda z:z[1])[0]:.6g} (critical candidates plus endpoints).")
                infl_rows = []
                for root in infl_candidates:
                    left = numeric_value(d2, x, root-1e-4)
                    right = numeric_value(d2, x, root+1e-4)
                    changes = np.isfinite(left) and np.isfinite(right) and left*right < 0
                    if changes:
                        infl_rows.append({"x (approx.)": root, "f(x)": numeric_value(expr, x, root), "concavity changes?": "Yes (numerical sign check)"})
                if infl_rows:
                    st.dataframe(pd.DataFrame(infl_rows), use_container_width=True)
                else:
                    st.write("No verified inflection points found in the interval; zeros of f″ alone are not enough.")
                # Numerical monotonicity and concavity summaries
                sample_mid = (xx[:-1]+xx[1:])/2
                d1vals = safe_curve(d1, x, sample_mid)
                d2vals = safe_curve(d2, x, sample_mid)
                inc = np.isfinite(d1vals) & (d1vals > 1e-6)
                dec = np.isfinite(d1vals) & (d1vals < -1e-6)
                conc_up = np.isfinite(d2vals) & (d2vals > 1e-6)
                conc_down = np.isfinite(d2vals) & (d2vals < -1e-6)
                st.write(f"Numerically sampled interval: increasing at {inc.mean()*100:.1f}% of sampled subintervals; decreasing at {dec.mean()*100:.1f}%; concave up at {conc_up.mean()*100:.1f}%; concave down at {conc_down.mean()*100:.1f}%.")
            # Average rates, range, end behavior
            if course in ["Honors Algebra 2", "Precalculus"]:
                st.subheader("Function behavior")
                finite_vals = yy[np.isfinite(yy)]
                if len(finite_vals):
                    st.write(f"Approximate plotted range on [{interval_low:g}, {interval_high:g}]: [{fmt(np.min(finite_vals))}, {fmt(np.max(finite_vals))}] {dep_unit}.")
                if a != b and np.isfinite(avg_rate):
                    st.write(f"Average rate of change: {fmt(avg_rate)} {dep_unit}/{indep_unit}.")
                st.write(f"End behavior is function-specific. Over the selected finite interval, the graph shows the trend; for polynomial functions, the leading term determines long-run end behavior.")
            # discontinuity/domain caution
            try:
                domain = sp.calculus.util.continuous_domain(expr, x, sp.Interval(interval_low, interval_high))
                st.write(f"**Domain/continuity on selected interval:** {sp.sstr(domain)} (symbolic result where SymPy can determine it).")
            except Exception:
                st.write("**Domain/discontinuities:** inspect gaps in the graph; symbolic domain analysis was not available for this expression.")
            # Show observed points separately when data-driven model is selected
            if function_source == "Generated polynomial fit":
                st.caption("This function comes from a historical-data polynomial fit. It is descriptive, not causal; extrapolated values may be implausible.")

with tab_compare:
    st.subheader("Same-indicator country comparison")
    if not compare_names:
        st.write("Select one or more comparison countries in the sidebar.")
    else:
        figc, axc = plt.subplots(figsize=(10, 5))
        axc.plot(valid_df["year"], valid_df["value"], marker="o", markersize=3, label=f"{country_name} (working data)")
        compare_rows = []
        for name in compare_names:
            dfc, msgc, _ = fetch_world_bank(COUNTRIES[name], indicator_code, start_year, end_year)
            dfc = dfc.dropna(subset=["value"]) if not dfc.empty else dfc
            if dfc.empty:
                st.warning(f"{name}: {msgc}")
                continue
            dfc["year"] = dfc["year"].astype(int)
            dfc["value"] = dfc["value"].astype(float)
            axc.plot(dfc["year"], dfc["value"], marker=".", markersize=3, label=name)
            compare_rows.append({"country": name, "first year": int(dfc.year.min()), "last year": int(dfc.year.max()), "observations": len(dfc), "message": msgc})
        axc.set_xlabel("Year")
        axc.set_ylabel(f"{indicator_name} ({units})")
        axc.set_title(f"Country comparison: {indicator_name}")
        axc.grid(True, alpha=.25)
        axc.legend()
        figc.tight_layout()
        st.pyplot(figc, use_container_width=True)
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)
        st.caption("Countries can have different coverage, missing years, definitions, or statistical capacity. Compare like-for-like indicators and avoid treating correlation or differences as proof of a cause.")

# Export current selection/model/data/results, with actual CSV/XLSX/PDF formats.
with tab_export:
    st.subheader("Downloads")
    summary = pd.DataFrame([{
        "country": country_name, "country_code": COUNTRIES[country_name], "indicator": indicator_name,
        "indicator_code": indicator_code, "definition": definition, "units": units,
        "source": api_url, "retrieved_at": retrieved_at, "requested_start": start_year,
        "requested_end": end_year, "actual_start": model["min_year"], "actual_end": model["max_year"],
        "observations": len(valid_df), "degree": degree, "base_year": model["base"],
        "equation": equation_text(model, "t"), "in_sample_R2": model["r2"], "in_sample_MAE": model["mae"],
        "mean": model["mean"], "edited_user_scenario": edited_status, "course": course,
        "research_question": st.session_state.get("research_question", "")
    }])
    working_export = valid_df.copy()
    working_export["fitted"] = [poly_value(model, y) for y in working_export["year"]]
    working_export["residual"] = working_export["value"] - working_export["fitted"]
    result_export = st.session_state.get("batch_results", pd.DataFrame(columns=["year", "estimate", "units", "classification"]))
    if st.session_state.get("last_calculation") and st.session_state["last_calculation"].get("signature") == signature:
        single_y = st.session_state["last_calculation"]["single"]
        single_export = pd.DataFrame([{"year": single_y, "estimate": poly_value(model, single_y), "units": units,
                                       "classification": "within fitted historical range" if model["min_year"] <= single_y <= model["max_year"] else "outside fitted historical range (extrapolation)"}])
    else:
        single_export = pd.DataFrame(columns=["year", "estimate", "units", "classification"])
    combined_results = pd.concat([single_export.assign(result_type="single"), result_export.assign(result_type="batch")], ignore_index=True)
    csv_bytes = working_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download working data CSV", data=csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        working_export.to_excel(writer, sheet_name="Working Data", index=False)
        combined_results.to_excel(writer, sheet_name="Calculated Years", index=False)
        for sheet_name in ["Summary", "Working Data", "Calculated Years"]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, 20, 22)
    st.download_button("Download Excel workbook (.xlsx)", data=xlsx_buffer.getvalue(), file_name="hispanic_heritage_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    # PDF chart and report
    pdf_buffer = io.BytesIO()
    pdf = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
             Paragraph(f"{country_name} · {indicator_name} ({units})", styles["Heading2"]),
             Paragraph(f"Research question: {st.session_state.get('research_question') or 'Not entered'}", styles["BodyText"]),
             Paragraph(f"Source: World Bank API · Indicator {indicator_code} · Retrieved {retrieved_at}", styles["BodyText"]),
             Paragraph(f"Definition: {definition}", styles["BodyText"]),
             Paragraph(f"Coverage: {model['min_year']}–{model['max_year']} · n={len(valid_df)} · degree={degree} · base year={model['base']}", styles["BodyText"]),
             Paragraph(f"Equation: {equation_text(model, 't')}", styles["BodyText"]),
             Paragraph(f"In-sample R²={fmt(model['r2'])}; in-sample MAE={fmt(model['mae'])} {units}; mean={fmt(model['mean'])} {units}", styles["BodyText"]),
             Paragraph(f"Data status: {'USER-MODIFIED SCENARIO' if edited_status else 'source observations'}", styles["BodyText"]),
             Spacer(1, 8)]
    pdf_fig, pdf_ax = plt.subplots(figsize=(9, 3.4))
    pdf_ax.scatter(valid_df["year"], valid_df["value"], label="Observed")
    pdf_x = np.linspace(model["min_year"], model["max_year"] + horizon, 400)
    pdf_y = np.array([poly_value(model, yy) for yy in pdf_x])
    pdf_ax.plot(pdf_x[pdf_x <= model["max_year"]], pdf_y[pdf_x <= model["max_year"]], label="Fit")
    if horizon:
        pdf_ax.plot(pdf_x[pdf_x > model["max_year"]], pdf_y[pdf_x > model["max_year"]], "--", label="Extrapolation")
    pdf_ax.axvline(model["max_year"], color="gray", linestyle=":")
    pdf_ax.set_xlabel("Year"); pdf_ax.set_ylabel(f"{indicator_name} ({units})"); pdf_ax.legend(); pdf_ax.grid(alpha=.2)
    pdf_fig.tight_layout()
    chart_buffer = io.BytesIO()
    pdf_fig.savefig(chart_buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(pdf_fig)
    chart_buffer.seek(0)
    story.append(Image(chart_buffer, width=650, height=245))
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    story.append(Paragraph("The polynomial summarizes the selected observations; it does not establish causation. Fit metrics are in-sample. Extrapolation may be implausible, and residuals/MAE are not a validated 95% prediction interval. Country coverage and statistical definitions can differ.", styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Data preview (first 20 rows)", styles["Heading2"]))
    preview = [["Year", "Observed", "Fitted", "Residual"]] + [[str(int(r.year)), fmt(r.value), fmt(r.fitted), fmt(r.residual)] for r in working_export.head(20).itertuples()]
    tbl = Table(preview, repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9e8f5")),("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(tbl)
    if not combined_results.empty:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Calculated-year results", styles["Heading2"]))
        rows = [["Year", "Estimate", "Units", "Classification", "Type"]] + [[str(r.year), fmt(r.estimate), str(r.units), str(r.classification), str(r.result_type)] for r in combined_results.head(80).itertuples()]
        tbl2 = Table(rows, repeatRows=1)
        tbl2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9e8f5")),("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.append(tbl2)
    pdf.build(story)
    st.download_button("Download report PDF", data=pdf_buffer.getvalue(), file_name="hispanic_heritage_explorer_report.pdf", mime="application/pdf")
    st.subheader("Verification checklist")
    for item in [
        "Source is World Bank API; nulls are excluded and no random values are substituted.",
        "Working table can be edited and reset to original source observations.",
        "Duplicate/non-integer years and invalid values block fitting.",
        "Degree d requires at least d+1 distinct valid years.",
        "Equation uses t = year − base year; R² and MAE are labeled in-sample.",
        "Single and batch estimates use the same fitted polynomial transformation.",
        "CSV, XLSX, and PDF are generated in their real file formats.",
        "Outside-range predictions are labeled extrapolations; no 95% interval is claimed."
    ]:
        st.checkbox(item, value=False, key="verify_" + str(abs(hash(item))))

st.divider()
st.caption("Educational research tool. Verify data definitions and values at the linked World Bank source before drawing conclusions. Do not infer historical causes from curve shape alone.")
