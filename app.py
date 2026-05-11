from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="UBER NYC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import html
import json
import urllib.request

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Main app canvas (matches Streamlit theme backgroundColor)
MAIN_BG = "#0B0C15"
TEXT_SILVER = "#BFC1C9"
TEXT_MUTED = "#8d919c"

ACCENT = "#22c55e"
ACCENT_SOFT = "#4ade80"
DEEP = "#15803d"
ACCENT_FORECAST = "#5cb85c"
ACCENT_RULE = "#8d919c"

# Semantic colors for deltas / KPI direction (slightly darker than prior emerald)
GOOD_GREEN = "#16a34a"
BAD_RED = "#ef4444"

CHART_BG = MAIN_BG
GRID_D = "#34354a"
AXIS_D = "#707484"
LABEL_D = TEXT_SILVER

LAG_DAYS = (1, 7, 14)
FEATURE_COLS = [f"lag_{lag}" for lag in LAG_DAYS] + ["dow", "month"]

# Cap how many days ahead we roll the model (horizon = length of selected date range)
MAX_USER_FORECAST_HORIZON = 366

def _data_bounds(df_in: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(df_in["date"].min()), pd.Timestamp(df_in["date"].max())


def date_range_from_preset(
    preset: str,
    d_min: pd.Timestamp,
    d_max: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Anchor relative ranges on the latest day in the dataset (GM-style 'as of' end)."""
    end = pd.Timestamp(d_max).normalize()
    start = end
    if preset == "Today":
        start = end
    elif preset == "Yesterday":
        start = end - pd.Timedelta(days=1)
        end = start
    elif preset == "Past 7 Days":
        start = end - pd.Timedelta(days=6)
    elif preset == "Past 14 Days":
        start = end - pd.Timedelta(days=13)
    elif preset == "Past 30 Days":
        start = end - pd.Timedelta(days=29)
    else:
        start = pd.Timestamp(d_min).normalize()
        end = pd.Timestamp(d_max).normalize()
    if start > end:
        start, end = end, start
    start = max(start, pd.Timestamp(d_min).normalize())
    end = min(end, pd.Timestamp(d_max).normalize())
    return start, end


def kpi_metric_html(label: str, value: str, *, tone: str | None) -> str:
    """KPI tile: bordered box, ▲/▼ + value color by tone."""
    if tone == "good":
        val_c = GOOD_GREEN
        tri = f'<span style="font-size:0.75rem;line-height:1;margin-right:0.18rem;color:{GOOD_GREEN};">▲</span>'
    elif tone == "bad":
        val_c = BAD_RED
        tri = f'<span style="font-size:0.75rem;line-height:1;margin-right:0.18rem;color:{BAD_RED};">▼</span>'
    else:
        val_c = TEXT_SILVER
        tri = '<span style="display:inline-block;width:0.75rem;margin-right:0.18rem;"></span>'
    return (
        f'<div class="kpi-tile-html" style="height:3.55rem;min-height:3.55rem;padding:0.35rem 0.5rem;'
        f"margin:0;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;"
        f"border:1px solid {GRID_D};border-radius:8px;background:{MAIN_BG};"
        f'">'
        f'<div style="color:#8d919c;font-size:0.55rem;line-height:1.2;">{html.escape(label)}</div>'
        f'<div style="display:flex;align-items:center;color:{val_c};font-size:0.82rem;font-weight:600;line-height:1.2;">'
        f"{tri}<span>{html.escape(value)}</span></div>"
        f"</div>"
    )


def top_zones_table_html(
    zdf: pd.DataFrame,
    *,
    height_px: int,
) -> str:
    """Scrollable HTML table: zone, borough, total rides (filter window), WoW %."""
    th = (
        '<thead><tr style="color:#8d919c;font-size:0.58rem;text-align:left;">'
        "<th style='padding:0.28rem 0.35rem;border:none;'>Zone</th>"
        "<th style='padding:0.28rem 0.35rem;border:none;'>Borough</th>"
        "<th style='padding:0.28rem 0.35rem;border:none;text-align:right;'>Total rides</th>"
        "<th style='padding:0.28rem 0.35rem;border:none;text-align:right;'>WoW %</th>"
        "</tr></thead>"
    )

    rows: list[str] = []
    for _, r in zdf.iterrows():
        z = html.escape(str(r.get("zone", "")))
        b = html.escape(str(r.get("borough", "")))
        tot = float(r.get("rides_total", 0) or 0)
        wow = r.get("wow_pct", np.nan)
        if pd.notna(wow):
            w = float(wow)
            if w > 0:
                w_c, w_a = GOOD_GREEN, "▲ "
            elif w < 0:
                w_c, w_a = BAD_RED, "▼ "
            else:
                w_c, w_a = TEXT_SILVER, ""
            w_s = f"{w:+.1f}%"
        else:
            w_c, w_a, w_s = TEXT_SILVER, "", "—"
        tds = (
            f"<td style='padding:0.28rem 0.35rem;border:none;color:#BFC1C9;'>{z}</td>"
            f"<td style='padding:0.28rem 0.35rem;border:none;color:#8d919c;'>{b}</td>"
            f"<td style='padding:0.28rem 0.35rem;border:none;text-align:right;font-weight:600;"
            f"color:#BFC1C9;'>{tot:,.0f}</td>"
            f"<td style='padding:0.28rem 0.35rem;border:none;text-align:right;font-weight:600;"
            f"color:{w_c};'>{w_a}{html.escape(w_s)}</td>"
        )
        rows.append(f"<tr>{tds}</tr>")

    body = "<tbody>" + "".join(rows) + "</tbody>" if rows else "<tbody></tbody>"
    return (
        f'<div style="max-height:{height_px}px;overflow:auto;border:none;border-radius:0;'
        f"background:{MAIN_BG};font-family:system-ui,sans-serif;\">"
        f'<table style="width:100%;border-collapse:collapse;border:none;font-size:0.72rem;">{th}{body}</table></div>'
    )


def _chart_title(text: str) -> alt.TitleParams:
    """Match section headers (Map, Table info — …): muted, semibold, uppercase."""
    return alt.TitleParams(
        text=text.upper(),
        color=TEXT_MUTED,
        fontSize=11,
        fontWeight=600,
        anchor="start",
        offset=1,
    )


def _finalize_dark(chart: alt.Chart) -> alt.Chart:
    """Charts on main canvas (no card border); padding so titles and axes are not clipped."""
    inset = {"left": 10, "right": 10, "top": 6, "bottom": 22}
    return (
        chart.properties(padding=inset)
        .configure_view(strokeWidth=0, fill=CHART_BG)
        .configure(background=CHART_BG)
        .configure_title(
            color=TEXT_MUTED,
            fontSize=11,
            fontWeight=600,
            anchor="start",
            offset=1,
        )
        .configure_axis(
            gridColor=GRID_D,
            domainColor=AXIS_D,
            labelColor=LABEL_D,
            titleColor=LABEL_D,
            labelPadding=4,
            titlePadding=6,
        )
        .configure_legend(
            labelColor=LABEL_D,
            titleColor=LABEL_D,
        )
    )


# Approximate centroids for borough-level map (source data has no lat/lon per row)
BOROUGH_LAT_LON: dict[str, tuple[float, float]] = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "EWR": (40.6895, -74.1745),
    "Unknown": (40.7128, -74.0060),
}


def borough_map_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    g = filtered_df.groupby("borough", as_index=False)["n"].sum()
    g = g.rename(columns={"n": "rides"})
    lat, lon = [], []
    for b in g["borough"]:
        ll = BOROUGH_LAT_LON.get(str(b), (40.7128, -74.0060))
        lat.append(ll[0])
        lon.append(ll[1])
    g["lat"] = lat
    g["lon"] = lon
    # Circle radius in meters (scaled for visibility)
    mx = float(g["rides"].max()) if len(g) else 1.0
    g["size"] = (np.sqrt(g["rides"] / max(mx, 1.0)) * 1800 + 400).astype(int)
    return g


@st.cache_data(show_spinner="Loading borough boundaries…")
def nyc_borough_boundaries_geojson() -> dict:
    url = (
        "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/"
        "NYC_Borough_Boundary/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
    )
    with urllib.request.urlopen(url, timeout=90) as resp:
        return json.loads(resp.read().decode())


def merge_metrics_into_borough_geojson(
    geojson: dict,
    rides_by_borough: pd.DataFrame,
    filtered: pd.DataFrame,
) -> dict:
    gj = json.loads(json.dumps(geojson))
    rrides = rides_by_borough.set_index("borough")["rides"].to_dict()
    nrows = filtered.groupby("borough").size().to_dict()
    total = float(sum(rrides.values())) or 1.0
    for feat in gj["features"]:
        prop = feat["properties"]
        name = str(prop["BoroName"])
        rides = int(rrides.get(name, 0))
        prop["rides"] = rides
        prop["pct_of_filtered"] = round(100.0 * rides / total, 2)
        prop["record_count"] = int(nrows.get(name, 0))
    return gj


def render_borough_zones_map(
    filtered: pd.DataFrame,
    rides_by_borough: pd.DataFrame,
    *,
    map_height: int = 240,
) -> None:
    """Basemap (light, dimmed in CSS) + borough outlines + circles (hover = metrics)."""
    try:
        gj_raw = nyc_borough_boundaries_geojson()
        gj = merge_metrics_into_borough_geojson(gj_raw, rides_by_borough, filtered)
    except Exception as exc:
        st.warning(f"Could not load borough boundaries ({exc}). Using point map.")
        map_df = borough_map_df(filtered)
        if hasattr(st, "map_dataframe"):
            try:
                st.map_dataframe(
                    map_df,
                    latitude="lat",
                    longitude="lon",
                    size="size",
                    zoom=9,
                    use_container_width=True,
                )
            except TypeError:
                st.map_dataframe(
                    map_df,
                    latitude="lat",
                    longitude="lon",
                    zoom=9,
                    use_container_width=True,
                )
        else:
            st.map(map_df[["lat", "lon"]])
        return

    # Green on basemap: soft borough tint + readable outlines
    fr, fg, fb = 34, 197, 94
    fill_rgb = [fr, fg, fb, 58]
    line_rgb = [55, 58, 72, 235]
    layer_geo = pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color=fill_rgb,
        get_line_color=line_rgb,
        line_width_min_pixels=1.5,
        pickable=False,
        auto_highlight=False,
    )

    trip_counts = filtered.groupby("borough").size().rename("record_count").reset_index()
    merged = rides_by_borough.merge(trip_counts, on="borough", how="left")
    merged["record_count"] = merged["record_count"].fillna(0).astype(int)
    total_r = float(merged["rides"].sum()) or 1.0
    mx = float(merged["rides"].max()) or 1.0
    merged["pct_of_filtered"] = (merged["rides"] / total_r * 100.0).round(2)
    if len(filtered):
        dmin = pd.to_datetime(filtered["date"]).min()
        dmax = pd.to_datetime(filtered["date"]).max()
        n_days = max(1, int((dmax - dmin).days) + 1)
    else:
        n_days = 1
    merged["avg_daily"] = (merged["rides"].astype(float) / n_days).round(1)

    circle_rows: list[dict] = []
    for _, row in merged.iterrows():
        b = str(row["borough"])
        rides = int(row["rides"])
        ll = BOROUGH_LAT_LON.get(b, (40.7128, -74.0060))
        rad = float(np.sqrt(rides / mx) * 9500.0 + 2200.0)
        circle_rows.append(
            {
                "BoroName": b,
                "rides": rides,
                "lat": ll[0],
                "lon": ll[1],
                "record_count": int(row["record_count"]),
                "pct_of_filtered": float(row["pct_of_filtered"]),
                "avg_daily": float(row["avg_daily"]),
                "size": rad,
            }
        )
    circles_df = pd.DataFrame(circle_rows)

    layer_circles = pdk.Layer(
        "ScatterplotLayer",
        data=circles_df,
        stroked=False,
        get_position=["lon", "lat"],
        get_radius="size",
        get_fill_color=[fr, fg, fb, 200],
        pickable=True,
    )
    layers = [layer_geo, layer_circles]

    view = pdk.ViewState(latitude=40.73, longitude=-73.95, zoom=9.1, pitch=0)
    tip = {
        "html": (
            "<div style=\"background:#ffffff;color:#0f172a;padding:10px 12px;border-radius:8px;"
            "border:1px solid #cbd5e1;font-family:system-ui,sans-serif;line-height:1.55;"
            "box-shadow:0 4px 14px rgba(15,23,42,0.18);\">"
            "<b style=\"color:#151624;font-size:14px;\">{BoroName}</b><br/>"
            "<span style=\"color:#5c6470;\">Total rides</span> "
            "<b style=\"color:#166534;\">{rides}</b><br/>"
            "<span style=\"color:#5c6470;\">Share of filtered total</span> "
            "<b style=\"color:#151624;\">{pct_of_filtered}%</b><br/>"
            "<span style=\"color:#5c6470;\">Trip records</span> "
            "<b style=\"color:#151624;\">{record_count}</b><br/>"
            "<span style=\"color:#5c6470;\">Avg rides / day</span> "
            "<b style=\"color:#151624;\">{avg_daily}</b>"
            "</div>"
        ),
        "style": {
            "backgroundColor": "#ffffff",
            "color": "#0f172a",
        },
    }
    map_style = getattr(pdk.map_styles, "LIGHT", None) or "mapbox://styles/mapbox/light-v10"
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=map_style,
        tooltip=tip,
    )
    st.pydeck_chart(deck, use_container_width=True, height=map_height)


def hour_ampm(h: int) -> str:
    h = int(h) % 24
    if h == 0:
        return "12 AM"
    if h < 12:
        return f"{h} AM"
    if h == 12:
        return "12 PM"
    return f"{h - 12} PM"


HOUR_LABELS = [hour_ampm(h) for h in range(24)]


def label_to_hour(label: str) -> int:
    return HOUR_LABELS.index(label)


def top_zones_with_wow(
    filt: pd.DataFrame,
    *,
    last_dates: set[pd.Timestamp],
    prev_dates: set[pd.Timestamp],
    top_n: int = 15,
) -> pd.DataFrame:
    """Zones ranked by rides in the latest 7-day window when available; WoW vs prior 7 days."""
    if not last_dates or not prev_dates:
        out = (
            filt.groupby(["zone", "borough"], as_index=False)["n"]
            .sum()
            .rename(columns={"n": "rides_rank"})
            .sort_values("rides_rank", ascending=False)
            .head(top_n)
            .copy()
        )
        out["rides_prior_7d"] = np.nan
        out["wow_pct"] = np.nan
        out["l7d_change"] = np.nan
        return out.rename(columns={"rides_rank": "rides_last_7d"})
    z_last = (
        filt[filt["date"].isin(last_dates)]
        .groupby(["zone", "borough"], as_index=False)["n"]
        .sum()
        .rename(columns={"n": "rides_last7"})
    )
    z_prev = (
        filt[filt["date"].isin(prev_dates)]
        .groupby(["zone", "borough"], as_index=False)["n"]
        .sum()
        .rename(columns={"n": "rides_prev7"})
    )
    merged = z_last.merge(z_prev, on=["zone", "borough"], how="outer").fillna(0.0)
    merged["wow_pct"] = np.where(
        merged["rides_prev7"] > 0,
        (merged["rides_last7"] - merged["rides_prev7"]) / merged["rides_prev7"] * 100.0,
        np.nan,
    )
    merged["l7d_change"] = merged["rides_last7"] - merged["rides_prev7"]
    merged = merged.sort_values("rides_last7", ascending=False).head(top_n)
    return merged.rename(
        columns={"rides_last7": "rides_last_7d", "rides_prev7": "rides_prior_7d"}
    )


@st.cache_data
def load_data():
    path = Path(__file__).resolve().parent / "location_features.csv.zip"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def prepare_daily_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values("date").drop_duplicates("date", keep="last")
    if daily.empty:
        return daily
    idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    s = daily.set_index("date")["n"].astype(float).reindex(idx, fill_value=0.0)
    out = s.reset_index()
    out.columns = ["date", "n"]
    return out


def daily_window_series(
    filtered_df: pd.DataFrame,
    start_d,
    end_d,
) -> pd.DataFrame:
    """Dense daily totals for [start_d, end_d] inclusive (zeros for missing days)."""
    idx = pd.date_range(pd.Timestamp(start_d).normalize(), pd.Timestamp(end_d).normalize(), freq="D")
    g = filtered_df.groupby("date", as_index=False)["n"].sum()
    if g.empty:
        return pd.DataFrame({"date": idx, "n": 0.0})
    s = g.set_index("date")["n"].astype(float).reindex(idx, fill_value=0.0).reset_index()
    s.columns = ["date", "n"]
    return s


def _volume_tone(cur: float, prev: float | None) -> str | None:
    if prev is None:
        return None
    if prev == 0.0:
        return "good" if cur > 0 else None
    if cur > prev:
        return "good"
    if cur < prev:
        return "bad"
    return None


def _date_axis_mmdd(label_angle: int = -38) -> alt.Axis:
    """Temporal x-axis labels as mm/dd (no weekday)."""
    return alt.Axis(format="%m/%d", labelAngle=label_angle, labelOverlap=True)


def add_lag_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for lag in LAG_DAYS:
        d[f"lag_{lag}"] = d["n"].shift(lag)
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    return d


def build_training_matrix(prepared: pd.DataFrame) -> pd.DataFrame:
    d = add_lag_calendar(prepared)
    return d.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def fit_best_forecaster(_prep_key: tuple) -> dict | None:
    """Ridge regression on lag + calendar features (scaled); CV R² from TimeSeriesSplit."""
    if not _prep_key:
        return None
    prepared = pd.DataFrame(_prep_key, columns=["date_str", "n"])
    prepared["date"] = pd.to_datetime(prepared["date_str"])
    prepared = prepared[["date", "n"]].sort_values("date").reset_index(drop=True)

    train_df = build_training_matrix(prepared)
    n_samples = len(train_df)
    if n_samples < 28:
        return None

    X = train_df[FEATURE_COLS]
    y = train_df["n"]
    n_splits = max(2, min(5, max(2, n_samples // 14)))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ridge",
                Ridge(alpha=8.0, random_state=0),
            ),
        ]
    )
    cv_scores = cross_val_score(
        pipe, X, y, cv=tscv, scoring="r2", n_jobs=-1
    )
    mean_r2 = float(np.nanmean(cv_scores))
    pipe.fit(X, y)
    return {
        "model_name": "RidgeRegression",
        "estimator": pipe,
        "cv_r2": mean_r2,
        "cv_splits": n_splits,
        "runner_up": [],
    }


def recursive_forecast(
    estimator,
    prepared: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """One-day-ahead model rolled forward with lags mixing history and predictions."""
    extended = prepared.copy().reset_index(drop=True)
    rows = []
    for _ in range(horizon):
        next_date = extended["date"].iloc[-1] + pd.Timedelta(days=1)
        vals = extended["n"].values
        if len(vals) < max(LAG_DAYS):
            break
        row = pd.DataFrame(
            [
                {
                    "dow": next_date.dayofweek,
                    "month": next_date.month,
                    "lag_1": vals[-1],
                    "lag_7": vals[-7],
                    "lag_14": vals[-14],
                }
            ]
        )[FEATURE_COLS]
        y_hat = max(0.0, float(estimator.predict(row)[0]))
        extended = pd.concat(
            [extended, pd.DataFrame({"date": [next_date], "n": [y_hat]})],
            ignore_index=True,
        )
        rows.append({"date": next_date, "n": y_hat})
    fc = pd.DataFrame(rows)
    cutoff = prepared["date"].max()
    return fc, cutoff


def prepared_fingerprint(prepared: pd.DataFrame) -> tuple:
    p = prepared.sort_values("date")
    return tuple(zip(p["date"].dt.strftime("%Y-%m-%d"), p["n"].astype(float)))


def train_forecast_bundle(_prep_key: tuple) -> dict:
    """Fit Ridge forecaster on full prepared daily series (caller slices chart & roll-forward)."""
    if not _prep_key:
        return {"ok": False, "reason": "empty", "prepared_full": None, "fit": None}

    prepared = pd.DataFrame(_prep_key, columns=["date_str", "n"])
    prepared["date"] = pd.to_datetime(prepared["date_str"])
    prepared = prepared[["date", "n"]].sort_values("date").reset_index(drop=True)

    fit = fit_best_forecaster(_prep_key)
    if fit is None:
        return {
            "ok": False,
            "reason": "insufficient_history",
            "prepared_full": prepared,
            "fit": None,
        }

    return {
        "ok": True,
        "prepared_full": prepared,
        "fit": fit,
    }


st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { display: none; }
    [data-testid="stAppViewContainer"] .main { margin-left: 0; }
    .block-container {
      padding-top: 1.1rem !important;
      padding-bottom: 1.35rem !important;
      padding-left: 2.85rem !important;
      padding-right: 2.85rem !important;
      max-width: 100%;
    }
    div[data-testid="stVerticalBlock"] > div { gap: 0.12rem !important; }
    h1 { font-size: 0.92rem !important; font-weight: 600 !important; line-height: 1.05 !important;
         margin: 0 0 0 !important; padding: 0 !important; color: #e8eaef !important; }
    h2 { border: none !important; box-shadow: none !important; }
    h3 { font-size: 0.95rem !important; color: #BFC1C9 !important; margin-bottom: 0.15rem !important;
         border: none !important; box-shadow: none !important; }
    h4 { font-size: 0.82rem !important; color: #8d919c !important; margin-bottom: 0.12rem !important; }
    div[data-testid="stCaption"] {
      margin-top: 0 !important; margin-bottom: 0.02rem !important;
      padding-top: 0 !important; padding-bottom: 0 !important; line-height: 1.15 !important;
      color: #8d919c !important;
    }
    div[data-testid="stDataFrame"] > div {
      border: none !important;
      border-radius: 0 !important;
      background: #0B0C15 !important;
      box-shadow: none !important;
    }
    div[data-testid="stMetric"] {
      border-radius: 8px !important;
      padding: 0.35rem 0.5rem !important;
      border: 1px solid #34354a !important;
      background: #0B0C15 !important;
      box-shadow: none !important;
      height: 3.55rem !important;
      min-height: 3.55rem !important;
      max-height: 3.55rem !important;
      box-sizing: border-box !important;
      display: flex !important;
      flex-direction: column !important;
      justify-content: center !important;
    }
    .kpi-tile-html {
      height: 3.55rem !important;
      min-height: 3.55rem !important;
      max-height: 3.55rem !important;
    }
    div[data-testid="stMetric"] label { color: #8d919c !important; font-size: 0.55rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
      color: #BFC1C9 !important; font-size: 0.82rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
      font-size: 0.65rem !important; color: #22c55e !important;
    }
    .stDateInput label, .stSelectbox label, .stMultiSelect label, div[data-testid="stSelectSlider"] > label {
      font-size: 0.6rem !important; margin-bottom: 0.02rem !important;
    }
    /* Filter row: pull date / multiselect / slider closer */
    div[data-testid="stHorizontalBlock"] {
      margin-top: 0 !important; margin-bottom: 0 !important;
      padding-top: 0 !important; padding-bottom: 0 !important;
      gap: 0.2rem !important;
    }
    div[data-testid="stSelectSlider"] {
      margin-top: -0.25rem !important; margin-bottom: -0.45rem !important;
      padding-top: 0 !important; padding-bottom: 0 !important;
    }
    div[data-testid="stAltairChart"],
    div[data-testid="stAltairChart"] > div {
      border: none !important;
      box-shadow: none !important;
      background: transparent !important;
    }
    /* Pull chart blocks slightly closer (left column is all Altair; map/table use other widgets) */
    div[data-testid="stAltairChart"] {
      margin-top: -0.14rem !important;
      margin-bottom: -0.2rem !important;
    }
    /* Deck map: dim basemap slightly (CSS filter on light Mapbox tiles) */
    iframe[title="streamlit_deck_gl.streamlit_deck_gl"],
    div[data-testid="stDeckGlJsonChart"] {
      margin-top: 0 !important;
      margin-bottom: 0 !important;
    }
    div[data-testid="stDeckGlJsonChart"] {
      filter: brightness(0.8) contrast(1.06) saturate(0.93);
      border-radius: 4px;
      overflow: hidden;
    }
    .stDateInput [data-baseweb="input"] { min-height: 1.65rem !important; font-size: 0.78rem !important; }
    .stMultiSelect [data-baseweb="select"] > div {
      min-height: 1.22rem !important; max-height: 4.2rem !important; font-size: 0.68rem !important;
      overflow-y: auto !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
      background: #22c55e !important;
      color: #0B0C15 !important; font-size: 0.62rem !important;
      padding: 0.02rem 0.22rem !important; min-height: 1.05rem !important; line-height: 1.1 !important;
    }
    div[data-testid="stSlider"] [role="slider"],
    div[data-testid="stSelectSlider"] [role="slider"] {
      background: linear-gradient(180deg, #5cb85c 0%, #22c55e 45%, #15803d 100%) !important;
      border: 1px solid #0B0C15 !important;
    }
    div[data-testid="column"] .stDateInput, div[data-testid="column"] .stSelectbox,
    div[data-testid="column"] .stMultiSelect {
      margin-top: 0 !important; margin-bottom: -0.35rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_data()

# KPIs: fixed from full dataset (do not change with filters)
_daily_kpi = df.groupby("date", as_index=False)["n"].sum().sort_values("date").reset_index(drop=True)
total_rides_static = int(df["n"].sum())
avg_daily_rides_static = float(_daily_kpi["n"].mean()) if len(_daily_kpi) else 0.0
if len(_daily_kpi) >= 7:
    l7d_rides_static = float(_daily_kpi.tail(7)["n"].sum())
else:
    l7d_rides_static = float(_daily_kpi["n"].sum()) if len(_daily_kpi) else 0.0
prev7_rides_static: float | None = None
wow_pct_static: float | None = None
if len(_daily_kpi) >= 14:
    prev7_rides_static = float(_daily_kpi.iloc[-14:-7]["n"].sum())
    if prev7_rides_static > 0:
        wow_pct_static = (l7d_rides_static - prev7_rides_static) / prev7_rides_static * 100.0
l7d_tone_static = _volume_tone(l7d_rides_static, prev7_rides_static)
if wow_pct_static is None:
    wow_tone_static: str | None = None
elif wow_pct_static > 0:
    wow_tone_static = "good"
elif wow_pct_static < 0:
    wow_tone_static = "bad"
else:
    wow_tone_static = None

borough_options = sorted(df["borough"].unique())

d_min_ts, d_max_ts = _data_bounds(df)
d_min_d = d_min_ts.date()
d_max_d = d_max_ts.date()
_date_presets = [
    "Today",
    "Yesterday",
    "Past 7 Days",
    "Past 14 Days",
    "Past 30 Days",
    "Custom",
]
_borough_pick_options = ["All Boroughs"] + list(borough_options) + ["Custom"]

# --- Filter row: prominent brand + filters ---
_brand, _fc1, _fc2, _fc3 = st.columns([1.35, 1.0, 1.0, 1.95], gap="small")
with _brand:
    st.markdown(
        f'<p style="margin:0;padding:0.2rem 0 0 0;line-height:1;font-size:2.35rem;font-weight:800;'
        f"color:#e8eaef;letter-spacing:0.03em;text-transform:uppercase;\">UBER NYC</p>",
        unsafe_allow_html=True,
    )
with _fc1:
    date_preset = st.selectbox(
        "Date range",
        _date_presets,
        index=2,
        key="flt_date_preset",
    )
with _fc2:
    borough_pick = st.selectbox(
        "Borough",
        _borough_pick_options,
        index=0,
        key="flt_borough_pick",
    )
with _fc3:
    hour_sel = st.select_slider(
        "Hour range",
        options=HOUR_LABELS,
        value=(HOUR_LABELS[0], HOUR_LABELS[-1]),
        label_visibility="visible",
        key="flt_hours",
    )

if date_preset == "Custom" or borough_pick == "Custom":
    _ex_a, _ex_b = st.columns(2, gap="medium")
    with _ex_a:
        if date_preset == "Custom":
            st.markdown("**Custom dates**")
            _sd_c, _ed_c = st.columns(2, gap="small")
            with _sd_c:
                start_date = st.date_input(
                    "Start",
                    d_min_d,
                    min_value=d_min_d,
                    max_value=d_max_d,
                    key="flt_start_custom",
                )
            with _ed_c:
                end_date = st.date_input(
                    "End",
                    d_max_d,
                    min_value=d_min_d,
                    max_value=d_max_d,
                    key="flt_end_custom",
                )
        else:
            _rs, _re = date_range_from_preset(date_preset, d_min_ts, d_max_ts)
            start_date = _rs.date()
            end_date = _re.date()
    with _ex_b:
        if borough_pick == "Custom":
            st.markdown("**Custom boroughs**")
            boroughs = st.multiselect(
                "Choose one or more",
                borough_options,
                default=borough_options,
                key="flt_boroughs_custom",
                label_visibility="collapsed",
            )
        elif borough_pick == "All Boroughs":
            boroughs = list(borough_options)
        else:
            boroughs = [borough_pick]
else:
    _rs, _re = date_range_from_preset(date_preset, d_min_ts, d_max_ts)
    start_date = _rs.date()
    end_date = _re.date()
    if borough_pick == "All Boroughs":
        boroughs = list(borough_options)
    else:
        boroughs = [borough_pick]

if start_date > end_date:
    st.error("Start date is after end date.")
    st.stop()
_ha = label_to_hour(hour_sel[0])
_hb = label_to_hour(hour_sel[1])
h0, h1 = min(_ha, _hb), max(_ha, _hb)

filtered = df[
    (df["date"] >= pd.to_datetime(start_date))
    & (df["date"] <= pd.to_datetime(end_date))
    & (df["borough"].isin(boroughs))
    & (df["hour"] >= h0)
    & (df["hour"] <= h1)
]

if filtered.empty:
    st.warning(
        "No rows match the current filters. Pick at least one borough, "
        "widen the date range, or adjust the hour range."
    )
    st.stop()

daily = filtered.groupby("date", as_index=False)["n"].sum()

# Full date range (borough + hour only) — used to train the forecaster
filtered_all_time = df[
    (df["borough"].isin(boroughs))
    & (df["hour"] >= h0)
    & (df["hour"] <= h1)
]
daily_all = filtered_all_time.groupby("date", as_index=False)["n"].sum()
prepared_all = prepare_daily_calendar(daily_all)

rides_by_borough = filtered.groupby("borough", as_index=False)["n"].sum().rename(
    columns={"n": "rides"}
)

hourly = filtered.groupby("hour", as_index=False)["n"].sum()
hourly["hour_label"] = hourly["hour"].map(hour_ampm)
_hour_grid = pd.DataFrame({"hour": np.arange(h0, h1 + 1, dtype=int)})
hourly_plot = _hour_grid.merge(hourly, on="hour", how="left")
hourly_plot["n"] = hourly_plot["n"].fillna(0.0)
hourly_plot["hour_label"] = hourly_plot["hour"].map(hour_ampm)

dow_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
by_dow = filtered.groupby("day_of_week", observed=False)["n"].sum()
by_dow = by_dow.reindex(dow_order, fill_value=0.0)
by_dow_df = by_dow.reset_index()
by_dow_df.columns = ["day_of_week", "n"]
by_dow_df["l7d_pct_change"] = np.nan
by_dow_df["chg_lbl"] = ""

prepared = prepare_daily_calendar(daily)
_hist_plot_df = daily_window_series(filtered, start_date, end_date)

# WoW windows: current = selected date range; prior = same-length block immediately before it.
# Uses full df (borough + hour) for the prior block so "Past 7 days" still has a comparison week.
_ws = pd.Timestamp(start_date).normalize()
_we = pd.Timestamp(end_date).normalize()
_period_days = int((_we - _ws).days) + 1
_prev_end = _ws - pd.Timedelta(days=1)
_prev_start = _ws - pd.Timedelta(days=_period_days)
_prev_start = max(_prev_start, pd.Timestamp(d_min_ts).normalize())
wow_cur_date_set = set(pd.date_range(_ws, _we, freq="D"))
wow_prev_date_set = (
    set(pd.date_range(_prev_start, _prev_end, freq="D"))
    if _prev_end >= _prev_start
    else set()
)
filtered_for_wow = df[
    (df["date"] >= _prev_start)
    & (df["date"] <= _we)
    & (df["borough"].isin(boroughs))
    & (df["hour"] >= h0)
    & (df["hour"] <= h1)
]

# L7D % change by weekday — labels on bar chart (same WoW windows)
if wow_cur_date_set and wow_prev_date_set:
    _dow_l7 = filtered_for_wow[filtered_for_wow["date"].isin(wow_cur_date_set)]
    _dow_p7 = filtered_for_wow[filtered_for_wow["date"].isin(wow_prev_date_set)]
    _n_l = _dow_l7.groupby("day_of_week", observed=False)["n"].sum().reindex(dow_order, fill_value=0.0)
    _n_p = _dow_p7.groupby("day_of_week", observed=False)["n"].sum().reindex(dow_order, fill_value=0.0)
    _pv = _n_p.to_numpy(dtype=float)
    _lv = _n_l.to_numpy(dtype=float)
    _pct = np.where(_pv > 0.0, (_lv - _pv) / _pv * 100.0, np.nan)
    by_dow_df["l7d_pct_change"] = _pct
    by_dow_df["chg_lbl"] = [
        f"{v:+.1f}%" if v == v and np.isfinite(v) else ""
        for v in _pct
    ]

zones_top_df = top_zones_with_wow(
    filtered_for_wow,
    last_dates=wow_cur_date_set,
    prev_dates=wow_prev_date_set,
    top_n=15,
)
_ztot = filtered.groupby(["zone", "borough"], as_index=False)["n"].sum().rename(
    columns={"n": "rides_total"}
)
zones_top_df = zones_top_df.merge(_ztot, on=["zone", "borough"], how="left")
zones_top_df["rides_total"] = zones_top_df["rides_total"].fillna(0.0)

m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    st.metric("Total rides", f"{total_rides_static:,}")
with m2:
    st.metric("Avg daily rides", f"{avg_daily_rides_static:,.0f}")
with m3:
    st.markdown(
        kpi_metric_html(
            "WoW % change",
            f"{wow_pct_static:+.1f}%" if wow_pct_static is not None else "—",
            tone=wow_tone_static,
        ),
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        kpi_metric_html(
            "L7D rides",
            f"{l7d_rides_static:,.0f}",
            tone=l7d_tone_static,
        ),
        unsafe_allow_html=True,
    )

# Plot heights (px): sized to use most of a typical laptop viewport; width follows columns
chart_h_main = 185
chart_h_combined = 305
map_h = 430
dow_h = 245

_y_n = alt.Y("n:Q", title="Rides", scale=alt.Scale(zero=True, padding=0.12))
# Vega-Lite gradient dict (Altair 6+ no longer uses gradientType on alt.Gradient)
_cy_area_fill: dict = {
    "gradient": "linear",
    "x1": 1,
    "x2": 1,
    "y1": 1,
    "y2": 0,
    "stops": [
        {"offset": 0, "color": CHART_BG},
        {"offset": 0.55, "color": "#4ade80"},
        {"offset": 1, "color": "#5cb85c"},
    ],
}

_hour_sort = [hour_ampm(h) for h in range(int(h0), int(h1) + 1)]
_hour_enc = alt.Chart(hourly_plot).encode(
    x=alt.X("hour_label:N", sort=_hour_sort, title="Hour"),
    y=_y_n,
    y2=alt.value(0),
)
hourly_chart = _finalize_dark(
    (
        _hour_enc.mark_area(
            interpolate="monotone",
            opacity=0.42,
            color=_cy_area_fill,
        )
        + alt.Chart(hourly_plot)
        .mark_line(
            interpolate="monotone",
            strokeWidth=2.2,
            clip=False,
            color=ACCENT,
        )
        .encode(
            x=alt.X("hour_label:N", sort=_hour_sort, title="Hour"),
            y=_y_n,
        )
        + alt.Chart(hourly_plot)
        .mark_point(color=ACCENT_SOFT, size=7, clip=False)
        .encode(
            x=alt.X("hour_label:N", sort=_hour_sort, title="Hour"),
            y=_y_n,
        )
    ).properties(height=chart_h_main, title=_chart_title("Rides — by hour"))
)

_dow_x_enc = alt.X(
    "day_of_week:N",
    sort=dow_order,
    title="",
    scale=alt.Scale(paddingInner=0.4),
)
_dow_y_enc = alt.Y("n:Q", title="Rides", scale=alt.Scale(zero=True, padding=0.1))
_dow_bar = (
    alt.Chart(by_dow_df)
    .mark_bar(color=ACCENT, clip=False, size=22)
    .encode(x=_dow_x_enc, y=_dow_y_enc)
)
_dow_lbl = (
    alt.Chart(by_dow_df)
    .transform_filter("datum.chg_lbl != ''")
    .mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=8,
        fontWeight=600,
        color=LABEL_D,
    )
    .encode(
        x=_dow_x_enc,
        y=_dow_y_enc,
        text=alt.Text("chg_lbl:N"),
    )
)
dow_chart = (
    _finalize_dark(
        alt.layer(_dow_bar, _dow_lbl)
        .resolve_scale(x="shared", y="shared")
        .properties(
            height=dow_h,
            padding={"bottom": 56, "left": 4, "right": 6},
            title=_chart_title("Rides — by weekday"),
        )
    )
    .configure_axisX(
        labelAngle=-48,
        labelOverlap=False,
        labelPadding=5,
        labelLimit=0,
    )
)

# Forecast: train on full history (borough + hour); horizon = length of selected date range;
# chart x-axis = selected window + the same number of days ahead.
_ts_start = pd.Timestamp(start_date).normalize()
_ts_end = pd.Timestamp(end_date).normalize()
forecast_horizon = int((_ts_end - _ts_start).days + 1)
forecast_horizon = max(1, min(forecast_horizon, MAX_USER_FORECAST_HORIZON))

_fc_train = train_forecast_bundle(prepared_fingerprint(prepared_all))
combined_daily_fc: alt.Chart | None = None
fc_df = pd.DataFrame()
if _fc_train["ok"] and _fc_train.get("fit") is not None and _fc_train.get("prepared_full") is not None:
    _pa_full = _fc_train["prepared_full"]
    prepared_roll = _pa_full[_pa_full["date"] <= _ts_end].copy().reset_index(drop=True)
    if len(prepared_roll) >= max(LAG_DAYS) and forecast_horizon > 0:
        fc_df, _ = recursive_forecast(
            _fc_train["fit"]["estimator"],
            prepared_roll,
            forecast_horizon,
        )

if (
    _fc_train["ok"]
    and not fc_df.empty
    and len(_hist_plot_df)
):
    _dom_end = _ts_end + pd.Timedelta(days=forecast_horizon)
    _x_domain = [
        pd.Timestamp(_ts_start).to_pydatetime(),
        pd.Timestamp(_dom_end).to_pydatetime(),
    ]
    _x_enc = alt.X(
        "date:T",
        title="Date",
        scale=alt.Scale(domain=_x_domain),
        axis=_date_axis_mmdd(),
    )
    _y_fc = alt.Y("n:Q", title="Rides", scale=alt.Scale(zero=True, padding=0.12))
    _hist_ln = (
        alt.Chart(_hist_plot_df)
        .mark_line(
            interpolate="monotone",
            strokeWidth=2.5,
            color=ACCENT,
            clip=False,
        )
        .encode(x=_x_enc, y=_y_fc)
    )
    _fc_ln = (
        alt.Chart(fc_df)
        .mark_line(
            strokeDash=[6, 4],
            interpolate="monotone",
            strokeWidth=2.5,
            color=DEEP,
            clip=False,
        )
        .encode(x=_x_enc, y=_y_fc)
    )
    _bridge_fc = pd.DataFrame(
        {
            "date": [_hist_plot_df["date"].iloc[-1], fc_df["date"].iloc[0]],
            "n": [
                float(_hist_plot_df["n"].iloc[-1]),
                float(fc_df["n"].iloc[0]),
            ],
        }
    )
    _conn_fc = (
        alt.Chart(_bridge_fc)
        .mark_line(strokeWidth=1.2, color=ACCENT_SOFT, opacity=0.75, clip=False)
        .encode(x=_x_enc, y=_y_fc)
    )
    _rule_fc = pd.DataFrame({"cutoff": [_ts_end]})
    _vline_fc = (
        alt.Chart(_rule_fc)
        .mark_rule(
            strokeDash=[4, 3],
            strokeWidth=1.5,
            color=AXIS_D,
            opacity=0.9,
        )
        .encode(
            x=alt.X(
                "cutoff:T",
                scale=alt.Scale(domain=_x_domain),
                axis=_date_axis_mmdd(),
            )
        )
    )
    combined_daily_fc = _finalize_dark(
        alt.layer(_hist_ln, _conn_fc, _fc_ln, _vline_fc)
        .resolve_scale(x="shared", y="shared")
        .properties(
            height=chart_h_combined,
            title=_chart_title("Daily rides — forecast"),
        )
    )

_enc_fb = alt.Chart(_hist_plot_df).encode(
    x=alt.X("date:T", title="Date", axis=_date_axis_mmdd()),
    y=_y_n,
    y2=alt.value(0),
)
daily_rides_fallback = _finalize_dark(
    (
        _enc_fb.mark_area(
            interpolate="monotone",
            opacity=0.42,
            color=_cy_area_fill,
        )
        + alt.Chart(_hist_plot_df)
        .mark_line(
            interpolate="monotone",
            strokeWidth=2.2,
            clip=False,
            color=ACCENT,
        )
        .encode(x=alt.X("date:T", title="Date", axis=_date_axis_mmdd()), y=_y_n)
        + alt.Chart(_hist_plot_df)
        .mark_point(color=ACCENT_SOFT, size=7, clip=False)
        .encode(x=alt.X("date:T", title="Date", axis=_date_axis_mmdd()), y=_y_n)
    ).properties(
        height=chart_h_combined,
        title=_chart_title("Daily rides — forecast"),
    )
)

# Wider left, narrower right; middle column adds gutter between sides
_c_left, _c_mid, _c_right = st.columns([13, 1, 11], gap="large")
with _c_left:
    if not _fc_train["ok"]:
        st.warning(
            "Could not fit the forecast model on **full daily history** (same borough & hour slice) — "
            "need roughly **four weeks** of distinct calendar days with daily totals after filling gaps. "
            f"(reason: **{_fc_train.get('reason', 'unknown')}**)"
        )
    st.altair_chart(
        combined_daily_fc if combined_daily_fc is not None else daily_rides_fallback,
        use_container_width=True,
    )
    st.altair_chart(hourly_chart, use_container_width=True)
    st.altair_chart(dow_chart, use_container_width=True)

with _c_mid:
    st.empty()

with _c_right:
    st.markdown(
        f'<p style="margin:0 0 0.15rem 0;padding:0;font-size:0.72rem;font-weight:600;'
        f"color:{TEXT_MUTED};letter-spacing:0.03em;text-transform:uppercase;\">"
        "Map</p>",
        unsafe_allow_html=True,
    )
    render_borough_zones_map(filtered, rides_by_borough, map_height=map_h)
    st.markdown(
        f'<p style="margin:0.12rem 0 0.1rem 0;padding:0;font-size:0.72rem;font-weight:600;'
        f"color:{TEXT_MUTED};letter-spacing:0.03em;text-transform:uppercase;\">"
        "Table info — top zones</p>",
        unsafe_allow_html=True,
    )
    # Max table height ≈ left chart stack minus map (fudge for titles / Streamlit chrome)
    _plot_stack = chart_h_combined + chart_h_main + dow_h
    _tbl_h = max(195, min(305, _plot_stack - map_h - 88))
    _zraw = zones_top_df.copy()
    st.markdown(
        top_zones_table_html(_zraw, height_px=_tbl_h),
        unsafe_allow_html=True,
    )
