import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="Retail Sales Intelligence",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Sidebar — force light mode so labels are readable */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}
[data-testid="stSidebar"] * {
    color: #1f2937 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #374151 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

.prediction-box {
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 1rem;
}
.prediction-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-bottom: 4px; }
.prediction-value { font-size: 2.6rem; font-weight: 700; color: #1e40af; margin: 0; }
.prediction-meta { font-size: 0.82rem; color: #6b7280; margin-top: 4px; }

.analysis-box {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 1rem;
}
.analysis-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #3b82f6; margin-bottom: 8px; }

.section-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 0.4rem; }

.drift-alert { background: #fff1f2; border: 1px solid #fecdd3; border-left: 4px solid #e11d48; border-radius: 6px; padding: 14px 18px; color: #9f1239; font-size: 0.9rem; }
.drift-ok { background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; border-radius: 6px; padding: 14px 18px; color: #14532d; font-size: 0.9rem; }

.stButton > button { background-color: #1e40af; color: white; border: none; border-radius: 6px; padding: 0.5rem 1.5rem; font-weight: 500; width: 100%; }
.stButton > button:hover { background-color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("RETAIL_API_KEY", "retail-ops-elite-key-2024")
HEADERS = {"X-API-KEY": API_KEY}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Retail Sales Intelligence")
    st.markdown("<small style='color:#6c757d'>ML + GenAI Platform</small>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "Daily Forecast",
        "Weekly / Monthly Forecast",
        "Model Monitoring",
        "API Reference"
    ])
    st.markdown("---")
    try:
        requests.get(f"{API_URL}/health", timeout=2)
        st.markdown("<span style='color:#16a34a; font-size:0.85rem;'>● API Connected</span>", unsafe_allow_html=True)
    except:
        st.markdown("<span style='color:#dc2626; font-size:0.85rem;'>● API Offline</span>", unsafe_allow_html=True)
        st.caption("Run: uvicorn app_fastapi:app --reload")


# ═════════════════════════════════════════════════════════════════════════════
# STORE CONFIG MAPS (Human readable -> API code)
# ═════════════════════════════════════════════════════════════════════════════
store_type_map = {
    "Small Convenience Store": "a",
    "Large Shopping Center": "b",
    "Standard Supermarket": "c",
    "Extra-Large Hypermarket": "d"
}
assortment_map = {
    "Basic (Essential Items Only)": "a",
    "Extended (Wide Selection)": "b",
    "Extra (Premium + Niche Items)": "c"
}
holiday_map = {
    "No Holiday": "0",
    "Public Holiday": "a",
    "Easter Holiday": "b",
    "Christmas": "c"
}


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DAILY FORECAST
# ═════════════════════════════════════════════════════════════════════════════
if page == "Daily Forecast":
    st.title("Daily Sales Forecast")
    st.markdown("Predict sales for a specific store on a specific date. All conditions below are for **that exact date**.")
    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<p class="section-label">Store Identity</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
        with c2:
            forecast_date = st.date_input("Date to Forecast", value=datetime.now())

        st.markdown('<p class="section-label" style="margin-top:1rem">Store Profile</p>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            store_type_label = st.selectbox("Store Size", list(store_type_map.keys()))
        with c4:
            assortment_label = st.selectbox("Product Range", list(assortment_map.keys()))

        competition_km = st.slider("Distance to Nearest Competitor (km)", 0.1, 50.0, 5.0, 0.1,
            help="How far away is the nearest competing store?")

        st.markdown('<p class="section-label" style="margin-top:1rem">Conditions on This Date</p>', unsafe_allow_html=True)
        st.caption("These fields describe what is happening ON the date you selected above.")
        c5, c6 = st.columns(2)
        with c5:
            promo = st.selectbox("Is Promotion Running?", ["Yes", "No"],
                help="Is there an active discount or promotional campaign on this date?")
            state_holiday_label = st.selectbox("State Holiday on This Date", list(holiday_map.keys()),
                help="Is there a regional/public holiday on this date?")
        with c6:
            school_holiday = st.selectbox("School Holiday on This Date", ["No", "Yes"],
                help="Are schools closed on this date?")

        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("Generate Forecast")

    with right:
        if generate:
            payload = {
                "store": int(store_id),
                "DayOfWeek": int(forecast_date.weekday()) + 1,
                "Date": str(forecast_date),
                "open": 1,  # Always open — no point forecasting a closed store
                "Promo": 1 if promo == "Yes" else 0,
                "StateHoliday": holiday_map[state_holiday_label],
                "SchoolHoliday": 1 if school_holiday == "Yes" else 0,
                "StoreType": store_type_map[store_type_label],
                "Assortment": assortment_map[assortment_label],
                "CompetitionDistance": float(competition_km * 1000)
            }
            with st.spinner("Running prediction..."):
                try:
                    resp = requests.post(f"{API_URL}/forecast", json=payload, headers=HEADERS, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()

                        # Prediction box
                        st.markdown(f"""
                        <div class="prediction-box">
                            <p class="prediction-label">Predicted Daily Sales</p>
                            <p class="prediction-value">€{data['prediction']:,.0f}</p>
                            <p class="prediction-meta">Store {store_id} &nbsp;·&nbsp; {str(forecast_date)} &nbsp;·&nbsp; {data['model_info'].get('algorithm', 'Model')}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # SHAP Feature Importance Chart
                        if data.get("explanation"):
                            shap_data = data["explanation"]
                            features = list(shap_data.keys())
                            values = list(shap_data.values())
                            colors = ["#e11d48" if v > 0 else "#2563eb" for v in values]

                            fig = go.Figure(go.Bar(
                                x=values,
                                y=features,
                                orientation='h',
                                marker_color=colors,
                                text=[f"{v:+.1f}" for v in values],
                                textposition='outside'
                            ))
                            fig.update_layout(
                                title="Top 5 Factors Influencing This Prediction",
                                xaxis_title="SHAP Impact on Sales (€)",
                                yaxis_title="",
                                height=300,
                                margin=dict(l=10, r=10, t=40, b=10),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family="Inter", size=12),
                                xaxis=dict(zeroline=True, zerolinecolor='#e5e7eb'),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Red bars push sales up. Blue bars pull sales down.")
                        else:
                            st.info("SHAP explanation not available for this prediction.")

                        # AI Analysis
                        if data.get("analysis"):
                            st.markdown('<div class="analysis-box"><p class="analysis-label">AI Business Analysis</p></div>', unsafe_allow_html=True)
                            st.markdown(data["analysis"])
                        
                        if data.get("latency"):
                            st.caption(f"Response time: {data['latency']}")

                    elif resp.status_code == 422:
                        st.error("Validation error.")
                        st.json(resp.json())
                    else:
                        st.error(f"API error {resp.status_code}")
                        st.text(resp.text)

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Start it with: uvicorn app_fastapi:app --reload")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.markdown("""
            <div style="background:#f9fafb; border:1px dashed #d1d5db; border-radius:8px; padding:40px; text-align:center; color:#9ca3af; margin-top:2rem;">
                Fill in the form and click Generate Forecast
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WEEKLY / MONTHLY FORECAST
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Weekly / Monthly Forecast":
    st.title("Multi-Period Sales Forecast")
    st.markdown("Forecast total expected sales over a range of days. Use the What-If tab to compare scenarios.")
    st.divider()

    tab1, tab2 = st.tabs(["Period Forecast", "What-If Scenario Analysis"])

    with tab1:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<p class="section-label">Store Identity</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                r_store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1, key="range_store")
            with c2:
                r_start_date = st.date_input("Start Date", value=datetime.now(), key="range_date")

            period_label = st.selectbox("Forecast Period", [
                "Next 7 Days (Weekly)",
                "Next 14 Days (Bi-Weekly)",
                "Next 30 Days (Monthly)",
            ])
            period_map = {"Next 7 Days (Weekly)": 7, "Next 14 Days (Bi-Weekly)": 14, "Next 30 Days (Monthly)": 30}
            days = period_map[period_label]

            st.markdown('<p class="section-label" style="margin-top:1rem">Store Profile</p>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                r_store_type_label = st.selectbox("Store Size", list(store_type_map.keys()), key="r_stype")
            with c4:
                r_assortment_label = st.selectbox("Product Range", list(assortment_map.keys()), key="r_assim")

            r_competition_km = st.slider("Nearest Competitor Distance (km)", 0.1, 50.0, 5.0, 0.1, key="r_comp")

            st.markdown('<p class="section-label" style="margin-top:1rem">Period Conditions</p>', unsafe_allow_html=True)
            r_promo = st.selectbox("Promotion Running During This Period?", ["No", "Yes"], key="range_promo")

            st.markdown("<br>", unsafe_allow_html=True)
            run_range = st.button("Run Period Forecast", key="run_range_btn")

        with right:
            if run_range:
                payload = {
                    "store": int(r_store_id),
                    "start_date": str(r_start_date),
                    "days": days,
                    "open_status": 1,
                    "promo_active": 1 if r_promo == "Yes" else 0,
                    "store_type": store_type_map[r_store_type_label],
                    "assortment": assortment_map[r_assortment_label],
                    "competition_distance": float(r_competition_km * 1000)
                }
                with st.spinner(f"Generating {days}-day forecast..."):
                    try:
                        resp = requests.post(f"{API_URL}/forecast/range", json=payload, headers=HEADERS, timeout=120)
                        if resp.status_code == 200:
                            data = resp.json()

                            m1, m2 = st.columns(2)
                            m1.metric("Total Estimated Sales", f"€{data['total_estimated_sales']:,.0f}")
                            avg_daily = data['total_estimated_sales'] / days
                            m2.metric("Average Daily Sales", f"€{avg_daily:,.0f}")
                            st.caption(f"Store {r_store_id}  |  {str(r_start_date)} to {data['end_date']}  |  {'Promo Active' if r_promo == 'Yes' else 'No Promo'}")

                            daily = data.get("daily_forecasts", [])
                            if daily:
                                df_daily = pd.DataFrame(daily)
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df_daily["date"], y=df_daily["prediction"],
                                    mode="lines+markers",
                                    line=dict(color="#3b82f6", width=2),
                                    marker=dict(size=5),
                                    fill='tozeroy', fillcolor='rgba(59,130,246,0.08)',
                                    name="Predicted Sales"
                                ))
                                fig.update_layout(
                                    title=f"Daily Sales — Store {r_store_id}",
                                    xaxis_title="Date", yaxis_title="Sales (€)",
                                    height=300, paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(family="Inter", size=12),
                                    xaxis=dict(gridcolor='#f3f4f6'),
                                    yaxis=dict(gridcolor='#f3f4f6'),
                                    margin=dict(l=10, r=10, t=40, b=10)
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Export to CSV
                                csv = df_daily.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download Forecast as CSV",
                                    data=csv,
                                    file_name=f"store_{r_store_id}_forecast.csv",
                                    mime="text/csv"
                                )

                            if data.get("strategic_analysis"):
                                st.markdown('<div class="analysis-box"><p class="analysis-label">Strategic Business Analysis</p></div>', unsafe_allow_html=True)
                                st.markdown(data["strategic_analysis"])

                            if data.get("latency"):
                                st.caption(f"Response time: {data['latency']}")

                        else:
                            st.error(f"API error {resp.status_code}")
                            st.text(resp.text)

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.markdown("""
                <div style="background:#f9fafb; border:1px dashed #d1d5db; border-radius:8px; padding:40px; text-align:center; color:#9ca3af; margin-top:2rem;">
                    Configure the period and click Run Period Forecast
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("**What-If Scenario Analysis**")
        st.markdown("Compare the impact of running a promotion vs not running one for the same store and period. This shows you exactly how much revenue a promotion generates.")

        c1, c2, c3 = st.columns(3)
        with c1:
            wi_store = st.number_input("Store ID", min_value=1, max_value=1115, value=1, key="wi_store")
        with c2:
            wi_date = st.date_input("Start Date", value=datetime.now(), key="wi_date")
        with c3:
            wi_days = st.selectbox("Period", [7, 14, 30], format_func=lambda x: f"{x} days", key="wi_days")

        c4, c5 = st.columns(2)
        with c4:
            wi_store_type = st.selectbox("Store Size", list(store_type_map.keys()), key="wi_stype")
        with c5:
            wi_competition = st.slider("Competitor Distance (km)", 0.1, 50.0, 5.0, key="wi_comp")

        run_whatif = st.button("Compare Scenarios", key="run_whatif")

        if run_whatif:
            base_payload = {
                "store": int(wi_store), "start_date": str(wi_date), "days": int(wi_days),
                "open_status": 1, "promo_active": 0,
                "store_type": store_type_map[wi_store_type],
                "assortment": "a",
                "competition_distance": float(wi_competition * 1000)
            }
            promo_payload = {**base_payload, "promo_active": 1}

            with st.spinner("Running both scenarios in parallel..."):
                try:
                    r_base = requests.post(f"{API_URL}/forecast/range", json=base_payload, headers=HEADERS, timeout=120)
                    r_promo = requests.post(f"{API_URL}/forecast/range", json=promo_payload, headers=HEADERS, timeout=120)

                    if r_base.status_code == 200 and r_promo.status_code == 200:
                        d_base = r_base.json()
                        d_promo = r_promo.json()

                        base_total = d_base['total_estimated_sales']
                        promo_total = d_promo['total_estimated_sales']
                        uplift = promo_total - base_total
                        uplift_pct = (uplift / base_total) * 100 if base_total > 0 else 0

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Without Promotion", f"€{base_total:,.0f}")
                        m2.metric("With Promotion", f"€{promo_total:,.0f}")
                        m3.metric("Revenue Uplift", f"€{uplift:,.0f}", delta=f"{uplift_pct:+.1f}%")

                        # Comparison chart
                        df_base = pd.DataFrame(d_base['daily_forecasts'])
                        df_promo = pd.DataFrame(d_promo['daily_forecasts'])

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_base["date"], y=df_base["prediction"],
                            name="No Promotion", line=dict(color="#9ca3af", width=2, dash='dash')
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_promo["date"], y=df_promo["prediction"],
                            name="With Promotion", line=dict(color="#3b82f6", width=2),
                            fill='tonexty', fillcolor='rgba(59,130,246,0.08)'
                        ))
                        fig.update_layout(
                            title="Promotion Impact on Daily Sales",
                            xaxis_title="Date", yaxis_title="Sales (€)",
                            height=320, paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter", size=12),
                            xaxis=dict(gridcolor='#f3f4f6'),
                            yaxis=dict(gridcolor='#f3f4f6'),
                            legend=dict(orientation="h", y=-0.2),
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("The shaded area between lines represents the additional revenue generated by the promotion.")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")




# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MONITORING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Monitoring":
    st.title("Model Health Monitoring")
    st.markdown("Detects when production data starts to differ from training data, signaling potential accuracy loss.")
    st.divider()

    left, right = st.columns([2, 1], gap="large")

    with left:
        if st.button("Run Drift Analysis"):
            with st.spinner("Comparing production traffic vs training baseline..."):
                try:
                    resp = requests.get(f"{API_URL}/monitor/drift", timeout=90)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data['status'] == 'success':
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Drift Share", f"{data['drift_share'] * 100:.1f}%")
                            m2.metric("Columns Drifted", f"{int(data['drifted_columns_count'])}")
                            m3.metric("Action Required", "Retrain Model" if data['requires_retraining'] else "None")

                            st.markdown("<br>", unsafe_allow_html=True)
                            if data['requires_retraining']:
                                st.markdown('<div class="drift-alert"><strong>Drift Detected</strong><br>Production data has shifted from the training baseline. Model retraining is recommended to maintain accuracy.</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="drift-ok"><strong>Model Stable</strong><br>Data distributions are within acceptable limits. No action required.</div>', unsafe_allow_html=True)
                        else:
                            st.info(data.get('message', 'No prediction data yet. Make a forecast first.'))
                    else:
                        st.error(f"API error {resp.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API.")

    with right:
        st.markdown("**What is data drift?**")
        st.markdown("""
        The model was trained on historical data from 2013–2015. When current data (e.g., competition distances, promo patterns) changes significantly, predictions become less reliable.

        This page detects that shift before it causes damage.
        """)
        st.markdown("**To test the monitoring:**")
        st.code("python scripts/simulate_customers.py", language="bash")
        st.caption("Sends 20 requests with modified competition distances to simulate real-world drift.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — API REFERENCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "API Reference":
    st.title("API Reference")
    st.markdown("Backend endpoint documentation for the Retail Sales Intelligence API.")
    st.divider()

    with st.expander("POST /forecast — Single Day Prediction"):
        st.markdown("Predicts sales for a single store on a specific date. Returns prediction, SHAP feature importance, and AI analysis.")
        st.code("""{
  "store": 1,
  "DayOfWeek": 3,
  "Date": "2024-06-15",
  "open": 1,
  "Promo": 1,
  "StateHoliday": "0",
  "SchoolHoliday": 0,
  "StoreType": "a",
  "Assortment": "a",
  "CompetitionDistance": 5000.0
}""", language="json")

    with st.expander("POST /forecast/range — Multi-Day Prediction"):
        st.markdown("Recursively predicts sales over a range of days (7 = weekly, 30 = monthly).")
        st.code("""{
  "store": 1,
  "start_date": "2024-06-15",
  "days": 7,
  "open_status": 1,
  "promo_active": 0
}""", language="json")

    with st.expander("GET /monitor/drift — Data Drift Analysis"):
        st.markdown("Compares live inference data against training baseline using Evidently AI.")
        st.code("""{
  "status": "success",
  "drift_share": 0.6,
  "drifted_columns_count": 6,
  "requires_retraining": true
}""", language="json")

    with st.expander("GET /health — Health Check"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=2)
            st.success("API is online")
            st.json(r.json())
        except:
            st.warning("API is currently offline.")
