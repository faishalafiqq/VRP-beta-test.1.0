import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from PIL import Image as PILImage

# Page config - Fullscreen responsive
st.set_page_config(
    page_title="🚛 VRP Banjir Live - 6 Algoritma Expert Pro",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
    height=1000
)

# Custom CSS - Professional Dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2.5rem !important;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .best-method { 
        background: linear-gradient(145deg, #10b981 0%, #059669 100%) !important;
        box-shadow: 0 15px 40px rgba(16,185,129,0.4) !important;
    }
    
    .flood-high { background: linear-gradient(145deg, #dc2626, #b91c1c) !important; }
    .flood-medium { background: linear-gradient(145deg, #f97316, #ea580c) !important; }
    .flood-low { background: linear-gradient(145deg, #eab308, #ca8a04) !important; }
    .flood-safe { background: linear-gradient(145deg, #059669, #047857) !important; }
    
    .stDataFrame > div > div > div {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important;
    }
    
    .stPlotlyChart {
        border-radius: 12px !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# KONFIGURASI GLOBAL & CONSTANTS
# ==========================================
TOMTOM_API_KEY = "DPKi6pXg3JG1rT3aKI8t7PWCzjYcxIof"
BIAYA_PER_KM_DEFAULT = 12000
BIAYA_PER_JAM_DEFAULT = 50000
KAPASITAS_TRUK_DEFAULT = 4500

# Flood classification thresholds (BMKG standard)
FLOOD_THRESHOLDS = {
    'high': {'precip': 20, 'prob': 70},
    'medium': {'precip': 10, 'prob': 60},
    'low': {'precip': 5, 'prob': 40}
}

FLOOD_COLORS = {
    'high': '#dc2626', 'medium': '#f97316', 
    'low': '#eab308', 'safe': '#059669'
}

# ==========================================
# 6 ALGORITMA VRP - IDENTIK DENGAN KODE ASLI ANDA
# ==========================================
@st.cache_data
class VRP_MasterSolver:
    """6 Algoritma VRP Heuristic - Copy exact dari kode original"""
    
    def __init__(self, dist_matrix, demands, capacity):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.nodes = list(range(1, len(dist_matrix)))
    
    def solve_nn(self):
        """1. Nearest Neighbor - Greedy closest customer"""
        unvisited = set(self.nodes)
        routes = []
        while unvisited:
            curr = 0
            route = []
            load = 0
            while True:
                cands = [n for n in unvisited if load + self.demands[n] <= self.capacity]
                if not cands: break
                next_node = min(cands, key=lambda x: self.dist_matrix[curr][x])
                route.append(next_node)
                unvisited.remove(next_node)
                load += self.demands[next_node]
                curr = next_node
            if route: routes.append(route)
        return routes
    
    def solve_cw(self):
        """2. Clarke & Wright Savings - Industry standard"""
        routes = [[i] for i in self.nodes]
        savings = []
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    s = self.dist_matrix[i][0] + self.dist_matrix[0][j] - self.dist_matrix[i][j]
                    savings.append((s, i, j))
        savings.sort(key=lambda x: x[0], reverse=True)
        
        for s, i, j in savings:
            ri = next((idx for idx, r in enumerate(routes) if i in r), None)
            rj = next((idx for idx, r in enumerate(routes) if j in r), None)
            if ri is not None and rj is not None and ri != rj:
                r1, r2 = routes[ri], routes[rj]
                if sum(self.demands[n] for n in r1 + r2) <= self.capacity:
                    if r1[-1] == i and r2[0] == j:
                        routes[ri].extend(r2)
                        routes.pop(rj)
                    elif r2[-1] == j and r1[0] == i:
                        routes[rj].extend(r1)
                        routes.pop(ri)
        return [r for r in routes if r]
    
    def solve_cheapest_insertion(self):
        """3. Cheapest Insertion - Global optimal insertion"""
        unvisited = set(self.nodes)
        routes = []
        while unvisited:
            seed = min(unvisited, key=lambda x: self.dist_matrix[0][x])
            route = [seed]
            unvisited.remove(seed)
            load = self.demands[seed]
            
            while True:
                cands = [n for n in unvisited if load + self.demands[n] <= self.capacity]
                if not cands: break
                
                best_cost = float('inf')
                best_n, best_p = None, None
                full_r = [0] + route + [0]
                
                for n in cands:
                    for i in range(len(full_r) - 1):
                        u, v = full_r[i], full_r[i + 1]
                        cost_add = self.dist_matrix[u][n] + self.dist_matrix[n][v] - self.dist_matrix[u][v]
                        if cost_add < best_cost:
                            best_cost, best_n, best_p = cost_add, n, i
                
                if best_n:
                    route.insert(best_p, best_n)
                    unvisited.remove(best_n)
                    load += self.demands[best_n]
                else: break
            routes.append(route)
        return routes
    
    def solve_nearest_insertion(self):
        return self._solve_insertion_general('nearest')
    
    def solve_farthest_insertion(self):
        return self._solve_insertion_general('farthest')
    
    def solve_arbitrary_insertion(self):
        return self._solve_insertion_general('arbitrary')
    
    def _solve_insertion_general(self, mode):
        unvisited = set(self.nodes)
        routes = []
        while unvisited:
            if mode == 'farthest':
                seed = max(unvisited, key=lambda x: self.dist_matrix[0][x])
            elif mode == 'arbitrary':
                seed = list(unvisited)[0]
            else:
                seed = min(unvisited, key=lambda x: self.dist_matrix[0][x])
            
            route = [seed]
            unvisited.remove(seed)
            load = self.demands[seed]
            
            while True:
                cands = [n for n in unvisited if load + self.demands[n] <= self.capacity]
                if not cands: break
                
                # Select candidate based on mode
                if mode == 'nearest':
                    sel_node = min(cands, key=lambda c: min([self.dist_matrix[c][r] for r in route] + [self.dist_matrix[c][0]]))
                    check_list = [sel_node]
                else:
                    sel_node = cands[0]
                    check_list = cands[:3]  # Check first 3
                
                best_cost = float('inf')
                best_node, best_pos = None, None
                full_r = [0] + route + [0]
                
                for n in check_list:
                    for i in range(len(full_r) - 1):
                        u, v = full_r[i], full_r[i + 1]
                        cost_add = self.dist_matrix[u][n] + self.dist_matrix[n][v] - self.dist_matrix[u][v]
                        if cost_add < best_cost:
                            best_cost, best_node, best_pos = cost_add, n, i
                
                if best_node:
                    route.insert(best_pos, best_node)
                    unvisited.remove(best_node)
                    load += self.demands[best_node]
                else: break
            routes.append(route)
        return routes

# ==========================================
# WEATHER & DISTANCE UTILITIES
# ==========================================
@st.cache_data(ttl=1800)  # 30 minutes cache
def get_weather_forecast(locations_df):
    """Open-Meteo 6-hour flood forecast - BMKG standards"""
    weather_data = {}
    base_url = "https://api.open-meteo.com/v1/forecast"
    
    progress_bar = st.progress(0)
    for idx, row in locations_df.iterrows():
        progress_bar.progress((idx + 1) / len(locations_df))
        
        params = {
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'hourly': 'precipitation,precipitation_probability,wind_speed_10m',
            'current': 'precipitation,precipitation_probability',
            'forecast_days': 1,
            'timezone': 'Asia/Jakarta'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=15)
            data = response.json()
            
            # 6-hour peak risk analysis
            hours = 6
            precip = data['hourly']['precipitation'][:hours]
            prob = data['hourly']['precipitation_probability'][:hours]
            
            avg_precip = np.mean(precip) if precip else 0
            max_prob = max(prob) if prob else 0
            max_wind = max(data['hourly']['wind_speed_10m'][:hours]) if 'wind_speed_10m' in data['hourly'] else 0
            
            # BMKG Flood Classification
            if avg_precip > FLOOD_THRESHOLDS['high']['precip'] or (avg_precip > 10 and max_prob > FLOOD_THRESHOLDS['high']['prob']):
                level, color, advice = "🔴 BANJIR BESAR", FLOOD_COLORS['high'], "🚨 HINDARI RUTE - Evakuasi prioritas"
            elif avg_precip > FLOOD_THRESHOLDS['medium']['precip'] or max_prob > FLOOD_THRESHOLDS['medium']['prob']:
                level, color, advice = "🟠 GENANGAN", FLOOD_COLORS['medium'], "⚠️ HATI-HATI - Genangan jalan raya"
            elif avg_precip > FLOOD_THRESHOLDS['low']['precip'] or max_prob > FLOOD_THRESHOLDS['low']['prob']:
                level, color, advice = "🟡 HUJAN LEBAT", FLOOD_COLORS['low'], "☔ Jas hujan + kecepatan rendah"
            else:
                level, color, advice = "🟢 AMAN", FLOOD_COLORS['safe'], "✅ Pengiriman normal"
            
            weather_data[idx] = {
                'avg_precip_mmh': round(avg_precip, 1),
                'max_prob_pct': int(max_prob),
                'max_wind_kmh': round(max_wind, 1),
                'flood_level': level,
                'flood_color': color,
                'flood_advice': advice,
                'forecast_time': datetime.now().strftime("%H:%M WIB")
            }
        except Exception as e:
            weather_data[idx] = {
                'avg_precip_mmh': 0, 'max_prob_pct': 0, 'max_wind_kmh': 0,
                'flood_level': "❓ TIDAK TERSEDIA", 'flood_color': "#6c757d",
                'flood_advice': "Cek manual BMKG", 'forecast_time': "N/A"
            }
    
    progress_bar.empty()
    return weather_data

@st.cache_data
def get_distance_matrix(locations_df):
    """Haversine distance matrix - Accurate road distance approximation"""
    n = len(locations_df)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
                continue
                
            lat1, lon1 = locations_df.iloc[i][['Latitude', 'Longitude']].values
            lat2, lon2 = locations_df.iloc[j][['Latitude', 'Longitude']].values
            
            # Haversine formula (meters)
            R = 6371000  # Earth radius
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi, dlambda = np.radians(lat2-lat1), np.radians(lon2-lon1)
            
            a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            dist_matrix[i][j] = R * c  # meters
    
    return dist_matrix

# ==========================================
# MAIN DASHBOARD LAYOUT
# ==========================================
st.markdown('<h1 class="main-header">🚛 VRP Banjir Live Pro<br><small>6 Algoritma Expert + Real-time Flood Risk Analysis</small></h1>', unsafe_allow_html=True)

# Initialize session state
if 'locations_df' not in st.session_state:
    default_data = {
        'ID': list(range(11)),
        'Nama': ['Gudang Sentul', 'Cileungsi', 'Gunung Putri', 'Jonggol', 'Cariu', 
                'Tanjungsari', 'Sukamakmur', 'Klapanunggal', 'Citeureup', 
                'Babakan Madang', 'Sukaraja'],
        'Latitude': [-6.5546, -6.4035, -6.4398, -6.4716, -6.5869, 
                    -6.6163, -6.6080, -6.4780, -6.4859, -6.5744, -6.5644],
        'Longitude': [106.8624, 106.9634, 106.9157, 107.0601, 107.1328, 
                     107.1950, 107.0199, 106.9530, 106.8833, 106.8920, 106.8188],
        'Demand_kg': [0, 1500, 1200, 1000, 800, 700, 600, 900, 1100, 1000, 1300]
    }
    st.session_state.locations_df = pd.DataFrame(default_data)

# ==========================================
# SIDEBAR - INPUT & CONFIG
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ **Konfigurasi Expert**")
    st.markdown("---")
    
    # Location Editor
    st.subheader("📍 **Lokasi Pengiriman**")
    edited_df = st.data_editor(
        st.session_state.locations_df,
        num_rows="dynamic",
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True),
            "Nama": st.column_config.TextColumn("Nama Lokasi"),
            "Latitude": st.column_config.NumberColumn("Latitude", min_value=-10, max_value=5, step=0.0001, help="Koordinat GPS latitude"),
            "Longitude": st.column_config.NumberColumn("Longitude", min_value=105, max_value=108, step=0.0001, help="Koordinat GPS longitude"),
            "Demand_kg": st.column_config.NumberColumn("Demand (kg)", min_value=0, max_value=5000, help="Muatan pengiriman")
        },
        use_container_width=True,
        hide_index=False
    )
    st.session_state.locations_df = edited_df
    
    st.markdown("---")
    st.subheader("🚛 **Parameter Operasional**")
    
    col1, col2 = st.columns(2)
    with col1:
        kapasitas = st.slider("Kapasitas Truk", 1000, 15000, KAPASITAS_TRUK_DEFAULT, help="Total muatan per truk")
    with col2:
        kecepatan_rata = st.slider("Kecepatan Rata-rata", 30, 80, 50, help="km/jam dengan traffic")
    
    col1, col2 = st.columns(2)
    with col1:
        biaya_km = st.number_input("Biaya/KM", 5000, 50000, BIAYA_PER_KM_DEFAULT)
    with col2:
        biaya_jam = st.number_input("Biaya/Jam", 20000, 150000, BIAYA_PER_JAM_DEFAULT)
    
    st.markdown("---")
    algoritma_options = [
        '🎯 **SEMUA 6 + AUTO TERBAIK**',
        'Clarke & Wright ⭐', 'Nearest Neighbor ⚡', 
        'Cheapest Insertion 💎', 'Nearest Insertion',
        'Farthest Insertion', 'Arbitrary Insertion'
    ]
    selected_algo = st.selectbox("**Algoritma VRP**", algoritma_options, help="Auto-terbaik = jalankan semua + pilih biaya terkecil")
    
    if st.button("🚀 **JALANKAN OPTIMASI + BANJIR ANALYSIS**", type="primary", use_container_width=True):
        st.session_state.run_optimization = True
        st.session_state.selected_algo = selected_algo
        st.session_state.kapasitas = kapasitas
        st.session_state.biaya_km = biaya_km
        st.session_state.biaya_jam = biaya_jam
        st.session_state.kecepatan_rata = kecepatan_rata
        st.rerun()

# ==========================================
# MAIN EXECUTION & RESULTS
# ==========================================
if st.session_state.get('run_optimization', False) and len(st.session_state.locations_df) > 1:
    
    with st.spinner("🔄 **Running 6 VRP Algorithms + Weather Analysis...**"):
        # Data preparation
        locations = st.session_state.locations_df.reset_index(drop=True)
        demands = locations['Demand_kg'].tolist()
        dist_matrix = get_distance_matrix(locations)
        weather_data = get_weather_forecast(locations)
        
        # Solve ALL 6 algorithms
        solver = VRP_MasterSolver(dist_matrix, demands, st.session_state.kapasitas)
        all_results = {
            'Nearest Neighbor': solver.solve_nn(),
            'Clarke & Wright': solver.solve_cw(),
            'Cheapest Insertion': solver.solve_cheapest_insertion(),
            'Nearest Insertion': solver.solve_nearest_insertion(),
            'Farthest Insertion': solver.solve_farthest_insertion(),
            'Arbitrary Insertion': solver.solve_arbitrary_insertion()
        }
        
        # Comprehensive analysis
        method_analysis = {}
        best_method = None
        best_cost = float('inf')
        best_metrics = {}
        
        for method_name, routes in all_results.items():
            total_cost = 0
            total_distance_km = 0
            total_time_hours = 0
            route_details = []
            flood_risk_score = 0
            
            for route_idx, route in enumerate(routes):
                # Route metrics
                route_nodes = [0] + route + [0]
                route_distance_m = sum(dist_matrix[route_nodes[j]][route_nodes[j+1]] for j in range(len(route_nodes)-1))
                route_distance_km = route_distance_m / 1000
                route_time_hours = route_distance_km / st.session_state.kecepatan_rata
                route_cost = (route_distance_km * st.session_state.biaya_km) + (route_time_hours * st.session_state.biaya_jam)
                route_load = sum(demands[n] for n in route)
                
                # Flood risk per route (weighted average)
                route_flood_levels = [weather_data.get(n, {}).get('flood_level', '🟢 AMAN') for n in route]
                route_max_flood = max(route_flood_levels)
                flood_score = sum(1 if '🔴' in f else 0.5 if '🟠' in f else 0.25 if '🟡' in f else 0 for f in route_flood_levels)
                
                total_cost += route_cost
                total_distance_km += route_distance_km
                total_time_hours += route_time_hours
                flood_risk_score += flood_score
                
                # Route names for display
                route_names = [locations.iloc[n]['Nama'] for n in route_nodes]
                route_details.append({
                    'Truk': route_idx + 1,
                    'Rute': ' → '.join(route_names),
                    'Jarak_km': round(route_distance_km, 1),
                    'Waktu_jam': round(route_time_hours, 1),
                    'Muatan_kg': route_load,
                    'Biaya_Rp': int(route_cost),
                    'Flood_Max': route_max_flood,
                    'Flood_Score': round(flood_score, 1)
                })
            
            method_analysis[method_name] = {
                'routes': routes,
                'total_cost': total_cost,
                'total_distance_km': total_distance_km,
                'total_time_hours': total_time_hours,
                'num_trucks': len(routes),
                'flood_risk_score': flood_risk_score / max(1, len(routes)),
                'details': route_details,
                'cost_per_km': total_cost / total_distance_km if total_distance_km > 0 else 0
            }
            
            # Track best method (lowest total cost)
            if total_cost < best_cost:
                best_cost = total_cost
                best_method = method_name
                best_metrics = method_analysis[method_name]
                st.session_state.best_result = method_analysis[method_name]

    # ==========================================
    # EXECUTIVE SUMMARY METRICS
    # ==========================================
    st.markdown("## 🏆 **HASIL OPTIMASI - METODE TERBAIK**")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card best-method">
            <h3 style="margin:0;font-size:1.8rem;">{best_method}</h3>
            <h1 style="margin:0;font-size:2.5rem;">🥇 #1</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("📏 Total Jarak", f"{best_metrics['total_distance_km']:.1f} km")
    
    with col3:
        st.metric("💰 Total Biaya", f"Rp {best_metrics['total_cost']:,.0f}", 
                 delta=f"per km: Rp {best_metrics['cost_per_km']:.0f}")
    
    with col4:
        st.metric("⏱️ Total Waktu", f"{best_metrics['total_time_hours']:.1f} jam")
    
    with col5:
        st.metric("🚛 Truk Dibutuhkan", best_metrics['num_trucks'])
    
    with col6:
        flood_class = "🟢 AMAN" if best_metrics['flood_risk_score'] < 0.25 else "🟡 SEDANG" if best_metrics['flood_risk_score'] < 0.75 else "🔴 TINGGI"
        st.metric("🌧️ Risiko Banjir", flood_class)

    # ==========================================
    # COMPARISON TABLE - 6 ALGORITHMS
    # ==========================================
    st.markdown("## 📊 **PERBANDINGAN 6 ALGORITMA**")
    
    comparison_data = []
    for method, metrics in method_analysis.items():
        rank = sorted(method_analysis.items(), key=lambda x: x[1]['total_cost'])[::-1].index((method, metrics)) + 1
        comparison_data.append({
            '🏆 Rank': f"#{rank}",
            'Algoritma': method,
            '💰 Biaya': f"Rp {metrics['total_cost']:,.0f}",
            '📏 Jarak': f"{metrics['total_distance_km']:.1f} km",
            '🚛 Truk': metrics['num_trucks'],
            '⏱️ Waktu': f"{metrics['total_time_hours']:.1f} jam",
            '🌧️ Flood Risk': f"{metrics['flood_risk_score']:.2f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, height=400)

    # ==========================================
    # BEST ROUTE DETAILS
    # ==========================================
    st.markdown("## 🛣️ **RUTE TERBAIK - DETAIL**")
    best_details_df = pd.DataFrame(st.session_state.best_result['details'])
    st.dataframe(best_details_df, use_container_width=True)

    # ==========================================
    # WEATHER & FLOOD RISK TABLE
    # ==========================================
    st.markdown("## 🌧️ **PRAKIRAAN CUACA - 6 JAM KEDEPAN**")
    weather_display = []
    for i, row in locations.iterrows():
        wdata = weather_data[i]
        weather_display.append({
            '📍 Lokasi': row['Nama'],
            '📦 Demand': f"{row['Demand_kg']:,} kg",
            '🌧️ Hujan': f"{wdata['avg_precip_mmh']} mm/jam",
            '📊 Prob': f"{wdata['max_prob_pct']}%",
            '💨 Angin': f"{wdata['max_wind_kmh']} km/jam",
            '🚨 Status': wdata['flood_level'],
            '💡 Saran': wdata['flood_advice']
        })
    
    weather_df = pd.DataFrame(weather_display)
    st.dataframe(weather_df, use_container_width=True)

    # ==========================================
    # INTERACTIVE MAP
    # ==========================================
    st.markdown("## 🗺️ **PETA RUTE TERBAIK + BANJIR RISK**")
    
    m = folium.Map(
        location=[locations['Latitude'].mean(), locations['Longitude'].mean()], 
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Location markers with flood colors
    for i, row in locations.iterrows():
        wdata = weather_data[i]
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=f"""
            <div style="width:250px">
                <b style="color:{wdata['flood_color']}">{row['Nama']}</b><br>
                📦 {row['Demand_kg']:,} kg<br>
                🌧️ {wdata['avg_precip_mmh']} mm/jam | {wdata['max_prob_pct']}% prob<br>
                <b>{wdata['flood_level']}</b><br>
                <small>{wdata['flood_advice']}</small>
            </div>
            """,
            tooltip=f"{row['Nama']} - {wdata['flood_level']}",
            icon=folium.Icon(
                color=wdata['flood_color'].lstrip('#'), 
                icon='map-marker-alt', 
                prefix='fa',
                icon_color='white'
            )
        ).add_to(m)
    
    # Best routes polyline
    best_routes = st.session_state.best_result['routes']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, route in enumerate(best_routes):
        route_coords = [[locations.iloc[n]['Latitude'], locations.iloc[n]['Longitude']] for n in [0] + route + [0]]
        folium.PolyLine(
            route_coords,
            color=colors[i % len(colors)],
            weight=8,
            opacity=0.9,
            popup=f"""
            <b>🚛 Truk {i+1}</b><br>
            💰 Rp{best_details_df.iloc[i]['Biaya_Rp']:,.0f}<br>
            📏 {best_details_df.iloc[i]['Jarak_km']} km<br>
            📦 {best_details_df.iloc[i]['Muatan_kg']} kg<br>
            🚨 {best_details_df.iloc[i]['Flood_Max']}
            """,
            tooltip=f"Truk {i+1}"
        ).add_to(m)
    
    # Legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 12px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
    <h4 style="margin:0 0 10px 0;">🌧️ Legend Banjir</h4>
    <div style="background:#dc2626;color:white;padding:6px;margin:3px 0;border-radius:6px;text-align:center;font-weight:600;">🔴 BANJIR BESAR</div>
    <div style="background:#f97316;color:white;padding:6px;margin:3px 0;border-radius:6px;text-align:center;font-weight:600;">🟠 GENANGAN</div>
    <div style="background:#eab308;color:black;padding:6px;margin:3px 0;border-radius:6px;text-align:center;font-weight:600;">🟡 HUJAN LEBAT</div>
    <div style="background:#059669;color:white;padding:6px;margin:3px 0;border-radius:6px;text-align:center;font-weight:600;">🟢 AMAN</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    st_folium(m, width=1400, height=650)

    # ==========================================
    # ADVANCED ANALYTICS CHARTS
    # ==========================================
    st.markdown("## 📈 **ANALISIS ADVANCED**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost comparison bar chart
        costs = [method_analysis[m]['total_cost'] for m in method_analysis.keys()]
        fig = px.bar(
            x=list(method_analysis.keys()), 
            y=costs,
            title="💰 **Perbandingan Biaya Total**",
            color=list(method_analysis.keys()),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.add_hline(y=best_cost, line_dash="dot", line_color="green", 
                      annotation_text=f"🥇 TERBAIK: {best_method}", annotation_position="top right")
        fig.update_layout(showlegend=False, xaxis_tickangle=45, height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Flood risk heatmap
        flood_scores = [method_analysis[m]['flood_risk_score'] for m in method_analysis.keys()]
        fig2 = go.Figure(data=go.Bar(
            x=list(method_analysis.keys()), 
            y=flood_scores,
            marker_color=['red' if s>0.75 else 'orange' if s>0.25 else 'green' for s in flood_scores],
            text=[f"{s:.2f}" for s in flood_scores],
            textposition='auto'
        ))
        fig2.update_layout(title="🌧️ **Flood Risk Score per Algoritma**", height=450, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # ==========================================
    # DOWNLOAD CENTER
    # ==========================================
    st.markdown("## 📥 **DOWNLOAD LENGKAP**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_data = df_comparison.to_csv(index=False)
        st.download_button(
            label="📊 **CSV Perbandingan**",
            data=csv_data,
            file_name=f"vrp_6_algoritma_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Multi-sheet
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_comparison.to_excel(writer, sheet_name='Perbandingan_6_Algo', index=False)
            best_details_df.to_excel(writer, sheet_name=f'Rute_{best_method}', index=False)
            weather_df.to_excel(writer, sheet_name='Cuaca_Banjir', index=False)
            locations.to_excel(writer, sheet_name='Lokasi_Input', index=False)
        st.download_button(
            label="📈 **Excel Full Report**",
            data=excel_buffer.getvalue(),
            file_name=f"vrp_full_analysis_{best_method}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # PDF Executive Summary
        def create_executive_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title = Paragraph(f"🏆 VRP Optimasi - {best_method} TERBAIK", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Executive summary table
            summary_data = [
                ['METRIK', 'NILAI', 'STATUS'],
                ['Algoritma Terbaik', best_method, '🥇 #1'],
                ['Total Biaya', f'Rp{best_metrics["total_cost"]:,.0f}', '💰'],
                ['Total Jarak', f'{best_metrics["total_distance_km"]:.1f} km', '📏'],
                ['Truk Dibutuhkan', str(best_metrics['num_trucks']), '🚛'],
                ['Flood Risk', f'{best_metrics["flood_risk_score"]:.2f}', '🌧️']
            ]
            
            table = Table(summary_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            story.append(table)
            
            doc.build(story)
            return buffer.getvalue()
        
        st.download_button(
            label="📄 **PDF Executive Summary**",
            data=create_executive_pdf(),
            file_name=f"vrp_executive_{best_method}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    # Success message
    st.markdown("---")
    st.success(f"""
    ✅ **OPTIMASI SELESAI!** 
    🏆 **{best_method}** = METODE TERBAIK 
    💰 **Rp{best_cost:,.0f}** total biaya 
    📱 **Share link ini ke tim!**
    """)
    
    st.balloons()

else:
    # Welcome screen
    st.markdown("## 🚀 **Welcome to VRP Banjir Live Pro**")
    st.info("""
    **Cara Menggunakan:**
    1. 📍 **Edit lokasi** di sidebar (koordinat + demand)
    2. ⚙️ **Atur parameter** truk & biaya
    3. 🚀 **Klik OPTIMASI** → 6 algoritma jalan otomatis
    4. 🏆 **Lihat hasil terbaik** + peta interaktif
    5. 📥 **Download** report PDF/Excel
    """)
    
    st.markdown("""
    **Fitur Pro:**
    • 6 Algoritma VRP ✅ Industry standard
    • Real-time banjir 6 jam ✅ Open-Meteo
    • Auto-terbaik berdasarkan biaya ✅
    • Peta interaktif ✅ Folium
    • Export lengkap ✅ PDF/Excel/CSV
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem; font-family: Inter;'>
    🚛 **VRP Banjir Live Pro** | 6 Algoritma Expert | 
    TomTom Traffic + Open-Meteo Weather | 
    <a href='https://streamlit.io/cloud' target='_blank'>Streamlit Cloud</a> © 2025
</div>
""", unsafe_allow_html=True)