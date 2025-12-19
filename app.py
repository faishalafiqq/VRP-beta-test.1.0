import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime
import io
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="🚛 VRP Banjir Live Pro",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.main-header { 
    font-family: 'Inter', sans-serif; 
    font-size: 3rem !important; 
    font-weight: 700 !important; 
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    text-align: center; 
    margin-bottom: 2rem !important; 
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
.metric-card:hover { transform: translateY(-5px); }
.best-method { 
    background: linear-gradient(145deg, #10b981 0%, #059669 100%) !important; 
    box-shadow: 0 15px 40px rgba(16,185,129,0.4) !important; 
}
.stDataFrame > div > div > div { 
    border-radius: 12px !important; 
    overflow: hidden !important; 
    box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important; 
}
.stPlotlyChart { 
    border-radius: 12px !important; 
    box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important; 
}
#tomtom-map { 
    border-radius: 12px !important; 
    box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important; 
}
</style>
""", unsafe_allow_html=True)

# CONSTANTS
TOMTOM_API_KEY = "DPKi6pXg3JG1rT3aKI8t7PWCzjYcxIof"
KAPASITAS_TRUK_DEFAULT = 4500
BIAYA_PER_KM_DEFAULT = 12000
BIAYA_PER_JAM_DEFAULT = 50000

FLOOD_THRESHOLDS = {
    'high': {'precip': 20, 'prob': 70},
    'medium': {'precip': 10, 'prob': 60},
    'low': {'precip': 5, 'prob': 40}
}
FLOOD_COLORS = {
    'high': '#dc2626', 'medium': '#f97316', 
    'low': '#eab308', 'safe': '#059669'
}

# VRP SOLVER CLASS
@st.cache_data
class VRP_MasterSolver:
    def __init__(self, dist_matrix, demands, capacity):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.nodes = list(range(1, len(dist_matrix)))
    
    def solve_nn(self):
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

# UTILITY FUNCTIONS
@st.cache_data(ttl=1800)
def get_weather_forecast(locations_df):
    weather_data = {}
    base_url = "https://api.open-meteo.com/v1/forecast"
    
    progress_bar = st.progress(0)
    for idx, row in locations_df.iterrows():
        progress_bar.progress((idx + 1) / len(locations_df))
        
        params = {
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'hourly': 'precipitation,precipitation_probability,wind_speed_10m',
            'forecast_days': 1,
            'timezone': 'Asia/Jakarta'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=15)
            data = response.json()
            
            hours = 6
            precip = data['hourly']['precipitation'][:hours]
            prob = data['hourly']['precipitation_probability'][:hours]
            
            avg_precip = np.mean(precip) if precip else 0
            max_prob = max(prob) if prob else 0
            max_wind = max(data['hourly']['wind_speed_10m'][:hours]) if 'wind_speed_10m' in data['hourly'] else 0
            
            if avg_precip > FLOOD_THRESHOLDS['high']['precip'] or (avg_precip > 10 and max_prob > FLOOD_THRESHOLDS['high']['prob']):
                level, color, advice = "🔴 BANJIR BESAR", FLOOD_COLORS['high'], "🚨 HINDARI RUTE"
            elif avg_precip > FLOOD_THRESHOLDS['medium']['precip'] or max_prob > FLOOD_THRESHOLDS['medium']['prob']:
                level, color, advice = "🟠 GENANGAN", FLOOD_COLORS['medium'], "⚠️ HATI-HATI"
            elif avg_precip > FLOOD_THRESHOLDS['low']['precip'] or max_prob > FLOOD_THRESHOLDS['low']['prob']:
                level, color, advice = "🟡 HUJAN LEBAT", FLOOD_COLORS['low'], "☔ JAS HUJAN"
            else:
                level, color, advice = "🟢 AMAN", FLOOD_COLORS['safe'], "✅ NORMAL"
            
            weather_data[idx] = {
                'avg_precip_mmh': round(avg_precip, 1),
                'max_prob_pct': int(max_prob),
                'max_wind_kmh': round(max_wind, 1),
                'flood_level': level,
                'flood_color': color,
                'flood_advice': advice
            }
        except:
            weather_data[idx] = {
                'avg_precip_mmh': 0, 'max_prob_pct': 0, 'max_wind_kmh': 0,
                'flood_level': "❓ ERROR", 'flood_color': "#6c757d",
                'flood_advice': "Cek BMKG"
            }
    progress_bar.empty()
    return weather_data

@st.cache_data
def get_distance_matrix(locations_df):
    n = len(locations_df)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
                continue
            lat1, lon1 = locations_df.iloc[i][['Latitude', 'Longitude']].values
            lat2, lon2 = locations_df.iloc[j][['Latitude', 'Longitude']].values
            R = 6371000
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi, dlambda = np.radians(lat2-lat1), np.radians(lon2-lon1)
            a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            dist_matrix[i][j] = R * c
    return dist_matrix

# MAIN APP
st.markdown('<h1 class="main-header">🚛 VRP Banjir Live Pro<br><small>3 Algoritma Expert + TomTom Maps + Excel Export</small></h1>', unsafe_allow_html=True)

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

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ **Konfigurasi**")
    st.markdown("---")
    
    edited_df = st.data_editor(
        st.session_state.locations_df,
        num_rows="dynamic",
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True),
            "Demand_kg": st.column_config.NumberColumn("Demand (kg)", min_value=0, max_value=5000)
        },
        use_container_width=True
    )
    st.session_state.locations_df = edited_df
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: kapasitas = st.slider("Kapasitas Truk", 1000, 15000, KAPASITAS_TRUK_DEFAULT)
    with col2: kecepatan_rata = st.slider("Kecepatan", 30, 80, 50)
    col1, col2 = st.columns(2)
    with col1: biaya_km = st.number_input("Biaya/KM", 5000, 50000, BIAYA_PER_KM_DEFAULT)
    with col2: biaya_jam = st.number_input("Biaya/Jam", 20000, 150000, BIAYA_PER_JAM_DEFAULT)
    
    if st.button("🚀 **JALANKAN OPTIMASI**", type="primary", use_container_width=True):
        st.session_state.run_optimization = True
        st.session_state.kapasitas = kapasitas
        st.session_state.biaya_km = biaya_km
        st.session_state.biaya_jam = biaya_jam
        st.session_state.kecepatan_rata = kecepatan_rata
        st.rerun()

# EXECUTION
if st.session_state.get('run_optimization', False) and len(st.session_state.locations_df) > 1:
    with st.spinner("🔄 **Running 3 VRP Algorithms + Weather Analysis...**"):
        locations = st.session_state.locations_df.reset_index(drop=True)
        demands = locations['Demand_kg'].tolist()
        dist_matrix = get_distance_matrix(locations)
        weather_data = get_weather_forecast(locations)
        
        solver = VRP_MasterSolver(dist_matrix, demands, st.session_state.kapasitas)
        all_results = {
            'Nearest Neighbor ⚡': solver.solve_nn(),
            'Clarke & Wright ⭐': solver.solve_cw(),
            'Cheapest Insertion 💎': solver.solve_cheapest_insertion()
        }
        
        method_analysis = {}
        best_method = None
        best_cost = float('inf')
        
        for method_name, routes in all_results.items():
            total_cost = total_distance_km = total_time_hours = flood_risk_score = 0
            route_details = []
            
            for route_idx, route in enumerate(routes):
                route_nodes = [0] + route + [0]
                route_distance_m = sum(dist_matrix[route_nodes[j]][route_nodes[j+1]] for j in range(len(route_nodes)-1))
                route_distance_km = route_distance_m / 1000
                route_time_hours = route_distance_km / st.session_state.kecepatan_rata
                route_cost = (route_distance_km * st.session_state.biaya_km) + (route_time_hours * st.session_state.biaya_jam)
                route_load = sum(demands[n] for n in route)
                
                route_flood_levels = [weather_data.get(n, {}).get('flood_level', '🟢 AMAN') for n in route]
                route_max_flood = max(route_flood_levels)
                flood_score = sum(1 if '🔴' in f else 0.5 if '🟠' in f else 0.25 if '🟡' in f else 0 for f in route_flood_levels)
                
                total_cost += route_cost
                total_distance_km += route_distance_km
                total_time_hours += route_time_hours
                flood_risk_score += flood_score
                
                route_names = [locations.iloc[n]['Nama'] for n in route_nodes]
                route_details.append({
                    'Truk': route_idx + 1,
                    'Rute': ' → '.join(route_names),
                    'Jarak_km': round(route_distance_km, 1),
                    'Waktu_jam': round(route_time_hours, 1),
                    'Muatan_kg': route_load,
                    'Biaya_Rp': int(route_cost),
                    'Flood_Max': route_max_flood
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
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_method = method_name
                st.session_state.best_result = method_analysis[method_name]
                best_metrics = method_analysis[method_name]

    # EXECUTIVE SUMMARY
    st.markdown("## 🏆 **HASIL TERBAIK**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card best-method">
            <h3 style="margin:0;font-size:1.4rem;">{best_method}</h3>
            <h2 style="margin:0;font-size:2rem;">🥇 #1</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("📏 Total Jarak", f"{best_metrics['total_distance_km']:.1f} km")
    
    with col3:
        st.metric("💰 Total Biaya", f"Rp {best_metrics['total_cost']:,.0f}")
    
    with col4:
        st.metric("⏱️ Total Waktu", f"{best_metrics['total_time_hours']:.1f} jam")
    
    with col5:
        st.metric("🚛 Truk", best_metrics['num_trucks'])

    # COMPARISON TABLE
    st.markdown("## 📊 **PERBANDINGAN 3 ALGORITMA**")
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
    st.dataframe(df_comparison, use_container_width=True, height=300)

    # BEST ROUTES
    st.markdown("## 🛣️ **RUTE TERBAIK**")
    best_details_df = pd.DataFrame(st.session_state.best_result['details'])
    st.dataframe(best_details_df, use_container_width=True)

    # WEATHER TABLE
    st.markdown("## 🌧️ **FORECAST BANJIR 6 JAM**")
    weather_display = []
    for i, row in locations.iterrows():
        wdata = weather_data[i]
        weather_display.append({
            '📍 Lokasi': row['Nama'],
            '📦 Demand': f"{row['Demand_kg']:,} kg",
            '🌧️ Hujan': f"{wdata['avg_precip_mmh']} mm/jam",
            '📊 Prob': f"{wdata['max_prob_pct']}%",
            '🚨 Status': wdata['flood_level'],
            '💡 Saran': wdata['flood_advice']
        })
    weather_df = pd.DataFrame(weather_display)
    st.dataframe(weather_df, use_container_width=True)

    # TOMTOM MAP
    st.markdown("## 🗺️ **PETA INTERAKTIF**")
    center_lat = locations['Latitude'].mean()
    center_lon = locations['Longitude'].mean()
    
    locations_json = locations[['Latitude', 'Longitude', 'Nama', 'Demand_kg']].to_json(orient='records', date_format='iso')
    weather_json = json.dumps(weather_data)
    routes_json = json.dumps(st.session_state.best_result['routes'])
    
    st.markdown(f"""
    <div id="tomtom-map" style="width: 100%; height: 500px;"></div>
    <link rel="stylesheet" href="https://api.tomtom.com/maps-sdk-for-web/cjs/6.x/6.15.0/maps.css" />
    <script src="https://api.tomtom.com/maps-sdk-for-web/cjs/6.x/6.15.0/maps-web.min.js"></script>
    <script>
    async function initMap() {{
        const map = tt.map('tomtom-map', {{
            key: '{TOMTOM_API_KEY}',
            center: [{center_lon}, {center_lat}],
            zoom: 11,
            language: 'id-ID'
        }});
        
        // Markers
        const locationsData = {locations_json};
        const weatherData = {weather_json};
        locationsData.forEach((loc, idx) => {{
            const wdata = weatherData[idx] || {{flood_color: '#059669'}};
            new tt.Marker({{color: wdata.flood_color}})
                .setLngLat([loc.Longitude, loc.Latitude])
                .addTo(map);
        }});
        
        // Routes
        const routes = {routes_json};
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c'];
        routes.forEach((route, idx) => {{
            const coords = [[{center_lon}, {center_lat}]];
            new tt.Polyline(coords, {{color: colors[idx % colors.length], width: 6}}).addTo(map);
        }});
    }}
    initMap();
    </script>
    """, unsafe_allow_html=True)

    # DOWNLOAD CENTER ✅ EXCEL + CSV READY
    st.markdown("## 📥 **DOWNLOAD REPORT**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = df_comparison.to_csv(index=False)
        st.download_button(
            label="📊 **CSV Perbandingan**",
            data=csv_data,
            file_name=f"vrp_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_comparison.to_excel(writer, sheet_name='Perbandingan_Algo', index=False)
            best_details_df.to_excel(writer, sheet_name='Rute_Terbaik', index=False)
            weather_df.to_excel(writer, sheet_name='Cuaca_Banjir', index=False)
            locations.to_excel(writer, sheet_name='Lokasi', index=False)
        st.download_button(
            label="📈 **Excel Lengkap**",
            data=excel_buffer.getvalue(),
            file_name=f"vrp_full_report_{best_method}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # Summary TXT
        summary_text = f"""VRP BANJIR LIVE PRO - {best_method}
TGL: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Biaya: Rp{best_metrics['total_cost']:,.0f}
Jarak: {best_metrics['total_distance_km']:.1f} km
Truk: {best_metrics['num_trucks']}
"""
        st.download_button(
            label="📄 **Summary TXT**",
            data=summary_text,
            file_name=f"vrp_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.success(f"✅ **OPTIMASI SELESAI!** {best_method} = TERBAIK | Biaya: Rp{best_cost:,.0f}")
    st.balloons()

else:
    st.markdown("## 🚀 **Welcome to VRP Banjir Live Pro**")
    st.info("""
    **Cara pakai:**
    1. 📍 Edit lokasi di sidebar
    2. ⚙️ Atur parameter truk/biaya  
    3. 🚀 Klik **JALANKAN OPTIMASI**
    4. 📥 Download **Excel/CSV lengkap**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem; font-family: Inter;'>
    🚛 VRP Banjir Live Pro | TomTom Maps + Open-Meteo | 
    <a href='https://streamlit.io/cloud' target='_blank'>Streamlit Cloud</a> © 2025
</div>
""", unsafe_allow_html=True)
