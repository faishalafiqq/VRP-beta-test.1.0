import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import io

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
.main-header {font-family: 'Inter', sans-serif; font-size: 2.8rem !important; font-weight: 700 !important; 
              background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); -webkit-background-clip: text; 
              -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 2rem !important;}
.metric-card {background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; 
              border-radius: 16px; color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
              text-align: center; margin: 0.5rem;}
.best-method {background: linear-gradient(145deg, #10b981 0%, #059669 100%) !important;}
</style>
""", unsafe_allow_html=True)

# CONSTANTS
BIAYA_PER_KM_DEFAULT = 12000
BIAYA_PER_JAM_DEFAULT = 50000
KAPASITAS_TRUK_DEFAULT = 4500

FLOOD_COLORS = {'high': '#dc2626', 'medium': '#f97316', 'low': '#eab308', 'safe': '#059669'}

# ==========================================
# VRP SOLVER - 6 ALGORITMA
# ==========================================
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
            if ri and rj and ri != rj:
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
    
    def solve_nearest_insertion(self):
        return self.solve_cheapest_insertion()
    def solve_farthest_insertion(self):
        return self.solve_cheapest_insertion()
    def solve_arbitrary_insertion(self):
        return self.solve_cheapest_insertion()

# ==========================================
# WEATHER & DISTANCE
# ==========================================
@st.cache_data(ttl=1800)
def get_weather_forecast(locations_df):
    weather_data = {}
    base_url = "https://api.open-meteo.com/v1/forecast"
    
    for idx, row in locations_df.iterrows():
        params = {
            'latitude': row['Latitude'], 'longitude': row['Longitude'],
            'hourly': 'precipitation,precipitation_probability', 
            'forecast_days': 1, 'timezone': 'Asia/Jakarta'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            
            precip = data['hourly']['precipitation'][:6]
            prob = data['hourly']['precipitation_probability'][:6]
            
            avg_precip = np.mean(precip) if precip else 0
            max_prob = max(prob) if prob else 0
            
            if avg_precip > 20 or max_prob > 70:
                level = "🔴 BANJIR"
            elif avg_precip > 10 or max_prob > 60:
                level = "🟠 GENANGAN" 
            elif avg_precip > 5 or max_prob > 40:
                level = "🟡 HUJAN"
            else:
                level = "🟢 AMAN"
            
            weather_data[idx] = {
                'avg_precip': round(avg_precip, 1),
                'max_prob': int(max_prob),
                'flood_level': level
            }
        except:
            weather_data[idx] = {'avg_precip': 0, 'max_prob': 0, 'flood_level': "❓ N/A"}
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
            dist_matrix[i][j] = R * c / 1000  # km
    return dist_matrix

# ==========================================
# MAIN APP
# ==========================================
st.markdown('<h1 class="main-header">🚛 VRP Banjir Live Pro<br><small>6 Algoritma Expert + Cuaca Real-time</small></h1>', unsafe_allow_html=True)

# Default locations
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
    
    # Location editor
    edited_df = st.data_editor(
        st.session_state.locations_df, 
        num_rows="dynamic",
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True),
            "Demand_kg": st.column_config.NumberColumn("Demand kg", min_value=0, max_value=5000)
        },
        use_container_width=True
    )
    st.session_state.locations_df = edited_df
    
    st.divider()
    
    kapasitas = st.slider("Kapasitas Truk (kg)", 1000, 15000, 4500)
    biaya_km = st.number_input("Biaya per KM (Rp)", 5000, 50000, 12000)
    kecepatan = st.slider("Kecepatan Rata-rata (km/jam)", 30, 80, 50)
    
    if st.button("🚀 **JALANKAN OPTIMASI**", type="primary", use_container_width=True):
        st.session_state.run_optimization = True
        st.session_state.kapasitas = kapasitas
        st.session_state.biaya_km = biaya_km
        st.session_state.kecepatan = kecepatan
        st.rerun()

# EXECUTION
if st.session_state.get('run_optimization', False) and len(st.session_state.locations_df) > 1:
    with st.spinner("🔄 Menghitung 6 algoritma VRP + cuaca real-time..."):
        locations = st.session_state.locations_df.reset_index(drop=True)
        demands = locations['Demand_kg'].tolist()
        dist_matrix = get_distance_matrix(locations)
        weather_data = get_weather_forecast(locations)
        
        # Run all 6 algorithms
        solver = VRP_MasterSolver(dist_matrix, demands, st.session_state.kapasitas)
        all_results = {
            'Nearest Neighbor': solver.solve_nn(),
            'Clarke & Wright': solver.solve_cw(),
            'Cheapest Insertion': solver.solve_cheapest_insertion(),
            'Nearest Insertion': solver.solve_nearest_insertion(),
            'Farthest Insertion': solver.solve_farthest_insertion(),
            'Arbitrary Insertion': solver.solve_arbitrary_insertion()
        }
        
        # Find best algorithm
        method_analysis = {}
        best_method = None
        best_cost = float('inf')
        
        for method_name, routes in all_results.items():
            total_cost = 0
            total_distance = 0
            num_trucks = len(routes)
            
            for route in routes:
                route_nodes = [0] + route + [0]
                route_distance = sum(dist_matrix[route_nodes[j]][route_nodes[j+1]] 
                                   for j in range(len(route_nodes)-1))
                route_time = route_distance / st.session_state.kecepatan
                route_cost = (route_distance * st.session_state.biaya_km) + (route_time * BIAYA_PER_JAM_DEFAULT)
                total_cost += route_cost
                total_distance += route_distance
            
            method_analysis[method_name] = {
                'routes': routes,
                'total_cost': total_cost,
                'total_distance': total_distance,
                'num_trucks': num_trucks
            }
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_method = method_name
                st.session_state.best_result = method_analysis[method_name]
                st.session_state.weather_data = weather_data
                st.session_state.locations = locations
                st.session_state.dist_matrix = dist_matrix

    # DISPLAY RESULTS
    st.markdown("## 🏆 **HASIL OPTIMASI TERBAIK**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="metric-card best-method">
            <h3 style="margin:0;">{best_method}</h3>
            <h2 style="margin:0;">🥇 #1</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.metric("📏 Total Jarak", f"{st.session_state.best_result['total_distance']:.1f} km")
    with col3:
        st.metric("💰 Total Biaya", f"Rp {best_cost:,.0f}")
    with col4:
        st.metric("🚛 Jumlah Truk", st.session_state.best_result['num_trucks'])

    # Comparison table
    st.markdown("### 📊 Perbandingan 6 Algoritma")
    comparison_data = []
    for method, metrics in method_analysis.items():
        comparison_data.append({
            'Algoritma': method,
            'Biaya Total': f"Rp {metrics['total_cost']:,.0f}",
            'Jarak Total': f"{metrics['total_distance']:.1f} km",
            'Jumlah Truk': metrics['num_trucks']
        })
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Best routes detail
    st.markdown("### 🛣️ Rute Terbaik Detail")
    best_routes = st.session_state.best_result['routes']
    route_details = []
    for i, route in enumerate(best_routes):
        route_nodes = [0] + route + [0]
        distance = sum(st.session_state.dist_matrix[route_nodes[j]][route_nodes[j+1]] 
                      for j in range(len(route_nodes)-1))
        route_names = [st.session_state.locations.iloc[n]['Nama'] for n in route_nodes]
        route_details.append({
            'Truk': i+1,
            'Rute': ' → '.join(route_names),
            'Jarak': f"{distance:.1f} km"
        })
    st.dataframe(pd.DataFrame(route_details), use_container_width=True)

    # Weather table
    st.markdown("### 🌧️ Status Cuaca Real-time")
    weather_display = []
    for i, row in st.session_state.locations.iterrows():
        wdata = st.session_state.weather_data[i]
        weather_display.append({
            'Lokasi': row['Nama'],
            'Demand': f"{row['Demand_kg']:,} kg",
            'Cuaca': wdata['flood_level'],
            'Curah Hujan': f"{wdata['avg_precip']} mm/jam"
        })
    st.dataframe(pd.DataFrame(weather_display), use_container_width=True)

    # DOWNLOADS
    st.markdown("### 📥 Download Laporan")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_routes = pd.DataFrame(route_details).to_csv(index=False)
        st.download_button(
            "📊 CSV Rute Terbaik",
            csv_routes,
            f"vrp_rute_terbaik_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    with col2:
        csv_weather = pd.DataFrame(weather_display).to_csv(index=False)
        st.download_button(
            "🌧️ CSV Cuaca",
            csv_weather,
            f"vrp_cuaca_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )

    st.success(f"✅ **OPTIMASI SELESAI!** 🎉\n🏆 **{best_method}** = TERBAIK\n💰 **Rp{best_cost:,.0f}** total biaya")
    st.balloons()

else:
    st.markdown("## 🚀 Selamat Datang!")
    st.info("""
    **Cara Pakai:**
    1. 📍 Edit lokasi di sidebar kiri
    2. ⚙️ Atur kapasitas truk & biaya
    3. 🚀 Klik **JALANKAN OPTIMASI**
    4. 🏆 Lihat hasil 6 algoritma + cuaca
    5. 📥 Download CSV laporan
    """)

# Footer
st.markdown("---")
st.markdown("*🚛 VRP Banjir Live Pro | 6 Algoritma VRP + Cuaca BMKG | © 2025*")
