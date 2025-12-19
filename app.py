import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import io

st.set_page_config(
    page_title="🚛 VRP Banjir Live Pro - Leaflet Map",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.main-header { 
    font-family: 'Inter', sans-serif; font-size: 3rem !important; font-weight: 700 !important; 
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; 
    margin-bottom: 2rem !important; 
}
.metric-card { 
    background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); 
    padding: 1.5rem; border-radius: 16px; color: white; 
    box-shadow: 0 10px 30px rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.1); 
    transition: transform 0.3s ease; 
}
.metric-card:hover { transform: translateY(-5px); }
.best-method { 
    background: linear-gradient(145deg, #10b981 0%, #059669 100%) !important; 
    box-shadow: 0 15px 40px rgba(16,185,129,0.4) !important; 
}
.stDataFrame > div > div > div { 
    border-radius: 12px !important; overflow: hidden !important; 
    box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important; 
}
</style>
""", unsafe_allow_html=True)

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

# 6 VRP ALGORITHMS
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
                
                if mode == 'nearest':
                    sel_node = min(cands, key=lambda c: min([self.dist_matrix[c][r] for r in route] + [self.dist_matrix[c][0]]))
                    check_list = [sel_node]
                else:
                    check_list = cands[:3]
                
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

def get_weather_forecast(locations_df):
    weather_data = {}
    base_url = "https://api.open-meteo.com/v1/forecast"
    progress_bar = st.progress(0)
    
    for idx, row in locations_df.iterrows():
        progress_bar.progress((idx + 1) / len(locations_df))
        params = {
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'hourly': 'precipitation,precipitation_probability',
            'forecast_days': 1,
            'timezone': 'Asia/Jakarta'
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            hours = 6
            precip = data['hourly']['precipitation'][:hours]
            prob = data['hourly']['precipitation_probability'][:hours]
            
            avg_precip = np.mean(precip) if precip else 0
            max_prob = max(prob) if prob else 0
            
            if avg_precip > FLOOD_THRESHOLDS['high']['precip'] or max_prob > FLOOD_THRESHOLDS['high']['prob']:
                level = "🔴 BANJIR BESAR"
            elif avg_precip > FLOOD_THRESHOLDS['medium']['precip'] or max_prob > FLOOD_THRESHOLDS['medium']['prob']:
                level = "🟠 GENANGAN"
            elif avg_precip > FLOOD_THRESHOLDS['low']['precip'] or max_prob > FLOOD_THRESHOLDS['low']['prob']:
                level = "🟡 HUJAN LEBAT"
            else:
                level = "🟢 AMAN"
            
            weather_data[idx] = {
                'avg_precip': round(avg_precip, 1),
                'max_prob': int(max_prob),
                'flood_level': level
            }
        except:
            weather_data[idx] = {'avg_precip': 0, 'max_prob': 0, 'flood_level': "❓ ERROR"}
    progress_bar.empty()
    return weather_data

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

def create_leaflet_map(locations, routes, weather_data):
    """Create Leaflet HTML Map"""
    center_lat = locations['Latitude'].mean()
    center_lon = locations['Longitude'].mean()
    
    # Build markers
    markers_js = ""
    for i, row in locations.iterrows():
        w = weather_data.get(i, {})
        color = '#dc2626' if '🔴' in w.get('flood_level', '') else '#f97316' if '🟠' in w.get('flood_level', '') else '#eab308' if '🟡' in w.get('flood_level', '') else '#059669'
        
        markers_js += f"""
        L.circleMarker([{row['Latitude']}, {row['Longitude']}], {{
            radius: 8,
            fillColor: '{color}',
            color: 'white',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }}).bindPopup(`<b>{row['Nama']}</b><br>
        📦 {row['Demand_kg']:,} kg<br>
        🌧️ {w.get('avg_precip', 0)} mm/jam<br>
        {w.get('flood_level', '🟢 AMAN')}`).addTo(map);
        """
    
    # Build routes
    routes_js = ""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for route_idx, route in enumerate(routes):
        route_coords = [[locations.iloc[n]['Latitude'], locations.iloc[n]['Longitude']] for n in [0] + route + [0]]
        coords_str = str(route_coords).replace("'", "")
        routes_js += f"""
        L.polyline({route_coords}, {{
            color: '{colors[route_idx % len(colors)]}',
            weight: 4,
            opacity: 0.8
        }}).bindPopup('Truk {route_idx + 1}').addTo(map);
        """
    
    html_map = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; }}
            #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
            .legend {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 9999;
                font-family: Inter, sans-serif;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <div class="legend">
            <b>🌧️ Legend Banjir</b><br>
            <span style="color: #dc2626; font-weight: bold;">🔴 BANJIR BESAR</span><br>
            <span style="color: #f97316; font-weight: bold;">🟠 GENANGAN</span><br>
            <span style="color: #eab308; font-weight: bold;">🟡 HUJAN LEBAT</span><br>
            <span style="color: #059669; font-weight: bold;">🟢 AMAN</span>
        </div>
        <script>
            const map = L.map('map').setView([{center_lat}, {center_lon}], 11);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '© OpenStreetMap'
            }}).addTo(map);
            
            // Markers
            {markers_js}
            
            // Routes
            {routes_js}
        </script>
    </body>
    </html>
    """
    return html_map

# MAIN
st.markdown('<h1 class="main-header">🚛 VRP Banjir Live Pro<br><small>6 Algoritma + Leaflet Map + Flood Alert + Excel</small></h1>', unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown("## ⚙️ **KONFIGURASI**")
    
    edited_df = st.data_editor(
        st.session_state.locations_df,
        num_rows="dynamic",
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True),
            "Demand_kg": st.column_config.NumberColumn("Demand (kg)")
        },
        use_container_width=True
    )
    st.session_state.locations_df = edited_df
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: kapasitas = st.slider("💪 Kapasitas", 1000, 15000, KAPASITAS_TRUK_DEFAULT)
    with col2: kecepatan = st.slider("🚗 Kecepatan", 30, 80, 50)
    col1, col2 = st.columns(2)
    with col1: biaya_km = st.number_input("💰 Biaya/KM", 5000, 50000, BIAYA_PER_KM_DEFAULT)
    with col2: biaya_jam = st.number_input("⏰ Biaya/Jam", 20000, 150000, BIAYA_PER_JAM_DEFAULT)
    
    if st.button("🚀 **JALANKAN 6 ALGORITMA**", type="primary", use_container_width=True):
        st.session_state.run_optimization = True
        st.session_state.kapasitas = kapasitas
        st.session_state.kecepatan_rata = kecepatan
        st.session_state.biaya_km = biaya_km
        st.session_state.biaya_jam = biaya_jam
        st.rerun()

if st.session_state.get('run_optimization', False) and len(st.session_state.locations_df) > 1:
    
    with st.spinner("🔄 **Optimasi 6 Algoritma + Weather Analysis...**"):
        locations = st.session_state.locations_df.reset_index(drop=True)
        demands = locations['Demand_kg'].tolist()
        dist_matrix = get_distance_matrix(locations)
        weather_data = get_weather_forecast(locations)
        
        solver = VRP_MasterSolver(dist_matrix, demands, st.session_state.kapasitas)
        all_results = {
            'Nearest Neighbor ⚡': solver.solve_nn(),
            'Clarke-Wright ⭐': solver.solve_cw(),
            'Cheapest Insertion 💎': solver.solve_cheapest_insertion(),
            'Nearest Insertion': solver.solve_nearest_insertion(),
            'Farthest Insertion': solver.solve_farthest_insertion(),
            'Arbitrary Insertion': solver.solve_arbitrary_insertion()
        }
        
        method_analysis = {}
        best_method = None
        best_cost = float('inf')
        
        for method_name, routes in all_results.items():
            total_cost = total_dist = total_time = total_flood = 0
            route_details = []
            
            for r_idx, route in enumerate(routes):
                nodes = [0] + route + [0]
                dist_m = sum(dist_matrix[nodes[j]][nodes[j+1]] for j in range(len(nodes)-1))
                dist_km = dist_m / 1000
                time_h = dist_km / st.session_state.kecepatan_rata
                cost = (dist_km * st.session_state.biaya_km) + (time_h * st.session_state.biaya_jam)
                load = sum(demands[n] for n in route)
                
                flood_levels = [weather_data.get(n, {}).get('flood_level', '🟢 AMAN') for n in route]
                flood_max = max(flood_levels)
                flood_sc = sum(1 if '🔴' in f else 0.5 if '🟠' in f else 0.25 if '🟡' in f else 0 for f in flood_levels)
                
                total_cost += cost
                total_dist += dist_km
                total_time += time_h
                total_flood += flood_sc
                
                names = [locations.iloc[n]['Nama'] for n in nodes]
                route_details.append({
                    'Truk': r_idx + 1,
                    'Rute': ' → '.join(names),
                    'Jarak': f"{dist_km:.1f} km",
                    'Waktu': f"{time_h:.1f} jam",
                    'Biaya': f"Rp {int(cost):,}",
                    'Muatan': f"{load:,} kg",
                    'Banjir': flood_max
                })
            
            method_analysis[method_name] = {
                'cost': total_cost,
                'dist': total_dist,
                'time': total_time,
                'truk': len(routes),
                'details': route_details,
                'routes': routes,
                'flood_risk': total_flood / max(1, len(routes))
            }
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_method = method_name
                st.session_state.best_result = method_analysis[method_name]

    st.markdown("## 🏆 **HASIL OPTIMASI**")
    cols = st.columns(6)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card best-method">
            <h4>{best_method}</h4>
            <h2>🥇 #1</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]: st.metric("📏 Jarak", f"{st.session_state.best_result['dist']:.1f} km")
    with cols[2]: st.metric("💰 Biaya", f"Rp {int(best_cost):,}")
    with cols[3]: st.metric("⏱️ Waktu", f"{st.session_state.best_result['time']:.1f} jam")
    with cols[4]: st.metric("🚛 Truk", st.session_state.best_result['truk'])
    with cols[5]: 
        risk = st.session_state.best_result['flood_risk']
        st.metric("🌧️ Risk", "🟢" if risk < 0.3 else "🟡" if risk < 0.7 else "🔴")

    st.markdown("## 📊 **PERBANDINGAN 6 ALGORITMA**")
    comp_data = []
    for rank, (method, data) in enumerate(sorted(method_analysis.items(), key=lambda x: x[1]['cost']), 1):
        comp_data.append({
            f"#{rank}": method,
            "Biaya": f"Rp {int(data['cost']):,}",
            "Jarak": f"{data['dist']:.1f} km",
            "Truk": data['truk'],
            "Flood": f"{data['flood_risk']:.2f}"
        })
    st.dataframe(pd.DataFrame(comp_data), height=300, use_container_width=True)

    st.markdown("## 🛣️ **RUTE TERBAIK**")
    st.dataframe(pd.DataFrame(st.session_state.best_result['details']), use_container_width=True)

    st.markdown("## 🌧️ **FORECAST BANJIR**")
    weather_rows = []
    for i, row in locations.iterrows():
        w = weather_data[i]
        weather_rows.append({
            '📍 Lokasi': row['Nama'],
            '📦 Demand': f"{row['Demand_kg']:,} kg",
            '🌧️ Hujan': f"{w['avg_precip']} mm/jam",
            '📊 Prob': f"{w['max_prob']}%",
            '🚨 Status': w['flood_level']
        })
    st.dataframe(pd.DataFrame(weather_rows), use_container_width=True)

    st.markdown("## 🗺️ **PETA INTERAKTIF - LEAFLET**")
    leaflet_html = create_leaflet_map(locations, st.session_state.best_result['routes'], weather_data)
    st.components.v1.html(leaflet_html, height=600)

    st.markdown("## 📥 **DOWNLOAD**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = pd.DataFrame(comp_data).to_csv(index=False)
        st.download_button(
            "📊 CSV",
            csv,
            f"vrp_6algo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            pd.DataFrame(comp_data).to_excel(writer, '6_Algoritma', index=False)
            pd.DataFrame(st.session_state.best_result['details']).to_excel(writer, 'Rute', index=False)
            pd.DataFrame(weather_rows).to_excel(writer, 'Cuaca', index=False)
        st.download_button(
            "📈 Excel",
            excel_buffer.getvalue(),
            f"vrp_full_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        summary = f"VRP BANJIR - {best_method}\nBiaya: Rp{int(best_cost):,}\nJarak: {st.session_state.best_result['dist']:.1f} km\nTruk: {st.session_state.best_result['truk']}"
        st.download_button(
            "📄 Summary",
            summary,
            f"vrp_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
            use_container_width=True
        )
    
    st.success(f"✅ **SELESAI!** {best_method} = TERBAIK | Rp{int(best_cost):,}")
    st.balloons()

else:
    st.info("👈 Edit lokasi → Klik JALANKAN")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#64748b;padding:2rem">🚛 VRP 6 Algo + Leaflet Map + Excel | © 2025</div>', unsafe_allow_html=True)
