import streamlit as st
import pandas as pd
import numpy as np
import googlemaps
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import plotly.express as px  # NOUVEAU POUR LES GRAPHIQUES

# --- CONFIGURATION ---
st.set_page_config(page_title="Mobilité CHU Rennes", layout="wide", page_icon="🏥")
st.title("Dashboard Mobilité - CHU Rennes")

# --- DONNÉES DE RÉFÉRENCE (ADEME & EMISSIONS) ---

# 1. Facteurs Mobilité Pendulaire (kg CO2/km)
EMISSIONS_DEPLACEMENT = {
    "Voiture Légère (Solo)": 0.12, 
    "SUV (Solo)": 0.22, 
    "Covoiturage (3 pers)": 0.04, 
    "Transports en commun": 0.05,
    "Vélo / Marche": 0.0
}

# 2. Facteurs Flotte Véhicules (g CO2/km) - Estimations basées sur moyennes ADEME
# Source : https://carlabelling.ademe.fr/chiffrescles/r/moyenneEmissionCo2Gamme
ADEME_REF = {
    "citadines": 110,    # Moyenne ~Segment B
    "berlines": 130,     # Moyenne ~Segment C/D
    "suv": 155,          # Moyenne Tout-Terrain/SUV
    "utilitaires": 185,  # VUL Diesel moyen
    "materiel": 0        # Treuils, remorques (pas de moteur thermique propre)
}
VALEUR_DEFAUT = 140

# --- PARAMÈTRES DE LA CARTE (BRETAGNE) ---
BZH_COORDS = [48.1172, -1.6777] # Rennes
MIN_ZOOM = 8 
MAX_BOUNDS = True
MIN_LAT, MAX_LAT = 47.0, 49.0
MIN_LON, MAX_LON = -5.5, -0.5
CHU_COORDS = [48.1206, -1.6800] # Pontchaillou

def create_base_map():
    """Crée une carte folium configurée pour la Bretagne."""
    m = folium.Map(
        location=BZH_COORDS,
        zoom_start=9,
        min_zoom=MIN_ZOOM,
        max_bounds=MAX_BOUNDS,
        min_lat=MIN_LAT,
        max_lat=MAX_LAT,
        min_lon=MIN_LON,
        max_lon=MAX_LON,
        tiles="OpenStreetMap" 
    )
    return m

def get_dist_km(lat1, lon1, lat2, lon2):
    """Calcul distance à vol d'oiseau."""
    R = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    """Charge les données de mobilité pendulaire."""
    API_KEY = "AIzaSyBvlr2tCs3RdAaELTMRShYKIJsHv8Vy8W8" # Attention: clé exposée
    try:
        # Tente de charger le fichier de mobilité
        df = pd.read_excel('data/MOBILITE_AVEC_TOUT.xlsx')
    except FileNotFoundError:
        return pd.DataFrame() # Retourne vide si absent pour éviter crash global

    if not df.empty and 'lat' not in df.columns:
        # Géocodage (partie existante conservée)
        gmaps = googlemaps.Client(key=API_KEY)
        def get_coords(ville):
            try:
                res = gmaps.geocode(ville)
                loc = res[0]['geometry']['location']
                return loc['lat'], loc['lng']
            except:
                return None, None
        
        coords = df['Commune_FR'].apply(get_coords)
        df['lat'] = [c[0] for c in coords]
        df['lon'] = [c[1] for c in coords]

    return df.dropna(subset=['lat', 'lon'])

@st.cache_data
def load_fleet_data():
    """Charge les données de la flotte de véhicules."""
    try:
        # Adaptez le chemin si nécessaire
        df = pd.read_excel('data/VEHICULES_segmentation_llm.xlsx')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# Chargement
df = load_data()
df_flotte = load_fleet_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Paramètres")
    jour_choisi = st.selectbox("Jour analysé", ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche'])
    st.divider()
    
    # Calculateur (Code existant conservé)
    st.header("Calculateur Trajet")
    if not df.empty:
        calc_ville = st.selectbox("Ville de départ", sorted(df['Commune'].unique()))
        calc_dest = st.radio("Destination", ["CHU Pontchaillou", "Hôpital Sud"])
        calc_mode = st.selectbox("Moyen de Transport", list(EMISSIONS_DEPLACEMENT.keys()))
        
        if calc_ville and calc_dest:
            col_dist_cible = "distance_CHU_pontchaillou_voiture" if calc_dest == "CHU Pontchaillou" else "distance_CHU_sud_voiture"
            dist_km = df.loc[df['Commune'] == calc_ville, col_dist_cible].values[0]
            co2_trajet = dist_km * 2 * EMISSIONS_DEPLACEMENT[calc_mode]
            st.write(f"📍 Distance A/R : **{dist_km*2:.1f} km**")
            st.metric(f"Empreinte CO₂", f"{co2_trajet:.2f} kg")

# --- PRÉPARATION DATA JOUR ---
if not df.empty:
    col_nb_emp = f"nb_employes_{jour_choisi}"
    col_dist_ref = "distance_CHU_pontchaillou_voiture"
    df['Employes_Jour'] = df[col_nb_emp].fillna(0)
    df['CO2_Actuel'] = df[col_dist_ref] * 2 * df['Employes_Jour'] * 0.218
    df_jour = df[df['Employes_Jour'] > 0].copy()
else:
    df_jour = pd.DataFrame()

# --- INTERFACE PRINCIPALE ---
tab1, tab2, tab3 = st.tabs(["Carte de Chaleur", "Simulation Covoiturage", "Analyse Flotte Véhicules"])

# =========================================================
# ONGLET 1 : HEATMAP
# =========================================================
with tab1:
    if df_jour.empty:
        st.warning("Données de mobilité non disponibles.")
    else:
        total_co2 = df_jour['CO2_Actuel'].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("CO₂ Total (Jour)", f"{total_co2/10000:.2f} Tonnes")
        c2.metric("Employés Mobiles", f"{int(df_jour['Employes_Jour'].sum())}")
        c3.metric("Ville la plus polluante", df_jour.loc[df_jour['CO2_Actuel'].idxmax(), 'Commune'])

        m = create_base_map()
        HeatMap(df_jour[['lat', 'lon', 'CO2_Actuel']].values.tolist(), radius=15, blur=10, gradient={0.4: 'blue', 0.6: 'lime', 1: 'red'}).add_to(m)
        folium.Marker(CHU_COORDS, tooltip="Pontchaillou", icon=folium.Icon(color="gray", icon="hospital-o", prefix='fa')).add_to(m)
        st_folium(m, width="100%", height=600)

# =========================================================
# ONGLET 2 : OPTIMISATION
# =========================================================
with tab2:
    if df_jour.empty:
        st.warning("Données de mobilité non disponibles.")
    else:
        st.markdown("##### Regroupement par Hubs (Trajet Direct)")
        col_s1, col_s2 = st.columns(2)
        nb_hubs = col_s1.slider("Nombre de Hubs", 5, 50, 25)
        taux_remp = col_s2.slider("Personnes par voiture", 1, 5, 3)

        # Clustering
        kmeans = KMeans(n_clusters=nb_hubs, random_state=42, n_init=10)
        df_jour['cluster'] = kmeans.fit_predict(df_jour[['lat', 'lon']], sample_weight=df_jour['Employes_Jour'])
        centers = kmeans.cluster_centers_

        m2 = create_base_map()
        couleurs = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'darkblue', 'pink']
        co2_optimise = 0
        
        folium.Marker(CHU_COORDS, tooltip="CHU", icon=folium.Icon(color="black", icon="hospital-o", prefix='fa')).add_to(m2)

        for i, center in enumerate(centers):
            groupe = df_jour[df_jour['cluster'] == i]
            dist_hub_chu = get_dist_km(center[0], center[1], CHU_COORDS[0], CHU_COORDS[1])
            
            # Si Hub < 20km, pas de covoit
            if dist_hub_chu < 20:
                for idx, row in groupe.iterrows():
                    co2_optimise += row[col_dist_ref] * 2 * row['Employes_Jour'] * 0.218
                continue

            carpoolers = []
            cluster_color = couleurs[i % len(couleurs)]

            for idx, row in groupe.iterrows():
                d_commune_hub = get_dist_km(row['lat'], row['lon'], center[0], center[1])
                d_commune_chu = get_dist_km(row['lat'], row['lon'], CHU_COORDS[0], CHU_COORDS[1])
                
                # Règle de détour (5%)
                if (d_commune_hub + dist_hub_chu) > (d_commune_chu * 1.05):
                    co2_optimise += row[col_dist_ref] * 2 * row['Employes_Jour'] * 0.218
                else:
                    carpoolers.append(row)
                    folium.PolyLine([[row['lat'], row['lon']], center], color=cluster_color, weight=1, opacity=0.6).add_to(m2)

            nb_carpoolers = sum([r['Employes_Jour'] for r in carpoolers])
            if nb_carpoolers > 0:
                folium.Marker(center, icon=folium.Icon(color=cluster_color, icon='car', prefix='fa'), tooltip=f"Hub {i}").add_to(m2)
                # Calcul navettes
                avg_dist = np.mean([r[col_dist_ref] for r in carpoolers])
                nb_voitures = np.ceil(nb_carpoolers / taux_remp)
                co2_optimise += nb_voitures * avg_dist * 2 * 0.218 * 1.05

        gain = total_co2 - co2_optimise
        k1, k2, k3 = st.columns(3)
        k1.metric("CO₂ Actuel", f"{total_co2/10000:.1f} T")
        k2.metric("CO₂ Optimisé", f"{co2_optimise/10000:.1f} T")
        k3.metric("Gain", f"-{gain/10000:.1f} T", delta=f"{(gain/total_co2)*100:.1f}%")
        st_folium(m2, width="100%", height=500)

# =========================================================
# ONGLET 3 : ANALYSE FLOTTE VÉHICULES (NOUVEAU)
# =========================================================
with tab3:
    st.header("Analyse de la Flotte Véhicules")
    
    if df_flotte.empty:
        st.error("❌ Le fichier 'VEHICULES_segmentation_llm.xlsx' est introuvable ou vide.")
    else:
        # 1. Nettoyage et Préparation
        # On renomme pour faciliter la lecture si besoin, ou on utilise direct
        if 'segment_llm' in df_flotte.columns:
            # Nettoyage des segments (minuscules, strip)
            df_flotte['Segment_Clean'] = df_flotte['segment_llm'].astype(str).str.lower().str.strip()
            
            # Mapping des émissions CO2 (g/km)
            def get_emission(seg):
                # On cherche si un mot clé est dans le segment (ex: "utilitaire" dans "utilitaires légers")
                for key, val in ADEME_REF.items():
                    if key in seg:
                        return val
                return VALEUR_DEFAUT if seg not in ['nan', 'none', ''] else 0

            df_flotte['Emission_g_km'] = df_flotte['Segment_Clean'].apply(get_emission)
            
            # On filtre pour ne garder que ce qui a un segment valide pour l'affichage
            df_viz = df_flotte[df_flotte['segment_llm'].notna()].copy()
            
            # --- KPI GLOBAL ---
            # On exclut le "matériel" (Emission = 0) pour calculer la moyenne véhicule
            avg_co2 = df_viz[df_viz['Emission_g_km'] > 0]['Emission_g_km'].mean()
            
            col_k1, col_k2, col_k3 = st.columns(3)
            col_k1.metric("Nombre de Véhicules/Matériel", len(df_viz))
            col_k2.metric("Moyenne Émissions Flotte", f"{avg_co2:.0f} gCO₂/km", help="Moyenne hors matériel (treuils, etc.)")
            col_k3.markdown("") # Placeholder contextuel pour l'échelle
            
            st.divider()

            # --- VISUALISATIONS ---
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.subheader("Composition de la Flotte")
                # Groupement par segment
                counts = df_viz['segment_llm'].value_counts().reset_index()
                counts.columns = ['Segment', 'Nombre']
                
                fig_pie = px.pie(counts, values='Nombre', names='Segment', 
                                 title='Répartition par Segment',
                                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_g2:
                st.subheader("Impact CO₂ estimé par Segment")
                st.caption("Basé sur les moyennes ADEME par gamme (g/km)")
                
                # On exclut le matériel (0 émission) pour ce graph pour ne pas fausser l'échelle
                df_co2 = df_viz[df_viz['Emission_g_km'] > 0]
                
                # Moyenne par segment
                avg_by_seg = df_co2.groupby('segment_llm')['Emission_g_km'].mean().reset_index()
                avg_by_seg = avg_by_seg.sort_values('Emission_g_km')
                
                fig_bar = px.bar(avg_by_seg, x='segment_llm', y='Emission_g_km',
                                 title="Émissions moyennes (g CO₂/km)",
                                 labels={'segment_llm': 'Segment', 'Emission_g_km': 'g CO₂/km'},
                                 color='Emission_g_km',
                                 color_continuous_scale='RdYlGn_r') # Vert au Rouge
                

                st.plotly_chart(fig_bar, use_container_width=True)

            st.write("---")
            st.write("**Aperçu des données détaillées :**")
            st.dataframe(df_viz[['Libellé du bien (fi)', 'segment_llm', 'Emission_g_km']].head(10))
            
        else:
            st.error("La colonne 'segment_llm' est absente du fichier.")