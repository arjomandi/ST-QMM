import osmnx as ox #ignore: F401
import networkx as nx
from pipeline_config import ensure_project_dirs, raw_graph_path

# Define the Hard-Case Locations (Center coordinates)
# We use precise lat/lon to guarantee we center exactly on the overlapping junctions.
hard_case_locations = {
    "Rozelle_Interchange_NSW": (-33.8702, 151.1722),
    "West_Gate_Tunnel_VIC": (-37.824208, 144.871139),   # Southern West Gate Tunnel portal
    "NorthConnex_NSW": (-33.7589, 151.0464),
    "Light_Horse_Interchange_NSW": (-33.7980, 150.8540),
    "Domain_Tunnel_VIC": (-37.8257, 144.9799),          # Melbourne CBD Domain Tunnel area
    "M80_Princes_Freeway_VIC": (-37.8276, 144.8169)
}

def pull_hard_case_maps(locations_dict, radius_meters=2000):
    """
    Downloads and saves the heavy-vehicle navigable road network for specific coordinates.
    radius_meters: 2000m is optimal to capture the long overlapping entry/exit ramps.
    """
    # OSMnx config() is deprecated in recent versions. Logging and caching are handled automatically.
    ensure_project_dirs()
    
    for name, (lat, lon) in locations_dict.items():
        print(f"--- Fetching 2D Map Graph for: {name} ---")
        
        try:
            # network_type="drive" filters out walking/bike paths and keeps heavy vehicle roads
            # simplify=True cleans up unnecessary intermediate nodes so the PQC doesn't get overloaded
            G = ox.graph_from_point(
                (lat, lon), 
                dist=radius_meters, 
                network_type='drive', 
                simplify=True
            )
            
            # Save the graphml file locally so we can load it instantly in Step 3
            file_name = raw_graph_path(name)
            ox.save_graphml(G, filepath=str(file_name))
            
            print(f"Successfully saved {file_name}")
            print(f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)}\n")
            
            # Optional: If you want to visualize the spaghetti complexity right away
            # ox.plot_graph(G, node_size=1, edge_color='cyan', bgcolor='black', save=True, filepath=f"{name}_plot.png")
            
        except Exception as e:
            print(f"Failed to fetch {name}. Error: {e}\n")

if __name__ == "__main__":
    # Execute the pull
    pull_hard_case_maps(hard_case_locations, radius_meters=2000) # Bumped to 2km for massive interchanges
