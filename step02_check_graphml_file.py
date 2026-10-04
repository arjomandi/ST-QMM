import networkx as nx
import os
import pandas as pd
import math
import re
from pipeline_config import REPORTS_DIR, ensure_project_dirs, metric_crs_for_case, unified_graph_path
try:
	from shapely import wkt as shapely_wkt
except ImportError:
	shapely_wkt = None


def to_int_safely(value, default=0):
	if value is None:
		return default
	try:
		return int(value)
	except (TypeError, ValueError):
		try:
			return int(float(str(value).strip()))
		except (TypeError, ValueError):
			return default

CASE_NAMES = [
	"Rozelle_Interchange_NSW",
	"West_Gate_Tunnel_VIC",
	"NorthConnex_NSW",
	"Light_Horse_Interchange_NSW",
	"Domain_Tunnel_VIC",
	"M80_Princes_Freeway_VIC",
]

# M80/Princes is retained as an elevated multi-level control site; the other
# five study graphs must contain an audited subterranean state.
EXPECTED_TUNNEL_CASES = set(CASE_NAMES) - {"M80_Princes_Freeway_VIC"}

def to_flag(value):
	"""Convert various value types to boolean flag.

	Handles numeric 0/0.0 -> False, non-zero -> True, and both boolean
	strings ("true"/"yes"/"y") and numeric strings ("1.0"/"0.0") as
	produced by networkx.read_graphml round-tripping float attributes.
	"""
	if value is None:
		return False
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(math.isfinite(float(value)) and float(value) != 0.0)
	s = str(value).strip().lower()
	if s in {"", "false", "no", "n", "off", "none", "nan", "null"}:
		return False
	if s in {"true", "yes", "y", "on"}:
		return True
	try:
		number = float(s)
		return math.isfinite(number) and number != 0.0
	except (TypeError, ValueError):
		return False


def _attribute_text(value):
	if isinstance(value, (list, tuple, set)):
		return " ".join(_attribute_text(item) for item in value)
	return str(value or "").strip().lower()


def is_truck_restricted(data):
	"""Return true for edges that must not be used as synthetic truck truth."""
	highway = _attribute_text(data.get("highway"))
	geoscape_used = to_flag(data.get("source_geoscape", 0))
	status = " ".join(
		part for part in [
			_attribute_text(data.get("status")),
			_attribute_text(data.get("geoscape_status")) if geoscape_used else "",
		] if part
	)
	construction = _attribute_text(data.get("construction"))
	access_values = " ".join(
		_attribute_text(data.get(key)) for key in ("access", "vehicle", "motor_vehicle", "hgv")
	)
	if geoscape_used:
		access_values += " " + _attribute_text(data.get("geoscape_access_type"))
	trafficable = _attribute_text(data.get("geoscape_trafficable")) if geoscape_used else ""
	if "construction" in highway or construction not in {"", "no", "none", "nan"}:
		return True
	if any(token in status for token in ("closed", "proposed", "under construction")):
		return True
	access_tokens = set(re.sub(r"[^a-z0-9]+", " ", access_values).split())
	if access_tokens.intersection({"no", "private"}):
		return True
	if "management only" in access_values:
		return True
	if trafficable and trafficable not in {"2wd", "nan", "none"}:
		return True
	return False


def validate_case_graph(case_name):
	ensure_project_dirs()
	graphml_path = str(unified_graph_path(case_name))
	if not os.path.exists(graphml_path):
		print(f"{case_name}: missing file {graphml_path}")
		return {
			"case_study": case_name,
			"graphml_path": graphml_path,
			"exists": False,
			"valid_graph": False,
			"nodes": 0,
			"edges": 0,
			"tunnel_edges": 0,
			"bridge_edges": 0,
			"negative_layer_edges": 0,
			"missing_xy_nodes": 0,
			"missing_length_edges": 0,
			"validation_passed": False,
		}

	G = nx.read_graphml(graphml_path)
	if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
		print(f"{case_name}: invalid graph with zero nodes/edges")
		return {
			"case_study": case_name,
			"graphml_path": graphml_path,
			"exists": True,
			"valid_graph": False,
			"nodes": int(G.number_of_nodes()),
			"edges": int(G.number_of_edges()),
			"tunnel_edges": 0,
			"bridge_edges": 0,
			"negative_layer_edges": 0,
			"missing_xy_nodes": 0,
			"missing_length_edges": 0,
			"validation_passed": False,
		}

	tunnel_edges = 0
	bridge_edges = 0
	negative_layer_edges = 0
	missing_length_edges = 0
	missing_geometry_edges = 0
	invalid_numeric_edges = 0
	semantic_conflicts = 0
	geometry_endpoint_mismatches = 0
	reverse_oriented_geometries = 0
	geoscape_matched_edges = 0
	geoscape_confident_edges = 0
	geoscape_ambiguous_edges = 0
	missing_geoscape_provenance = 0
	unsafe_portal_snaps = 0
	injected_two_way_without_reverse = 0
	truck_restricted_edges = 0
	construction_edges = 0

	if G.is_multigraph():
		edge_iter = G.edges(keys=True, data=True)
	else:
		edge_iter = ((u, v, 0, d) for u, v, d in G.edges(data=True))

	for _u, _v, _k, d in edge_iter:
		if is_truck_restricted(d):
			truck_restricted_edges += 1
		if "construction" in _attribute_text(d.get("highway")):
			construction_edges += 1
		is_geoscape = to_flag(d.get("source_geoscape", 0))
		if is_geoscape:
			geoscape_matched_edges += 1
			if (
				not str(d.get("geoscape_road_pid", "")).strip()
				or not str(d.get("geoscape_match_method", "")).strip()
			):
				missing_geoscape_provenance += 1
		if to_flag(d.get("geoscape_match_confident", 0)):
			geoscape_confident_edges += 1
		if to_flag(d.get("geoscape_match_ambiguous", 0)):
			geoscape_ambiguous_edges += 1
		if to_flag(d.get("geoscape_injected", 0)):
			try:
				limit = float(d.get("portal_snap_max_distance_m", 30.0))
			except (TypeError, ValueError):
				limit = 30.0
			for prefix in ("start", "end"):
				if to_flag(d.get(f"portal_{prefix}_snapped", 0)):
					try:
						distance = float(d.get(f"portal_{prefix}_nearest_node_distance_m"))
						if not math.isfinite(distance) or distance > limit:
							unsafe_portal_snaps += 1
					except (TypeError, ValueError):
						unsafe_portal_snaps += 1
			if not to_flag(d.get("oneway", 0)):
				reverse_bundle = G.get_edge_data(_v, _u, default={})
				if G.is_multigraph():
					reverse_candidates = reverse_bundle.values()
				else:
					reverse_candidates = [reverse_bundle] if reverse_bundle else []
				road_pid = str(d.get("geoscape_road_pid", ""))
				# A reverse-direction edge covering the same authoritative
				# road_pid satisfies two-way completeness whether it was
				# itself Geoscape-injected or is a pre-existing OSM edge
				# later cross-tagged with the same road_pid during
				# OSM-Geoscape matching: what matters is that the reverse
				# path physically exists in the graph, not which pass added
				# it. Requiring geoscape_injected specifically on the
				# reverse candidate produced a false positive whenever OSM
				# already modelled the opposite direction of a two-way
				# Geoscape tunnel/road as its own (often one-way-tagged)
				# edge.
				has_matching_reverse = any(
					str(candidate.get("geoscape_road_pid", "")) == road_pid
					for candidate in reverse_candidates
					if road_pid
				)
				if not has_matching_reverse:
					injected_two_way_without_reverse += 1
		layer = to_int_safely(d.get("layer", 0), default=0)
		is_tunnel = to_flag(d.get("tunnel", 0))
		is_bridge = to_flag(d.get("bridge", 0))
		if is_tunnel or layer < 0:
			tunnel_edges += 1
		if is_bridge or layer > 0:
			bridge_edges += 1
		if layer < 0:
			negative_layer_edges += 1
		if d.get("length") is None:
			missing_length_edges += 1
		try:
			length = float(d.get("length"))
			bearing = float(d.get("bearing"))
			if not (length > 0 and 0 <= bearing <= 360):
				invalid_numeric_edges += 1
		except (TypeError, ValueError):
			invalid_numeric_edges += 1
		if is_tunnel and (layer >= 0 or is_bridge):
			semantic_conflicts += 1
		if is_bridge and layer <= 0:
			semantic_conflicts += 1
		geometry_text = d.get("geometry")
		if not geometry_text:
			missing_geometry_edges += 1
		elif shapely_wkt is not None:
			try:
				geometry = shapely_wkt.loads(str(geometry_text))
				ux, uy = float(G.nodes[_u]["x"]), float(G.nodes[_u]["y"])
				vx, vy = float(G.nodes[_v]["x"]), float(G.nodes[_v]["y"])
				start = geometry.coords[0]
				end = geometry.coords[-1]
				forward = ((start[0]-ux)**2+(start[1]-uy)**2)**0.5 + ((end[0]-vx)**2+(end[1]-vy)**2)**0.5
				reverse = ((start[0]-vx)**2+(start[1]-vy)**2)**0.5 + ((end[0]-ux)**2+(end[1]-uy)**2)**0.5
				if forward > 2.0:
					geometry_endpoint_mismatches += 1
					if reverse <= 2.0:
						reverse_oriented_geometries += 1
			except Exception:
				geometry_endpoint_mismatches += 1

	missing_xy_nodes = sum(
		1
		for _node, data in G.nodes(data=True)
		if data.get("x") is None or data.get("y") is None
	)

	print(
		f"{case_name}: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
		f"tunnel_edges={tunnel_edges} bridge_edges={bridge_edges} "
		f"negative_layer_edges={negative_layer_edges} missing_xy_nodes={missing_xy_nodes} "
		f"missing_length_edges={missing_length_edges} geoscape_matched={geoscape_matched_edges} "
		f"geoscape_confident={geoscape_confident_edges} geoscape_ambiguous={geoscape_ambiguous_edges} "
		f"truck_restricted_edges={truck_restricted_edges}"
	)

	expected_crs = metric_crs_for_case(case_name)
	crs_matches = str(G.graph.get("crs", "")).upper() == expected_crs
	directed = bool(G.is_directed())
	weak_components = nx.number_weakly_connected_components(G) if directed else nx.number_connected_components(G)
	case_tunnel_warning = case_name in EXPECTED_TUNNEL_CASES and tunnel_edges == 0
	passed = (
		missing_xy_nodes == 0 and missing_length_edges == 0 and
		missing_geometry_edges == 0 and invalid_numeric_edges == 0 and
		semantic_conflicts == 0 and geometry_endpoint_mismatches == 0 and
		reverse_oriented_geometries == 0 and missing_geoscape_provenance == 0 and
		unsafe_portal_snaps == 0 and injected_two_way_without_reverse == 0 and
		crs_matches and directed and weak_components == 1 and not case_tunnel_warning
	)
	return {
		"case_study": case_name,
		"graphml_path": graphml_path,
		"exists": True,
		"valid_graph": True,
		"nodes": int(G.number_of_nodes()),
		"edges": int(G.number_of_edges()),
		"tunnel_edges": int(tunnel_edges),
		"bridge_edges": int(bridge_edges),
		"negative_layer_edges": int(negative_layer_edges),
		"missing_xy_nodes": int(missing_xy_nodes),
		"missing_length_edges": int(missing_length_edges),
		"missing_geometry_edges": int(missing_geometry_edges),
		"invalid_numeric_edges": int(invalid_numeric_edges),
		"semantic_conflicts": int(semantic_conflicts),
		"geometry_endpoint_mismatches": int(geometry_endpoint_mismatches),
		"reverse_oriented_geometries": int(reverse_oriented_geometries),
		"geoscape_matched_edges": int(geoscape_matched_edges),
		"geoscape_confident_edges": int(geoscape_confident_edges),
		"geoscape_ambiguous_edges": int(geoscape_ambiguous_edges),
		"missing_geoscape_provenance": int(missing_geoscape_provenance),
		"unsafe_portal_snaps": int(unsafe_portal_snaps),
		"injected_two_way_without_reverse": int(injected_two_way_without_reverse),
		"truck_restricted_edges": int(truck_restricted_edges),
		"construction_edges": int(construction_edges),
		"directed": directed,
		"weak_components": int(weak_components),
		"crs": str(G.graph.get("crs", "")),
		"expected_crs": expected_crs,
		"crs_matches": bool(crs_matches),
		"case_tunnel_warning": bool(case_tunnel_warning),
		"validation_passed": bool(passed),
	}


if __name__ == "__main__":
	all_ok = True
	rows = []
	for case_name in CASE_NAMES:
		result = validate_case_graph(case_name)
		rows.append(result)
		all_ok = all_ok and bool(result.get("validation_passed", False))

	out_csv = REPORTS_DIR / "step02_graphml_validation_all_cases.csv"
	pd.DataFrame(rows).to_csv(out_csv, index=False)
	print(f"Saved Step02 validation CSV: {out_csv}")

	if not all_ok:
		raise SystemExit(1)
