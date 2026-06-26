"""HotSpotter-style local matching utilities."""

from .hotspotter_pipeline import (
    HotspotterConfig,
    HotspotterFeature,
    HotspotterFeatureExtractionReport,
    HotspotterIndex,
    HotspotterPrescoreCandidate,
    create_match_board,
    extract_hesaff_features,
    extract_hesaff_features_with_report,
    prescore_results_to_dataframe,
    query_hotspotter_all,
    rank_results_to_dataframe,
    unique_pair_results_to_dataframe,
)

__all__ = [
    "HotspotterConfig",
    "HotspotterFeature",
    "HotspotterFeatureExtractionReport",
    "HotspotterIndex",
    "HotspotterPrescoreCandidate",
    "create_match_board",
    "extract_hesaff_features",
    "extract_hesaff_features_with_report",
    "prescore_results_to_dataframe",
    "query_hotspotter_all",
    "rank_results_to_dataframe",
    "unique_pair_results_to_dataframe",
]
