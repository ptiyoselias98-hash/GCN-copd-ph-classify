"""
Quantitative feature extraction for COPD-PH evolution analysis.

Extracts three categories of features:
  1. Vascular remodeling: BV5, BV10, pruning index, diameter distribution
  2. Lung parenchyma: LAA%, mean lung density, density histogram
  3. Airway remodeling: wall area %, Pi10, wall thickness ratio
"""

import numpy as np
from scipy import ndimage
from typing import Dict, List, Optional, Tuple


class VascularQuantifier:
    """Quantify pulmonary vascular remodeling features."""

    def __init__(self, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing)

    def compute_blood_volume_metrics(
        self,
        vessel_mask: np.ndarray,
        lung_mask: np.ndarray,
        bv5_threshold: float = 5.0,
        bv10_threshold: float = 10.0
    ) -> Dict[str, float]:
        """
        Compute BV5 and BV10 — blood volume in small vessels.

        BV5: fraction of vessel volume in vessels with cross-section < 5mm²
        BV10: fraction of vessel volume in vessels with cross-section < 10mm²

        These are key markers of distal vessel pruning in PH.
        """
        vessel_binary = (vessel_mask > 0).astype(np.uint8)
        lung_binary = (lung_mask > 0).astype(np.uint8)

        # Restrict to vessels within lung
        vessel_in_lung = vessel_binary * lung_binary

        # Distance transform → approximate local vessel radius
        dt = ndimage.distance_transform_edt(vessel_in_lung, sampling=self.spacing)
        vessel_diameters = 2.0 * dt  # diameter at each voxel

        voxel_vol = float(np.prod(self.spacing))  # mm³ per voxel
        lung_vol = float(np.sum(lung_binary)) * voxel_vol

        # Total vessel volume
        total_vessel_voxels = np.sum(vessel_in_lung > 0)
        total_vessel_vol = total_vessel_voxels * voxel_vol

        # BV5: vessels with diameter < 5mm
        small5 = (vessel_diameters > 0) & (vessel_diameters < bv5_threshold)
        bv5_vol = float(np.sum(small5)) * voxel_vol

        # BV10: vessels with diameter < 10mm
        small10 = (vessel_diameters > 0) & (vessel_diameters < bv10_threshold)
        bv10_vol = float(np.sum(small10)) * voxel_vol

        # Normalize by lung volume
        bv5_ratio = bv5_vol / max(lung_vol, 1e-6)
        bv10_ratio = bv10_vol / max(lung_vol, 1e-6)

        return {
            'bv5': bv5_ratio,
            'bv10': bv10_ratio,
            'bv5_absolute_ml': bv5_vol / 1000.0,
            'bv10_absolute_ml': bv10_vol / 1000.0,
            'total_vessel_volume_ml': total_vessel_vol / 1000.0,
            'lung_volume_ml': lung_vol / 1000.0,
            'vessel_to_lung_ratio': total_vessel_vol / max(lung_vol, 1e-6)
        }

    def compute_pruning_index(
        self,
        branches: List[Dict],
        branch_features: List[Dict],
        diameter_threshold: float = 2.0
    ) -> Dict[str, float]:
        """
        Compute pruning index — ratio of small distal vessels to total.

        A higher pruning index indicates more distal vessel loss,
        characteristic of PH progression.
        """
        if not branches or not branch_features:
            return {'pruning_index': 0.0, 'num_total_branches': 0,
                    'num_small_branches': 0}

        total = len(branch_features)
        small = sum(1 for f in branch_features
                    if f.get('diameter', 0) < diameter_threshold)

        diameters = [f.get('diameter', 0) for f in branch_features]

        return {
            'pruning_index': small / max(total, 1),
            'num_total_branches': total,
            'num_small_branches': small,
            'mean_diameter': float(np.mean(diameters)) if diameters else 0.0,
            'std_diameter': float(np.std(diameters)) if diameters else 0.0,
            'median_diameter': float(np.median(diameters)) if diameters else 0.0
        }

    def compute_artery_vein_ratio(
        self,
        artery_mask: np.ndarray,
        vein_mask: np.ndarray,
        lung_mask: np.ndarray
    ) -> Dict[str, float]:
        """Compute artery-to-vein volume ratio within the lung."""
        lung = lung_mask > 0
        artery_vol = float(np.sum((artery_mask > 0) & lung)) * float(np.prod(self.spacing))
        vein_vol = float(np.sum((vein_mask > 0) & lung)) * float(np.prod(self.spacing))

        return {
            'artery_volume_ml': artery_vol / 1000.0,
            'vein_volume_ml': vein_vol / 1000.0,
            'artery_vein_ratio': artery_vol / max(vein_vol, 1e-6)
        }


class ParenchymaQuantifier:
    """Quantify lung parenchymal damage features."""

    def __init__(self, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing)

    def compute_laa(
        self,
        ct_volume: np.ndarray,
        lung_mask: np.ndarray,
        threshold: float = -950.0
    ) -> Dict[str, float]:
        """
        Compute Low Attenuation Area percentage (LAA%).

        LAA% = fraction of lung voxels below -950 HU.
        Higher LAA% indicates more emphysematous destruction.
        """
        lung = lung_mask > 0
        lung_voxels = ct_volume[lung]

        if len(lung_voxels) == 0:
            return {'laa_pct': 0.0, 'mean_lung_density': 0.0}

        below_threshold = np.sum(lung_voxels < threshold)
        laa_pct = float(below_threshold) / float(len(lung_voxels)) * 100.0

        return {
            'laa_pct': laa_pct,
            'mean_lung_density': float(np.mean(lung_voxels)),
            'std_lung_density': float(np.std(lung_voxels)),
            'percentile_15': float(np.percentile(lung_voxels, 15)),
            'lung_volume_ml': float(np.sum(lung)) * float(np.prod(self.spacing)) / 1000.0
        }

    def compute_density_histogram(
        self,
        ct_volume: np.ndarray,
        lung_mask: np.ndarray,
        bins: int = 100,
        range_hu: Tuple[float, float] = (-1100.0, -200.0)
    ) -> Dict[str, np.ndarray]:
        """Compute HU density histogram within the lung."""
        lung_voxels = ct_volume[lung_mask > 0]
        hist, bin_edges = np.histogram(lung_voxels, bins=bins, range=range_hu)
        hist_normalized = hist / max(np.sum(hist), 1)

        return {
            'histogram': hist_normalized,
            'bin_edges': bin_edges,
            'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2
        }


class AirwayQuantifier:
    """Quantify airway remodeling features."""

    def __init__(self, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing)

    def compute_airway_metrics(
        self,
        airway_mask: np.ndarray,
        ct_volume: np.ndarray,
        lung_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute airway remodeling metrics.

        Features:
          - Wall Area % (WA%): wall area / total airway area
          - Wall Thickness ratio: wall thickness / outer diameter
          - Airway count: total visible airway branches

        Note: Precise Pi10 requires per-airway cross-section analysis.
        This provides population-level approximations.
        """
        airway = (airway_mask > 0).astype(np.uint8)
        lung = lung_mask > 0

        airway_in_lung = airway & lung

        # Total airway volume
        voxel_vol = float(np.prod(self.spacing))
        airway_vol = float(np.sum(airway_in_lung)) * voxel_vol

        # Estimate lumen vs wall using distance transform + thresholding
        dt = ndimage.distance_transform_edt(airway_in_lung, sampling=self.spacing)

        # Inner 50% by radius → lumen, outer 50% → wall (approximation)
        if np.max(dt) > 0:
            median_radius = np.median(dt[dt > 0])
            lumen = dt > median_radius
            wall = airway_in_lung & ~lumen

            lumen_vol = float(np.sum(lumen)) * voxel_vol
            wall_vol = float(np.sum(wall)) * voxel_vol

            wa_pct = wall_vol / max(airway_vol, 1e-6) * 100.0
            wt_ratio = wall_vol / max(lumen_vol + wall_vol, 1e-6)
        else:
            wa_pct = 0.0
            wt_ratio = 0.0

        # Airway density in CT
        airway_hu = ct_volume[airway_in_lung] if np.any(airway_in_lung) else np.array([0.0])

        # Connected component count → approximate branch count
        labeled, num_components = ndimage.label(airway_in_lung)

        return {
            'wall_area_pct': wa_pct,
            'wall_thickness_ratio': wt_ratio,
            'airway_volume_ml': airway_vol / 1000.0,
            'airway_count': num_components,
            'mean_airway_hu': float(np.mean(airway_hu)),
            'airway_to_lung_ratio': airway_vol / max(
                float(np.sum(lung)) * voxel_vol, 1e-6
            )
        }


def extract_all_features(
    ct_volume: np.ndarray,
    artery_mask: np.ndarray,
    vein_mask: np.ndarray,
    lung_mask: np.ndarray,
    airway_mask: np.ndarray,
    branches: List[Dict],
    branch_features: List[Dict],
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> Dict[str, Dict]:
    """
    Extract all quantitative features for a single patient.

    Returns:
        Dictionary with keys 'vascular', 'parenchyma', 'airway',
        each containing a dict of feature values.
    """
    vq = VascularQuantifier(spacing)
    pq = ParenchymaQuantifier(spacing)
    aq = AirwayQuantifier(spacing)

    # Combine artery + vein for total vessel mask
    vessel_mask = ((artery_mask > 0) | (vein_mask > 0)).astype(np.uint8)

    vascular = {}
    vascular.update(vq.compute_blood_volume_metrics(vessel_mask, lung_mask))
    vascular.update(vq.compute_pruning_index(branches, branch_features))
    vascular.update(vq.compute_artery_vein_ratio(artery_mask, vein_mask, lung_mask))

    parenchyma = pq.compute_laa(ct_volume, lung_mask)

    airway = aq.compute_airway_metrics(airway_mask, ct_volume, lung_mask)

    return {
        'vascular': vascular,
        'parenchyma': parenchyma,
        'airway': airway
    }
