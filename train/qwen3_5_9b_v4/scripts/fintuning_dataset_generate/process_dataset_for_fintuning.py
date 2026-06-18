from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


GEOMETRY_RELATIONS = {
    -1: "vertical_length",
    -2: "horizontal_length",
    -7: "lead_thickness",
    -10: "diameter",
    -11: "corner_radius_or_notch",
    -12: "groove_depth_or_protrusion_length",
    -15: "noise",
}

POSITION_RELATIONS = {
    -3: "top",
    -4: "left",
    -5: "bottom",
    -6: "right",
    -8: "vertical_centerline",
    -9: "horizontal_centerline",
}

VERTICAL_ANCHOR_CODES = {-3, -5, -9}
HORIZONTAL_ANCHOR_CODES = {-4, -6, -8}
VERTICAL_GEOMETRY_CODES = {-1}
HORIZONTAL_GEOMETRY_CODES = {-2}


def bbox_to_qwen_1000(
    bbox: Sequence[float] | Iterable[float],
    image_width: float,
    image_height: float,
) -> List[int]:
    """
    Convert an image-space bounding box into Qwen's 0-1000 integer space.
    """
    coords = list(bbox)
    if len(coords) != 4:
        raise ValueError("bbox must contain exactly 4 values: [x1, y1, x2, y2]")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be greater than 0")

    x1, y1, x2, y2 = map(float, coords)

    def scale(value: float, size: float) -> int:
        scaled = (value / size) * 1000.0
        scaled = min(max(scaled, 0.0), 1000.0)
        return int(round(scaled))

    return [
        scale(x1, image_width),
        scale(y1, image_height),
        scale(x2, image_width),
        scale(y2, image_height),
    ]


def decode_dimension_pairs(
    pair1: Sequence[int] | Iterable[int],
    pair2: Sequence[int] | Iterable[int],
) -> Dict[str, Any]:
    """
    Decode two dimension relation pairs into a structured description.

    Rules:
    - One pair must be positive and represents object ids.
    - The other pair must be negative and represents either:
      1. ordered anchor positions, such as [-3, -5]
      2. a geometry code, such as [-1, -1]
    - If the positive pair is [x, x], the dimension describes one object.
    - If the positive pair is [x, y] and x != y, the dimension describes
      the spacing between two objects.
    """
    first = [int(value) for value in pair1]
    second = [int(value) for value in pair2]

    if len(first) != 2 or len(second) != 2:
        raise ValueError("each input pair must contain exactly 2 integers")

    pair1_positive = all(value >= 0 for value in first)
    pair2_positive = all(value >= 0 for value in second)
    pair1_negative = all(value < 0 for value in first)
    pair2_negative = all(value < 0 for value in second)

    if pair1_positive and pair2_negative:
        object_pair = first
        relation_pair = second
    elif pair2_positive and pair1_negative:
        object_pair = second
        relation_pair = first
    else:
        raise ValueError(
            "inputs must contain exactly one positive pair and one negative pair"
        )

    object_a, object_b = object_pair
    same_object = object_a == object_b
    dimension_type = "size" if same_object else "distance"

    result: Dict[str, Any] = {
        "object_pair": object_pair,
        "relation_pair": relation_pair,
        "dimension_type": dimension_type,
        "target_ids": [object_a] if same_object else [object_a, object_b],
        "is_single_object": same_object,
        "dimension_orientation": "unknown",
    }

    relation_a, relation_b = relation_pair

    if relation_a == relation_b and relation_a in GEOMETRY_RELATIONS:
        result["relation_type"] = "geometry"
        result["geometry_code"] = relation_a
        result["geometry_label"] = GEOMETRY_RELATIONS[relation_a]
        if relation_a in VERTICAL_GEOMETRY_CODES:
            result["dimension_orientation"] = "vertical"
        elif relation_a in HORIZONTAL_GEOMETRY_CODES:
            result["dimension_orientation"] = "horizontal"
        return result

    if relation_a not in POSITION_RELATIONS or relation_b not in POSITION_RELATIONS:
        raise ValueError(f"unknown relation pair: {relation_pair}")

    result["relation_type"] = "anchors"
    result["start_object_id"] = object_a
    result["end_object_id"] = object_b
    result["start_anchor_code"] = relation_a
    result["end_anchor_code"] = relation_b
    result["start_anchor"] = POSITION_RELATIONS[relation_a]
    result["end_anchor"] = POSITION_RELATIONS[relation_b]
    if relation_a in VERTICAL_ANCHOR_CODES and relation_b in VERTICAL_ANCHOR_CODES:
        result["dimension_orientation"] = "vertical"
    elif relation_a in HORIZONTAL_ANCHOR_CODES and relation_b in HORIZONTAL_ANCHOR_CODES:
        result["dimension_orientation"] = "horizontal"
    return result
