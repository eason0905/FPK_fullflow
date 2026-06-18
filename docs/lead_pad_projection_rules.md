# Lead Pad Projection Rules

This document describes the current deterministic rules used by the final
comparison review gallery when projecting partial evidence from `side`,
`front`, `lead`, and `land_detail` views onto top/bottom/land package pads.

The output is display evidence only. It does not change GT alignment or model
predictions.

## Coordinate Convention

- Main-view graph coordinates come from the selected top/bottom/land package
  graph.
- `x` means horizontal in the main-view overlay.
- `y` means vertical in the main-view overlay.
- Dimension values are physical units.
- Graph unit scale is physical units per reconstructed graph pixel.

## Flowchart

```mermaid
flowchart TD
  A[Partial evidence dimension] --> B{source view}
  B -->|side| C[projection_axis = y]
  B -->|front| D[projection_axis = x]
  B -->|lead / land_detail| E[projection_axis = pad radial axis]

  C --> F{anchors}
  D --> F
  E --> F

  F -->|center + left/right| G[semantics = lead_ground_contact_length]
  F -->|left_edge + right_edge, source = lead| H[semantics = lead_pad_length]
  F -->|left_edge + right_edge, source = side| I[semantics = lead_ground_contact_length]
  F -->|left_edge + right_edge, source = front| J[base semantics = pad_width]
  F -->|other| K[ignore dimension]

  J --> L{front value > main pad extent on projection axis}
  L -->|yes| M[semantics = lead_ground_contact_length]
  L -->|no| N[semantics = pad_width]

  G --> O[draw from pad outer edge toward outline center]
  I --> O
  M --> O
  H --> P[draw centered on pad along projection axis]
  N --> P

  O --> Q{projection axis is tangential to pad side?}
  Q -->|pad is top/bottom and axis = y| R[align to top/bottom pad outer edge]
  Q -->|pad is left/right and axis = y| S[center on horizontal pad centerline]
  Q -->|pad is left/right and axis = x| T[align to left/right pad outer edge]
  Q -->|pad is top/bottom and axis = x| U[center on vertical pad centerline]
```

## Current Practical Meaning

- `side` is treated as a vertical-length evidence source.
  - Example: if a package pad is on the left/right side of the outline,
    side-view evidence still draws a vertical lead pad length centered on the
    pad horizontal centerline.
  - Example: if a package pad is on the top/bottom side of the outline,
    side-view evidence draws a vertical lead pad length aligned to the top or
    bottom pad outer edge.
- `front` is treated as a horizontal-length evidence source.
  - If the front dimension is larger than the visible top/bottom pad width,
    the gallery treats it as contact length hidden under the package body.
    The new lead pad aligns with the original pad outer edge and extends inward.
- `lead` and `land_detail` remain radial by default because they can describe
  local detail rather than a global side/front orientation.

## Known Risk

These rules are deterministic heuristics for review visualization. If a future
case uses `side` to describe a true horizontal pad width, the current rule will
draw it vertically and should be split with an additional classifier.
