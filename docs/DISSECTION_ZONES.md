# Suggested dissection zones (Go / No-Go)

PRISM-Endo overlays a **green Go zone** and a **red No-Go zone** on cholecystectomy video, offline and live.

## Basis in the literature

| Source | What it says | How it is used here |
|---|---|---|
| Madani et al., *Artificial Intelligence for Intraoperative Guidance: Using Semantic Segmentation to Identify Surgical Anatomy in Laparoscopic Cholecystectomy*, Ann Surg 2022 (GoNoGoNet) | Deep learning on 2,627 frames from 290 videos identifies safe (Go) and dangerous (No-Go) zones of dissection; green / red overlay | Same idea and colours |
| SAGES Safe Cholecystectomy program; Strasberg, Critical View of Safety | Clear the fibro-fatty tissue of the hepatocystic triangle, staying on the gallbladder; only two structures should enter the gallbladder | Go = fibro-fatty tissue close to the gallbladder |
| Rouvière's sulcus / R4U line (e.g. Hugh et al.; Singh & Ohri) | Dissect above the R4U line; the area below it, where the common bile duct runs, is a danger zone | No-Go = deep tissue far from the gallbladder |

## How the zones are computed

`src/safety/dissection_zones.py`, from the tissue segmentation (cholecystectomy model only, when the gallbladder covers at
least 1 % of the frame). Distances are fractions of the image diagonal (`configs/pipeline.yaml → safety.dissection_zones`).

- **Go**: fat / connective tissue within 6 % of the gallbladder edge, at least 3 % away from any danger structure.
- **No-Go**: cystic duct, hepatic vein, blood and bowel plus a 3 % margin, and fat / connective tissue more than 16 % from
  the gallbladder.

Each frame records `go_zone_frac`, `nogo_zone_frac`, `<tip>_zone` (`go` / `no_go` / empty) and `tip_in_nogo_zone`; the
dashboard shows a *Tip in No-Go zone* timeline track and events, and the live view marks each tip card and logs an event.

## How far to trust it

GoNoGoNet learned its zones from expert annotations; PRISM-Endo has none, so these zones are a **rule-based
approximation** that has **not been compared with expert annotations**. Rouvière's sulcus and the common bile duct are
not segmented, and the cystic duct class has almost no held-out test frames. Use the overlay to review and discuss
video, not to decide where to cut.
