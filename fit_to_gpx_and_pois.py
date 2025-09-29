#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIT -> GPX conversie + POI generatie + SSML (intro/outro en per POI)

Gebruik:
  1) Zorg voor Python 3.10+
  2) Installeer dependencies (een van de twee FIT libs is genoeg):
       pip install fitdecode gpxpy numpy
     (of)
       pip install fitparse gpxpy numpy
  3) Run:
       python fit_to_gpx_and_pois.py --fit "2025-09-26-13-10-14.fit" --outdir "./out" --pace-kmh 4 --poi-count 10

Wat doet het script?
  - Leest de FIT file en extraheert trackpoints (lat, lon, ele, tijd).
  - Schrijft GPX met het volledige spoor.
  - Berekent cumulatieve afstand en verdeelt POI's gelijkmatig over de route (afstand-gestuurd).
  - Berekent ETA per POI (op basis van gemeten tijden; valt terug op pace-kmh als geen tijd beschikbaar).
  - Genereert kalm-enthousiaste SSML bestanden voor: intro, per POI, en outro.
  - Maakt manifest.json + pois.csv voor jouw app/MVP.

Auteur: ChatGPT
"""
import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta, timezone

# Probeer beide bibliotheken; minstens één moet werken
FIT_BACKEND = None
try:
    import fitdecode  # type: ignore
    FIT_BACKEND = "fitdecode"
except Exception:
    try:
        from fitparse import FitFile  # type: ignore
        FIT_BACKEND = "fitparse"
    except Exception:
        FIT_BACKEND = None

import numpy as np  # type: ignore
import gpxpy  # type: ignore
import gpxpy.gpx  # type: ignore
import csv
import json

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c  # meters

def cumulative_distance(points: List[Tuple[float,float]]) -> List[float]:
    dists = [0.0]
    for i in range(1, len(points)):
        d = haversine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
        dists.append(dists[-1] + d)
    return dists

def read_fit_fitdecode(path: Path):
    lats, lons, eles, times = [], [], [], []
    with fitdecode.FitReader(str(path)) as rd:
        for frame in rd:
            if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == 'record':
                lat = frame.get_value('position_lat')
                lon = frame.get_value('position_long')
                ele = frame.get_value('altitude')
                t = frame.get_value('timestamp')
                # FIT semicercles -> degrees
                def semi_to_deg(x):
                    return x * (180.0 / 2**31) if x is not None else None
                lat = semi_to_deg(lat)
                lon = semi_to_deg(lon)
                if lat is not None and lon is not None:
                    lats.append(lat)
                    lons.append(lon)
                    eles.append(ele if ele is not None else 0.0)
                    times.append(t if isinstance(t, datetime) else None)
    return lats, lons, eles, times

def read_fit_fitparse(path: Path):
    lats, lons, eles, times = [], [], [], []
    ff = FitFile(str(path))
    for record in ff.get_messages('record'):
        vals = {d.name: d.value for d in record}
        lat = vals.get('position_lat')
        lon = vals.get('position_long')
        ele = vals.get('altitude', 0.0)
        t = vals.get('timestamp', None)
        def semi_to_deg(x):
            return x * (180.0 / 2**31) if x is not None else None
        lat = semi_to_deg(lat)
        lon = semi_to_deg(lon)
        if lat is not None and lon is not None:
            lats.append(lat)
            lons.append(lon)
            eles.append(ele if ele is not None else 0.0)
            times.append(t if isinstance(t, datetime) else None)
    return lats, lons, eles, times

def fit_to_track(path: Path):
    if FIT_BACKEND is None:
        raise RuntimeError("Geen FIT-bibliotheek beschikbaar. Installeer fitdecode of fitparse.")
    if FIT_BACKEND == "fitdecode":
        return read_fit_fitdecode(path)
    else:
        return read_fit_fitparse(path)

def make_gpx(lats, lons, eles, times, creator="fit-to-gpx"):
    gpx = gpxpy.gpx.GPX()
    gpx.creator = creator
    trk = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(trk)
    seg = gpxpy.gpx.GPXTrackSegment()
    trk.segments.append(seg)
    for lat, lon, ele, t in zip(lats, lons, eles, times):
        # gpxpy expects timezone-aware or naive; we keep naive
        seg.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, elevation=ele, time=t))
    return gpx

def interpolate_time_from_pace(start_time: Optional[datetime], dist_m: float, pace_kmh: float) -> datetime:
    if start_time is None:
        start_time = datetime(1970,1,1, tzinfo=timezone.utc)
    hours = dist_m / (pace_kmh * 1000.0)
    delta = timedelta(hours=hours)
    return start_time + delta

def ssml_wrap(text: str) -> str:
    return f'<speak><prosody rate="85%" pitch="+0st" volume="+0.0dB">{text}</prosody></speak>'

POI_TITLES = [
    "Landschap & waterstaat",
    "Ontstaan van nederzettingen",
    "Ambachten en handel",
    "Architectuur in de regio",
    "Natuur & biodiversiteit",
    "Polders, dijken en gemalen",
    "Oude routes & verbindingen",
    "Oorlogssporen en herdenking",
    "Bestuur & gemeentefusies",
    "Moderne tijd & toekomst"
]

POI_TEMPLATES = [
    "Dit landschap is gevormd door water en wind. Let op sloten, kades en hoogtes: ze vertellen hoe mensen het water sturen. Een klein weetje: veel polderwegen liggen net hoger dan omliggende velden.",
    "Veel dorpen ontstonden bij kruisingen van wegen en water. Luister hoe handel en ambacht bewoners trokken—vaak rond een kerk of plein als hart van de gemeenschap.",
    "Ambachten bloeiden waar grondstoffen en routes elkaar raken. Denk aan molenaars, smeden, scheepstimmerlieden of linnenwevers—ieder gebonden aan seizoen en plaats.",
    "Kijk naar metselwerk, dakvormen en gevelopeningen. Baksteen en nuchtere lijnen domineren vaak; wederopbouw en modernisering lieten ook hun sporen na.",
    "Let op vogels aan randen van water en riet. Fluisterstil bewegen vaak waterhoentjes; in sloten zie je soms libellen. Tip: loop even langzamer en luister naar lagen van geluid.",
    "Polderen is samenwerken: dijken houden water buiten, sloten voeren af, gemalen regelen peil. Een metafoor: het landschap ademt—in natte tijden trager, in droge sneller.",
    "Routes vormen ritme: karrenpaden, trekvaarten, kanalen of spoorlijnen verplaatsen mensen en ideeën. Oude tracés wonen soms voort als stille fietspaden.",
    "Sporen van oorlog vragen bedachtzaamheid. Monumenten zijn ankerpunten van herinnering; ze nodigen uit tot stilstaan, kijken, en zachtjes verdergaan.",
    "Gemeenten veranderden mee met tijd en taken. Fusies bundelden diensten; grenzen verschoof men om samen sterker te staan in zorg, onderwijs en beheer.",
    "Vandaag draait veel om balans: natuurontwikkeling, energie, wonen en recreatie. Kijk om je heen: vernieuwing en behoud kunnen elkaar versterken."
]

def build_pois(lats, lons, times, poi_count: int, pace_kmh: float):
    assert len(lats) == len(lons) == len(times)
    if len(lats) < 2:
        raise RuntimeError("Te weinig trackpunten in FIT.")
    points = list(zip(lats, lons))
    cum = cumulative_distance(points)  # meters
    total_m = cum[-1]
    # Doelafstanden voor POIs gelijkmatig verdeeld
    targets = np.linspace(total_m * 0.09, total_m * 0.99, poi_count)  # start niet op 0 om dubbel met intro te voorkomen
    pois = []
    t0 = None
    # kies starttijd
    for t in times:
        if isinstance(t, datetime):
            t0 = t
            break
    for i, tm in enumerate(targets):
        # vind dichtstbijzijnde index
        idx = int(np.argmin([abs(d - tm) for d in cum]))
        km = round(cum[idx] / 1000.0, 2)
        lat, lon = lats[idx], lons[idx]
        # ETA
        if times[idx] is not None and isinstance(times[idx], datetime):
            eta = times[idx]
        else:
            eta = interpolate_time_from_pace(t0, cum[idx], pace_kmh)
        # duur 90-110 s, afwisselend
        dur_s = 100 if i % 2 == 0 else 95
        title = POI_TITLES[i % len(POI_TITLES)]
        text = POI_TEMPLATES[i % len(POI_TEMPLATES)]
        ssml = ssml_wrap(f"{title}. {text}")
        pois.append({
            "id": f"POI-{i+1:02d}",
            "km": float(km),
            "lat": float(lat),
            "lon": float(lon),
            "eta": eta.astimezone(timezone.utc).isoformat() if isinstance(eta, datetime) else None,
            "duration_s": dur_s,
            "title": title,
            "ssml": ssml
        })
    return pois, total_m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="Pad naar FIT bestand")
    ap.add_argument("--outdir", default="./out", help="Uitvoermap")
    ap.add_argument("--pace-kmh", type=float, default=4.0, help="Pace km/h (fallback voor ETA)")
    ap.add_argument("--poi-count", type=int, default=10, help="Aantal POIs")
    args = ap.parse_args()

    fit_path = Path(args.fit)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Lees FIT
    lats, lons, eles, times = fit_to_track(fit_path)

    # Schrijf GPX
    gpx = make_gpx(lats, lons, eles, times)
    gpx_path = outdir / (fit_path.stem + ".gpx")
    with gpx_path.open("w", encoding="utf-8") as f:
        f.write(gpx.to_xml())

    # POIs + SSML
    pois, total_m = build_pois(lats, lons, times, args.poi_count, args.pace_kmh)

    # Intro/Outro SSML
    intro_text = ("Welkom bij deze wandeling. We nemen u in een rustig tempo mee door het landschap, "
                  "met korte verhalen over de omgeving, afgewisseld met muziek uit de jaren tachtig, "
                  "taalprikkels Noors en ontspannen stretchmomenten. "
                  "Wandel op uw gemak, kijk om u heen en luister wanneer we een interessant punt passeren. "
                  "Aan het begin starten we met een korte warming-up. Klaar? Dan gaan we op pad.")
    outro_text = ("We zijn bijna aan het einde van de route. Neem even de tijd om na te voelen hoe uw lichaam is "
                  "opgewarmd. Straks sluiten we af met enkele rustige stretches. "
                  "Dank voor het meewandelen. Heeft u iets nieuws ontdekt of geleerd—een woord Noors, een stukje "
                  "geschiedenis? Bewaar het gevoel van ruimte en ritme. Tot de volgende keer, en fijne dag verder.")
    intro_ssml = ssml_wrap(intro_text)
    outro_ssml = ssml_wrap(outro_text)

    # Output mappen
    ssml_dir = outdir / "ssml"
    ssml_dir.mkdir(exist_ok=True)

    # Schrijf SSML files
    (ssml_dir / "intro.ssml").write_text(intro_ssml, encoding="utf-8")
    (ssml_dir / "outro.ssml").write_text(outro_ssml, encoding="utf-8")
    for poi in pois:
        (ssml_dir / f"{poi['id']}.ssml").write_text(poi["ssml"], encoding="utf-8")

    # CSV met POIs
    csv_path = outdir / "pois.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","km","lat","lon","eta_utc","duration_s","title","ssml_file"])
        for poi in pois:
            w.writerow([poi["id"], poi["km"], poi["lat"], poi["lon"], poi["eta"], poi["duration_s"], poi["title"], f"ssml/{poi['id']}.ssml"])

    # Manifest
    manifest = {
        "route": {
            "meters": round(float(total_m), 1),
            "km": round(float(total_m)/1000.0, 3)
        },
        "gpx": gpx_path.name,
        "pace_kmh": args.pace_kmh,
        "ssml": {
            "intro": "ssml/intro.ssml",
            "outro": "ssml/outro.ssml"
        },
        "pois": [
            {
                "id": poi["id"],
                "km": poi["km"],
                "lat": poi["lat"],
                "lon": poi["lon"],
                "eta_utc": poi["eta"],
                "duration_s": poi["duration_s"],
                "title": poi["title"],
                "ssml": f"ssml/{poi['id']}.ssml"
            } for poi in pois
        ]
    }
    with (outdir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"✅ GPX geschreven: {gpx_path}")
    print(f"✅ POIs CSV: {csv_path}")
    print(f"✅ SSML map: {ssml_dir}")
    print(f"✅ Manifest: {outdir / 'manifest.json'}")
    print("Klaar.")

if __name__ == "__main__":
    main()
