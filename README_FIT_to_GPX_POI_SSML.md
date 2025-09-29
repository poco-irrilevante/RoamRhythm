# FIT → GPX + POI + SSML (rustig-enthousiaste tour)

Dit pakket bevat een **lokaal** uitvoerbaar script dat jouw `.FIT`-route converteert naar `.GPX`, POI’s gelijkmatig verdeelt over de afstand, en **SSML**-bestanden maakt (intro/outro + per POI) in een **rustige, enthousiasmerende** stijl.

## Snelstart

1) **Python 3.10+** installeren
2) In een terminal/cmd:

```bash
pip install fitdecode gpxpy numpy
# (of als fitdecode niet werkt)
pip install fitparse gpxpy numpy
```

3) Run het script (voorbeeld):

```bash
python fit_to_gpx_and_pois.py --fit "2025-09-26-13-10-14.fit" --outdir "./out" --pace-kmh 4 --poi-count 10
```

Uitvoer:
- `out/<bestandsnaam>.gpx` — GPX van jouw FIT
- `out/pois.csv` — overzicht met id, km, lat, lon, ETA, duur, titel, SSML-pad
- `out/ssml/intro.ssml` en `out/ssml/outro.ssml`
- `out/ssml/POI-XX.ssml` per POI
- `out/manifest.json` — samenvattend manifest voor je app/MVP

## Details

- **Pace fallback**: als je FIT geen timestamps heeft, wordt ETA berekend via `--pace-kmh` (standaard 4 km/u).
- **POI verdeling**: gelijkmatig over totale afstand, met een kleine offset aan het begin/einde.
- **SSML toon**: `<prosody rate="85%">` voor kalm tempo; tekst is generiek regionaal NL (later kun je dit vervangen door route-specifieke historie uit OSM/Wikidata).

## PowerShell-voorbeeld (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install fitdecode gpxpy numpy
python .\fit_to_gpx_and_pois.py --fit ".\2025-09-26-13-10-14.fit" --outdir ".\out" --pace-kmh 4 --poi-count 10
```

## Tip: route-specifieke historie
Wanneer je GPX en POI’s hebt, kun je met een LLM per POI de **feitelijke** historie genereren (Wikidata/Wikipedia/erfgoed-API’s) en de SSML vervangen door de definitieve voiceover-teksten.
