"""Build the v2 data-driven hantavirus case panel.

This script assembles the first reproducible case panel for calibration from
NCBI GenBank/BioSample metadata, CDC NNDSS state-week surveillance tables, and
WHO Disease Outbreak News. HTTPS calls use requests with certifi verification
and are cached under /Volumes/HELFRICH-GD/TEG_data/inputs/cases so the git repo
stays small.

Assumptions:
    * NCBI records with explicit lat_lon are treated as point-level evidence.
      NCBI records with only geo_loc_name are retained at a country centroid and
      receive lower confidence rather than being promoted to precise points.
    * CDC NNDSS HPS tables expose cumulative annual fields more reliably than
      current-week fields. Incident state-week counts are recovered by
      differencing state cumulative totals, with dates assigned to the ISO week
      Monday for the reported MMWR year/week.
    * WHO Disease Outbreak News events are country-level unless the title/body
      cannot be matched to one Natural Earth country, in which case they are
      skipped rather than geocoded by guesswork.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import certifi
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "data" / "outputs" / "cases_panel_v1.parquet"
MANIFEST_PATH = REPO_ROOT / "data" / "outputs" / "cases_panel_v1_manifest.json"
EXTERNAL_CACHE = Path("/Volumes/HELFRICH-GD/TEG_data/inputs/cases")

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_USER_AGENT = "helfrich-teg/0.2 (case-panel loader; ianthelfrich@gmail.com)"
NATURAL_EARTH_COUNTRIES_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)
CENSUS_STATE_CENTERS_URL = (
    "https://www2.census.gov/geo/docs/reference/cenpop2020/"
    "CenPop2020_Mean_ST.txt"
)
WHO_DON_URL = "https://www.who.int/api/news/diseaseoutbreaknews"

CDC_NNDSS_IDS = [
    "tfcp-ufzp",
    "bhxw-k5sb",
    "azpx-5hzx",
    "a9xa-yrhn",
    "chmz-4uae",
]

CDC_CUMULATIVE_COLUMNS = {
    "tfcp-ufzp": [
        "hantavirus_pulmonary_syndrome_4",
    ],
    "bhxw-k5sb": [
        "hantavirus_pulmonary_syndrome_all_serotypes_cum_2019",
        "hantavirus_pulmonary_syndrome_cum_2019",
    ],
    "azpx-5hzx": [
        "hantavirus_pulmonary_syndrome_all_serotypes_cum_2019",
        "hantavirus_pulmonary_syndrome_cum_2019",
    ],
    "a9xa-yrhn": [
        "hantavirus_pulmonary_syndrome_4",
    ],
    "chmz-4uae": [
        "hantavirus_pulmonary_syndrome_4",
    ],
}

NON_STATE_AREAS = {
    "TOTAL",
    "US RESIDENTS",
    "NON-US RESIDENTS",
    "NEW ENGLAND",
    "MIDDDLE ATLANTIC",
    "MIDDLE ATLANTIC",
    "EAST NORTH CENTRAL",
    "WEST NORTH CENTRAL",
    "SOUTH ATLANTIC",
    "EAST SOUTH CENTRAL",
    "WEST SOUTH CENTRAL",
    "MOUNTAIN",
    "PACIFIC",
    "AMERICAN SAMOA",
    "GUAM",
    "NORTHERN MARIANA ISLANDS",
    "PUERTO RICO",
    "U.S. VIRGIN ISLANDS",
    "VIRGIN ISLANDS",
}

COUNTRY_ALIASES = {
    "USA": "United States of America",
    "US": "United States of America",
    "U.S.A.": "United States of America",
    "UNITED STATES": "United States of America",
    "UNITED STATES OF AMERICA": "United States of America",
    "RUSSIA": "Russia",
    "RUSSIAN FEDERATION": "Russia",
    "SOUTH KOREA": "South Korea",
    "KOREA": "South Korea",
    "REPUBLIC OF KOREA": "South Korea",
    "DEMOCRATIC REPUBLIC OF THE CONGO": "Democratic Republic of the Congo",
    "DRC": "Democratic Republic of the Congo",
    "CZECH REPUBLIC": "Czechia",
    "VIET NAM": "Vietnam",
    "LAO PDR": "Laos",
    "BOLIVIA": "Bolivia",
    "VENEZUELA": "Venezuela",
    "IRAN": "Iran",
    "TANZANIA": "Tanzania",
    "SYRIA": "Syria",
    "MOLDOVA": "Moldova",
}


@dataclass
class InputManifest:
    path: str
    source_url: str
    pulled_at_utc: str
    sha256: str
    size_bytes: int
    license: str
    citation: str
    description: str


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _file_mtime_utc(path: Path) -> str:
    if not path.exists():
        return _now_utc()
    return (
        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entry(
    path: Path,
    source_url: str,
    license_text: str,
    citation: str,
    description: str,
) -> InputManifest:
    return InputManifest(
        path=str(path),
        source_url=source_url,
        pulled_at_utc=_file_mtime_utc(path),
        sha256=sha256_file(path) if path.exists() else "",
        size_bytes=path.stat().st_size if path.exists() else 0,
        license=license_text,
        citation=citation,
        description=description,
    )


def _request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": NCBI_USER_AGENT})
    session.verify = certifi.where()
    return session


def _get_with_cache(
    session: requests.Session,
    url: str,
    cache_path: Path,
    *,
    params: dict[str, Any] | None = None,
    refresh: bool = False,
    binary: bool = False,
    empty_payload: bytes | str = "{}",
    timeout: int = 60,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return cache_path
    try:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.content if binary else response.text
    except Exception as exc:
        print(f"warning: fetch failed for {url}: {exc}", file=sys.stderr)
        payload = empty_payload
    if binary:
        cache_path.write_bytes(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    else:
        cache_path.write_text(payload if isinstance(payload, str) else payload.decode("utf-8"), encoding="utf-8")
    return cache_path


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text.lower() in {"not provided", "missing", "unknown", "na", "n/a"}:
        return None
    return text


def _parse_count(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "N", "NC", "NN", "NP"}:
        return None
    text = text.replace(",", "")
    try:
        out = int(float(text))
    except ValueError:
        return None
    return out if out >= 0 else None


def _normalize_country_name(country: str | None) -> str | None:
    country = _clean_text(country)
    if not country:
        return None
    country = country.split(";")[0].strip()
    country = re.sub(r"\s+", " ", country)
    return COUNTRY_ALIASES.get(country.upper(), country)


def _extract_country_admin1(geo_loc: str | None) -> tuple[str | None, str | None]:
    geo_loc = _clean_text(geo_loc)
    if not geo_loc:
        return None, None
    if ":" in geo_loc:
        country, rest = geo_loc.split(":", 1)
        admin1 = rest.split(",", 1)[0].strip() or None
    else:
        country, admin1 = geo_loc, None
    country = _normalize_country_name(country)
    return country, _clean_text(admin1)


def parse_lat_lon(value: str | None) -> tuple[float, float] | None:
    """Parse INSDC-style lat_lon strings into (lon, lat)."""
    value = _clean_text(value)
    if not value:
        return None
    text = value.replace(",", " ").replace(";", " ")
    hemi_match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*([NS])\s+([-+]?\d+(?:\.\d+)?)\s*([EW])",
        text,
        flags=re.IGNORECASE,
    )
    if hemi_match:
        lat = float(hemi_match.group(1))
        lon = float(hemi_match.group(3))
        if hemi_match.group(2).upper() == "S":
            lat *= -1
        if hemi_match.group(4).upper() == "W":
            lon *= -1
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lon, lat
    signed = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(signed) >= 2:
        lat = float(signed[0])
        lon = float(signed[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lon, lat
    return None


def parse_collection_date(value: str | None) -> pd.Timestamp | None:
    """Return a coarse but sortable date from common collection_date strings."""
    text = _clean_text(value)
    if not text:
        return None
    text = text.replace("?", "").strip()
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    iso_day = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso_day:
        return pd.Timestamp(int(iso_day.group(1)), int(iso_day.group(2)), int(iso_day.group(3)))
    iso_month = re.search(r"\b(\d{4})-(\d{1,2})\b", text)
    if iso_month:
        return pd.Timestamp(int(iso_month.group(1)), int(iso_month.group(2)), 15)
    year_only = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if year_only:
        year = int(year_only.group(1))
        month_lookup = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        month = 7
        lowered = text.lower()
        for key, val in month_lookup.items():
            if key in lowered:
                month = val
                break
        return pd.Timestamp(year, month, 15 if month != 7 else 1)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).tz_localize(None) if getattr(parsed, "tzinfo", None) else pd.Timestamp(parsed)


def _iso_week_monday(year: int, week: int) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(date.fromisocalendar(year, week, 1))
    except ValueError:
        return None


def ensure_country_centroids(
    session: requests.Session,
    *,
    refresh: bool = False,
) -> tuple[dict[str, tuple[float, float]], list[InputManifest]]:
    path = EXTERNAL_CACHE / "boundaries" / "ne_110m_admin_0_countries.zip"
    _get_with_cache(
        session,
        NATURAL_EARTH_COUNTRIES_URL,
        path,
        refresh=refresh,
        binary=True,
        empty_payload=b"",
        timeout=90,
    )
    manifests = [
        _manifest_entry(
            path,
            NATURAL_EARTH_COUNTRIES_URL,
            "Public domain",
            "Natural Earth. 1:110m Cultural Vectors, Admin 0 Countries.",
            "Country polygons used to assign country-level centroids.",
        )
    ]
    if not path.exists() or path.stat().st_size == 0:
        return {}, manifests
    with ZipFile(path) as zf:
        if not any(name.endswith(".shp") for name in zf.namelist()):
            return {}, manifests
    countries = gpd.read_file(f"zip://{path}")
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    countries = countries.to_crs("EPSG:4326")
    points = countries.geometry.representative_point()
    lookup: dict[str, tuple[float, float]] = {}
    for idx, row in countries.iterrows():
        point = points.iloc[idx]
        names = {
            row.get("ADMIN"),
            row.get("NAME"),
            row.get("NAME_LONG"),
            row.get("SOVEREIGNT"),
            row.get("FORMAL_EN"),
        }
        for name in names:
            clean = _normalize_country_name(str(name)) if pd.notna(name) else None
            if clean:
                lookup[clean] = (float(point.x), float(point.y))
                lookup[clean.upper()] = (float(point.x), float(point.y))
    return lookup, manifests


def ensure_state_centers(
    session: requests.Session,
    *,
    refresh: bool = False,
) -> tuple[dict[str, tuple[float, float]], list[InputManifest]]:
    path = EXTERNAL_CACHE / "boundaries" / "CenPop2020_Mean_ST.txt"
    _get_with_cache(
        session,
        CENSUS_STATE_CENTERS_URL,
        path,
        refresh=refresh,
        empty_payload="STATEFP,STNAME,POPULATION,LATITUDE,LONGITUDE\n",
        timeout=60,
    )
    manifests = [
        _manifest_entry(
            path,
            CENSUS_STATE_CENTERS_URL,
            "U.S. federal public domain",
            "U.S. Census Bureau. 2020 Centers of Population by State.",
            "Population-weighted state centroids for NNDSS disaggregation.",
        )
    ]
    lookup: dict[str, tuple[float, float]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["STNAME"].upper()
                lookup[name] = (float(row["LONGITUDE"]), float(row["LATITUDE"]))
    except Exception as exc:
        print(f"warning: failed to parse Census state centers: {exc}", file=sys.stderr)
    return lookup, manifests


def _country_point(
    country: str | None,
    centroids: dict[str, tuple[float, float]],
) -> tuple[float | None, float | None]:
    country = _normalize_country_name(country)
    if not country:
        return None, None
    point = centroids.get(country) or centroids.get(country.upper())
    if not point:
        return None, None
    return point


def _qualifiers_from_genbank_record(record: str) -> dict[str, str]:
    qualifiers: dict[str, str] = {}
    for match in re.finditer(r'/([A-Za-z0-9_]+)="((?:[^"]|\n\s+)*)"', record):
        key = match.group(1)
        value = re.sub(r"\s+", " ", match.group(2)).strip()
        qualifiers.setdefault(key, value)
    return qualifiers


def _accession_from_record(record: str) -> str | None:
    match = re.search(r"^ACCESSION\s+(\S+)", record, flags=re.MULTILINE)
    return match.group(1) if match else None


def _fetch_nuccore_flatfiles(session: requests.Session, ids: list[str]) -> str:
    """Fetch GenBank flatfiles, splitting batches that NCBI rejects."""
    if not ids:
        return ""
    last_error: Exception | str = "empty response"
    params = {
        "db": "nuccore",
        "id": ",".join(ids),
        "rettype": "gb",
        "retmode": "text",
    }
    for attempt in range(2):
        try:
            response = session.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=120)
            response.raise_for_status()
            text = response.text
            if text.strip():
                return text
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 + attempt * 0.5)
    if len(ids) == 1:
        print(f"warning: NCBI efetch failed for nuccore id {ids[0]}: {last_error}", file=sys.stderr)
        return ""
    midpoint = len(ids) // 2
    left = _fetch_nuccore_flatfiles(session, ids[:midpoint])
    time.sleep(0.35)
    right = _fetch_nuccore_flatfiles(session, ids[midpoint:])
    return "\n".join(part for part in (left, right) if part)


def fetch_ncbi_nuccore_ids(
    session: requests.Session,
    *,
    refresh: bool = False,
    retmax: int = 20000,
) -> tuple[list[str], list[InputManifest]]:
    params = {
        "db": "nuccore",
        "term": "hantavirus[organism]",
        "retmode": "json",
        "retmax": str(retmax),
    }
    path = EXTERNAL_CACHE / "ncbi" / "nuccore_hantavirus_esearch.json"
    _get_with_cache(
        session,
        f"{NCBI_BASE}/esearch.fcgi",
        path,
        params=params,
        refresh=refresh,
        empty_payload='{"esearchresult":{"idlist":[],"count":"0"}}',
        timeout=90,
    )
    data = _read_json(path, {"esearchresult": {"idlist": []}})
    ids = [str(x) for x in data.get("esearchresult", {}).get("idlist", [])]
    manifests = [
        _manifest_entry(
            path,
            f"{NCBI_BASE}/esearch.fcgi?db=nuccore&term=hantavirus[organism]&retmode=json",
            "NCBI public data; see NCBI disclaimer and data usage policies",
            "NCBI Nucleotide database, E-utilities esearch for hantavirus[organism].",
            "Nuccore UID list used for GenBank source-feature extraction.",
        )
    ]
    return ids, manifests


def parse_ncbi_nuccore(
    session: requests.Session,
    country_centroids: dict[str, tuple[float, float]],
    *,
    refresh: bool = False,
    retmax: int = 20000,
    batch_size: int = 100,
) -> tuple[pd.DataFrame, list[InputManifest]]:
    ids, manifests = fetch_ncbi_nuccore_ids(session, refresh=refresh, retmax=retmax)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        cache_path = EXTERNAL_CACHE / "ncbi" / "nuccore_gb" / f"gb_{start:06d}.txt"
        if not cache_path.exists() or refresh:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(_fetch_nuccore_flatfiles(session, batch), encoding="utf-8")
            time.sleep(0.35)
        manifests.append(
            _manifest_entry(
                cache_path,
                f"{NCBI_BASE}/efetch.fcgi?db=nuccore&rettype=gb&retmode=text",
                "NCBI public data; see NCBI disclaimer and data usage policies",
                "NCBI Nucleotide database, GenBank flatfile source features.",
                f"GenBank flatfile batch {start // batch_size + 1}.",
            )
        )
        text = cache_path.read_text(encoding="utf-8", errors="replace") if cache_path.exists() else ""
        for record in text.split("\n//"):
            if "FEATURES" not in record:
                continue
            qualifiers = _qualifiers_from_genbank_record(record)
            virus = _clean_text(qualifiers.get("organism"))
            collection = parse_collection_date(qualifiers.get("collection_date"))
            geo_loc = qualifiers.get("geo_loc_name") or qualifiers.get("country")
            country, admin1 = _extract_country_admin1(geo_loc)
            coords = parse_lat_lon(qualifiers.get("lat_lon"))
            confidence = 1.0
            if coords is None:
                lon, lat = _country_point(country, country_centroids)
                coords = (lon, lat) if lon is not None and lat is not None else None
                confidence = 0.35
            if coords is None or collection is None:
                continue
            rows.append(
                {
                    "date": collection,
                    "source": "genbank_nuccore",
                    "virus_species": virus or "Orthohantavirus",
                    "country": country,
                    "admin1": admin1,
                    "lon": coords[0],
                    "lat": coords[1],
                    "count": 1,
                    "confidence": confidence,
                    "_dedupe_key": "|".join(
                        [
                            (qualifiers.get("isolate") or _accession_from_record(record) or "").lower(),
                            (virus or "").lower(),
                            str(collection.date()),
                            (geo_loc or "").lower(),
                            qualifiers.get("lat_lon") or "",
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows), manifests


def parse_ncbi_biosample(
    session: requests.Session,
    country_centroids: dict[str, tuple[float, float]],
    *,
    refresh: bool = False,
    batch_size: int = 200,
) -> tuple[pd.DataFrame, list[InputManifest]]:
    search_path = EXTERNAL_CACHE / "ncbi" / "biosample_hantavirus_esearch.json"
    _get_with_cache(
        session,
        f"{NCBI_BASE}/esearch.fcgi",
        search_path,
        params={
            "db": "biosample",
            "term": "hantavirus",
            "retmode": "json",
            "retmax": "5000",
        },
        refresh=refresh,
        empty_payload='{"esearchresult":{"idlist":[],"count":"0"}}',
        timeout=90,
    )
    manifests = [
        _manifest_entry(
            search_path,
            f"{NCBI_BASE}/esearch.fcgi?db=biosample&term=hantavirus&retmode=json",
            "NCBI public data; see NCBI disclaimer and data usage policies",
            "NCBI BioSample database, E-utilities esearch for hantavirus.",
            "BioSample UID list used for sample metadata extraction.",
        )
    ]
    ids = [
        str(x)
        for x in _read_json(search_path, {"esearchresult": {"idlist": []}})
        .get("esearchresult", {})
        .get("idlist", [])
    ]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        cache_path = EXTERNAL_CACHE / "ncbi" / "biosample_xml" / f"biosample_{start:06d}.xml"
        if not cache_path.exists() or refresh:
            _get_with_cache(
                session,
                f"{NCBI_BASE}/efetch.fcgi",
                cache_path,
                params={"db": "biosample", "id": ",".join(batch), "retmode": "xml"},
                refresh=True,
                empty_payload="<?xml version=\"1.0\"?><BioSampleSet />",
                timeout=90,
            )
            time.sleep(0.35)
        manifests.append(
            _manifest_entry(
                cache_path,
                f"{NCBI_BASE}/efetch.fcgi?db=biosample&retmode=xml",
                "NCBI public data; see NCBI disclaimer and data usage policies",
                "NCBI BioSample database, XML sample attributes.",
                f"BioSample XML batch {start // batch_size + 1}.",
            )
        )
        try:
            root = ET.fromstring(cache_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            print(f"warning: failed to parse {cache_path}: {exc}", file=sys.stderr)
            continue
        for bs in root.findall(".//BioSample"):
            attrs: dict[str, str] = {}
            for attr in bs.findall(".//Attribute"):
                key = attr.attrib.get("attribute_name") or attr.attrib.get("harmonized_name")
                if key and attr.text:
                    attrs[key.lower()] = _clean_text(attr.text) or ""
            organism_el = bs.find("./Description/Organism")
            virus = None
            if organism_el is not None:
                virus = organism_el.attrib.get("taxonomy_name")
            geo_loc = attrs.get("geo_loc_name") or attrs.get("geo loc name")
            country, admin1 = _extract_country_admin1(geo_loc)
            lat_lon = attrs.get("lat_lon") or attrs.get("lat lon") or attrs.get("latitude and longitude")
            coords = parse_lat_lon(lat_lon)
            confidence = 1.0
            if coords is None:
                lon, lat = _country_point(country, country_centroids)
                coords = (lon, lat) if lon is not None and lat is not None else None
                confidence = 0.35
            collection = parse_collection_date(attrs.get("collection_date") or attrs.get("collection date"))
            if coords is None or collection is None:
                continue
            accession = bs.attrib.get("accession") or bs.attrib.get("id") or ""
            isolate = attrs.get("isolate") or accession
            rows.append(
                {
                    "date": collection,
                    "source": "genbank_biosample",
                    "virus_species": virus or "Orthohantavirus",
                    "country": country,
                    "admin1": admin1,
                    "lon": coords[0],
                    "lat": coords[1],
                    "count": 1,
                    "confidence": confidence,
                    "_dedupe_key": "|".join(
                        [
                            isolate.lower(),
                            (virus or "").lower(),
                            str(collection.date()),
                            (geo_loc or "").lower(),
                            lat_lon or "",
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows), manifests


def parse_cdc_nndss(
    session: requests.Session,
    state_centers: dict[str, tuple[float, float]],
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[InputManifest]]:
    rows: list[dict[str, Any]] = []
    manifests: list[InputManifest] = []
    for dataset_id in CDC_NNDSS_IDS:
        cache_path = EXTERNAL_CACHE / "cdc_nndss" / f"{dataset_id}.json"
        _get_with_cache(
            session,
            f"https://data.cdc.gov/resource/{dataset_id}.json",
            cache_path,
            params={"$limit": "50000"},
            refresh=refresh,
            empty_payload="[]",
            timeout=90,
        )
        manifests.append(
            _manifest_entry(
                cache_path,
                f"https://data.cdc.gov/resource/{dataset_id}.json",
                "Public domain unless otherwise noted by data.cdc.gov",
                f"CDC NNDSS Socrata dataset {dataset_id}.",
                "Hantavirus pulmonary syndrome cumulative state-week rows.",
            )
        )
        data = _read_json(cache_path, [])
        parsed_rows: list[dict[str, Any]] = []
        for row in data:
            area = _clean_text(row.get("reporting_area"))
            if not area:
                continue
            area_upper = area.upper()
            if area_upper in NON_STATE_AREAS or area_upper not in state_centers:
                continue
            year = _parse_count(row.get("mmwr_year"))
            week = _parse_count(row.get("mmwr_week"))
            if year is None or week is None:
                continue
            cumulative = None
            for col in CDC_CUMULATIVE_COLUMNS.get(dataset_id, []):
                cumulative = _parse_count(row.get(col))
                if cumulative is not None:
                    break
            parsed_rows.append(
                {
                    "dataset_id": dataset_id,
                    "state": area_upper,
                    "year": year,
                    "week": week,
                    "cumulative": cumulative,
                }
            )
        if not parsed_rows:
            continue
        table = pd.DataFrame(parsed_rows).drop_duplicates(["state", "year", "week"], keep="last")
        table = table.sort_values(["state", "year", "week"])
        table["cumulative"] = pd.to_numeric(table["cumulative"], errors="coerce")
        table["cumulative"] = table.groupby(["state", "year"])["cumulative"].ffill().fillna(0).astype(int)
        table["previous"] = table.groupby(["state", "year"])["cumulative"].shift(1).fillna(0).astype(int)
        table["incident"] = (table["cumulative"] - table["previous"]).clip(lower=0)
        for item in table.loc[table["incident"] > 0].to_dict("records"):
            dt = _iso_week_monday(int(item["year"]), int(item["week"]))
            if dt is None:
                continue
            lon, lat = state_centers[item["state"]]
            rows.append(
                {
                    "date": dt,
                    "source": "cdc_nndss",
                    "virus_species": "Hantavirus pulmonary syndrome",
                    "country": "United States of America",
                    "admin1": item["state"].title(),
                    "lon": lon,
                    "lat": lat,
                    "count": int(item["incident"]),
                    "confidence": 0.65,
                    "_dedupe_key": f"cdc|{item['state']}|{item['year']}|{item['week']}",
                }
            )
    return pd.DataFrame(rows), manifests


def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _country_from_event_text(text: str, country_centroids: dict[str, tuple[float, float]]) -> str | None:
    lowered = text.lower()
    if "multi-country" in lowered or "americas" in lowered:
        return None
    country_names = sorted(
        {
            name
            for name in country_centroids
            if name and name == name.title() and len(name) > 3
        },
        key=len,
        reverse=True,
    )
    for country in country_names:
        if re.search(rf"\b{re.escape(country)}\b", text, flags=re.IGNORECASE):
            return country
    for alias, canonical in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
            return canonical
    return None


def parse_who_don(
    session: requests.Session,
    country_centroids: dict[str, tuple[float, float]],
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[InputManifest]]:
    queries = [
        (
            "title",
            f"{WHO_DON_URL}?$filter=contains(tolower(Title),'hanta')&$top=50",
            "Disease Outbreak News items with hantavirus text in Title.",
        ),
        (
            "summary",
            f"{WHO_DON_URL}?$filter=contains(tolower(Summary),'hanta')&$top=50",
            "Disease Outbreak News items with hantavirus text in Summary.",
        ),
    ]
    manifests: list[InputManifest] = []
    items_by_id: dict[str, dict[str, Any]] = {}
    for label, url, description in queries:
        cache_path = EXTERNAL_CACHE / "who" / f"diseaseoutbreaknews_hantavirus_{label}.json"
        _get_with_cache(
            session,
            url,
            cache_path,
            refresh=refresh,
            empty_payload='{"value":[]}',
            timeout=90,
        )
        manifests.append(
            _manifest_entry(
                cache_path,
                url,
                "WHO website terms and conditions",
                "World Health Organization. Disease Outbreak News API.",
                description,
            )
        )
        data = _read_json(cache_path, {"value": []})
        for item in data.get("value", []):
            key = str(item.get("Id") or item.get("UrlName") or len(items_by_id))
            items_by_id[key] = item
    rows: list[dict[str, Any]] = []
    for item in items_by_id.values():
        text = " ".join(
            [
                _strip_html(item.get("Title")),
                _strip_html(item.get("TitleSuffix")),
                _strip_html(item.get("Summary")),
                _strip_html(item.get("Overview")),
                _strip_html(item.get("Epidemiology")),
            ]
        )
        if "hanta" not in text.lower():
            continue
        country = _country_from_event_text(text, country_centroids)
        lon, lat = _country_point(country, country_centroids)
        if lon is None or lat is None:
            continue
        dt = pd.to_datetime(item.get("PublicationDate") or item.get("PublicationDateAndTime"), errors="coerce")
        if pd.isna(dt):
            continue
        rows.append(
            {
                "date": pd.Timestamp(dt).tz_localize(None) if getattr(dt, "tzinfo", None) else pd.Timestamp(dt),
                "source": "who_don",
                "virus_species": "Hantavirus",
                "country": country,
                "admin1": None,
                "lon": lon,
                "lat": lat,
                "count": 1,
                "confidence": 0.30,
                "_dedupe_key": f"who|{item.get('Id') or item.get('UrlName')}",
            }
        )
    return pd.DataFrame(rows), manifests


def _dedupe_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    columns = ["date", "source", "virus_species", "country", "admin1", "lon", "lat", "count", "confidence"]
    if not frames:
        return pd.DataFrame(columns=columns)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date", "lon", "lat"])
    panel = panel[(panel["lon"].between(-180, 180)) & (panel["lat"].between(-90, 90))]
    panel["count"] = panel["count"].fillna(1).astype(int).clip(lower=1)
    panel["confidence"] = panel["confidence"].astype(float).clip(0, 1)
    panel["_source_priority"] = panel["source"].map(
        {
            "genbank_biosample": 4,
            "genbank_nuccore": 3,
            "cdc_nndss": 2,
            "who_don": 1,
        }
    ).fillna(0)
    panel["_dedupe_key"] = panel["_dedupe_key"].fillna(
        panel.apply(
            lambda row: "|".join(
                [
                    str(row["source"]),
                    str(row["date"].date()),
                    str(row["country"]),
                    str(row["admin1"]),
                    f"{row['lon']:.4f}",
                    f"{row['lat']:.4f}",
                ]
            ),
            axis=1,
        )
    )
    panel = panel.sort_values(["confidence", "_source_priority"], ascending=False)
    panel = panel.drop_duplicates("_dedupe_key", keep="first")
    return panel[columns].sort_values(["date", "source", "country"]).reset_index(drop=True)


def write_manifest(manifests: list[InputManifest], panel: gpd.GeoDataFrame) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    unique: dict[str, InputManifest] = {}
    for entry in manifests:
        unique[entry.path] = entry
    payload = {
        "created_at_utc": _now_utc(),
        "output": {
            "path": str(OUTPUT_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(OUTPUT_PATH) if OUTPUT_PATH.exists() else "",
            "size_bytes": OUTPUT_PATH.stat().st_size if OUTPUT_PATH.exists() else 0,
            "row_count": int(len(panel)),
            "crs": str(panel.crs),
        },
        "inputs": [entry.__dict__ for entry in sorted(unique.values(), key=lambda x: x.path)],
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_cases_panel(*, refresh: bool = False, ncbi_retmax: int = 20000) -> gpd.GeoDataFrame:
    session = _request_session()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXTERNAL_CACHE.mkdir(parents=True, exist_ok=True)

    country_centroids, country_manifest = ensure_country_centroids(session, refresh=refresh)
    state_centers, state_manifest = ensure_state_centers(session, refresh=refresh)

    manifests: list[InputManifest] = []
    manifests.extend(country_manifest)
    manifests.extend(state_manifest)

    frames: list[pd.DataFrame] = []
    nuccore, nuccore_manifest = parse_ncbi_nuccore(
        session,
        country_centroids,
        refresh=refresh,
        retmax=ncbi_retmax,
    )
    frames.append(nuccore)
    manifests.extend(nuccore_manifest)

    biosample, biosample_manifest = parse_ncbi_biosample(
        session,
        country_centroids,
        refresh=refresh,
    )
    frames.append(biosample)
    manifests.extend(biosample_manifest)

    cdc, cdc_manifest = parse_cdc_nndss(session, state_centers, refresh=refresh)
    frames.append(cdc)
    manifests.extend(cdc_manifest)

    who, who_manifest = parse_who_don(session, country_centroids, refresh=refresh)
    frames.append(who)
    manifests.extend(who_manifest)

    panel = _dedupe_panel(frames)
    geometry = [Point(xy) for xy in zip(panel["lon"], panel["lat"])]
    gdf = gpd.GeoDataFrame(panel, geometry=geometry, crs="EPSG:4326")
    gdf.to_parquet(OUTPUT_PATH, index=False)
    write_manifest(manifests, gdf)
    return gdf


def panel_summary(gdf: gpd.GeoDataFrame) -> str:
    if gdf.empty:
        return "N points: 0, sources: {}, countries: 0, date range: NA"
    sources = dict(sorted(Counter(gdf["source"]).items()))
    countries = int(gdf["country"].dropna().nunique())
    dates = pd.to_datetime(gdf["date"], errors="coerce").dropna()
    date_range = f"{dates.min().date()} to {dates.max().date()}" if not dates.empty else "NA"
    return f"N points: {len(gdf):,}, sources: {sources}, countries: {countries}, date range: {date_range}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TEG hantavirus cases panel.")
    parser.add_argument("--refresh", action="store_true", help="Refresh upstream caches.")
    parser.add_argument(
        "--ncbi-retmax",
        type=int,
        default=20000,
        help="Maximum nuccore records to fetch from NCBI.",
    )
    args = parser.parse_args()
    gdf = build_cases_panel(refresh=args.refresh, ncbi_retmax=args.ncbi_retmax)
    print(panel_summary(gdf))
    if not gdf.empty:
        point_fraction = float((gdf["confidence"] >= 0.95).mean())
        print(f"Point-level coordinate fraction: {point_fraction:.1%}")
        print(f"Saved: {OUTPUT_PATH}")
        print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
