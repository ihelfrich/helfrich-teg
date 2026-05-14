"""Administrative boundary helpers for areal exposure evidence.

This module caches Natural Earth Admin 1 states/provinces and provides a
conservative label matcher for case records that have country and admin1 text
but no point-level coordinates. Matches are exact after light normalization and
simple descriptor stripping. Ambiguous or locality-like labels remain unmatched
instead of being promoted to fake local exposure.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import certifi
import geopandas as gpd
import requests


NE_ADMIN1_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
NE_ADMIN1_CACHE = Path("/Volumes/HELFRICH-GD/TEG_data/inputs/boundaries/ne_10m_admin_1_states_provinces.zip")

COUNTRY_ALIASES = {
    "united states": "united states of america",
    "usa": "united states of america",
    "u s a": "united states of america",
    "russian federation": "russia",
    "czech republic": "czechia",
    "viet nam": "vietnam",
    "republic of korea": "south korea",
}

ADMIN_DESCRIPTOR_RE = re.compile(
    r"\b("
    r"province|prov|state|region|oblast|krai|kray|republic|respublika|"
    r"department|departement|county|island|autonomous|municipality|city"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AdminMatch:
    status: str
    index: int | None
    match_key: str | None
    matched_name: str | None
    matched_name_en: str | None
    iso_3166_2: str | None
    adm1_code: str | None


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60, verify=certifi.where()) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_ne_admin1(*, download: bool = True) -> Path:
    """Return the cached Natural Earth Admin 1 zip path."""
    if NE_ADMIN1_CACHE.exists():
        return NE_ADMIN1_CACHE
    if not download:
        raise FileNotFoundError(NE_ADMIN1_CACHE)
    _download(NE_ADMIN1_URL, NE_ADMIN1_CACHE)
    with ZipFile(NE_ADMIN1_CACHE) as zf:
        if not any(name.lower().endswith(".shp") for name in zf.namelist()):
            raise FileNotFoundError(f"No shapefile found inside {NE_ADMIN1_CACHE}")
    return NE_ADMIN1_CACHE


def normalize_label(value: object) -> str:
    """Normalize free-text country/admin labels for exact matching."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return ""
    text = (
        text.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("Ä", "a")
        .replace("Ö", "o")
        .replace("Ü", "u")
        .replace("ß", "ss")
    )
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return COUNTRY_ALIASES.get(text, text)


def _strip_admin_descriptors(value: str) -> str:
    text = ADMIN_DESCRIPTOR_RE.sub(" ", value)
    text = re.sub(r"\b(the|of|and)\b", " ", text, flags=re.IGNORECASE)
    return normalize_label(text)


def admin_label_variants(value: object) -> set[str]:
    """Return conservative normalized variants for an admin label."""
    raw = "" if value is None else str(value).strip()
    if not raw:
        return set()
    pieces = {raw}
    pieces.add(re.sub(r"\([^)]*\)", " ", raw))
    if "," in raw:
        pieces.add(raw.split(",", 1)[0])

    variants: set[str] = set()
    for piece in pieces:
        base_norm = normalize_label(piece)
        germanized = (
            str(piece)
            .replace("ae", "a")
            .replace("oe", "o")
            .replace("ue", "u")
            .replace("Ae", "A")
            .replace("Oe", "O")
            .replace("Ue", "U")
        )
        for norm in {base_norm, normalize_label(germanized)}:
            if norm:
                variants.add(norm)
                stripped = _strip_admin_descriptors(norm)
                if stripped:
                    variants.add(stripped)
    return variants


def load_admin1_boundaries(*, download: bool = True) -> gpd.GeoDataFrame:
    """Load Natural Earth Admin 1 boundaries with normalized country labels."""
    path = ensure_ne_admin1(download=download)
    gdf = gpd.read_file(f"zip://{path}")
    gdf = gdf.loc[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf["_country_norm"] = gdf["admin"].map(normalize_label)
    return gdf


def _candidate_values(row) -> set[str]:
    values: set[str] = set()
    for col in [
        "name",
        "name_en",
        "name_alt",
        "name_local",
        "gn_name",
        "gns_name",
        "woe_name",
        "postal",
        "iso_3166_2",
        "abbrev",
    ]:
        value = row.get(col)
        if value is None:
            continue
        for part in str(value).split("|"):
            values.update(admin_label_variants(part))
    return {value for value in values if value}


def build_admin1_index(gdf: gpd.GeoDataFrame) -> dict[tuple[str, str], list[int]]:
    """Build a country-scoped normalized label index for Admin 1 rows."""
    index: dict[tuple[str, str], list[int]] = {}
    for idx, row in gdf.iterrows():
        country = row["_country_norm"]
        for value in _candidate_values(row):
            index.setdefault((country, value), []).append(int(idx))
    return index


def match_admin1(
    country: object,
    admin1: object,
    boundaries: gpd.GeoDataFrame,
    index: dict[tuple[str, str], list[int]],
) -> AdminMatch:
    """Match a country/admin1 label to one Natural Earth Admin 1 polygon."""
    country_norm = normalize_label(country)
    if not country_norm:
        return AdminMatch("missing_country", None, None, None, None, None, None)

    candidate_hits: list[tuple[str, list[int]]] = []
    for key in admin_label_variants(admin1):
        hits = sorted(set(index.get((country_norm, key), [])))
        if hits:
            candidate_hits.append((key, hits))
    if not candidate_hits:
        return AdminMatch("unmatched", None, None, None, None, None, None)

    unique_hits = sorted({hit for _, hits in candidate_hits for hit in hits})
    if len(unique_hits) != 1:
        key = sorted(candidate_hits, key=lambda item: (len(item[1]), -len(item[0])))[0][0]
        return AdminMatch("ambiguous", None, key, None, None, None, None)

    hit = unique_hits[0]
    key = sorted(
        (key for key, hits in candidate_hits if hit in hits),
        key=len,
        reverse=True,
    )[0]
    row = boundaries.loc[hit]
    return AdminMatch(
        "matched",
        int(hit),
        key,
        None if row.get("name") is None else str(row.get("name")),
        None if row.get("name_en") is None else str(row.get("name_en")),
        None if row.get("iso_3166_2") is None else str(row.get("iso_3166_2")),
        None if row.get("adm1_code") is None else str(row.get("adm1_code")),
    )
