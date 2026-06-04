"""
geonames.py — Postgres-backed replacement for the Elasticsearch GeonamesService.

Drop-in replacement for the original mordecai3 geonames.py. The public API
(GeonamesService methods and their return shapes) is identical so that
geoparse.py requires no changes.

Key differences from the ES version:
- Candidate lookup queries the gazetteer table directly (place_name rows)
  rather than an Elasticsearch index.
- alternate_name_count and admin1_name/country_code3 come from
  geonames_place_stats (a separate normalised table populated in step 02).
- Fuzzy search uses pg_trgm similarity rather than ES fuzziness parameter.
- The DataExtent check queries Postgres instead of ES.
- Connection pool uses psycopg (v3) ConnectionPool.

One mordecai dependency removed: elasticsearch + elasticsearch_dsl.
New dependency added: psycopg[pool] (psycopg v3).
All other mordecai dependencies (spacy, torch, jellyfish, numpy) unchanged.
"""

import logging
import re
from enum import IntEnum

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# DataExtent (unchanged from original)
# ---------------------------------------------------------------------------

class DataExtent(IntEnum):
    NA   = 0  # connection problems or missing tables
    NONE = 1
    TEST = 2
    ALL  = 3


# ---------------------------------------------------------------------------
# Connection pool helper
# ---------------------------------------------------------------------------

def setup_pg_pool(dsn: str, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    """
    Create a psycopg v3 ConnectionPool.

    Parameters
    ----------
    dsn : str
        libpq connection string, e.g.
        "host=localhost dbname=geoindex user=postgres password=secret"
    min_size, max_size : int
        Pool size limits. For multiprocessing workloads set max_size >= workers.

    Returns
    -------
    psycopg_pool.ConnectionPool
    """
    return ConnectionPool(dsn, min_size=min_size, max_size=max_size)


# ---------------------------------------------------------------------------
# lat/lon from H3 index
# ---------------------------------------------------------------------------
# mordecai uses lat/lon only for display and for the document geographic
# prior computed in geoparse.py. We derive them from the H3 cell centroid.

try:
    import h3 as h3lib
    def _h3_to_latlon(h3_index: str):
        lat, lon = h3lib.cell_to_latlng(h3_index)
        return float(lat), float(lon)
except ImportError:
    logger.warning("h3 library not available; lat/lon will be 0.0")
    def _h3_to_latlon(h3_index: str):
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch(conn: psycopg.Connection, sql: str, params) -> list[dict]:
    """Execute a query and return all rows as plain dicts."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _format_rows(rows: list[dict]) -> list[dict]:
    """
    Convert raw DB rows into the dict shape that res_formatter() in
    geoparse.py and _format_country_results() below expect.

    The ES version stored alternativenames as a list in the index.
    Here we get it from a GROUP BY array_agg in the query.

    The returned shape mirrors the ES _source dict exactly so that
    res_formatter() in geoparse.py requires no changes.
    """
    seen = set()
    out = []
    for r in rows:
        gid = r['geonames_id']
        if gid in seen:
            continue
        seen.add(gid)
        lat, lon = _h3_to_latlon(r['h3_index'])
        out.append({
            'geonameid':        str(gid),
            'name':             r['name'],
            # asciiname: best approximation without storing it separately;
            # good enough for the ascii_dist Levenshtein feature.
            'asciiname':        r['name'],
            'alternativenames': list(r['alternativenames'] or []),
            'feature_class':    r['feature_class'] or '',
            'feature_code':     r['feature_code'] or '',
            'country_code3':    r['country_code3'] or '',
            'admin1_name':      r['admin1_name'] or '',
            'admin2_name':      '',   # not used by the ranker
            'admin1_code':      '',   # not used by the ranker
            'admin2_code':      '',   # not used by the ranker
            'alt_name_length':  int(r['alternate_name_count'] or 0),
            'lat':              lat,
            'lon':              lon,
            'coordinates':      f"{lat},{lon}",
        })
    return out


def _format_country_results(formatted_rows: list[dict]) -> dict | None:
    """
    Return a single result dict in the shape that geoparse.py expects from
    the single-entry lookup methods. Mirrors the original exactly.
    """
    if not formatted_rows:
        return None
    r = formatted_rows[0]
    return {
        "extracted_name": "",
        "name":           r['name'],
        "lat":            r['lat'],
        "lon":            r['lon'],
        "admin1_name":    r['admin1_name'],
        "admin2_name":    "",
        "country_code3":  r['country_code3'],
        "feature_code":   r['feature_code'],
        "feature_class":  r['feature_class'],
        "geonameid":      r['geonameid'],
        "start_char":     "",
        "end_char":       "",
    }


def _clean_search_name(search_name: str) -> str:
    """
    Strip administrative suffixes that prevent correct candidate matching.
    Identical logic to the original; uses word-boundary regex to avoid
    partial-word stripping (e.g. "City" in "Mexico City").
    """
    search_name = re.sub(r"(?i)^the\s+",            "", search_name).strip()
    search_name = re.sub(r"(?i)\btribal district\b", "", search_name).strip()
    search_name = re.sub(r"(?i)\bcity\b",            "", search_name).strip()
    search_name = re.sub(r"(?i)\bdistrict\b",        "", search_name).strip()
    search_name = re.sub(r"(?i)\bmetropolis\b",      "", search_name).strip()
    search_name = re.sub(r"(?i)\bcounty\b",          "", search_name).strip()
    search_name = re.sub(r"(?i)\bregion\b",          "", search_name).strip()
    search_name = re.sub(r"(?i)\bprovince\b",        "", search_name).strip()
    search_name = re.sub(r"(?i)\bterritory\b",       "", search_name).strip()
    search_name = re.sub(r"(?i)\bbranch\b",          "", search_name).strip()
    search_name = re.sub(r"'s$",                     "", search_name).strip()
    if search_name.upper() == "US":
        search_name = "United States"
    return search_name


# ---------------------------------------------------------------------------
# GeonamesService
# ---------------------------------------------------------------------------

# SQL for single-entry lookups (no alternativenames self-join needed).
_SINGLE_ENTRY_SELECT = """
    SELECT DISTINCT ON (g.geonames_id)
        g.geonames_id,
        g.place_name                                     AS name,
        g.feature_class,
        g.feature_code,
        g.h3_index,
        COALESCE(ps.country_code3, '')                   AS country_code3,
        COALESCE(ps.admin1_name,   '')                   AS admin1_name,
        COALESCE(ps.alternate_name_count, 0)             AS alternate_name_count,
        ARRAY[]::TEXT[]                                  AS alternativenames
    FROM gazetteer g
    LEFT JOIN geonames_place_stats ps ON ps.geonames_id = g.geonames_id
"""


class GeonamesService:
    """
    Postgres-backed replacement for the ES GeonamesService.

    All public method signatures and return shapes are identical to the
    original so geoparse.py requires no changes.

    Parameters
    ----------
    pg_pool : psycopg_pool.ConnectionPool
        Created by setup_pg_pool(dsn).
    """

    def __init__(self, pg_pool: ConnectionPool):
        self._pool = pg_pool

    # ------------------------------------------------------------------
    # DataExtent check
    # ------------------------------------------------------------------

    def determine_data_extent(self) -> DataExtent:
        """
        Check whether the gazetteer has full data.
        Mirrors the original: looks for New York ADM1 (USA) and
        North Holland ADM1 (NLD).
        """
        try:
            usa = self.get_adm1_country_entry("New York", "USA")
            nld = self.get_adm1_country_entry("North Holland", "NLD")
            if usa and nld:
                return DataExtent.ALL
            elif nld:
                return DataExtent.TEST
            else:
                return DataExtent.NONE
        except Exception as e:
            logger.error("determine_data_extent failed: %s", e)
            return DataExtent.NA

    # ------------------------------------------------------------------
    # Single-entry lookups
    # ------------------------------------------------------------------

    def get_entry_by_id(self, geonameid: str) -> dict | None:
        """Return a single gazetteer entry by geonames_id."""
        sql = _SINGLE_ENTRY_SELECT + """
            WHERE g.geonames_id = %s
            ORDER BY g.geonames_id, g.importance DESC NULLS LAST
            LIMIT 1
        """
        with self._pool.connection() as conn:
            rows = _fetch(conn, sql, (int(geonameid),))
        return _format_country_results(_format_rows(rows))

    def get_adm1_country_entry(self,
                               adm1: str,
                               iso3c: str | None = None) -> dict | None:
        """Return the ADM1 entry for a state/province, optionally filtered by ISO3."""
        if iso3c:
            sql = _SINGLE_ENTRY_SELECT + """
                WHERE g.feature_code = 'ADM1'
                  AND ps.country_code3 = %s
                  AND lower(g.place_name) = lower(%s)
                ORDER BY g.geonames_id, g.importance DESC NULLS LAST
                LIMIT 1
            """
            params = (iso3c, adm1)
        else:
            sql = _SINGLE_ENTRY_SELECT + """
                WHERE g.feature_code = 'ADM1'
                  AND lower(g.place_name) = lower(%s)
                ORDER BY g.geonames_id, g.importance DESC NULLS LAST
                LIMIT 1
            """
            params = (adm1,)

        with self._pool.connection() as conn:
            rows = _fetch(conn, sql, params)
        return _format_country_results(_format_rows(rows))

    def get_country_entry(self, iso3c: str) -> dict | None:
        """Return the PCLI entry for a country given its ISO3 code."""
        sql = _SINGLE_ENTRY_SELECT + """
            WHERE g.feature_code = 'PCLI'
              AND ps.country_code3 = %s
            ORDER BY g.geonames_id, g.importance DESC NULLS LAST
            LIMIT 1
        """
        with self._pool.connection() as conn:
            rows = _fetch(conn, sql, (iso3c,))
        return _format_country_results(_format_rows(rows))

    def get_country_by_name(self, country_name: str) -> dict | None:
        """Return the PCLI entry for a country matched by name."""
        sql = _SINGLE_ENTRY_SELECT + """
            WHERE g.feature_code = 'PCLI'
              AND lower(g.place_name) = lower(%s)
            ORDER BY g.geonames_id, g.importance DESC NULLS LAST
            LIMIT 1
        """
        with self._pool.connection() as conn:
            rows = _fetch(conn, sql, (country_name,))
        return _format_country_results(_format_rows(rows))

    # ------------------------------------------------------------------
    # Main candidate search — the hot path
    # ------------------------------------------------------------------

    def search_by_name(self,
                       search_name: str,
                       max_results: int = 50,
                       fuzzy: int = 0,
                       limit_types: bool = False,
                       known_country: str | None = None) -> dict:
        """
        Search for candidate gazetteer entries matching search_name.

        Returns a dict in the ES response envelope shape so that
        res_formatter() in geoparse.py requires no changes:
            {'hits': {'hits': [{'_source': {...}}, ...]}}

        Parameters
        ----------
        search_name : str
            Toponym as extracted from the document.
        max_results : int
            Maximum candidates (default 50, matches ES default).
        fuzzy : int
            0 = exact case-insensitive match.
            1+ = pg_trgm similarity fallback with decreasing threshold.
        limit_types : bool
            Restrict to feature_class IN ('P', 'A').
        known_country : str or None
            ISO3 country code to restrict results.
        """
        search_name = _clean_search_name(search_name)

        with self._pool.connection() as conn:
            rows = self._search(conn, search_name, int(max_results),
                                fuzzy, limit_types, known_country)

        formatted = _format_rows(rows)
        return {'hits': {'hits': [{'_source': r} for r in formatted]}}

    def _search(self,
                conn: psycopg.Connection,
                search_name: str,
                max_results: int,
                fuzzy: int,
                limit_types: bool,
                known_country: str | None) -> list[dict]:
        """
        Core candidate query.

        - Inner subquery: one row per geonames_id, highest-importance match.
        - Outer query: joins geonames_place_stats for ranker features and
          self-joins gazetteer to aggregate all name variants as
          alternativenames[], mirroring the ES stored field.
        - Ordered by alternate_name_count DESC (prominence) matching the
          original ES sort on alt_name_length.

        Requires:
          CREATE INDEX ON gazetteer (lower(place_name));
          CREATE EXTENSION pg_trgm;
          CREATE INDEX ON gazetteer USING GIN (place_name gin_trgm_ops);
        """
        type_filter    = "AND inner_g.feature_class IN ('P', 'A')" if limit_types else ""
        country_filter = "AND ps.country_code3 = %(known_country)s" if known_country else ""

        if fuzzy == 0:
            match_filter = "AND lower(inner_g.place_name) = lower(%(search_name)s)"
        else:
            threshold = max(0.2, 0.5 - fuzzy * 0.15)
            match_filter = (
                f"AND similarity(inner_g.place_name, %(search_name)s) > {threshold}"
            )

        sql = f"""
            SELECT
                g.geonames_id,
                g.place_name                                         AS name,
                g.feature_class,
                g.feature_code,
                g.h3_index,
                COALESCE(ps.country_code3, '')                       AS country_code3,
                COALESCE(ps.admin1_name,   '')                       AS admin1_name,
                COALESCE(ps.alternate_name_count, 0)                 AS alternate_name_count,
                COALESCE(
                    array_agg(DISTINCT other.place_name)
                        FILTER (WHERE other.place_name IS NOT NULL
                                  AND lower(other.place_name) <> lower(g.place_name)),
                    ARRAY[]::TEXT[]
                )                                                    AS alternativenames
            FROM (
                SELECT DISTINCT ON (inner_g.geonames_id)
                    inner_g.geonames_id,
                    inner_g.place_name,
                    inner_g.feature_class,
                    inner_g.feature_code,
                    inner_g.h3_index
                FROM gazetteer inner_g
                WHERE inner_g.geonames_id IS NOT NULL
                  {match_filter}
                  {type_filter}
                ORDER BY inner_g.geonames_id, inner_g.importance DESC NULLS LAST
            ) g
            LEFT JOIN geonames_place_stats ps
                   ON ps.geonames_id = g.geonames_id
            LEFT JOIN gazetteer other
                   ON other.geonames_id = g.geonames_id
            WHERE TRUE
              {country_filter}
            GROUP BY
                g.geonames_id,
                g.place_name,
                g.feature_class,
                g.feature_code,
                g.h3_index,
                ps.country_code3,
                ps.admin1_name,
                ps.alternate_name_count
            ORDER BY ps.alternate_name_count DESC NULLS LAST
            LIMIT %(max_results)s
        """

        params = {
            'search_name':   search_name,
            'max_results':   max_results,
            'known_country': known_country,
        }
        return _fetch(conn, sql, params)
