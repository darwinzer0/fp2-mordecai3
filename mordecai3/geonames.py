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
- Connection is a psycopg2 connection pool rather than an ES client.
"""

import logging
import re
from enum import IntEnum

import psycopg2
import psycopg2.extras
from psycopg2 import pool

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# DataExtent
# ---------------------------------------------------------------------------

class DataExtent(IntEnum):
    NA   = 0  # connection problems or missing tables
    NONE = 1
    TEST = 2
    ALL  = 3


# ---------------------------------------------------------------------------
# Connection pool helper
# ---------------------------------------------------------------------------

def setup_pg_pool(dsn: str, minconn: int = 1, maxconn: int = 10):
    """
    Create a psycopg2 ThreadedConnectionPool.

    Parameters
    ----------
    dsn : str
        libpq connection string, e.g.
        "host=localhost dbname=geoindex user=postgres password=secret"
    minconn, maxconn : int
        Pool size limits. For multiprocessing workloads set maxconn >= workers.

    Returns
    -------
    psycopg2.pool.ThreadedConnectionPool
    """
    return pool.ThreadedConnectionPool(minconn, maxconn, dsn)


# ---------------------------------------------------------------------------
# lat/lon from H3 index
# ---------------------------------------------------------------------------
# mordecai uses lat/lon only for display and for the document geographic
# prior computed in geoparse.py. We derive them from the H3 cell centroid.

try:
    import h3 as h3lib
    def _h3_to_latlon(h3_index: str):
        lat, lon = h3lib.h3_to_geo(h3_index)
        return float(lat), float(lon)
except ImportError:
    logger.warning("h3 library not available; lat/lon will be 0.0")
    def _h3_to_latlon(h3_index: str):
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch(conn, sql: str, params) -> list:
    """Execute a query and return all rows as RealDictRow list."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _format_rows(rows: list) -> list:
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
            'geonameid':         str(gid),
            'name':              r['name'],
            # asciiname: best approximation available without storing it
            # separately; good enough for the ascii_dist Levenshtein feature.
            'asciiname':         r['name'],
            'alternativenames':  list(r['alternativenames'] or []),
            'feature_class':     r['feature_class'] or '',
            'feature_code':      r['feature_code'] or '',
            'country_code3':     r['country_code3'] or '',
            'admin1_name':       r['admin1_name'] or '',
            'admin2_name':       '',   # not used by the ranker
            'admin1_code':       '',   # not used by the ranker
            'admin2_code':       '',   # not used by the ranker
            'alt_name_length':   int(r['alternate_name_count'] or 0),
            'lat':               lat,
            'lon':               lon,
            # coordinates string format expected by _format_country_results
            'coordinates':       f"{lat},{lon}",
        })
    return out


def _format_country_results(formatted_rows: list) -> dict | None:
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
    Identical logic to the original; converted to use word-boundary regex
    to avoid partial-word stripping (e.g. "City" in "Mexico City" → "Mexico").
    """
    search_name = re.sub(r"(?i)^the\s+",       "", search_name).strip()
    search_name = re.sub(r"(?i)\btribal district\b", "", search_name).strip()
    search_name = re.sub(r"(?i)\bcity\b",       "", search_name).strip()
    search_name = re.sub(r"(?i)\bdistrict\b",   "", search_name).strip()
    search_name = re.sub(r"(?i)\bmetropolis\b", "", search_name).strip()
    search_name = re.sub(r"(?i)\bcounty\b",     "", search_name).strip()
    search_name = re.sub(r"(?i)\bregion\b",     "", search_name).strip()
    search_name = re.sub(r"(?i)\bprovince\b",   "", search_name).strip()
    search_name = re.sub(r"(?i)\bterritory\b",  "", search_name).strip()
    search_name = re.sub(r"(?i)\bbranch\b",     "", search_name).strip()
    search_name = re.sub(r"'s$",                "", search_name).strip()
    if search_name.upper() == "US":
        search_name = "United States"
    return search_name


# ---------------------------------------------------------------------------
# GeonamesService
# ---------------------------------------------------------------------------

class GeonamesService:
    """
    Postgres-backed replacement for the ES GeonamesService.

    All public method signatures and return shapes are identical to the
    original so geoparse.py requires no changes.

    Parameters
    ----------
    pg_pool : psycopg2.pool.ThreadedConnectionPool
        Created by setup_pg_pool(dsn).
    """

    def __init__(self, pg_pool):
        self._pool = pg_pool

    def _conn(self):
        return self._pool.getconn()

    def _release(self, conn):
        self._pool.putconn(conn)

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
    # Used by geoparse.py for in_rel parent resolution and hierarchy lookup.
    # These don't need the alternativenames array so we skip the self-join.
    # ------------------------------------------------------------------

    def get_entry_by_id(self, geonameid: str) -> dict | None:
        """Return a single gazetteer entry by geonames_id."""
        sql = """
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
            WHERE g.geonames_id = %s
            ORDER BY g.geonames_id, g.importance DESC NULLS LAST
            LIMIT 1
        """
        conn = self._conn()
        try:
            rows = _fetch(conn, sql, (int(geonameid),))
        finally:
            self._release(conn)
        return _format_country_results(_format_rows(rows))

    def get_adm1_country_entry(self,
                               adm1: str,
                               iso3c: str | None = None) -> dict | None:
        """Return the ADM1 entry for a state/province name, optionally filtered by ISO3."""
        if iso3c:
            sql = """
                SELECT DISTINCT ON (g.geonames_id)
                    g.geonames_id,
                    g.place_name    AS name,
                    g.feature_class,
                    g.feature_code,
                    g.h3_index,
                    COALESCE(ps.country_code3, '')       AS country_code3,
                    COALESCE(ps.admin1_name,   '')       AS admin1_name,
                    COALESCE(ps.alternate_name_count, 0) AS alternate_name_count,
                    ARRAY[]::TEXT[]                      AS alternativenames
                FROM gazetteer g
                LEFT JOIN geonames_place_stats ps ON ps.geonames_id = g.geonames_id
                WHERE g.feature_code = 'ADM1'
                  AND ps.country_code3 = %s
                  AND lower(g.place_name) = lower(%s)
                ORDER BY g.geonames_id, g.importance DESC NULLS LAST
                LIMIT 1
            """
            params = (iso3c, adm1)
        else:
            sql = """
                SELECT DISTINCT ON (g.geonames_id)
                    g.geonames_id,
                    g.place_name    AS name,
                    g.feature_class,
                    g.feature_code,
                    g.h3_index,
                    COALESCE(ps.country_code3, '')       AS country_code3,
                    COALESCE(ps.admin1_name,   '')       AS admin1_name,
                    COALESCE(ps.alternate_name_count, 0) AS alternate_name_count,
                    ARRAY[]::TEXT[]                      AS alternativenames
                FROM gazetteer g
                LEFT JOIN geonames_place_stats ps ON ps.geonames_id = g.geonames_id
                WHERE g.feature_code = 'ADM1'
                  AND lower(g.place_name) = lower(%s)
                ORDER BY g.geonames_id, g.importance DESC NULLS LAST
                LIMIT 1
            """
            params = (adm1,)

        conn = self._conn()
        try:
            rows = _fetch(conn, sql, params)
        finally:
            self._release(conn)
        return _format_country_results(_format_rows(rows))

    def get_country_entry(self, iso3c: str) -> dict | None:
        """Return the PCLI entry for a country given its ISO3 code."""
        sql = """
            SELECT DISTINCT ON (g.geonames_id)
                g.geonames_id,
                g.place_name    AS name,
                g.feature_class,
                g.feature_code,
                g.h3_index,
                COALESCE(ps.country_code3, '')       AS country_code3,
                COALESCE(ps.admin1_name,   '')       AS admin1_name,
                COALESCE(ps.alternate_name_count, 0) AS alternate_name_count,
                ARRAY[]::TEXT[]                      AS alternativenames
            FROM gazetteer g
            LEFT JOIN geonames_place_stats ps ON ps.geonames_id = g.geonames_id
            WHERE g.feature_code = 'PCLI'
              AND ps.country_code3 = %s
            ORDER BY g.geonames_id, g.importance DESC NULLS LAST
            LIMIT 1
        """
        conn = self._conn()
        try:
            rows = _fetch(conn, sql, (iso3c,))
        finally:
            self._release(conn)
        return _format_country_results(_format_rows(rows))

    def get_country_by_name(self, country_name: str) -> dict | None:
        """Return the PCLI entry for a country matched by name."""
        sql = """
            SELECT DISTINCT ON (g.geonames_id)
                g.geonames_id,
                g.place_name    AS name,
                g.feature_class,
                g.feature_code,
                g.h3_index,
                COALESCE(ps.country_code3, '')       AS country_code3,
                COALESCE(ps.admin1_name,   '')       AS admin1_name,
                COALESCE(ps.alternate_name_count, 0) AS alternate_name_count,
                ARRAY[]::TEXT[]                      AS alternativenames
            FROM gazetteer g
            LEFT JOIN geonames_place_stats ps ON ps.geonames_id = g.geonames_id
            WHERE g.feature_code = 'PCLI'
              AND lower(g.place_name) = lower(%s)
            ORDER BY g.geonames_id, g.importance DESC NULLS LAST
            LIMIT 1
        """
        conn = self._conn()
        try:
            rows = _fetch(conn, sql, (country_name,))
        finally:
            self._release(conn)
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

        conn = self._conn()
        try:
            rows = self._search(conn, search_name, int(max_results),
                                fuzzy, limit_types, known_country)
        finally:
            self._release(conn)

        formatted = _format_rows(rows)
        # Wrap in ES envelope — res_formatter iterates res['hits']['hits']
        return {'hits': {'hits': [{'_source': r} for r in formatted]}}

    def _search(self, conn, search_name: str, max_results: int,
                fuzzy: int, limit_types: bool,
                known_country: str | None) -> list:
        """
        Core candidate query.

        Strategy:
        - Inner subquery finds the best (highest importance) gazetteer row
          per geonames_id that matches the search name.
        - Outer query joins geonames_place_stats for ranker features and
          self-joins gazetteer to aggregate all name variants as
          alternativenames[], mirroring the ES stored field.
        - Results ordered by alternate_name_count DESC (prominence) to
          match the ES sort on alt_name_length.

        pg_trgm index (GIN on place_name) must exist for fuzzy queries to
        be fast. Exact queries use the btree index on place_name.
        """
        # Build optional filter clauses
        type_filter    = "AND inner_g.feature_class IN ('P', 'A')" if limit_types else ""
        country_filter = "AND ps.country_code3 = %(known_country)s" if known_country else ""

        if fuzzy == 0:
            match_filter = "AND lower(inner_g.place_name) = lower(%(search_name)s)"
        else:
            # pg_trgm similarity threshold: 0.35 for fuzzy=1, 0.2 for fuzzy=2+
            threshold = max(0.2, 0.5 - fuzzy * 0.15)
            match_filter = f"AND similarity(inner_g.place_name, %(search_name)s) > {threshold}"

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
                -- One row per geonames_id: highest-importance name variant
                -- that matched the search term.
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
            -- Collect all name variants for this geonames_id so
            -- res_formatter can compute Levenshtein across all of them.
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
