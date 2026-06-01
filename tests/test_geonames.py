def test_get_adm1_country_entry(geonames_service_test_data):
    nld = geonames_service_test_data.get_adm1_country_entry("North Holland", "NLD")
    assert nld is not None
    assert nld["geonameid"] == "2749879"

    # Without country filter
    nld2 = geonames_service_test_data.get_adm1_country_entry("North Holland", None)
    assert nld2 is not None
    assert nld2["geonameid"] == "2749879"

    # Non-existent entry
    xyz = geonames_service_test_data.get_adm1_country_entry("NonExistent", None)
    assert xyz is None


def test_get_country_entry(geonames_service_test_data):
    res = geonames_service_test_data.get_country_entry("SYR")
    assert res is not None
    assert res["feature_code"] == "PCLI"
    assert res["country_code3"] == "SYR"


def test_get_country_by_name(geonames_service_test_data):
    res = geonames_service_test_data.get_country_by_name("Cuba")
    assert res is not None
    assert res["feature_code"] == "PCLI"
    assert res["geonameid"] == "3562981"

    res = geonames_service_test_data.get_country_by_name("Syria")
    assert res is not None
    assert res["country_code3"] == "SYR"
    assert res["feature_code"] == "PCLI"

    # Non-existent country
    res = geonames_service_test_data.get_country_by_name("Atlantis")
    assert res is None


def test_get_entry_by_id(geonames_service_test_data):
    # Berlin
    res = geonames_service_test_data.get_entry_by_id("2950159")
    assert res is not None
    assert res["name"] is not None
    assert res["country_code3"] == "DEU"


def test_search_by_name_exact(geonames_service_all_data):
    res = geonames_service_all_data.search_by_name("Paris")
    hits = res["hits"]["hits"]
    assert len(hits) > 0
    names = [h["_source"]["name"] for h in hits]
    assert any("Paris" in n for n in names)


def test_search_by_name_known_country(geonames_service_all_data):
    res = geonames_service_all_data.search_by_name("Paris", known_country="FRA")
    hits = res["hits"]["hits"]
    assert len(hits) > 0
    assert all(h["_source"]["country_code3"] == "FRA" for h in hits)


def test_search_by_name_fuzzy(geonames_service_all_data):
    # Exact match finds nothing for a misspelling; fuzzy should recover
    exact = geonames_service_all_data.search_by_name("Belin")  # typo for Berlin
    fuzzy = geonames_service_all_data.search_by_name("Belin", fuzzy=1)
    assert len(fuzzy["hits"]["hits"]) >= len(exact["hits"]["hits"])


def test_search_by_name_limit_types(geonames_service_all_data):
    res = geonames_service_all_data.search_by_name("London", limit_types=True)
    hits = res["hits"]["hits"]
    assert all(h["_source"]["feature_class"] in ("P", "A") for h in hits)


def test_result_shape(geonames_service_all_data):
    """Every hit must have the fields that res_formatter expects."""
    res = geonames_service_all_data.search_by_name("Berlin")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        for field in ("name", "asciiname", "alternativenames", "feature_class",
                      "feature_code", "country_code3", "admin1_name",
                      "geonameid", "lat", "lon", "coordinates"):
            assert field in src, f"Missing field: {field}"
        assert isinstance(src["alternativenames"], list)


def test_determine_data_extent(geonames_service_all_data):
    from mordecai3.geonames import DataExtent
    extent = geonames_service_all_data.determine_data_extent()
    assert extent == DataExtent.ALL
