import pytest
import spacy

from mordecai3 import geoparse as geoparse_module
from mordecai3.geoparse import Geoparser, guess_in_rel, make_admin1_counts
from mordecai3.utils import check_spacy_model


if not check_spacy_model():
    pytest.skip("spaCy model not available", allow_module_level=True)


@pytest.fixture(scope="session", autouse=True)
def geo(geonames_service_all_data):
    return Geoparser(geonames=geonames_service_all_data)


def test_no_event_given(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    out = geo.geoparse_doc(text)
    assert out["event_location_raw"] == ""


def test_no_locs(geo):
    text = "President Obama expressed his hope for a peaceful resolution to the fighting."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"] == []


def test_three_locs(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in the cities of Homs and Aleppo."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["geonameid"] == "2950159"  # Berlin
    assert out["geolocated_ents"][1]["geonameid"] == "169577"   # Homs (city)
    assert out["geolocated_ents"][2]["geonameid"] == "170063"   # Aleppo (city)


def test_governorates(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo Governorates."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][1]["geonameid"] == "169575"   # Homs (governorate)
    assert out["geolocated_ents"][2]["geonameid"] == "170062"   # Aleppo (governorate)


@pytest.mark.skip(reason="messes up on capital-D District")
def test_district_upper_term(geo):
    text = "Afghanistan: Southern Radio, Television Highlights 22 February 2021. He added: 'Ten Taliban, including four Pakistani Nationals, were killed in clashes between the commandos and Taliban in Arghistan District on the night of 21 February."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["search_name"] == "Afghanistan"
    assert out["geolocated_ents"][1]["feature_code"] == "ADM2"
    assert out["geolocated_ents"][1]["geonameid"] == "7053299"


def test_district_lower_term(geo):
    text = "Afghanistan: Southern Radio, Television Highlights 22 February 2021. He added: 'Ten Taliban, including four Pakistani Nationals, were killed in clashes between the commandos and Taliban in Arghistan district on the night of 21 February."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["search_name"] == "Afghanistan"
    assert out["geolocated_ents"][1]["feature_code"] == "ADM2"
    assert out["geolocated_ents"][1]["geonameid"] == "7053299"


def test_miss_oxford(geo):
    text = "Ole Miss is located in Oxford."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["admin1_name"] == "Mississippi"
    assert out["geolocated_ents"][0]["geonameid"] == "4440076"


def test_uk_oxford(geo):
    text = "Oxford University, in the town of Oxford, is the best British university."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["geonameid"] == "2640729"


def test_uk_oxford2(geo):
    text = "Oxford is home to Oxford University, one of the best universities in the world."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["geonameid"] == "2640729"


def test_multi_sent(geo):
    text = """Gangster Kulveer Singh and his accomplice, Chamkaur Singh, were shot dead at Naruana village in Bathinda district on Wednesday morning.  Police said the two were shot dead at Singh's house at his native village by another accomplice, Manpreet Singh Manna, who also sustained a bullet injury and was undergoing treatment at the Bathinda Civil Hospital."""
    out = geo.geoparse_doc(text)
    # Just check it doesn't crash and returns some results
    assert "geolocated_ents" in out


def test_prague(geo):
    text = "A group of settlers in Oklahoma named their new town Prague."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["feature_code"] == "ADM1"
    assert out["geolocated_ents"][1]["admin1_name"] == "Oklahoma"

    text = "Barack Obama gave a speech on nuclear weapons in Prague."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["feature_code"] == "PPLC"
    assert out["geolocated_ents"][0]["country_code3"] == "CZE"


def test_pragues(geo):
    out = geo.geoparse_doc("I visited family in Prague.")
    assert out["geolocated_ents"][0]["geonameid"] == "3067696"
    assert out["geolocated_ents"][0]["country_code3"] == "CZE"

    out = geo.geoparse_doc("I visited family in Prague, Oklahoma.")
    assert out["geolocated_ents"][0]["geonameid"] == "4548393"
    assert out["geolocated_ents"][0]["admin1_name"] == "Oklahoma"


def test_geneva(geo):
    text = "On June 16, Russian President Vladimir Putin and his counterpart Joe Biden held talks in Geneva."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][-1]["country_code3"] == "CHE"


def test_geneva_il(geo):
    text = "On June 16, Russian President Vladimir Putin and his counterpart Joe Biden held talks in Geneva, Illinois."
    out = geo.geoparse_doc(text)
    assert out["geolocated_ents"][0]["geonameid"] == "4893591"


def test_index_error(geo):
    text = """Ukraine Reform Conference opens in Vilnius\n\nVILNIUS, Jul 07, BNS – Lithuanian and Ukrainian President Gitanas Nauseda and Volodymyr Zelensky will open the Ukraine Reform Conference in Vilnius on Wednesday."""
    out = geo.geoparse_doc(text)
    assert "geolocated_ents" in out


# -----------------------------------------------------------------------
# Component-level tests
# -----------------------------------------------------------------------

def test_adm1_count():
    out = [
        {"es_choices": [{"admin1_name": "MA"}, {"admin1_name": "England"}]},
        {"es_choices": [{"admin1_name": "MA"}]},
        {"es_choices": [{"admin1_name": "MA"}]},
    ]
    adm1_counts = make_admin1_counts(out)
    assert adm1_counts["MA"] == 1.0
    assert adm1_counts["England"] == float(1 / 3)


def test_rel(geo):
    doc = geo.nlp("I visited Paris, France.")
    assert guess_in_rel(doc.ents[0]) == "France"
    assert guess_in_rel([doc.ents[0][0]]) == "France"

    doc = geo.nlp("I visited Paris, Berlin, and Munich.")
    assert guess_in_rel(doc.ents[0]) == ""
    assert guess_in_rel([doc.ents[0][0]]) == ""


def test_adm1_country_lookup(geonames_service_all_data):
    svc = geonames_service_all_data

    res = svc.get_adm1_country_entry("Maine", None)
    assert res["geonameid"] == "4971068"

    res = svc.get_adm1_country_entry("Maine", "USA")
    assert res["geonameid"] == "4971068"

    res = svc.get_country_by_name("Cuba")
    assert res["feature_code"] == "PCLI"
    assert res["geonameid"] == "3562981"

    res = svc.get_country_by_name("Atlantis")
    assert res is None

    res = svc.get_country_by_name("Syria")
    assert res["country_code3"] == "SYR"
    assert res["feature_code"] == "PCLI"
