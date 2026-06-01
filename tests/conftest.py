import os
import pytest

from mordecai3.geonames import DataExtent, GeonamesService, setup_pg_pool
from mordecai3.logging import setup_logging

PG_DSN = os.getenv("PG_DSN", "host=localhost dbname=geoindex user=postgres")

setup_logging()


@pytest.fixture(scope="session")
def pg_pool():
    return setup_pg_pool(PG_DSN)


@pytest.fixture(scope="session")
def geonames_service(pg_pool):
    return GeonamesService(pg_pool)


@pytest.fixture(scope="session")
def geonames_service_test_data(geonames_service):
    if geonames_service.determine_data_extent() < DataExtent.TEST:
        pytest.skip("Geonames test data not available")
    return geonames_service


@pytest.fixture(scope="session")
def geonames_service_all_data(geonames_service):
    if geonames_service.determine_data_extent() < DataExtent.ALL:
        pytest.skip("Full geonames data not available")
    return geonames_service


@pytest.fixture(scope="session")
def geoparser_all_data(geonames_service_all_data):
    from mordecai3.geoparse import Geoparser
    return Geoparser(geonames=geonames_service_all_data)


@pytest.fixture(scope="session")
def geoparser_test_data(geonames_service_test_data):
    from mordecai3.geoparse import Geoparser
    return Geoparser(geonames=geonames_service_test_data)
