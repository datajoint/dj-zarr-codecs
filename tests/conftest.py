import logging
from typing import Generator, TypedDict

import pytest
from testcontainers.postgres import PostgresContainer
import datajoint as dj

logger = logging.getLogger(__name__)

class DBCreds(TypedDict):
    backend: str
    host: str
    user: str
    password: str

@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Start PostgreSQL container for the test session"""
    container = PostgresContainer(
        image="postgres:15",
        username="postgres",
        password="password",
        dbname="test",
    )
    container.start()

    host = container.get_container_host_ip()
    port = container.get_exposed_port(5432)
    logger.info(f"PostgreSQL container started at {host}:{port}")

    yield container

    container.stop()
    logger.info("PostgreSQL container stopped")

@pytest.fixture(scope="session")
def db_creds(postgres_container: PostgresContainer) -> DBCreds:
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)

    return {
            "backend": "postgresql",
            "host": f"{host}:{port}",
            "user": "postgres",
            "password": "password",
            }


@pytest.fixture(scope="function")
def dj_connection(db_creds: DBCreds) -> Generator[dj.Connection, None, None]:
    """Create connection for the specified backend.

    This fixture is function-scoped to ensure database.backend config
    is restored after each test, preventing config pollution between tests.
    """
    old_config = dict(dj.config._conf)

    # Configure backend
    dj.config["database.backend"] = db_creds["backend"]

    # Parse host:port
    host_port = db_creds["host"]
    host, port = host_port.rsplit(":", 1)

    dj.config["database.host"] = host
    dj.config["database.port"] = int(port)
    dj.config["database.user"] = db_creds["user"]
    dj.config["database.password"] = db_creds["password"]
    dj.config["database.use_tls"] = False
    dj.config["safemode"] = False
    dj.config["loglevel"] = "DEBUG"

    logger.info(f"Connecting to {db_creds['backend']} at {host_port}")

    connection = dj.Connection(
        host=host_port,
        user=db_creds["user"],
        password=db_creds["password"],
    )

    yield connection

    # Restore original config
    connection.close()
    for key, value in old_config.items():
        dj.config[key] = value

