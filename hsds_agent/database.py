from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/hsds_agent.sqlite")
DEFAULT_SAMPLE_PATH = Path("data/sample_hsds.json")

CITY_COORDINATES = {
    "bloomington": (39.1653, -86.5264),
    "47401": (39.1417, -86.5038),
    "47403": (39.1366, -86.5772),
    "47404": (39.1953, -86.5756),
    "47408": (39.2304, -86.4697),
}


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            website TEXT
        );

        CREATE TABLE IF NOT EXISTS services (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL REFERENCES organizations(id),
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            categories TEXT NOT NULL,
            languages TEXT NOT NULL DEFAULT '[]',
            eligibility TEXT,
            fees TEXT,
            url TEXT
        );

        CREATE TABLE IF NOT EXISTS locations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            address_1 TEXT NOT NULL,
            city TEXT NOT NULL,
            region TEXT NOT NULL,
            postal_code TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS service_at_locations (
            id TEXT PRIMARY KEY,
            service_id TEXT NOT NULL REFERENCES services(id),
            location_id TEXT NOT NULL REFERENCES locations(id)
        );

        CREATE TABLE IF NOT EXISTS phones (
            id TEXT PRIMARY KEY,
            service_id TEXT NOT NULL REFERENCES services(id),
            number TEXT NOT NULL,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS schedules (
            id TEXT PRIMARY KEY,
            service_id TEXT NOT NULL REFERENCES services(id),
            weekday TEXT NOT NULL,
            opens_at TEXT NOT NULL,
            closes_at TEXT NOT NULL,
            description TEXT
        );
        """
    )
    conn.commit()


def seed_from_json(
    conn: sqlite3.Connection, sample_path: Path = DEFAULT_SAMPLE_PATH
) -> None:
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    conn.executescript(
        """
        DELETE FROM schedules;
        DELETE FROM phones;
        DELETE FROM service_at_locations;
        DELETE FROM locations;
        DELETE FROM services;
        DELETE FROM organizations;
        """
    )
    conn.executemany(
        "INSERT INTO organizations(id, name, website) VALUES(:id, :name, :website)",
        data["organizations"],
    )
    conn.executemany(
        """
        INSERT INTO services(
            id, organization_id, name, description, categories, languages,
            eligibility, fees, url
        )
        VALUES(
            :id, :organization_id, :name, :description, :categories, :languages,
            :eligibility, :fees, :url
        )
        """,
        [_json_fields(row, ("categories", "languages")) for row in data["services"]],
    )
    conn.executemany(
        """
        INSERT INTO locations(
            id, name, address_1, city, region, postal_code, latitude, longitude
        )
        VALUES(
            :id, :name, :address_1, :city, :region, :postal_code, :latitude, :longitude
        )
        """,
        data["locations"],
    )
    conn.executemany(
        """
        INSERT INTO service_at_locations(id, service_id, location_id)
        VALUES(:id, :service_id, :location_id)
        """,
        data["service_at_locations"],
    )
    conn.executemany(
        """
        INSERT INTO phones(id, service_id, number, description)
        VALUES(:id, :service_id, :number, :description)
        """,
        data["phones"],
    )
    conn.executemany(
        """
        INSERT INTO schedules(id, service_id, weekday, opens_at, closes_at, description)
        VALUES(:id, :service_id, :weekday, :opens_at, :closes_at, :description)
        """,
        data["schedules"],
    )
    conn.commit()


def fetch_service_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT
                s.id AS service_id,
                s.name AS service_name,
                s.description,
                s.categories,
                s.languages,
                s.eligibility,
                s.fees,
                s.url AS service_url,
                o.name AS organization_name,
                o.website AS organization_website,
                l.address_1,
                l.city,
                l.region,
                l.postal_code,
                l.latitude,
                l.longitude,
                p.number AS phone
            FROM services s
            JOIN organizations o ON o.id = s.organization_id
            JOIN service_at_locations sal ON sal.service_id = s.id
            JOIN locations l ON l.id = sal.location_id
            LEFT JOIN phones p ON p.service_id = s.id
            """
        )
    )


def fetch_schedule(conn: sqlite3.Connection, service_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT weekday, opens_at, closes_at, description
            FROM schedules
            WHERE service_id = ?
            ORDER BY
                CASE weekday
                    WHEN 'monday' THEN 1
                    WHEN 'tuesday' THEN 2
                    WHEN 'wednesday' THEN 3
                    WHEN 'thursday' THEN 4
                    WHEN 'friday' THEN 5
                    WHEN 'saturday' THEN 6
                    WHEN 'sunday' THEN 7
                    ELSE 8
                END
            """,
            (service_id,),
        )
    )


def resolve_coordinates(location: str | None) -> tuple[float, float] | None:
    if not location:
        return None
    normalized = location.lower().strip()
    for key, coords in CITY_COORDINATES.items():
        if key in normalized:
            return coords
    return None


def distance_miles(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    radius = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    hav = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))


def _json_fields(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    converted = dict(row)
    for field in fields:
        converted[field] = json.dumps(converted[field])
    return converted
