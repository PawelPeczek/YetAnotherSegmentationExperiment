from dataclasses import dataclass


@dataclass
class ServiceSpecs:
    host: str
    port: int
    service_name: str
    version: str = "v1"


@dataclass
class ServiceJWT:
    token: str
    token_secret: str


