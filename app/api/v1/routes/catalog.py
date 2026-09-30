from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from app.catalog import get_catalog

router = APIRouter(prefix="/catalog", tags=["catalog"])

Component = Literal["configuration", "orientation", "localization", "movement"]


class LevelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    level: int
    name: str
    required_components: list[Component]


class SignOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    display_name: str
    level: int
    type: Literal["static", "dynamic"]
    required_components: list[Component]
    hold_time_ms: int | None
    max_duration_ms: int | None
    reference_asset: str | None
    components: dict[Component, str | None]
    validated: bool


class CatalogOut(BaseModel):
    catalog_version: str
    levels: list[LevelOut]
    signs: list[SignOut]


@router.get("/signs")
async def list_signs() -> CatalogOut:
    catalog = get_catalog()
    return CatalogOut(
        catalog_version=catalog.version,
        levels=[LevelOut.model_validate(level) for level in catalog.levels],
        signs=[SignOut.model_validate(sign) for sign in catalog.signs],
    )
