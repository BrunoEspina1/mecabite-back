"""Errores REST con el formato del contrato: `{"error": {"code", "message", "details"}}`."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            error_body(error.code, error.message, error.details), status_code=error.status
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = f"{field}: {first.get('msg', 'inválido')}" if field else "JSON inválido"
        return JSONResponse(
            error_body("INVALID_REQUEST", message, {"errors": jsonable_encoder(error.errors())}),
            status_code=400,
        )
