"""FastAPI dependency accessors."""

from fastapi import Request


def settings(request: Request):
    return request.app.state.settings


def registry(request: Request):
    return request.app.state.registry


def repository(request: Request):
    return request.app.state.repository
