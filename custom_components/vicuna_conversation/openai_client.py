"""Module for handling the open AI client library."""

from __future__ import annotations

from typing import Any, Mapping

import openai

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from .const import (
    CONF_BASE_URL,
)

_TIMEOUT = 10.0


def _create_client(config_entry_data: Mapping[str, Any]) -> openai.AsyncOpenAI:
    """Create a new OpenAI client."""
    return openai.AsyncOpenAI(
        api_key=config_entry_data[CONF_API_KEY],
        base_url=config_entry_data[CONF_BASE_URL],
    ).with_options(timeout=_TIMEOUT)


async def async_create_client(
    hass: HomeAssistant,
    config_entry_data: Mapping[str, Any],
) -> openai.AsyncOpenAI:
    """Create an OpenAI client and test the connection."""

    def validate_client() -> openai.AsyncOpenAI:
        """Get OpenAI client."""
        client = _create_client(config_entry_data)
        client.models.list()  # Ignore
        return client

    return await hass.async_add_executor_job(validate_client)
