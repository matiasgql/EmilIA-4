"""The OpenAI Conversation integration."""

from __future__ import annotations

import logging

import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.const import Platform

from .const import (
    CONF_BASE_URL,
    DOMAIN,
)

__all__ = [
    DOMAIN,
]

_LOGGER = logging.getLogger(__name__)
PLATFORMS = (Platform.CONVERSATION,)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    def create_client() -> openai.AsyncOpenAI:
        """Get OpenAI client."""
        client = openai.AsyncOpenAI(
            api_key=entry.data[CONF_API_KEY], base_url=entry.data[CONF_BASE_URL]
        )
        client.with_options(timeout=10.0).models.list()  # Ignore
        return client

    try:
        client = await hass.async_add_executor_job(create_client)
    except openai.AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
