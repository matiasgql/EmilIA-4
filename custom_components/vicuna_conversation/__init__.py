"""The OpenAI Conversation integration."""

from __future__ import annotations

import logging

import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.const import Platform

from .const import (
    DOMAIN,
)
from .openai_client import async_create_client, async_list_models

__all__ = [
    DOMAIN,
]

_LOGGER = logging.getLogger(__name__)
PLATFORMS = (Platform.CONVERSATION,)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    try:
        client = await async_create_client(hass, entry.data)
        await async_list_models(client)
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
