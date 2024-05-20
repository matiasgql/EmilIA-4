"""Tests for the config flow."""

from unittest.mock import patch


from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.core import HomeAssistant


from custom_components.vicuna_conversation.const import DOMAIN


async def test_config_flow(
    hass: HomeAssistant,
) -> None:
    """Test selecting a model in the configuration flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    assert result.get("errors") is None

    with patch(
        f"custom_components.{DOMAIN}.async_setup_entry", return_value=True
    ) as mock_setup:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "api_key": "sk-0000000000000000000",
                "base_url": "http://llama-cublas.llama:8000/v1",
            },
        )
        await hass.async_block_till_done()

    assert result.get("type") is FlowResultType.CREATE_ENTRY
    assert result.get("title") == "Custom OpenAI Conversation"
    assert result.get("data") == {
        "api_key": "sk-0000000000000000000",
        "base_url": "http://llama-cublas.llama:8000/v1",
    }

    assert result.get("options") == {}
    assert len(mock_setup.mock_calls) == 1
