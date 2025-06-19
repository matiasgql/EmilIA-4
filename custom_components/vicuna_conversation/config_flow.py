"""Config flow for OpenAI Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    TemplateSelector,
    SelectSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
)
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_BASE_URL,
    CONF_STREAMING,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODELS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    LOGGER,
)
from .openai_client import (
    async_create_client,
    async_list_models,
    async_validate_completions,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY, default=DEFAULT_API_KEY): str,
        vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
    }
)
STEP_MODELS_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}


def _recommended_model(models: list[str] | None) -> str:
    """Return the selected model from user input."""
    # Don't use the recommended model if there is a valid list of other
    # models to choose from.
    if not models:
        return RECOMMENDED_CHAT_MODEL
    for model in RECOMMENDED_CHAT_MODELS:
        if model in models:
            return model
    return models[0]


class OpenAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 1

    data: dict[str, Any] | None = None
    client: openai.AsyncOpenAI | None = None
    models: list[str] | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors = {}
        if user_input is not None:
            try:
                self.client = await async_create_client(self.hass, user_input)
                self.models = await async_list_models(self.client)
            except openai.APIConnectionError:
                errors["base"] = "cannot_connect"
            except openai.AuthenticationError:
                errors["base"] = "invalid_api_key"
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                self.data = user_input
                return await self.async_step_model()

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle selecting a model."""
        assert self.client is not None
        assert self.models is not None
        errors = {}
        if user_input is not None:
            model = user_input[CONF_CHAT_MODEL]
            success = await async_validate_completions(
                self.client,
                model=model,
                stream=False,
            )
            if not success:
                errors["base"] = "cannot_connect"
            else:
                stream_support = await async_validate_completions(
                    self.client,
                    model=model,
                    stream=True,
                )
                return self.async_create_entry(
                    title="Custom OpenAI",
                    data=self.data,  # type: ignore[arg-type]
                    options={
                        **RECOMMENDED_OPTIONS,
                        **user_input,
                        CONF_STREAMING: stream_support,
                    },
                )

        return self.async_show_form(
            step_id="model",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_CHAT_MODEL,
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=self.models,
                                translation_key=CONF_CHAT_MODEL,
                                mode=SelectSelectorMode.DROPDOWN,
                                custom_value=True,  # Allow manual model entry
                            ),
                        ),
                    }
                ),
                {
                    CONF_CHAT_MODEL: (user_input or {}).get(
                        CONF_CHAT_MODEL, _recommended_model(self.models)
                    ),
                },
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OpenAIOptionsFlow(config_entry)


class OpenAIOptionsFlow(OptionsFlow):
    """OpenAI config flow options handler."""

    models: list[str] | None = None

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if self.models is None:
            client = await async_create_client(self.hass, self.config_entry.data)
            self.models = await async_list_models(client)

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)
                return self.async_create_entry(title="", data=user_input)

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
                CONF_CHAT_MODEL: user_input[CONF_CHAT_MODEL],
            }

        schema = openai_config_option_schema(self.hass, options, self.models)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def openai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    models: list[str] | None = None,
) -> dict:
    """Return a schema for OpenAI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    schema = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_CHAT_MODEL,
            description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            default=options.get(CONF_CHAT_MODEL, _recommended_model(models)),
        ): SelectSelector(
            SelectSelectorConfig(
                options=models or [],
                translation_key=CONF_CHAT_MODEL,
                mode=SelectSelectorMode.DROPDOWN,
                custom_value=True,  # Allow manual model entry
            ),
        ),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    schema.update(
        {
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
        }
    )
    return schema
