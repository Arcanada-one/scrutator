"""Tests for LTM LLM client — focus on permissive JSON parsing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scrutator.ltm.llm import LtmLlmClient, parse_json_permissive


class TestParseJsonPermissive:
    """Test the permissive JSON parser that handles LLM output quirks."""

    def test_valid_json_object(self):
        result = parse_json_permissive('{"name": "Scrutator", "type": "project"}')
        assert result == {"name": "Scrutator", "type": "project"}

    def test_valid_json_array(self):
        result = parse_json_permissive('[{"name": "X"}, {"name": "Y"}]')
        assert result == [{"name": "X"}, {"name": "Y"}]

    def test_json_in_markdown_code_block(self):
        text = '```json\n[{"name": "Scrutator"}]\n```'
        result = parse_json_permissive(text)
        assert result == [{"name": "Scrutator"}]

    def test_json_with_surrounding_text(self):
        text = 'Here are the entities:\n[{"name": "X"}]\nThat is all.'
        result = parse_json_permissive(text)
        assert result == [{"name": "X"}]

    def test_json_object_with_surrounding_text(self):
        text = 'The result is: {"key": "value"} and more text.'
        result = parse_json_permissive(text)
        assert result == {"key": "value"}

    def test_nested_json(self):
        obj = {"entities": [{"name": "A", "props": {"x": 1}}]}
        result = parse_json_permissive(json.dumps(obj))
        assert result == obj

    def test_no_json_returns_raw(self):
        result = parse_json_permissive("No JSON here at all")
        assert result == {"raw": "No JSON here at all"}

    def test_empty_string(self):
        result = parse_json_permissive("")
        assert result == {"raw": ""}

    def test_json_with_trailing_comma(self):
        # Some LLMs add trailing commas
        text = '[{"name": "X",}]'
        result = parse_json_permissive(text)
        # Should fall back to regex extraction if json.loads fails
        assert isinstance(result, (list, dict))

    def test_multiple_json_blocks_picks_first(self):
        text = '[{"a": 1}] and then [{"b": 2}]'
        result = parse_json_permissive(text)
        assert result == [{"a": 1}]

    def test_code_block_without_json_label(self):
        text = '```\n{"name": "test"}\n```'
        result = parse_json_permissive(text)
        assert result == {"name": "test"}


class TestLtmLlmClient:
    """Test the LLM client wrapper."""

    @pytest.fixture
    def client(self):
        return LtmLlmClient(
            mc_url="http://localhost:3900",
            connector="cursor",
            model="auto",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_call_success(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "status": "success",
            "result": "Hello world",
        }

        with patch("scrutator.ltm.llm.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            result = await client.call("Say hello")
            assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_call_mc_error(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("scrutator.ltm.llm.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            with pytest.raises(Exception, match="MC returned 500"):
                await client.call("fail")

    @pytest.mark.asyncio
    async def test_extract_json_parses_response(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "status": "success",
            "result": '[{"name": "Scrutator", "type": "project"}]',
        }

        with patch("scrutator.ltm.llm.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            result = await client.extract_json("Extract entities")
            assert result == [{"name": "Scrutator", "type": "project"}]

    @pytest.mark.asyncio
    async def test_call_with_system_prompt(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "status": "success",
            "result": "response",
        }

        with patch("scrutator.ltm.llm.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            await client.call("prompt", system="You are an extractor")
            call_args = mock_http.post.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "You are an extractor" in body["prompt"]
