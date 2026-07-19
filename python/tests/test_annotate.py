#!/usr/bin/env python

"""
Tests for annotation functionality in mLLMCelltype.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mllmcelltype.annotate import (
    annotate_clusters,
    get_model_response,
)


# Test annotation functions
class TestAnnotation:
    """Test class for annotation functions."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_marker_genes_df, sample_marker_genes_dict, mock_api_response):
        """Set up test fixtures."""
        self.marker_genes_df = sample_marker_genes_df
        self.marker_genes_dict = sample_marker_genes_dict
        self.mock_api_response = mock_api_response

    @patch("mllmcelltype.annotate.load_api_key")
    @patch("mllmcelltype.annotate.get_default_model")
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters(self, mock_get_default_model, mock_load_api_key):
        """Test annotate_clusters function."""
        # Setup mocks
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        # Return a list because format_results function expects a list
        PROVIDER_FUNCTIONS["mock_provider"] = lambda *args, **kwargs: [
            "Cluster 1: T cells",
            "Cluster 2: B cells",
        ]
        mock_load_api_key.return_value = "test-key"
        mock_get_default_model.return_value = "mock_model"

        # Test with DataFrame input - disable cache
        result = annotate_clusters(
            marker_genes=self.marker_genes_df,
            species="human",
            provider="mock_provider",
            model="mock_model",
            tissue="blood",
            use_cache=False,  # disable cache
        )

        # Verify results
        assert isinstance(result, dict)
        assert "1" in result
        assert "2" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"

        # Test with dictionary input - disable cache
        result = annotate_clusters(
            marker_genes=self.marker_genes_dict,
            species="human",
            provider="mock_provider",
            model="mock_model",
            tissue="blood",
            use_cache=False,  # disable cache
        )

        # Verify results
        assert isinstance(result, dict)
        assert "1" in result
        assert "2" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"

    @patch("mllmcelltype.annotate.load_api_key")
    @patch("mllmcelltype.annotate.get_default_model")
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_numbered_list_on_zero_based_is_not_shifted(
        self,
        mock_get_default_model,
        mock_load_api_key,
    ):
        """Regression: a numbered-list answer on 0-based clusters must not shift.

        End-to-end guard for the parser bug where "1." / "2." ordinals on Seurat
        0-based clusters were read as cluster IDs, shifting every annotation one
        cluster and losing cluster 0.
        """
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        PROVIDER_FUNCTIONS["mock_provider"] = lambda *args, **kwargs: [
            "1. T cells",
            "2. B cells",
            "3. NK cells",
        ]
        mock_load_api_key.return_value = "test-key"
        mock_get_default_model.return_value = "mock_model"

        result = annotate_clusters(
            marker_genes={"0": ["CD3D"], "1": ["CD19"], "2": ["NCAM1"]},
            species="human",
            provider="mock_provider",
            model="mock_model",
            tissue="blood",
            use_cache=False,
        )

        assert result == {"0": "T cells", "1": "B cells", "2": "NK cells"}

    @patch("mllmcelltype.annotate.load_api_key")
    @patch("mllmcelltype.annotate.get_default_model")
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_mixed_cluster_key_types_are_merged(
        self,
        mock_get_default_model,
        mock_load_api_key,
    ):
        """Test mixed int/str cluster keys are merged before prompt generation."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        captured = {}

        def mock_provider_func(prompt, *_args, **_kwargs):
            captured["prompt"] = prompt
            return ["Cluster 1: T cells"]

        PROVIDER_FUNCTIONS["mock_provider"] = mock_provider_func
        mock_load_api_key.return_value = "test-key"
        mock_get_default_model.return_value = "mock_model"

        result = annotate_clusters(
            marker_genes={1: ["CD3D"], "1": ["IL7R"]},
            species="human",
            provider="mock_provider",
            model="mock_model",
            use_cache=False,
        )

        assert result == {"1": "T cells"}
        assert "prompt" in captured
        marker_section = captured["prompt"].split("Here are the marker genes for each cluster:")[-1]
        assert marker_section.count("Cluster 1:") == 1
        assert "CD3D" in captured["prompt"]
        assert "IL7R" in captured["prompt"]

    # Fix parameter name issues
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS")
    @patch("mllmcelltype.utils.load_api_key")
    @patch("mllmcelltype.annotate.load_api_key")
    @patch("mllmcelltype.annotate.get_default_model")
    def test_get_model_response(
        self,
        mock_get_default_model,
        mock_load_api_key,
        mock_utils_load_api_key,
        mock_provider_functions,
    ):
        """Test get_model_response function."""

        # Setup mocks
        def mock_provider_func(*args, **kwargs):
            return [
                "Cluster 1: T cells",
                "Cluster 2: B cells",
            ]

        # Set up PROVIDER_FUNCTIONS dictionary
        mock_provider_functions.get.return_value = mock_provider_func
        mock_provider_functions.__contains__.return_value = True

        mock_load_api_key.return_value = "test-key"
        mock_utils_load_api_key.return_value = "test-key"
        mock_get_default_model.return_value = "gpt-4o"

        # Test function
        result = get_model_response(
            prompt="Test prompt",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",  # Explicitly provide API key to avoid loading real keys
            use_cache=False,
        )

        # Verify results
        assert isinstance(result, (list, str))
        if isinstance(result, list):
            assert len(result) == 2
            assert "Cluster 1: T cells" in result[0]
            assert "Cluster 2: B cells" in result[1]
        else:
            assert "Cluster 1: T cells" in result
            assert "Cluster 2: B cells" in result

    # Fix API key issues - use patch.dict to ensure environment variables are properly mocked
    @patch.dict(os.environ, {}, clear=True)  # Clear all environment variables
    @patch("mllmcelltype.utils.load_api_key")
    @patch("mllmcelltype.annotate.load_api_key")
    def test_get_model_response_missing_api_key(self, mock_load_api_key, mock_utils_load_api_key):
        """Test get_model_response with missing API key."""
        # Ensure both load_api_key functions return None
        mock_load_api_key.return_value = None
        mock_utils_load_api_key.return_value = None

        # Test function with missing API key
        with pytest.raises(ValueError, match="API key not found"):
            get_model_response(
                prompt="Test prompt",
                provider="openai",
                model="gpt-4o",
                api_key=None,
                use_cache=False,
            )

    def test_get_model_response_invalid_provider_type(self):
        """Test get_model_response with non-string provider."""
        with pytest.raises(ValueError, match="provider must be a string"):
            get_model_response(
                prompt="Test prompt",
                provider=123,  # type: ignore[arg-type]
                model="gpt-4o",
                api_key="test-key",
                use_cache=False,
            )

    def test_get_model_response_unknown_provider(self):
        """Test get_model_response rejects unknown provider names."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_model_response(
                prompt="Test prompt",
                provider="unknown_provider",
                model="gpt-5.5",
                api_key="test-key",
                use_cache=False,
            )

    def test_annotate_clusters_unknown_provider(self):
        """Test annotate_clusters rejects unknown provider names."""
        with pytest.raises(ValueError, match="Unknown provider"):
            annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="unknown_provider",
                model="gpt-5.5",
                api_key="test-key",
                use_cache=False,
            )

    def test_get_model_response_rejects_known_provider_model_mismatch(self):
        """A known model family cannot be routed through a different provider."""
        provider_mock = MagicMock(return_value="Cluster 1: T cells")
        with (
            patch.dict(
                "mllmcelltype.annotate.PROVIDER_FUNCTIONS",
                {"openai": provider_mock},
                clear=True,
            ),
            pytest.raises(ValueError, match="provider/model mismatch"),
        ):
            get_model_response(
                prompt="Test prompt",
                provider="openai",
                model="claude-opus-4-7",
                api_key="test-key",
                use_cache=False,
            )

        provider_mock.assert_not_called()

    def test_annotate_clusters_rejects_known_provider_model_mismatch(self):
        """Direct annotation follows the same provider/model contract as consensus."""
        provider_mock = MagicMock(return_value=["Cluster 1: T cells"])
        with (
            patch.dict(
                "mllmcelltype.annotate.PROVIDER_FUNCTIONS",
                {"openai": provider_mock},
                clear=True,
            ),
            pytest.raises(ValueError, match="provider/model mismatch"),
        ):
            annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="openai",
                model="claude-opus-4-7",
                api_key="test-key",
                use_cache=False,
            )

        provider_mock.assert_not_called()

    def test_annotate_clusters_blank_provider_name(self):
        """Test annotate_clusters rejects blank provider names."""
        with pytest.raises(ValueError, match="provider must be a non-empty string"):
            annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="   ",
                model="gpt-5.5",
                api_key="test-key",
                use_cache=False,
            )

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_provider_error_is_propagated(self):
        """Test provider-side runtime errors are surfaced to caller for observability."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        PROVIDER_FUNCTIONS["mock_provider"] = MagicMock(
            side_effect=RuntimeError("upstream timeout")
        )

        with pytest.raises(RuntimeError, match="upstream timeout"):
            annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="mock_provider",
                model="mock_model",
                api_key="test-key",
                use_cache=False,
            )

    @pytest.mark.parametrize("invalid_response", [123, "", ["Error: failed"]])
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_rejects_invalid_provider_response(self, invalid_response):
        """Invalid provider payloads are rejected before cache or annotation parsing."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        PROVIDER_FUNCTIONS["mock_provider"] = MagicMock(return_value=invalid_response)
        with pytest.raises(ValueError):
            annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="mock_provider",
                model="mock-model",
                api_key="test-key",
                use_cache=False,
            )

    @patch("mllmcelltype.annotate.load_from_cache", return_value=123)
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_invalid_cache_payload_is_recomputed(self, mock_load):
        """An invalid cache value is a miss, not an annotation source."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        provider = MagicMock(return_value=["Cluster 1: T cells"])
        PROVIDER_FUNCTIONS["mock_provider"] = provider
        result = annotate_clusters(
            marker_genes={"1": ["CD3D"]},
            species="human",
            provider="mock_provider",
            model="mock-model",
            api_key="test-key",
            use_cache=True,
        )

        assert result == {"1": "T cells"}
        assert mock_load.called
        provider.assert_called_once()

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_empty_input_returns_empty_result_without_provider_call(self):
        """Test empty marker set returns empty mapping and skips provider call."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        provider_mock = MagicMock(return_value=["Cluster 1: T cells"])
        PROVIDER_FUNCTIONS["mock_provider"] = provider_mock

        result = annotate_clusters(
            marker_genes={},
            species="human",
            provider="mock_provider",
            model="mock_model",
            api_key="test-key",
            use_cache=False,
        )

        assert result == {}
        provider_mock.assert_not_called()

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS")
    @patch("mllmcelltype.annotate.get_default_model")
    def test_get_model_response_provider_with_whitespace(
        self,
        mock_get_default_model,
        mock_provider_functions,
    ):
        """Test provider normalization trims whitespace."""

        def mock_provider_func(*args, **kwargs):
            return "Cluster 1: T cells"

        mock_provider_functions.get.return_value = mock_provider_func
        mock_get_default_model.return_value = "gpt-5.5"

        result = get_model_response(
            prompt="Test prompt",
            provider="  openai  ",
            model="gpt-5.5",
            api_key="test-key",
            use_cache=False,
        )
        assert "T cells" in result

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS")
    @patch("mllmcelltype.annotate.get_default_model")
    def test_get_model_response_model_with_whitespace_uses_default(
        self,
        mock_get_default_model,
        mock_provider_functions,
    ):
        """Test whitespace-only model falls back to provider default model."""
        captured = {}

        def mock_provider_func(_prompt, model, _api_key, _base_url):
            captured["model"] = model
            return "Cluster 1: T cells"

        mock_provider_functions.get.return_value = mock_provider_func
        mock_get_default_model.return_value = "gpt-5.5"

        result = get_model_response(
            prompt="Test prompt",
            provider="openai",
            model="   ",
            api_key="test-key",
            use_cache=False,
        )
        assert "T cells" in result
        assert captured["model"] == "gpt-5.5"

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS")
    @patch("mllmcelltype.annotate.load_api_key")
    def test_get_model_response_api_key_with_whitespace_uses_loaded_key(
        self,
        mock_load_api_key,
        mock_provider_functions,
    ):
        """Test whitespace-only API key falls back to loaded key."""
        captured = {}

        def mock_provider_func(_prompt, _model, api_key, _base_url):
            captured["api_key"] = api_key
            return "Cluster 1: T cells"

        mock_provider_functions.get.return_value = mock_provider_func
        mock_load_api_key.return_value = "  loaded-key  "

        result = get_model_response(
            prompt="Test prompt",
            provider="openai",
            model="gpt-5.5",
            api_key="   ",
            use_cache=False,
        )
        assert "T cells" in result
        assert captured["api_key"] == "loaded-key"

    def test_get_model_response_invalid_api_key_type(self):
        """Test get_model_response rejects non-string api_key types."""
        with pytest.raises(ValueError, match="api_key must be a string"):
            get_model_response(
                prompt="Test prompt",
                provider="openai",
                model="gpt-5.5",
                api_key=123,  # type: ignore[arg-type]
                use_cache=False,
            )

    def test_get_model_response_invalid_model_type(self):
        """Test get_model_response rejects non-string model types."""
        with pytest.raises(ValueError, match="model must be a string"):
            get_model_response(
                prompt="Test prompt",
                provider="openai",
                model=123,  # type: ignore[arg-type]
                api_key="test-key",
                use_cache=False,
            )

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_invalid_marker_genes_type(self):
        """Test annotate_clusters rejects non-dict/non-DataFrame marker input."""
        with pytest.raises(ValueError, match="marker_genes must be a dict"):
            annotate_clusters(
                marker_genes=["CD3D", "IL7R"],  # type: ignore[arg-type]
                species="human",
                provider="mock_provider",
                model="mock_model",
                api_key="test-key",
                use_cache=False,
            )

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_all_empty_marker_lists_returns_unknown_without_api_call(self):
        """Test no-marker clusters skip API call and deterministically return Unknown."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        provider_mock = MagicMock(return_value=["Cluster 1: T cells"])
        PROVIDER_FUNCTIONS["mock_provider"] = provider_mock

        result = annotate_clusters(
            marker_genes={"1": [], "2": []},
            species="human",
            provider="mock_provider",
            model="mock_model",
            api_key="test-key",
            use_cache=False,
        )

        assert result == {"1": "Unknown", "2": "Unknown"}
        provider_mock.assert_not_called()

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_mixed_empty_and_nonempty_markers_merge_unknown(self):
        """Test empty-marker clusters are Unknown while valid clusters are model-annotated."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        captured = {}

        def provider_func(prompt, *_args, **_kwargs):
            captured["prompt"] = prompt
            return ["Cluster 1: T cells"]

        PROVIDER_FUNCTIONS["mock_provider"] = provider_func

        result = annotate_clusters(
            marker_genes={"1": ["CD3D"], "2": []},
            species="human",
            provider="mock_provider",
            model="mock_model",
            api_key="test-key",
            use_cache=False,
        )

        assert result["1"] == "T cells"
        assert result["2"] == "Unknown"
        assert "Cluster 1:" in captured["prompt"]
        assert "Cluster 2:" not in captured["prompt"]

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_dataframe_preserves_empty_marker_cluster_as_unknown(self):
        """Test DataFrame input with empty-marker cluster returns Unknown for that cluster."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        captured = {}

        def provider_func(prompt, *_args, **_kwargs):
            captured["prompt"] = prompt
            return ["Cluster 1: T cells"]

        PROVIDER_FUNCTIONS["mock_provider"] = provider_func

        import pandas as pd

        df = pd.DataFrame(
            {
                "cluster": ["1", "2"],
                "gene": ["CD3D", None],
            }
        )

        result = annotate_clusters(
            marker_genes=df,
            species="human",
            provider="mock_provider",
            model="mock_model",
            api_key="test-key",
            use_cache=False,
        )

        assert result == {"1": "T cells", "2": "Unknown"}
        assert "Cluster 1:" in captured["prompt"]
        assert "Cluster 2:" not in captured["prompt"]

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_cache_hit_skips_second_provider_call(self):
        """Test cache hit avoids repeat provider call and returns same annotations."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        with tempfile.TemporaryDirectory() as temp_dir:
            provider_mock = MagicMock(return_value=["Cluster 1: T cells"])
            PROVIDER_FUNCTIONS["mock_provider"] = provider_mock

            first = annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="mock_provider",
                model="mock_model",
                api_key="test-key",
                use_cache=True,
                cache_dir=temp_dir,
            )
            assert first == {"1": "T cells"}
            assert provider_mock.call_count == 1

            provider_mock.side_effect = RuntimeError("provider should not be called on cache hit")
            second = annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="mock_provider",
                model="mock_model",
                api_key="test-key",
                use_cache=True,
                cache_dir=temp_dir,
            )
            assert second == {"1": "T cells"}

    @patch("mllmcelltype.annotate.load_api_key", return_value=None)
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_cache_hit_does_not_require_api_key(self, mock_load_api_key):
        """Test cached annotations remain usable without current provider credentials."""
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        with tempfile.TemporaryDirectory() as temp_dir:
            provider_mock = MagicMock(return_value=["Cluster 1: T cells"])
            PROVIDER_FUNCTIONS["mock_provider"] = provider_mock

            first = annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="mock_provider",
                model="mock_model",
                api_key="test-key",
                use_cache=True,
                cache_dir=temp_dir,
            )
            second = annotate_clusters(
                marker_genes={"1": ["CD3D"]},
                species="human",
                provider="mock_provider",
                model="mock_model",
                api_key=None,
                use_cache=True,
                cache_dir=temp_dir,
            )

        assert first == second == {"1": "T cells"}
        assert provider_mock.call_count == 1
        mock_load_api_key.assert_not_called()

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS")
    def test_get_model_response_cache_hit_skips_second_provider_call(self, mock_provider_functions):
        """Test get_model_response serves cached value without second provider call."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider_mock = MagicMock(return_value=["line one", "line two"])
            mock_provider_functions.get.return_value = provider_mock

            first = get_model_response(
                prompt="test prompt",
                provider="openai",
                model="gpt-5.5",
                api_key="test-key",
                use_cache=True,
                cache_dir=temp_dir,
            )
            assert "line one" in first
            assert provider_mock.call_count == 1

            provider_mock.side_effect = RuntimeError("provider should not be called on cache hit")
            second = get_model_response(
                prompt="test prompt",
                provider="openai",
                model="gpt-5.5",
                api_key="test-key",
                use_cache=True,
                cache_dir=temp_dir,
            )
            assert second == first

    @patch("mllmcelltype.annotate.load_api_key", return_value=None)
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS")
    def test_get_model_response_cache_hit_does_not_require_api_key(
        self,
        mock_provider_functions,
        mock_load_api_key,
    ):
        """Test a cached raw response can be read after credentials are removed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider_mock = MagicMock(return_value=["line one", "line two"])
            mock_provider_functions.get.return_value = provider_mock

            first = get_model_response(
                prompt="test prompt",
                provider="openai",
                model="gpt-5.5",
                api_key="test-key",
                use_cache=True,
                cache_dir=temp_dir,
            )
            second = get_model_response(
                prompt="test prompt",
                provider="openai",
                model="gpt-5.5",
                api_key=None,
                use_cache=True,
                cache_dir=temp_dir,
            )

        assert second == first
        assert provider_mock.call_count == 1
        mock_load_api_key.assert_not_called()

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS")
    def test_get_model_response_normalizes_request_and_base_url(self, mock_provider_functions):
        """Test cache identity and provider calls use the same normalized inputs."""
        provider_mock = MagicMock(return_value=["response"])
        mock_provider_functions.get.return_value = provider_mock

        response = get_model_response(
            prompt="  test prompt  ",
            provider="openai",
            model="gpt-5.5",
            api_key="test-key",
            base_url="  https://proxy.example.com/v1/  ",
            use_cache=False,
        )

        assert response == "response"
        provider_mock.assert_called_once_with(
            "test prompt",
            "gpt-5.5",
            "test-key",
            "https://proxy.example.com/v1",
        )

    @pytest.mark.parametrize("prompt", [None, "", "   ", 123])
    def test_get_model_response_rejects_invalid_prompt(self, prompt):
        """Test raw model requests require unambiguous non-empty prompt text."""
        with pytest.raises(ValueError, match="prompt must be"):
            get_model_response(
                prompt=prompt,
                provider="openai",
                model="gpt-5.5",
                api_key="test-key",
                use_cache=False,
            )

    @pytest.mark.parametrize("use_cache", [None, 0, 1, "yes"])
    def test_get_model_response_rejects_non_boolean_cache_flag(self, use_cache):
        """Test cache behavior cannot depend on arbitrary truthiness."""
        with pytest.raises(ValueError, match="use_cache must be True or False"):
            get_model_response(
                prompt="test prompt",
                provider="openai",
                model="gpt-5.5",
                api_key="test-key",
                use_cache=use_cache,
            )

    @pytest.mark.parametrize("use_cache", [None, 0, 1, "yes"])
    def test_annotate_clusters_rejects_non_boolean_cache_flag(self, use_cache):
        """Test annotation cache behavior requires an actual boolean."""
        with pytest.raises(ValueError, match="use_cache must be True or False"):
            annotate_clusters(
                marker_genes={},
                species="human",
                provider="openai",
                use_cache=use_cache,
            )

    @pytest.mark.parametrize("species", [None, "", "   ", 123])
    def test_annotate_clusters_validates_species_even_without_clusters(self, species):
        """Test required request context has one contract for empty and non-empty inputs."""
        with pytest.raises(ValueError, match="species must be"):
            annotate_clusters(
                marker_genes={},
                species=species,
                provider="openai",
                use_cache=False,
            )

    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters_rejects_mapping_marker_values(self):
        """Test marker gene values must be sequence-like genes, not dictionaries."""
        with pytest.raises(ValueError, match="marker_genes values must be gene lists"):
            annotate_clusters(
                marker_genes={"1": {"gene": "CD3D"}},  # type: ignore[dict-item]
                species="human",
                provider="mock_provider",
                model="mock_model",
                api_key="test-key",
                use_cache=False,
            )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
