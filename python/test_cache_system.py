#!/usr/bin/env python3
"""
Test suite for mLLMCelltype cache system.
Tests cache key generation, provider detection, and cache isolation.
"""

import os
import shutil
import sys
import tempfile

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mllmcelltype.functions import get_provider
from mllmcelltype.utils import create_cache_key, load_from_cache, save_to_cache


class TestCacheSystem:
    """Test suite for cache system validation."""

    def __init__(self):
        self.temp_cache_dir = tempfile.mkdtemp()
        self.test_results = []

    def cleanup(self):
        """Clean up temporary cache directory."""
        if os.path.exists(self.temp_cache_dir):
            shutil.rmtree(self.temp_cache_dir)

    def test_cache_key_generation(self):
        """Test that cache keys are properly generated with provider information."""
        print("\n=== Testing Cache Key Generation ===")

        test_cases = [
            # (prompt, model, provider, expected_provider_in_key)
            ("Test prompt", "gpt-4o", "openai", "openai"),
            ("Test prompt", "claude-3-opus", "anthropic", "anthropic"),
            (
                "Test prompt",
                "openai/gpt-4o",
                "openai",
                "openrouter",
            ),  # Should normalize to openrouter
            (
                "Test prompt",
                "anthropic/claude-3-opus",
                "anthropic",
                "openrouter",
            ),  # Should normalize to openrouter
            ("Test prompt", "meta-llama/llama-3.1-405b", "openrouter", "openrouter"),
        ]

        for prompt, model, provider, _expected_provider in test_cases:
            key = create_cache_key(prompt, model, provider)

            # Verify that different providers produce different keys
            alt_provider = "different_provider"
            alt_key = create_cache_key(prompt, model, alt_provider)

            # For OpenRouter models, keys should be the same regardless of input provider
            if "/" in model:
                # Both should use openrouter as provider
                assert (
                    key == alt_key
                ), f"OpenRouter model {model} should produce same key regardless of provider input"
                print(f"✓ OpenRouter model {model} correctly normalized")
            else:
                # Different providers should produce different keys
                assert (
                    key != alt_key
                ), f"Different providers should produce different keys for model {model}"
                print(f"✓ Model {model} with provider {provider} produces unique key")

    def test_provider_detection(self):
        """Test that get_provider correctly identifies providers from model names."""
        print("\n=== Testing Provider Detection ===")

        test_cases = [
            # (model, expected_provider)
            ("gpt-4o", "openai"),
            ("claude-3-opus", "anthropic"),
            ("openai/gpt-4o", "openrouter"),
            ("anthropic/claude-3-opus", "openrouter"),
            ("meta-llama/llama-3.1-405b", "openrouter"),
            ("google/gemini-2.5-pro", "openrouter"),
            ("qwen-max-2025-01-25", "qwen"),
            ("deepseek-chat", "deepseek"),
        ]

        for model, expected_provider in test_cases:
            detected_provider = get_provider(model)
            assert (
                detected_provider == expected_provider
            ), f"Expected {expected_provider} for {model}, got {detected_provider}"
            print(f"✓ Model {model} correctly detected as {detected_provider}")

    def test_cache_isolation(self):
        """Test that different models don't share cache."""
        print("\n=== Testing Cache Isolation ===")

        prompt = "Analyze these marker genes: CD3D, CD3E, CD4"

        # Test case 1: Different providers, same prompt
        models = [
            ("gpt-4o", "openai"),
            ("claude-3-opus", "anthropic"),
            ("openai/gpt-4o", "openrouter"),
            ("anthropic/claude-3-opus", "openrouter"),
        ]

        cache_keys = []
        for model, provider in models:
            key = create_cache_key(prompt, model, provider)

            # Save test data
            test_data = {"model": model, "provider": provider, "result": f"Result from {model}"}
            save_to_cache(key, test_data, self.temp_cache_dir)

            # Verify we can load it back
            loaded = load_from_cache(key, self.temp_cache_dir)
            assert loaded == test_data, f"Cache data mismatch for {model}"

            cache_keys.append(key)
            print(f"✓ Cache saved and loaded correctly for {model} ({provider})")

        # Verify all keys are unique (except for OpenRouter normalization)
        unique_keys = set(cache_keys)
        # Debug: print all keys
        print(f"  Debug: Generated {len(cache_keys)} total keys, {len(unique_keys)} unique")
        for i, (model, provider) in enumerate(models):
            print(f"    - {model} ({provider}): {cache_keys[i][:16]}...")

        # We expect 4 unique keys:
        # 1. gpt-4o with openai
        # 2. claude-3-opus with anthropic
        # 3. openai/gpt-4o with openrouter
        # 4. anthropic/claude-3-opus with openrouter
        # Even though the last two are both openrouter, they have different model names
        expected_unique_count = 4
        assert (
            len(unique_keys) == expected_unique_count
        ), f"Expected {expected_unique_count} unique keys, got {len(unique_keys)}"
        print(f"✓ Cache keys are properly isolated: {len(unique_keys)} unique keys")

    def test_openrouter_models_normalization(self):
        """Test that OpenRouter models with different stated providers use same cache."""
        print("\n=== Testing OpenRouter Model Normalization ===")

        prompt = "Test prompt for OpenRouter"
        model = "openai/gpt-4o"

        # Create keys with different stated providers
        key1 = create_cache_key(prompt, model, "openai")
        key2 = create_cache_key(prompt, model, "openrouter")
        key3 = create_cache_key(prompt, model, "anything")

        # All should be the same since model contains "/"
        assert (
            key1 == key2 == key3
        ), "OpenRouter model keys should be identical regardless of stated provider"
        print(f"✓ OpenRouter model {model} produces consistent cache keys")

        # Save data with one key
        test_data = {"test": "data"}
        save_to_cache(key1, test_data, self.temp_cache_dir)

        # Should be able to load with any of the keys
        for key in [key1, key2, key3]:
            loaded = load_from_cache(key, self.temp_cache_dir)
            assert loaded == test_data, "Should load same data with any key"
        print("✓ Cache data accessible with any provider for OpenRouter models")

    def test_cache_collision_prevention(self):
        """Test that the fix prevents cache collisions between different models."""
        print("\n=== Testing Cache Collision Prevention ===")

        prompt = "Analyze markers: CD3D, CD3E"

        # These should have different cache keys
        test_cases = [
            ("gpt-4o", "openai", "OpenAI GPT-4o result"),
            ("openai/gpt-4o", "openrouter", "OpenRouter GPT-4o result"),
            ("openai/gpt-4o-mini", "openrouter", "OpenRouter GPT-4o-mini result"),
        ]

        # Save different results for each
        for model, provider, expected_result in test_cases:
            key = create_cache_key(prompt, model, provider)
            save_to_cache(key, expected_result, self.temp_cache_dir)

        # Verify each loads its own result
        for model, provider, expected_result in test_cases:
            key = create_cache_key(prompt, model, provider)
            loaded = load_from_cache(key, self.temp_cache_dir)
            assert loaded == expected_result, f"Cache collision detected for {model} ({provider})"
            print(f"✓ No cache collision for {model} ({provider})")

    def run_all_tests(self):
        """Run all tests and report results."""
        print("mLLMCelltype Cache System Tests")
        print("=" * 50)

        try:
            self.test_cache_key_generation()
            self.test_provider_detection()
            self.test_cache_isolation()
            self.test_openrouter_models_normalization()
            self.test_cache_collision_prevention()

            print("\n" + "=" * 50)
            print("✅ ALL TESTS PASSED!")
            print("The cache system is working correctly.")

        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            return False
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            return False
        finally:
            self.cleanup()

        return True


def test_real_scenario():
    """Test a real-world scenario similar to the reported issue."""
    print("\n" + "=" * 50)
    print("Testing Real-World Scenario")
    print("=" * 50)

    # Create a temporary cache directory
    temp_cache = tempfile.mkdtemp()

    try:
        prompt = "Identify cell type for markers: CD3D, CD3E, CD4, CD8A"

        # Simulate first attempt with regular models
        print("\n1. First attempt with regular models:")
        models_attempt1 = [
            ("gpt-4o", "openai"),
            ("claude-3-opus", "anthropic"),
            ("qwen-max-2025-01-25", "qwen"),
        ]

        for model, provider in models_attempt1:
            # Mock the response since we don't have API keys
            key = create_cache_key(prompt, model, provider)
            mock_response = f"Response from {model} via {provider}"
            save_to_cache(key, mock_response, temp_cache)
            print(f"   - Cached response for {model} ({provider})")

        # Simulate second attempt with OpenRouter models
        print("\n2. Second attempt with OpenRouter models:")
        models_attempt2 = [
            ("openai/gpt-4o-mini", "openrouter"),
            ("anthropic/claude-3-opus", "openrouter"),
        ]

        for model, provider in models_attempt2:
            # These should NOT reuse cache from attempt 1
            key = create_cache_key(prompt, model, provider)

            # Check if there's existing cache (there shouldn't be)
            existing = load_from_cache(key, temp_cache)
            if existing:
                print(f"   ❌ ERROR: Found unexpected cache for {model} ({provider})")
                print(f"      Cache content: {existing}")
                return False
            else:
                print(f"   ✓ No existing cache for {model} ({provider}) - correct!")

                # Save new response
                mock_response = f"Response from {model} via OpenRouter"
                save_to_cache(key, mock_response, temp_cache)

        print("\n✅ Real-world scenario test PASSED!")
        print("   Different models correctly use different cache entries.")

    finally:
        shutil.rmtree(temp_cache)

    return True


if __name__ == "__main__":
    # Run unit tests
    tester = TestCacheSystem()
    unit_tests_passed = tester.run_all_tests()

    # Run real-world scenario test
    scenario_passed = test_real_scenario()

    # Summary
    print("\n" + "=" * 50)
    print("FINAL TEST SUMMARY")
    print("=" * 50)
    print(f"Unit tests: {'✅ PASSED' if unit_tests_passed else '❌ FAILED'}")
    print(f"Scenario test: {'✅ PASSED' if scenario_passed else '❌ FAILED'}")

    if unit_tests_passed and scenario_passed:
        print("\n🎉 All tests passed! The cache system is working correctly.")
        print("\nThe cache system ensures:")
        print("1. OpenRouter models always use 'openrouter' as provider in cache keys")
        print("2. Different providers don't share cache entries")
        print("3. Cache collisions are prevented between similar model names")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
        sys.exit(1)
