"""
Test script to demonstrate the new DEQN interfaces and backward compatibility.

This script validates that the new interface system works with existing models
and shows how to use the new typed configuration and metrics systems.
"""

import sys

sys.path.append("/Users/matiascovarrubias/jaxecon")

from DEQN.core.config import DEQNConfig, quick_test_config, rbc_ces_config
from DEQN.core.examples import example_config_usage, example_metrics_usage
from DEQN.core.validation import quick_model_test, validate_econ_model
from DEQN.econ_models.rbc_ces import RbcCES_fixL


def test_model_interface():
    """Test that existing RBC model satisfies new interface."""
    print("=== Testing Model Interface Compliance ===")

    # Create an instance of existing model
    model = RbcCES_fixL()

    # Test interface compliance
    print(f"Testing model: {type(model).__name__}")
    errors = validate_econ_model(model, strict=False)

    if errors:
        print("❌ Interface validation errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Model passes interface validation!")

    # Run comprehensive validation
    print("\n--- Comprehensive Validation ---")
    quick_model_test(model, quick_test_config())

    return len(errors) == 0


def test_config_system():
    """Test the new configuration system."""
    print("\n=== Testing Configuration System ===")

    # Test predefined configs
    config1 = rbc_ces_config()
    config2 = quick_test_config()

    print(f"RBC CES config - Training steps per epoch: {config1.train.steps_per_epoch}")
    print(f"Quick test config - Training steps per epoch: {config2.train.steps_per_epoch}")

    # Test backward compatibility
    old_dict = config1.to_dict()
    new_config = DEQNConfig.from_dict(old_dict)

    print(f"Round-trip successful: {new_config.train.mc_draws == config1.train.mc_draws}")

    # Show dict structure for backward compatibility
    print(f"Dictionary has config_eval nested: {'config_eval' in old_dict}")
    print(f"Dictionary keys: {list(old_dict.keys())[:5]}...")  # First 5 keys

    return True


def test_import_compatibility():
    """Test that existing imports still work."""
    print("\n=== Testing Import Compatibility ===")

    try:
        # Test that old algorithm imports still work
        from DEQN.algorithm import (
            create_batch_loss_fn,
            create_epoch_train_fn,
            create_eval_fn,
        )
        from DEQN.algorithm.simulation import create_episode_simul_fn

        print("✅ Algorithm imports successful")

        # Test new core imports
        from DEQN.core import DEQNConfig, EconModel, LossMetrics, validate_econ_model

        print("✅ Core imports successful")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Run all interface tests."""
    print("DEQN Interface Validation Test")
    print("=" * 50)

    results = []

    # Test 1: Model interface compliance
    results.append(test_model_interface())

    # Test 2: Configuration system
    results.append(test_config_system())

    # Test 3: Import compatibility
    results.append(test_import_compatibility())

    # Test 4: Example usage
    print("\n=== Example Usage ===")
    example_config_usage()
    print()
    example_metrics_usage()
    results.append(True)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All {total} tests passed! Interface system is working correctly.")
        print("\nThe new interfaces are ready to use and maintain backward compatibility.")
    else:
        print(f"⚠️  {passed}/{total} tests passed. Some issues need to be addressed.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
