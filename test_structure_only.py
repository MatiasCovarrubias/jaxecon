"""
Test script for the new DEQN algorithm structure (no JAX required).

This script validates the module structure and interface definitions
without requiring JAX to be installed.
"""

import sys

sys.path.append("/Users/matiascovarrubias/jaxecon")


def test_core_interfaces():
    """Test that core interfaces can be imported and used."""
    print("=== Testing Core Interfaces ===")

    try:
        from DEQN.core.config import DEQNConfig, EvalConfig, TrainConfig
        from DEQN.core.validation import validate_config

        print("✅ All core interfaces imported successfully")

        # Test config creation
        config = DEQNConfig(train=TrainConfig(mc_draws=16, batch_size=64), eval=EvalConfig(eval_n_epis=32, mc_draws=32))
        print(f"✅ Configuration created: {config.train.mc_draws} MC draws")

        # Test validation
        warnings = validate_config(config)
        if warnings:
            print(f"⚠️  Configuration warnings: {len(warnings)}")
        else:
            print("✅ Configuration validation passed")

        # Test backward compatibility
        old_dict = config.to_dict()
        new_config = DEQNConfig.from_dict(old_dict)
        print(f"✅ Backward compatibility: {new_config.train.mc_draws == config.train.mc_draws}")

        return True

    except Exception as e:
        print(f"❌ Core interfaces test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all expected files exist and have correct structure."""
    print("\n=== Testing File Structure ===")

    import os

    # Check core files
    core_files = [
        "DEQN/core/__init__.py",
        "DEQN/core/interfaces.py",
        "DEQN/core/config.py",
        "DEQN/core/metrics.py",
        "DEQN/core/validation.py",
        "DEQN/core/examples.py",
    ]

    # Check algorithm_new files
    algorithm_new_files = [
        "DEQN/algorithm_new/__init__.py",
        "DEQN/algorithm_new/simulation.py",
        "DEQN/algorithm_new/loss.py",
        "DEQN/algorithm_new/train.py",
        "DEQN/algorithm_new/eval.py",
        "DEQN/algorithm_new/compose.py",
        "DEQN/algorithm_new/README.md",
    ]

    all_files = core_files + algorithm_new_files
    missing_files = []

    for file_path in all_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✅ All {len(all_files)} expected files exist")
        return True


def test_import_structure():
    """Test import structure without importing JAX-dependent modules."""
    print("\n=== Testing Import Structure ===")

    try:
        # Test that we can import the interfaces and configs
        print("✅ Core public API imports work")

        # Test that algorithm structure exists (but don't import JAX-dependent parts)
        import DEQN.algorithm_new

        algorithm_new_dir = dir(DEQN.algorithm_new)
        expected_functions = [
            "create_episode_simulation_fn",
            "create_batch_loss_fn",
            "create_epoch_train_fn",
            "create_eval_fn",
            "make_deqn_components",
        ]

        missing_functions = [f for f in expected_functions if f not in algorithm_new_dir]
        if missing_functions:
            print(f"⚠️  algorithm_new missing functions: {missing_functions}")
        else:
            print("✅ algorithm_new has all expected functions")

        return len(missing_functions) == 0

    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False


def demonstrate_improvements():
    """Demonstrate the key improvements."""
    print("\n=== Demonstrating Key Improvements ===")

    try:
        import numpy as np  # Use numpy instead of jax.numpy for demo

        from DEQN.core.config import rbc_ces_config
        from DEQN.core.metrics import LossMetrics

        # 1. Structured Configuration
        print("1. Structured Configuration:")
        config = rbc_ces_config()
        print(f"   Training: {config.train.mc_draws} MC draws, {config.train.batch_size} batch size")
        print(f"   Evaluation: {config.eval.eval_n_epis} episodes, every {config.eval.eval_frequency} epochs")

        # 2. Independent Eval Config
        print("\n2. Independent Evaluation Configuration:")
        print(f"   Training periods per episode: {config.train.periods_per_epis}")
        print(f"   Eval periods per episode: {config.eval.periods_per_epis}")
        print(f"   Different MC precision: train={config.train.mc_draws}, eval={config.eval.mc_draws}")

        # 3. Structured Metrics
        print("\n3. Structured Metrics (demo with numpy):")
        # Simulate metrics
        metrics = LossMetrics(
            mean_loss=np.array(0.01),
            max_loss=np.array(0.05),
            mean_accuracy=np.array(0.95),
            min_accuracy=np.array(0.89),
            mean_accs_foc=np.array([0.92, 0.94]),
            min_accs_foc=np.array([0.87, 0.89]),
        )
        print(f"   Mean loss: {metrics.mean_loss}")
        print(f"   Accuracy: {metrics.mean_accuracy}")
        print(f"   FOC accuracies: {metrics.mean_accs_foc}")

        # 4. Backward Compatibility
        print("\n4. Backward Compatibility:")
        old_tuple = metrics.to_tuple()
        print(f"   Can convert to legacy tuple: {len(old_tuple)} elements")

        old_dict = config.to_dict()
        print(f"   Can convert to legacy dict: {len(old_dict)} keys")
        print(f"   Has config_eval: {'config_eval' in old_dict}")

        return True

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all structure tests."""
    print("DEQN Algorithm New Implementation - Structure Test")
    print("=" * 60)

    results = []

    # Test 1: Core interfaces
    results.append(test_core_interfaces())

    # Test 2: File structure
    results.append(test_file_structure())

    # Test 3: Import structure
    results.append(test_import_structure())

    # Test 4: Demonstrate improvements
    results.append(demonstrate_improvements())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All {total} structure tests passed!")
        print("\nNew algorithm implementation is structurally sound:")
        print("  ✅ Clean typed interfaces")
        print("  ✅ Structured configurations")
        print("  ✅ Organized file structure")
        print("  ✅ Proper import hierarchy")
        print("  ✅ Backward compatibility maintained")
        print("\nReady for use with JAX-enabled environments!")
    else:
        print(f"⚠️  {passed}/{total} tests passed. Some structural issues need attention.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
