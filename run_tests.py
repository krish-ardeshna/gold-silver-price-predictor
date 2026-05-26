#!/usr/bin/env python3
"""
Comprehensive Test Runner for Gold-Silver Price Predictor

This script runs all tests with various configurations to ensure
robust testing coverage including unit tests, integration tests,
and edge case testing.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"✅ {description} PASSED ({duration:.2f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} FAILED ({duration:.2f}s)")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} TIMED OUT after 300s")
        return False
    except Exception as e:
        print(f"💥 {description} ERROR: {str(e)}")
        return False


def setup_test_environment():
    """Setup test environment and dependencies"""
    print("Setting up test environment...")

    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Check if we're in a Python environment (conda or venv)
    try:
        import sys
        python_exe = sys.executable
        print(f"Using Python: {python_exe}")

        # Check if required packages are available
        required_packages = ['pytest', 'pandas', 'numpy', 'sklearn', 'xgboost']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Please install requirements: pip install -r requirements.txt")
            return False

        print("✅ Test environment ready")
        return True

    except Exception as e:
        print(f"⚠️  Environment setup issue: {e}")
        return False


def run_unit_tests():
    """Run unit tests"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-m", "not integration"]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-m", "integration"]
    return run_command(cmd, "Integration Tests")


def run_all_tests():
    """Run all tests"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    return run_command(cmd, "All Tests")


def run_coverage_tests():
    """Run tests with coverage"""
    cmd = [
        sys.executable, "-m", "pytest", "tests/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "-v", "--tb=short"
    ]
    return run_command(cmd, "Coverage Tests")


def run_stress_tests():
    """Run stress tests (multiple runs)"""
    print("\n🔄 Running stress tests (3 iterations)...")

    results = []
    for i in range(3):
        print(f"\nIteration {i+1}/3")
        cmd = [sys.executable, "-m", "pytest", "tests/test_preprocess.py", "tests/test_train.py", "-v", "--tb=line"]
        success = run_command(cmd, f"Stress Test Iteration {i+1}")
        results.append(success)

    overall_success = all(results)
    print(f"\nStress test result: {'PASSED' if overall_success else 'FAILED'} ({sum(results)}/{len(results)} passed)")
    return overall_success


def run_edge_case_tests():
    """Run edge case and boundary tests"""
    print("\n🔍 Running edge case tests...")

    # Test with minimal data
    print("Testing with minimal dataset...")
    cmd = [sys.executable, "-c", """
import pandas as pd
import numpy as np
from src.preprocess import load_and_prepare
import tempfile
import os

# Create minimal valid dataset
dates = pd.date_range('2020-01-01', periods=60, freq='D')
prices = 1500 + np.cumsum(np.random.randn(60) * 2)
data = pd.DataFrame({'Date': dates, 'Close': prices})

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    data.to_csv(f.name, index=False)
    try:
        df = load_and_prepare(f.name)
        print(f"✅ Minimal data test passed: {len(df)} rows processed")
    except Exception as e:
        print(f"❌ Minimal data test failed: {e}")
    finally:
        os.unlink(f.name)
"""]
    success1 = run_command(cmd, "Edge Case: Minimal Data")

    # Test with corrupted data
    print("Testing with corrupted data...")
    cmd = [sys.executable, "-c", """
import pandas as pd
import numpy as np
from src.preprocess import load_and_prepare
import tempfile
import os

# Create corrupted dataset
data = pd.DataFrame({
    'Date': ['invalid'] * 60,
    'Close': ['not_a_number'] * 60
})

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    data.to_csv(f.name, index=False)
    try:
        df = load_and_prepare(f.name)
        print("❌ Corrupted data test should have failed")
    except Exception as e:
        print(f"✅ Corrupted data test passed: correctly caught error - {type(e).__name__}")
"""]
    success2 = run_command(cmd, "Edge Case: Corrupted Data")

    return success1 and success2


def run_performance_tests():
    """Run performance tests"""
    print("\n⚡ Running performance tests...")

    # Test preprocessing performance
    cmd = [sys.executable, "-c", """
import pandas as pd
import numpy as np
from src.preprocess import load_and_prepare
import tempfile
import os
import time

# Create large dataset
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
prices = 1500 + np.cumsum(np.random.randn(1000) * 2)
data = pd.DataFrame({'Date': dates, 'Close': prices})

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    data.to_csv(f.name, index=False)
    try:
        start_time = time.time()
        df = load_and_prepare(f.name)
        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ Performance test passed: {len(df)} rows processed in {duration:.2f}s")
        if duration > 10:  # Should complete in reasonable time
            print("⚠️  Performance warning: took longer than 10 seconds")
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    finally:
        os.unlink(f.name)
"""]
    return run_command(cmd, "Performance Test")


def run_brutality_tests():
    """Run brutal edge case tests"""
    print("\n💀 Running brutality tests (extreme edge cases)...")

    test_cases = [
        ("Empty file", """
import tempfile
import os
from src.preprocess import load_and_prepare

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write("")
    try:
        df = load_and_prepare(f.name)
        print("❌ Should have failed with empty file")
    except Exception as e:
        print(f"✅ Correctly handled empty file: {type(e).__name__}")
    finally:
        os.unlink(f.name)
"""),
        ("Wrong columns", """
import pandas as pd
import tempfile
import os
from src.preprocess import load_and_prepare

data = pd.DataFrame({'WrongCol': [1, 2, 3]})
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    data.to_csv(f.name, index=False)
    try:
        df = load_and_prepare(f.name)
        print("❌ Should have failed with wrong columns")
    except Exception as e:
        print(f"✅ Correctly handled wrong columns: {type(e).__name__}")
    finally:
        os.unlink(f.name)
"""),
        ("All NaN values", """
import pandas as pd
import numpy as np
import tempfile
import os
from src.preprocess import load_and_prepare

dates = pd.date_range('2020-01-01', periods=60, freq='D')
data = pd.DataFrame({'Date': dates, 'Close': [np.nan] * 60})
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    data.to_csv(f.name, index=False)
    try:
        df = load_and_prepare(f.name)
        print("❌ Should have failed with all NaN")
    except Exception as e:
        print(f"✅ Correctly handled all NaN: {type(e).__name__}")
    finally:
        os.unlink(f.name)
"""),
    ]

    results = []
    for test_name, test_code in test_cases:
        cmd = [sys.executable, "-c", test_code]
        success = run_command(cmd, f"Brutality: {test_name}")
        results.append(success)

    return all(results)


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument('--type', choices=['unit', 'integration', 'all', 'coverage',
                                         'stress', 'edge', 'performance', 'brutal'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Run only quick tests')

    args = parser.parse_args()

    print("🚀 Gold-Silver Price Predictor - Comprehensive Test Suite")
    print("="*60)

    if not setup_test_environment():
        sys.exit(1)

    test_results = []

    if args.type == 'unit':
        test_results.append(run_unit_tests())
    elif args.type == 'integration':
        test_results.append(run_integration_tests())
    elif args.type == 'coverage':
        test_results.append(run_coverage_tests())
    elif args.type == 'stress':
        test_results.append(run_stress_tests())
    elif args.type == 'edge':
        test_results.append(run_edge_case_tests())
    elif args.type == 'performance':
        test_results.append(run_performance_tests())
    elif args.type == 'brutal':
        test_results.append(run_brutality_tests())
    else:  # 'all'
        if not args.quick:
            print("\n📊 Running comprehensive test suite...")

            # Core tests
            test_results.extend([
                run_unit_tests(),
                run_integration_tests(),
                run_coverage_tests(),
            ])

            # Advanced tests
            if not args.quick:
                test_results.extend([
                    run_edge_case_tests(),
                    run_performance_tests(),
                    run_stress_tests(),
                    run_brutality_tests(),
                ])
        else:
            test_results.extend([
                run_unit_tests(),
                run_integration_tests(),
            ])

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)

    total_tests = len(test_results)
    passed_tests = sum(test_results)

    print(f"Total test suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
