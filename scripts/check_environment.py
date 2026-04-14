from __future__ import annotations

import importlib


PACKAGES_TO_CHECK = [
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "xrfm",
]


def check_package(package_name: str) -> None:
    """Try importing a package and print installation status and version."""
    try:
        module = importlib.import_module(package_name)
    except Exception as exc:
        print(f"{package_name}: NOT INSTALLED")
        print(f"  error_type: {type(exc).__name__}")
        print(f"  error_message: {exc}")
        return

    version = getattr(module, "__version__", "unknown")
    print(f"{package_name}: INSTALLED")
    print(f"  version: {version}")


def main() -> None:
    print("Environment package check")
    for package_name in PACKAGES_TO_CHECK:
        check_package(package_name)


if __name__ == "__main__":
    main()
