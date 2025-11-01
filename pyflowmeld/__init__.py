from pathlib import Path 


def find_package_root(start_file: Path, package_name: str = "pyflowmeld") -> Path:
    current = start_file.resolve()
    while current.name != package_name and current.parent != current:
        current = current.parent
    if current.name == package_name:
        return current
    raise RuntimeError(f"Could not find package root named '{package_name}' from '{start_file}'")
