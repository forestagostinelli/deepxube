from importlib import import_module
import os


def import_local_modules(root_dir: str, base_package: str = "") -> None:
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                module_name = os.path.join(root, file).replace('.py', '').replace('/', '.')
                if base_package:
                    module_name = base_package + '.' + module_name
                try:
                    import_module(module_name)
                except Exception as e:
                    print(f"Failed to import {module_name}: {e}")
