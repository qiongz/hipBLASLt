from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections import defaultdict
from collections.abc import Callable
from types import ModuleType

from ._utils import fail_open_enabled, warn


Hook = Callable[[ModuleType], None]

_POST_IMPORT_HOOKS: dict[str, list[Hook]] = defaultdict(list)


def _run_hook(fullname: str, hook: Hook, module: ModuleType) -> None:
    try:
        hook(module)
    except Exception as exc:
        if fail_open_enabled():
            warn(f"adapter hook for {fullname} failed: {exc}")
            return
        raise


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped_loader: importlib.abc.Loader) -> None:
        self._fullname = fullname
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        if hasattr(self._wrapped_loader, "create_module"):
            return self._wrapped_loader.create_module(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped_loader.exec_module(module)
        for hook in list(_POST_IMPORT_HOOKS.get(self._fullname, ())):
            _run_hook(self._fullname, hook, module)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        if fullname not in _POST_IMPORT_HOOKS:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None

        if not isinstance(spec.loader, _PostImportLoader):
            spec.loader = _PostImportLoader(fullname, spec.loader)
        return spec


_FINDER = _PostImportFinder()


def register_post_import_hook(fullname: str, hook: Hook) -> None:
    hooks = _POST_IMPORT_HOOKS[fullname]
    if hook in hooks:
        return
    hooks.append(hook)

    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)

    module = sys.modules.get(fullname)
    if module is not None:
        _run_hook(fullname, hook, module)
