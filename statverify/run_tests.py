#!/usr/bin/env python3
"""Запуск тестов без внешних зависимостей.

Тесты написаны под pytest и в CI выполняются им. Но у проекта уже был
случай отсутствия сетевого доступа к pypi, а верификатор, чьи тесты
невозможно прогнать в наличной среде, ничем не лучше непроверенного.
Поэтому здесь минимальная подмена pytest, покрывающая ровно то, чем
тесты пользуются: mark.parametrize, importorskip, skip.

Если настоящий pytest доступен, пользуйся им:  python3 -m pytest
"""

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


class Skipped(Exception):
    pass


def _install_pytest_stub():
    if importlib.util.find_spec("pytest") is not None:
        return False
    import types

    stub = types.ModuleType("pytest")

    def parametrize(argnames, argvalues, ids=None):
        names = [a.strip() for a in argnames.split(",")] \
            if isinstance(argnames, str) else list(argnames)

        def deco(fn):
            cases = []
            for i, vals in enumerate(argvalues):
                packed = vals if len(names) > 1 else (vals,)
                if callable(ids):
                    label = ids(vals)
                elif ids is not None:
                    label = ids[i]
                else:
                    label = str(i)
                cases.append((label, dict(zip(names, packed))))
            prev = getattr(fn, "_cases", None)
            if prev is None:
                fn._cases = cases
            else:                      # несколько parametrize на функции
                fn._cases = [(f"{a}-{b}", {**ka, **kb})
                             for a, ka in prev for b, kb in cases]
            return fn
        return deco

    mark = types.SimpleNamespace(parametrize=parametrize)

    def importorskip(name, reason=None):
        try:
            return importlib.import_module(name)
        except ImportError:
            raise Skipped(reason or f"нет модуля {name}")

    def skip(reason=""):
        raise Skipped(reason)

    stub.mark = mark
    stub.importorskip = importorskip
    stub.skip = skip
    stub.Skipped = Skipped
    sys.modules["pytest"] = stub
    return True


def main() -> int:
    stubbed = _install_pytest_stub()
    if stubbed:
        print("pytest не найден — используется встроенная подмена\n")

    passed = failed = skipped = 0
    failures = []

    for path in sorted((ROOT / "tests").glob("test_*.py")):
        name = path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Skipped as exc:
            print(f"{name:<18} ПРОПУЩЕН ({exc})")
            skipped += 1
            continue

        fns = [(n, f) for n, f in vars(module).items()
               if n.startswith("test_") and callable(f)]
        mod_pass = mod_fail = 0
        for fname, fn in fns:
            cases = getattr(fn, "_cases", [("", {})])
            for label, kwargs in cases:
                try:
                    fn(**kwargs)
                    mod_pass += 1
                except Skipped:
                    skipped += 1
                except Exception:
                    mod_fail += 1
                    tag = f"{name}::{fname}" + (f"[{label}]" if label else "")
                    failures.append((tag, traceback.format_exc()))
        passed += mod_pass
        failed += mod_fail
        mark = "ok" if mod_fail == 0 else f"{mod_fail} ПРОВАЛ"
        print(f"{name:<18} {mod_pass:>4} прошло   {mark}")

    if failures:
        print("\n" + "=" * 70)
        for tag, tb in failures[:10]:
            print(f"\n--- {tag}")
            print(tb.rstrip())
        if len(failures) > 10:
            print(f"\n... ещё {len(failures) - 10} провалов")

    print("\n" + "=" * 70)
    print(f"прошло {passed}, провалено {failed}, пропущено {skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
