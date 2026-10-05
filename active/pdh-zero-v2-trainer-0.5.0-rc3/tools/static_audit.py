"""无第三方依赖的语法、格式、安全与注释基础审计。"""
import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("pdh_v2.static_audit")
PROHIBITED_IMPORTS = {"pickle", "marshal", "shelve"}
PROHIBITED_CALLS = {"eval", "exec"}


def _python_files() -> tuple[Path, ...]:
    """返回正式源码、测试与工具文件，排除缓存和临时目录。"""
    roots = (ROOT / "src", ROOT / "tests", ROOT / "tools")
    return tuple(
        path
        for base in roots
        for path in sorted(base.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def _definition_lines(node: ast.AST) -> int:
    """返回定义占用行数；缺失end_lineno时安全返回一行。"""
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    return end - start + 1


def _audit_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """审计一个文件；硬错误与软复杂度警告分开返回。"""
    relative = str(path.relative_to(ROOT))
    raw = path.read_text(encoding="utf-8")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    lines = raw.splitlines()
    if len(lines) > 500:
        errors.append({"file": relative, "code": "FILE_OVER_500", "lines": len(lines)})
    elif len(lines) > 200:
        warnings.append({"file": relative, "code": "FILE_OVER_200", "lines": len(lines)})
    for number, line in enumerate(lines, 1):
        if len(line) > 100:
            errors.append({"file": relative, "line": number, "code": "LINE_OVER_100"})
        if line.endswith((" ", "\t")):
            errors.append({"file": relative, "line": number, "code": "TRAILING_SPACE"})
        if "\t" in line:
            errors.append({"file": relative, "line": number, "code": "TAB"})
    try:
        tree = ast.parse(raw, filename=relative)
    except SyntaxError as exc:
        errors.append({"file": relative, "line": exc.lineno, "code": "SYNTAX"})
        return errors, warnings
    if ast.get_docstring(tree) is None:
        errors.append({"file": relative, "code": "MODULE_DOCSTRING"})
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = _definition_lines(node)
            if length > 200:
                errors.append(
                    {"file": relative, "line": node.lineno, "code": "DEF_OVER_200"}
                )
            elif length > 30:
                warnings.append(
                    {
                        "file": relative,
                        "line": node.lineno,
                        "code": "DEF_OVER_30",
                        "name": node.name,
                        "lines": length,
                    }
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                relative.startswith(("src/", "tools/"))
                and not node.name.startswith("_")
                and ast.get_docstring(node) is None
            ):
                errors.append(
                    {
                        "file": relative,
                        "line": node.lineno,
                        "code": "PUBLIC_DOCSTRING",
                        "name": node.name,
                    }
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in PROHIBITED_IMPORTS:
                    errors.append(
                        {"file": relative, "line": node.lineno, "code": "UNSAFE_IMPORT"}
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in PROHIBITED_IMPORTS:
                errors.append(
                    {"file": relative, "line": node.lineno, "code": "UNSAFE_IMPORT"}
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in PROHIBITED_CALLS:
                errors.append(
                    {"file": relative, "line": node.lineno, "code": "UNSAFE_CALL"}
                )
    return errors, warnings


def audit() -> dict[str, Any]:
    """运行全仓基础静态审计并返回结构化报告。"""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    files = _python_files()
    for path in files:
        file_errors, file_warnings = _audit_file(path)
        errors.extend(file_errors)
        warnings.extend(file_warnings)
    return {
        "schema_version": 1,
        "files": len(files),
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
        "ruff": "NOT_INSTALLED",
        "mypy": "NOT_INSTALLED",
    }


def main() -> int:
    """CLI入口；可选报告路径禁止覆盖。"""
    parser = argparse.ArgumentParser(description="运行PDH-Zero v2无依赖静态基础审计。")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = audit()
    raw = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    try:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(raw)
    except OSError as exc:
        LOGGER.error(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 2
    LOGGER.info(raw.rstrip())
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
