"""
file_map.py - show which Python files in a project import each other.

    python file_map.py --root C:\\SCCS-UAV-Tracking
    python file_map.py --root C:\\SCCS-UAV-Tracking --focus C:\\SCCS-UAV-Tracking\\ground_station --exclude hardware_backups

--root     the project folder to scan (default: current folder)
--focus    only report on files in this folder
--exclude  folder names to skip

Writes file_map.txt (report) and file_map.html (diagram) into file_map_output/
next to this script.
"""
import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path

SKIP = {".git", "__pycache__", ".venv", "venv", "build", "dist"}


def find_files(root, exclude):
    """Every .py file under root, except those in skipped folders."""
    files = []
    for path in sorted(root.rglob("*.py")):
        folders = path.relative_to(root).parts[:-1]
        if not any(f in SKIP or f in exclude for f in folders) and path != Path(__file__).resolve():
            files.append(path)
    return files


def read_imports(path):
    """Yield (module, dots, names) for each import line in a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        print(f"skipped {path} (syntax error)")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):              # import a.b
            for alias in node.names:
                yield alias.name, 0, []
        elif isinstance(node, ast.ImportFrom):        # from a.b import c   /   from . import c
            yield node.module or "", node.level, [a.name for a in node.names]


def find_file(module, folders, files):
    """Turn "a.b.c" into a/b/c.py or a/b/c/__init__.py in one of the folders."""
    parts = [p for p in module.split(".") if p]
    for folder in folders:
        base = folder.joinpath(*parts)
        for candidate in (base.parent / (base.name + ".py"), base / "__init__.py"):
            if candidate in files:
                return candidate
    return None


def scan(root, files):
    """Return {file: files it imports} and {file: outside libraries it imports}."""
    imports, libraries = defaultdict(set), defaultdict(set)
    for path in files:
        for module, dots, names in read_imports(path):
            if dots:   # relative import: start in the file's folder, go up one per extra dot
                folders = [path.parents[dots - 1]]
            else:      # absolute import: try the file's folder, then each parent up to the root
                folders = [p for p in path.parents if p == root or root in p.parents]

            # "from pkg import mod" can name a file; otherwise the module itself is the file.
            targets = {find_file(f"{module}.{n}", folders, files) for n in names} - {None}
            if not targets:
                targets = {find_file(module, folders, files)} - {None}

            if targets - {path}:
                imports[path] |= targets - {path}
            elif not dots and module:
                libraries[path].add(module.split(".")[0])
    return imports, libraries


def write_report(root, shown, imports, used_by, libraries):
    name = lambda p: p.relative_to(root).as_posix()
    lines = [f"Root: {root}", f"Files: {len(shown)}", ""]
    for path in shown:
        lines.append(name(path))
        lines += [f"    imports    {name(t)}" for t in sorted(imports[path])]
        lines += [f"    used by    {name(u)}" for u in sorted(used_by[path])]
        if libraries[path]:
            lines.append(f"    libraries  {', '.join(sorted(libraries[path]))}")
        lines.append("")
    return "\n".join(lines)


def write_diagram(root, shown, imports, used_by, focus):
    """Mermaid flowchart: the shown files, their neighbours, and arrows for each import."""
    drawn = set(shown)
    for path in shown:
        drawn |= imports[path] | used_by[path]

    node = lambda p: "f_" + re.sub(r"\W", "_", p.relative_to(root).as_posix())
    top = lambda p: p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else "(root)"

    lines = ["flowchart LR"]
    for i, folder in enumerate(sorted({top(p) for p in drawn})):
        lines.append(f'  subgraph g{i}["{folder}"]')
        for p in sorted(drawn):
            if top(p) == folder:
                lines.append(f'    {node(p)}["{p.relative_to(root).as_posix()}"]')
        lines.append("  end")
    for source in sorted(drawn):
        lines += [f"  {node(source)} --> {node(t)}" for t in sorted(imports[source]) if t in drawn]
    if focus:
        lines.append("  classDef focus fill:#dbe8ff,stroke:#2c6bd2")
        lines += [f"  class {node(p)} focus" for p in shown]
    return HTML.replace("DIAGRAM", "\n".join(lines))


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>File map</title></head>
<body style="font-family: sans-serif">
<p>Each box is a Python file. An arrow A &rarr; B means A imports B. Blue boxes are the focused folder.</p>
<pre class="mermaid">
DIAGRAM
</pre>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, maxTextSize: 500000, maxEdges: 2000 });
</script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description="Show which Python files import each other.")
    parser.add_argument("folder", nargs="?", default=".", help="same as --root")
    parser.add_argument("--root", help="project folder to scan")
    parser.add_argument("--focus", help="only report on files in this folder")
    parser.add_argument("--exclude", nargs="*", default=[], help="folder names to skip")
    args = parser.parse_args()

    root = Path(args.root or args.folder).resolve()
    files = find_files(root, set(args.exclude))
    imports, libraries = scan(root, set(files))

    used_by = defaultdict(set)
    for source, targets in imports.items():
        for target in targets:
            used_by[target].add(source)

    shown, focus = files, None
    if args.focus:
        focus = Path(args.focus).resolve()
        if not focus.is_dir():                      # allow a focus written relative to the root
            focus = (root / args.focus).resolve()
        if not focus.is_dir():
            parser.error(f"focus folder not found: {args.focus}")
        shown = [f for f in files if focus in f.parents]

    out = Path(__file__).resolve().parent / "file_map_output"
    if focus:
        out = out / focus.name                      # e.g. file_map_output/drone_air
    out.mkdir(parents=True, exist_ok=True)
    report = write_report(root, shown, imports, used_by, libraries)
    (out / "file_map.txt").write_text(report, encoding="utf-8")
    (out / "file_map.html").write_text(write_diagram(root, shown, imports, used_by, focus), encoding="utf-8")

    print(report)
    print(f"Report:  {out / 'file_map.txt'}")
    print(f"Diagram: {out / 'file_map.html'}")


if __name__ == "__main__":
    main()
