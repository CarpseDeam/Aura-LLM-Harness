# Task: Duplicate File Finder

Write a Python module that walks a directory tree, identifies groups of files
with identical content, and writes a JSON report.

## Function signature

The module must define exactly one public function:

```python
def find_duplicates(directory: str, output_path: str) -> None:
    ...
```

`directory` is a path to a directory to walk recursively. `output_path` is the
path where the JSON report must be written (overwriting any existing file).
The function returns `None`.

## What counts as a duplicate

Two files are duplicates if and only if both are true:

- Their byte content is identical. Hash by full content; SHA-256 is recommended.
- Their size is strictly greater than zero. Empty files MUST NOT be reported as
  a duplicate group, even when more than one empty file exists.

A "duplicate group" is a set of two or more files that share identical content.
Files with content unique to themselves do not appear in the report.

## Walking behavior

Walk `directory` recursively. Include regular files only. Do not follow symlinks.

If a file cannot be read (permission denied, I/O error, etc.), skip it and
print a warning to `stderr` of the form:

```
warning: cannot read <path>: <reason>
```

Do not abort the run on such errors.

## Report shape

Write JSON to `output_path` with the following structure:

```json
[
  ["a.txt", "c.log", "nested/b.txt"],
  ["dup1.bin", "subdir/dup2.bin"]
]
```

- The top-level value is a list of groups.
- Each group is a list of file paths (strings) belonging to that group.
- Paths must be **relative to `directory`**, using forward slashes (`/`) as
  the path separator on every platform.
- Within each group: paths are sorted lexicographically (Python `sorted()`
  string ordering).
- Groups themselves are sorted by their first path (lexicographic).
- If there are no duplicate groups, write an empty list: `[]`.

## Output requirements

- The output file is valid UTF-8 JSON.
- The file must contain only the JSON document — no leading or trailing prose,
  no comments.
- Either compact or pretty-printed JSON is acceptable.

## Out of scope

- No CLI / `argparse` layer is required. The verifier imports `find_duplicates`
  directly and calls it.
- Standard library only — do not import any third-party packages.
