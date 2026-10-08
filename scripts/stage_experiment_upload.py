"""Stage an experiment's retained records for object storage, mirroring local paths.

Hard-links the files worth keeping into a staging tree (no copy) and writes a manifest
of what was staged and what was left out and why: symlinked model views, hard-link
data mirrors, re-creatable merged weights and superseded checkpoints. Push the staging
tree with ``ab push <staging>/<experiment>/ whai:open-audio-llm/runs/<experiment>/``.
"""
import argparse
import fnmatch
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--include', nargs='+', required=True,
                        help='paths under the experiment to stage (dirs or files)')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='glob patterns (relative to the experiment) of paths to leave out')
    parser.add_argument('--keep', nargs='*', default=[],
                        help='glob patterns that override --exclude (e.g. a retained checkpoint)')
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment.resolve()
    target = args.staging / root.name
    staged, skipped, links = [], [], []

    def excluded(relative):
        # fnmatch's * also matches "/", so "a/*" covers a whole subtree.
        if any(fnmatch.fnmatch(relative, pattern) for pattern in args.keep):
            return False
        return any(fnmatch.fnmatch(relative, pattern) for pattern in args.exclude)

    for item in args.include:
        base = root / item
        paths = [base] if base.is_file() else sorted(base.rglob('*'))
        for path in paths:
            relative = str(path.relative_to(root))
            if path.is_symlink():
                links.append({'path': relative, 'target': os.readlink(path)})
                continue
            if path.is_dir():
                continue
            if excluded(relative):
                skipped.append(relative)
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(path, destination)
            staged.append({'path': relative, 'bytes': path.stat().st_size})
    manifest = {'experiment': root.name, 'staged_files': len(staged),
                'staged_bytes': sum(s['bytes'] for s in staged), 'exclude_patterns': args.exclude,
                'keep_patterns': args.keep, 'skipped_files': len(skipped), 'symlinks': links,
                'files': staged}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: manifest[k] for k in ('staged_files', 'staged_bytes', 'skipped_files')}
                     | {'symlinks': len(links)}))


if __name__ == '__main__':
    main()
