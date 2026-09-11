# Present so pytest puts the repo root on sys.path: the pipeline is a flat
# set of top-level modules, so `import config` in tests/ only resolves when
# the rootdir is importable.
