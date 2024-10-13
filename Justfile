help:
  @just --list

# run pre-commit checks
check: author
  @git add .
  @uv run pre-commit

# compile documentation with pdoc
document:
  @uv run sphinx-build -M html "./docs/site" "./docs/site/_build"

# compile whitepaper with typst
author:
  @typst compile docs/whitepaper/whitepaper.typ docs/whitepaper/windmark.pdf  --root=docs

# run training pipeline
train:
  @uv run -m windmark

# clear pyflyte cache
clear:
  @uv run pyflyte local-cache clear

# obfuscate codebase with pyarmor
obfuscate:
  @uv run pyarmor generate -r windmark
  @cp -r ./dist/pyarmor_runtime_000000 ./dist/windmark && rm -R ./dist/pyarmor_runtime_000000
  @cp -r config ./dist/config
  @zip -r dist.zip dist

explore:
  @uv run -m windmark.devtools.explorer
