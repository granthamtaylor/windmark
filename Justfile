help:
  @just --list

# run pre-commit checks
check: publish
  @git add .
  @uv run pre-commit

# compile documentation
document:
  @uv run sphinx-build -M html "./docs/site" "./docs/site/_build"

# compile whitepaper
publish:
  @typst compile docs/whitepaper/whitepaper.typ docs/whitepaper/windmark.pdf  --root=docs

# run training pipeline
train: freeze && clean
  @uv run -m windmark

# reset pyflyte cache
reset:
  @uv run pyflyte local-cache clear

# obfuscate codebase with pyarmor
obfuscate: freeze
  @uv run pyarmor generate -r windmark
  @cp -r ./dist/pyarmor_runtime_000000 ./dist/windmark && rm -R ./dist/pyarmor_runtime_000000
  @cp -r config ./dist/config
  @zip -r dist.zip dist

# run model explorer TUI
explore:
  @uv run -m windmark.devtools.explorer

# freeze dependencies
freeze:
  @uv pip compile pyproject.toml > requirements.txt --python-platform linux

# clean up temporary files
clean:
  @rm -rf wandb
  @rm -rf outputs
