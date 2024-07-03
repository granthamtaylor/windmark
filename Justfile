_list:
  @just --list

# run pre-commit checks
check: whitepaper
  @git add .
  @poetry run pre-commit

# run pytest
test:
  @poetry run pytest ./windmark/tests

# start tensorboard
tensorboard:
  @poetry run tensorboard --logdir ./logs --host localhost --port 8888

# compile documentation with pdoc
doc:
  @poetry run sphinx-build -M html "./docs/site" "./docs/site/_build"

# compile whitepaper with typst
whitepaper:
  @typst compile docs/whitepaper/whitepaper.typ docs/whitepaper/windmark.pdf  --root=docs

# run training pipeline
train:
  @poetry run python windmark

# clear pyflyte cache
clear:
  @poetry run pyflyte local-cache clear

# obfuscate codebase with pyarmor
obfuscate:
  @poetry run pyarmor generate -r windmark
  @mv ./dist/gen/pyarmor_runtime_000000 ./dist/gen/windmark
  @cp -r config ./dist/config
  # @cp -r notebooks ./dist/notebooks
  # @cp -r data ./dist/data
  # @cp pyproject.toml ./dist/pyproject.toml
  # @cp .python-version ./dist/.python-version
  # @cp .pre-commit-config.yaml ./dist/.pre-commit-config.yaml
