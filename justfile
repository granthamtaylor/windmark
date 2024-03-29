_list:
  @just --list

# @poetry run pyflyte --verbose run windmark/pipelines/workflow.py pipeline

# run pre-commit checks
check:
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
  @typst watch docs/content/whitepaper.typst --root=docs

# run training pipeline
train:
  @poetry run python windmark/main.py
