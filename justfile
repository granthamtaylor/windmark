_list:
  @just --list

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
  @poetry run pdoc windmark/core/architecture.py -o ./docs/site

# compile whitepaper with typst
whitepaper:
  @typst watch docs/content/whitepaper.typst --root=docs

# run training pipeline
train:
  # @poetry run pyflyte --verbose run windmark/pipelines/workflow.py pipeline
  @poetry run python windmark/main.py
