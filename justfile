train:
  @poetry run pyflyte --verbose run windmark/pipelines/workflow.py pipeline 

check:
  @git add .
  @poetry run pre-commit

test:
  @poetry run pytest ./windmark/tests

tensorboard:
  @poetry run tensorboard --logdir ./logs --host localhost --port 8888

doc:
  @poetry run pdoc windmark/core/architecture.py -o ./docs/site

main:
  @poetry run python windmark/main.py

compile:
  typst compile docs/content/motivations.typst --root=docs
