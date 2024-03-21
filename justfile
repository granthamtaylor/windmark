train:
  @poetry run pyflyte --verbose run source/pipelines/workflow.py pipeline 

check:
  @git add .
  @poetry run pre-commit

test:
  @poetry run pytest ./source/tests

tensorboard:
  @poetry run tensorboard --logdir ./logs --host localhost --port 8888

doc:
  @poetry run pdoc source/core/architecture.py -o ./docs/site

main:
  @poetry run python ./source/pipelines/main.py

compile:
  typst compile docs/content/motivations.typst --root=docs