check:
  @git add .
  @poetry run pre-commit

train:
  @poetry run pyflyte run source/pipelines/workflow.py pipeline

doc:
  @poetry run pdoc ./source/core/architecture.py -o ./docs/site

test:
  @poetry run pytest ./source/tests