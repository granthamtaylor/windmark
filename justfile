check:
  @git add .
  @poetry run pre-commit

run:
  @poetry run pyflyte run source/pipelines/workflow.py pipeline
