train:
  @poetry run pyflyte run source/pipelines/workflow.py pipeline

check:
  @git add .
  @poetry run pre-commit

doc:
  @poetry run pdoc ./source/core/architecture.py -o ./docs/site

test:
  @poetry run pytest ./source/tests

# cov:
#   @poetry run pytest --cov=myproj ./source/tests
