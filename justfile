train:
  @poetry run pyflyte --verbose run source/pipelines/workflow.py pipeline 

check:
  @git add .
  @poetry run pre-commit

doc:
  @poetry run pdoc ./source/core/architecture.py -o ./docs/site

test:
  @poetry run pytest ./source/tests

tensorboard:
  @poetry run tensorboard --logdir ./logs --host localhost --port 8888

# cov:
#   @poetry run pytest --cov=myproj ./source/tests
