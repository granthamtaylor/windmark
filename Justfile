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
  @typst compile docs/whitepaper/whitepaper.typ docs/whitepaper/build/windmark.pdf  --root=docs
  @pandoc docs/whitepaper/whitepaper.typ -o docs/whitepaper/build/windmark.md
  @pandoc docs/whitepaper/whitepaper.typ -o docs/whitepaper/build/windmark.docx
  @pandoc docs/whitepaper/whitepaper.typ -o docs/whitepaper/build/windmark.html
  @zip -r whitepaper.zip docs

# run training pipeline
train:
  @poetry run python windmark/main.py

# clear pyflyte cache
clear:
  @poetry run pyflyte local-cache clear
