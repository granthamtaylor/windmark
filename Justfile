# run pre-commit checks
check: publish
  @git add .
  @uv run pre-commit

# compile whitepaper
publish:
  @typst compile docs/whitepaper/whitepaper.typ docs/whitepaper/windmark.pdf  --root=docs

# run training pipeline
train: && clean
  @uv run -m windmark

# reset pyflyte cache
reset:
  @uv run pyflyte local-cache clear

# run model explorer TUI
explore:
  @uv run -m windmark.devtools.explorer

# clean up temporary files
clean:
  @rm -rf wandb
  @rm -rf outputs
