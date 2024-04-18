from pathlib import Path


from windmark.core.orchestration import task


@task
def sanitize_ledger_path(ledger: str) -> str:
    assert Path(ledger).exists(), "ledger path does not exist"

    return ledger
