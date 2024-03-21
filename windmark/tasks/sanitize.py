from pathlib import Path

import flytekit as fk

@fk.task
def sanitize_ledger_path(ledger: str) -> str:
    
    assert Path(ledger).exists(), 'ledger path does not exist'
    
    return ledger