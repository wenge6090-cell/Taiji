#!/usr/bin/env python3
"""Quick import verification for DMN changes."""
import sys
sys.path.insert(0, '/root/vingobot')

from vingobot.goal.guizang_types import CangSeaMemory, CangSeaEntry, GuizangState, QiOperator

# Test save/load roundtrip
from pathlib import Path
import tempfile, os

sea = CangSeaMemory(max_entries=100)

# Add some entries
sea.add(CangSeaEntry(
    state_from=GuizangState.resting(),
    operator=QiOperator.SHENG,
    state_to=GuizangState(bits=0b001000),
    reward=0.3,
    summary="test transition",
    timestamp="2026-05-15T00:00:00",
))

# Hebbian learning
sea.hebbian_record(0, 0b001000, 0.3)

# Save
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "test_cang_sea.json"
    sea.save(path)
    assert path.is_file(), "Save failed"
    
    # Load
    loaded = CangSeaMemory.load(path, max_entries=100)
    assert loaded is not None, "Load returned None"
    assert loaded.size == 1, f"Expected 1 entry, got {loaded.size}"
    assert loaded.transition_weights[0][0b001000] > 0, "Hebbian weights not restored"
    
    print(f"OK: save/load roundtrip passed ({loaded.size} entries, weights preserved)")

# Test to_dict/from_dict
data = sea.to_dict()
assert data["version"] == 1
assert len(data["entries"]) == 1
restored = CangSeaMemory.from_dict(data)
assert restored.size == 1
print("OK: to_dict/from_dict roundtrip passed")

print("\nAll checks passed!")
