# M3: <title>

Status: spec (written before code; not edited after approval)
Roadmap ref: docs/ROADMAP.md#M3

## Goal
<1-2 sentences: what is true after this milestone that isn't true now>

## Scope
- <in-scope item>

## Out of scope
- <explicitly excluded item>

## Files
- Create: <path>
- Change: <path>
- Must NOT touch: <path/area>

## Acceptance criteria
Offline and deterministic. Mark costly commands [COSTS MONEY] + estimate.
1. `python -m uv run pytest -q tests/test_x.py` -> all pass
2. `python -m uv run python scripts/x.py --check` -> exit 0, prints "..."
3. `python -m uv run ruff check` -> clean

## Risks
- <risk> -> <mitigation>

## Deviations from roadmap
- <none | what the roadmap said vs. what the code actually needs>