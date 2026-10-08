"""Mutants for scripts/eval_api.py and the /health fields it relies on.

  python -m uv run python tools/mutation_check.py tests/mutants/eval_api.py

Every line breaks one rule on purpose; the tests must notice (see tools/mutation_check.py).
This file is not collected by pytest (its name does not start with `test_`).
"""

EVAL = "scripts/eval_api.py"
APP = "src/docextract/api/app.py"
T = ["tests/test_eval_api.py"]
TA = ["tests/test_eval_api.py", "tests/test_api.py"]

MUTANTS = [
    (
        EVAL,
        T,
        "the blind rule is skipped",
        "reason = blind_refusal(row, rows, labels_dir)\n"
        "        if reason is not None:\n            return finish",
        "reason = None\n        if reason is not None:\n            return finish",
    ),
    (
        EVAL,
        T,
        "the cost cap is not noticed",
        'return record, code == "budget_exceeded"',
        "return record, False",
    ),
    (
        EVAL,
        T,
        "the cap does not stop the run",
        "        if hit_cap:\n            stopped = True",
        "        if hit_cap:\n            stopped = False",
    ),
    (
        EVAL,
        T,
        "spend is the server's lifetime total",
        "budget_spent_usd=spent_after - spent_before",
        "budget_spent_usd=spent_after",
    ),
    (
        EVAL,
        T,
        "a 401 is a failed document",
        "if response.status_code == 401:",
        "if response.status_code == 4010:",
    ),
    (EVAL, T, "the doc id is not sent", 'data={"doc_id": doc_id}', "data={}"),
    (
        EVAL,
        T,
        "pages are sent in reverse",
        "for page in pages\n    ]",
        "for page in reversed(pages)\n    ]",
    ),
    (EVAL, T, "the model recorded is the client's", 'model=facts["model"]', 'model="client"'),
    (
        EVAL,
        T,
        "the prompt hash is not the server's",
        'prompt_sha256=facts["prompt_sha256"]',
        'prompt_sha256="0" * 64',
    ),
    (EVAL, T, "the settings are not the server's", 'settings=facts["settings"]', "settings={}"),
    (
        EVAL,
        T,
        "the token is dropped",
        'return {"Authorization": f"Bearer {token}"} if token else {}',
        "return {}",
    ),
    (
        EVAL,
        T,
        "an error without a record has no message",
        "record = _failed(doc_id, message)",
        "record = _failed(doc_id, '')",
    ),
    (
        EVAL,
        T,
        "a server error with a record is lost",
        'record = DocRecord.model_validate(body["record"])',
        "raise KeyError",
    ),
    (
        EVAL,
        T,
        "old servers are accepted",
        "if missing:\n        raise ApiUnavailable",
        "if False:\n        raise ApiUnavailable",
    ),
    (EVAL, T, "a non-JSON health is accepted", "or not isinstance(facts, dict)", "or False"),
    (
        EVAL,
        T,
        "an existing run folder is overwritten",
        "if (args.runs_dir / args.run_id).exists():",
        "if False:",
    ),
    (
        EVAL,
        T,
        "the dry run sends",
        "    if args.dry_run:\n"
        "        return dry_run(selected, rows, args.pages_dir, args.ground_truth_dir)\n",
        "    if args.dry_run:\n        pass\n",
    ),
    (EVAL, T, "a scoring failure is hidden", "code = code or scored", "code = code"),
    (
        APP,
        TA,
        "health reports the wrong prompt hash",
        '"prompt_sha256": prompt_sha256(SYSTEM_PROMPT)',
        '"prompt_sha256": "x"',
    ),
    (APP, TA, "health drops the cache setting", '"cache": cfg.llm_cache,', '"cache": "on",'),
    (
        APP,
        TA,
        "health reports the endpoint wrongly",
        '"endpoint": state.settings.llm_base_url',
        '"endpoint": ""',
    ),
]
