# OpenCode Pre-Submission Audit Prompt

You are performing a frozen, evidence-only pre-submission audit for the
HackerRank Orchestrate Multi-Modal Evidence Review repository.

Read `AGENTS.md` first and append your own entries to the shared transcript
at `%USERPROFILE%\\hackerrank_orchestrate\\log.txt`. Do not record secrets,
API keys, tokens, or claim-agent runtime traces.

## Operating Rules

- Do not edit product code, prompts, tests, outputs, or configuration.
- Do not refactor or add features. Report findings with severity and a
  recommended smallest fix only.
- Treat documentation and earlier plans as unverified until demonstrated by
  a current command, test result, log line, or file-and-line citation.
- Use the existing code under `code/`; the Claude extraction under `../claude`
  is reference material, not the submission implementation.
- Do not run live provider calls unless the user explicitly approves sending
  local claim text and images to the configured external providers.

## Audit Tracks

Run these independent tracks in parallel where available, then consolidate:

1. Architecture and rule conformance: verify deterministic evidence and
   decision flow, risk/evidence separation, CSV-driven requirements, and all
   three verdict states.
2. Data and output integrity: resolve every image and history record, inspect
   leakage risks, run `python code/check_output.py`, and validate CSV schema.
3. Security and adversarial behavior: injection, wrong object/part, duplicate
   image, conflicting evidence, quality failure, and manipulation handling.
4. QA and reliability: run `python -m pytest code/tests -q`, inspect failures,
   retry behavior, cache behavior, and fresh-evaluation reliability.
5. Documentation and interview readiness: compare README claims with actual
   executable behavior and identify claims that cannot be demonstrated.

## Required Report Format

Write findings in this order:

1. Executive Audit Summary
2. CRITICAL Findings table: finding, evidence, severity, smallest fix
3. HIGH Findings
4. MEDIUM / LOW Findings
5. Verified-Working Components
6. Adversarial Test Results table
7. Rubric-Mapped Readiness Scorecard
8. GO / NO-GO Recommendation
9. Prioritized Remediation Plan sorted by score impact divided by time

Every finding must cite a current command/test result, log line, or exact
file path and line number. Mark checks that cannot be performed as
`UNVERIFIED`; do not guess.
