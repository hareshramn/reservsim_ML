# Agent Execution Protocol

This repository uses a documentation-first workflow. Implementation starts only after documentation approval.

## Scope Boundary (Current Phase)

- Allowed: planning, specs, interface definitions, validation criteria, runbook drafting.
- Not allowed: simulator coding, CUDA kernels, training scripts, parsing code, or benchmark execution.
- Promotion rule: move to implementation only when `docs/11_definition_of_done.md` pre-implementation gate is approved.

## Working Agreement for Agents

1. Follow `docs/00_project_charter.md` constraints and non-goals.
2. Keep interfaces stable once locked in `docs/04_software_architecture.md`.
3. Prefer additive changes; do not remove agreed sections without rationale.
4. Update dependent docs when interface/schema changes are proposed.
5. Mark assumptions explicitly; avoid hidden defaults.

## Branch and Task Naming

- Branch format: `agent/<agent-id>/<task-id>-<short-topic>`
- Task ID format: `DOC-XX` (example: `DOC-04`)
- Suggested examples:
  - `agent/a/DOC-02-physics-math`
  - `agent/c/DOC-05-gpu-plan`

## Commit Message Convention

- Format: `<task-id>: <scope> - <result>`
- Examples:
  - `DOC-03: numerics - define IMPES pseudocode and CFL policy`
  - `DOC-07: validation - lock parity and speedup metrics`

## Verification Before Handoff

Each agent must verify:
1. Required headings are present for assigned file(s).
2. Cross-references point to existing docs.
3. Interface and schema statements are consistent with `docs/04_software_architecture.md`.
4. Open questions are documented in a dedicated section.

## Mandatory Handoff Note Template

```md
## Handoff Note
- Agent:
- Task ID:
- What was decided:
- Unresolved questions:
- Files touched:
- Acceptance evidence links:
```

## Merge Priority

1. Charter, problem, physics, numerics.
2. Architecture and interface locks.
3. GPU + ML plans.
4. Validation and artifact standards.
5. Timeline, risk, done criteria, runbook.

