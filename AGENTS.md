## Telegram UI invariants

These rules apply only to messages that include inline keyboards. Plain text messages must remain unmodified and must not be padded or height-normalized.

The Telegram UI is intentionally treated as a fixed-size menu:
- every message with buttons must render as exactly 3 text lines
- every message with buttons must render exactly 3 rows of buttons

The goal is a visually stable interface with zero layout shift between steps.

More importantly, the layout is not only fixed in size, but also **structurally predictable**:
- button positions follow consistent patterns across all menus
- the same conceptual actions tend to appear in the same areas
- missing options do not collapse layout — they are replaced structurally

This allows the user to build strong spatial memory and interact very quickly without re-scanning the UI.

---

### Text layout

Only real text lines represent content. Padding lines are artificial and exist purely for layout stability.

There are only two valid layouts:

#### Single-content messages
Used for prompts and simple instructions.

- the real content must be placed on the 3rd line (closest to buttons)
- the first two lines are padding

Layout:
- line 1: padding
- line 2: padding
- line 3: real content

Rationale:
The user’s attention is naturally focused near the buttons.

---

#### Context + prompt messages
Used when showing current selection state + next action.

- the state/context must be on line 1
- the prompt must be on line 3
- line 2 is always padding

Layout:
- line 1: context (e.g. `Selected: ...`)
- line 2: padding
- line 3: prompt

---

### Width reservation

Padding must use a single consistent mechanism tied to a configured width.

Do not introduce:
- manual spacing
- empty lines (`\n\n`)
- alternative padding techniques

The layout system must remain deterministic and uniform.

---

### Selected formatting

Selection state must be visually structured, not string-concatenated.

Rules:
- prefix (`Selected:`) is plain text
- each value is emphasized individually
- separators are plain and never emphasized

Conceptually:
- plain label
- alternating [value, separator, value, separator...]

This ensures readability and avoids visual noise.

---

## Button layout invariants

All inline keyboards must always have exactly 3 rows.

This is a strict invariant to preserve consistent menu height and interaction predictability.

---

### Structural consistency (core principle)

The UI is designed as a **fixed spatial grid**, not a dynamic list.

This means:
- each menu has a predefined set of possible button slots
- these slots are filled deterministically
- unavailable options do not remove slots — they are replaced with structural placeholders

As a result:
- store and fetch menus share the same layouts
- the user does not need to re-learn layouts between flows
- interaction becomes faster and more automatic

---

### Hierarchy and positioning

Buttons are arranged by role:

- **Top + middle rows** → selectable options (primary interaction space)
- **Bottom row** → navigation (Back or terminal action)

The layout must feel consistent across all menus.

---

### Back button

If present:
- it must always occupy the entire bottom row
- its position must never change between menus

This creates a stable navigation anchor.

---

### Fixed option grid (first two rows)

The first two rows form a **deterministic option grid**.

Options are not placed left-to-right sequentially.  
Instead, they follow a consistent spatial placement pattern optimized for usability.

#### Snake layout (default)

Options are placed starting from the **top-right corner**, then filled in a snake-like pattern across the first two rows:

- start at top-right
- go down
- then move left
- then up
- continue alternating direction while moving left

Conceptually:
- positions closer to the top-right are easier to reach and are filled first
- newer / more relevant / higher-priority items naturally occupy these positions

Additional rule:
- if the number of option slots is odd, the first row contains one fewer slot than the second row

This layout is:
- deterministic
- consistent across menus
- optimized for interaction speed

---

### Fixed layout across flows

Menus must not change shape depending on data availability.

Instead:
- a full set of possible options is defined for each menu
- available options are rendered normally
- unavailable options are replaced with dummy placeholders in the same positions

This ensures:
- store and fetch share identical layouts
- the UI does not shift when data changes
- the user can rely on position rather than re-reading labels

---

### Dummy buttons

Dummy buttons are purely structural.

They serve two purposes:
1. preserve layout height when there are too few buttons
2. preserve fixed option positions when options are unavailable

Strict rules:
- they must not affect logic, parsing, or state transitions
- their interaction must be inert
- they must not visually compete with real actions

They are part of layout, not behavior.

---

### Special actions

Some actions (e.g. “All”) are not domain values but UI-level actions.

They still participate in layout:
- they are treated as regular options in the grid
- they occupy fixed positions just like any other option
- differences between flows (e.g. store vs fetch) are handled via dummy substitution

This keeps layout uniform while allowing different behavior.

---

### Distribution principles

Layouts must be:
- deterministic
- consistent across menus
- stable across data states

Key rules:
- never collapse layout due to missing options
- never insert placeholders when enough real options exist
- always prefer structural consistency over compactness

---

### Directional ordering (UX rationale)

The layout intentionally leverages spatial ergonomics:

- top-right positions are easiest to reach and scan
- important or recent items tend to appear there
- movement follows predictable patterns

This allows users to:
- build muscle memory
- interact faster without scanning entire menus
- rely on position instead of text

---

## UI/domain separation

UI representation may differ from domain structure, but only at rendering time.

Rules:
- domain enums and values remain authoritative
- UI may reorder, group, or position values for usability
- such transformations must not affect:
  - storage
  - parsing
  - business logic

The UI layer is a projection, not a source of truth.

# Repository Agent Rules

## Tooling

- Use `uv` for dependency management and running commands.
- Pin exact versions when adding dependencies.
- Install the development environment with `uv sync --dev`.
- Repository uses a `src/` layout; tests must import the installed package.
- Do not modify `sys.path` in tests or use pytest/pythonpath hacks.
- Run tests with `uv run pytest`.

## Code Style

- Target modern Python. The minimum supported Python version is defined in `pyproject.toml` (`requires-python`) and must not be duplicated here.
- Prefer current language features over legacy compatibility patterns.
- Do not introduce backward-compatibility constructs unless explicitly required.
- Prefer explicit code over clever abstractions.
- Prefer public and documented library APIs over private/internal ones.
- Do not rely on private attributes or methods (for example names prefixed with `_`) unless explicitly requested or there is no viable public alternative.

- Ruff is the source of truth for code style.
- Do not introduce code that violates Ruff rules.
- Run `uv run ruff check --fix` and `uv run ruff format` before committing changes.

- Pyright is used for static type checking.
- Do not introduce code that fails Pyright type checking.
- Before finalizing changes, run `uv run pyright` and resolve type errors.

- Prefer fixing root causes over using `Any`, `cast`, suppressions, or fragile library internals.
- Prefer explicit None checks and proper type narrowing over unsafe assumptions.

- When a library intentionally relies on dynamic runtime behavior that static analysis cannot model
  (for example `pydantic-settings` loading values from environment variables),
  do not distort the architecture just to satisfy the type checker.

- In such cases, prefer a narrow and well-documented suppression over:
  - duplicating schemas or models
  - introducing parallel representations of the same data
  - using private/internal library APIs
  - significantly increasing code complexity

- Suppressions are allowed only when:
  - runtime behavior is correct and intentional
  - the limitation comes from the type checker, not from a real bug
  - the suppression is minimal in scope (prefer line-level over file-level when possible)
  - the suppression includes a short explanatory comment

Imports:

- Use absolute imports across the entire codebase.
- Do not use relative imports, even within the same package.
- Import paths should always start from the top-level package (e.g. `general_bot...`).

Formatting:

- Use single quotes for normal strings; use triple double quotes for triple strings.
- Use Google-style docstrings.

Exception documentation:

- If a function intentionally raises exceptions as part of its contract, document them in a `Raises:` section.
- Only document exceptions that callers are expected to handle or that represent meaningful API behavior.
- Do not document incidental internal exceptions from underlying libraries unless they are part of the method's intended behavior.

## Operating Assumptions

Audience and scale:

- Primary user: repository owner; possibly a few trusted users.
- Traffic: low and mostly sequential.
- Developer time is the most constrained resource.

Design principles:

- Prefer simplicity and explicit assumptions over defensive completeness.
- Fail fast on unhandled exceptions so cloud restart/alerts surface issues.
- Soft or unbounded buffering is acceptable when aligned with personal usage.
- Handlers may orchestrate logic pragmatically; avoid premature module splitting.
- Logs should be concise and human-readable (cloud platform adds timestamps).
- Optimize for common paths and maintainability over exhaustive edge-case guards.

Refactor triggers:

Structure or guard rails should be added only after clear pain signals:

- repeated production failures
- difficult debugging
- increasing message/user volume
- handlers becoming hard to modify safely
- memory or runtime limits reached

Accepted risks:

- Some edge cases may remain intentionally unhandled.
- Resource bounds and stronger isolation are introduced after incidents, not preemptively.
- Maintenance speed and clarity take priority over defensive completeness.

### Agent behavior

Agents should:

- Avoid proposing large architectural changes for hypothetical scale.
- Treat some tradeoffs as intentional when they reduce maintenance cost.
- Prefer incremental improvements preserving current behavior and simplicity.
- Suggest next steps only when naturally useful; keep them concise and optional.

### Plan mode

When working in plan/spec mode:

- Prefer asking clarifying questions over making assumptions.
- If the task is ambiguous, ask multiple questions before proposing a plan.
- Use numbered questions.

### Recommendation policy

Default to the smallest change that solves today's problem.

Classify recommendations:

- now — required due to active impact
- later — optional until a trigger appears

### Testing expectations

- Cover critical paths and previously observed regressions.
- Exhaustive edge-case testing is not required by default.
- Prefer fast unit tests; add integration tests only for high-risk flows.

## Code Review Expectations

Reviews should evaluate more than correctness.

### Architecture

- Ensure abstraction boundaries match intended layers.
- Infrastructure code must remain generic and domain-independent.
- Avoid unnecessary layers, indirection, or premature abstractions.
- Prefer small explicit modules over framework-like structures.

### Modern Python

- Prefer modern Python 3.13 idioms.
- Avoid legacy compatibility constructs.
- Favor readable control flow over clever patterns.

### API design

- Public APIs should remain minimal and stable.
- Keep helper methods private unless external use is clearly justified.
- Method naming should reflect the abstraction level (infra vs domain).
- Avoid expanding the public surface without clear benefit.

### Internal `__init__.py` policy

For internal packages prefer empty `__init__.py`.

Rules:

- Do not create package-level APIs for internal packages unless requested.
- Do not re-export symbols just for convenience.

Preferred:

from general_bot.infra.tasks import TaskScheduler
from general_bot.infra.tasks import TaskSupervisor
from general_bot.infra.s3 import S3Client

Avoid:

from general_bot.infra import TaskScheduler

Use re-exports only when intentionally defining a stable package API.

### Maintainability

- Implementations should remain straightforward and easy to modify.
- Avoid unnecessary configuration or genericity.
- Code should remain understandable after long gaps.

### Reliability

- Consider failure modes and resource lifecycles.
- Ensure cleanup exists for files, streams, or clients.
- Prefer fail-fast behavior over silent error masking.

### Naming

- Names should match abstraction level and responsibility.
- Infrastructure code should avoid domain semantics.

### Simplicity rule

Even if no bugs exist, verify:

- the solution is the simplest clear implementation
- abstractions are justified
- the public API can be smaller or clearer

Review feedback should be grouped as:

- critical issues
- important improvements
- optional polish

## Commit Messages

Use Conventional Commits.

Format:

type(scope): short description

Optional body explaining the reasoning.

Example:

feat(infra): add fail-fast detached task supervision

Introduce `TaskSupervisor` for detached asyncio tasks with centralized
exception handling and a one-shot failure hook.

### Types

Allowed types:

- feat — new capability or user-visible behavior
- fix — bug fix
- refactor — internal restructuring without behavior change
- perf — performance improvement without behavior change
- test — tests added or updated
- docs — documentation-only changes
- chore — repository maintenance, tooling, or runtime requirement changes

Choose the type that reflects the **intent of the change**, not the file modified.  
Do not invent new types.

Examples:

- chore: update `.gitignore`
- chore(deps): bump `httpx` to 0.28.0
- chore!: bump python to 3.14

### Breaking changes

Agents must evaluate whether a change is backward-incompatible.

Typical breaking changes:

- renamed or removed environment variables
- renamed or removed settings fields
- changed configuration formats or required values
- changed public APIs
- changed CLI flags or behavior
- changed persisted data formats or schemas
- changed runtime requirements (e.g. minimum Python version)

Breaking commits must use:

type!: description  
type(scope)!: description

and include a footer:

BREAKING CHANGE: describe the migration required.

Example:

feat!: add superuser-aware shutdown notifications

BREAKING CHANGE: replace `USER_ALLOWLIST` with `SUPERUSER_IDS` and `USER_IDS`.

Configuration or settings contract changes should be assumed breaking
unless proven otherwise.

### Subject rules

Subjects must:

- be lowercase
- use imperative mood
- be ≤72 characters
- describe the system-level behavior change

Avoid vague subjects such as:

- "update rules"
- "improve system"

Prefer concise, action-oriented phrasing:

- use "bump" for version updates (e.g. python, dependencies)
- avoid overly abstract or policy-like wording

Implementation details should appear in the commit body.

### Referencing code entities

Wrap file names, modules, classes, and functions in backticks.

Example:

Rename `MessageBuffer` to `ChatMessageBuffer`.

### Shell safety for commit messages

Backticks trigger command substitution inside double quotes.

Incorrect:

git commit -m "Refactor `TaskScheduler` API"

Preferred:

git commit -m 'refactor(services): make task scheduler key-agnostic'

Safest method for multiline commits:

git commit -F - <<'EOF'
refactor(services): make task scheduler key-agnostic

Replace `TaskScheduler` user-coupled API with a generic `Hashable` key.
EOF

Agents must:

- avoid backticks inside double-quoted commit messages
- use single quotes or heredocs when backticks appear

### Referencing canonical repository files

When a commit primarily modifies a well-known document such as:

- `AGENTS.md`
- `README.md`
- `CHANGELOG.md`

mention it directly in the subject when helpful.

Example:

docs: expand `AGENTS.md` commit guidelines

### Commit body guidance

Add a body when reasoning is important, for example when:

- multiple subsystems are affected
- a refactor changes conceptual structure
- a rename clarifies an abstraction
- the reason is not obvious from the subject

The body should explain **why**, not list the diff.

### Scope discipline

Scopes should represent stable architectural subsystems.

Valid scopes:

- app — runtime bootstrap and wiring
- handlers — Telegram routing and handlers
- services — application services/state containers
- infra — shared infrastructure utilities
- settings — configuration loading and validation
- deps — dependency updates

Avoid choosing scope based only on a single touched file.

### Root-level files

Commits affecting repository-level files usually omit scope.

Examples:

docs: update `README.md`  
chore: update `.gitignore`

Use `deps` only for dependency changes.

### Generic infrastructure

Generic reusable modules belong to the `infra` scope.

Example:

feat(infra): add fail-fast task supervisor

### When scope is unclear

If a change does not clearly belong to a subsystem, omit the scope.

Example:

feat: add task supervisor utility

### Commit coherence

Commits should remain conceptually coherent by subsystem and intent.

Do not combine unrelated changes in a single commit.  
If unrelated local modifications exist, commit only the files relevant to the requested change.
