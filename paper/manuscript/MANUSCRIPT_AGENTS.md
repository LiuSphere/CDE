# FACMAC + EA Microgrid Manuscript Workspace

## Scope

This manuscript studies continuous-action FACMAC with evolutionary augmentation
for cooperative microgrid scheduling. Paths in this document are relative to
the repository root, `C:/Users/xyx/Desktop/GDE-main`, unless stated otherwise.

- Main source: `paper/manuscript/main.tex`.
- Section sources: `paper/manuscript/sections/`.
- Editable figure sources: `paper/manuscript/figures/`, including TikZ diagrams
  included directly by the manuscript.
- Bibliography records: `paper/bibliography.bib`.
- Local reference PDFs: `paper/`; converted reading materials and research
  indexes: `paper/converted/`.
- Local Elsevier template examples and bibliography styles:
  `paper/templates/elsarticle/`.
- Generated PDFs, logs, auxiliary files, and figure exports:
  new output directories under `results/`.

Read and apply this file only when modifying manuscript text, equations,
figures, tables, or LaTeX formatting, as required by the project `AGENTS.md`.
It does not govern training, code changes, experiment execution, or standalone
read-only manuscript reviews. Review rules below concern self-checks performed
as part of an authorized manuscript modification. Explicit user requests to
inspect or edit this instruction file are separate authorized tasks.

Historical drafts, reports, and submission material are evidence of earlier
work, not authority for current claims or journal conventions. Check their
suitability before reusing their content.

When consulting local reference PDFs or `paper/converted/`, separate observed writing patterns from source
facts and current-manuscript recommendations. Use the papers to inform style
and presentation choices, but independently verify any technical fact, number,
method description, or conclusion before using it in the manuscript. A file's
presence in the reading corpus does not by itself justify citing it. These
papers may also support manuscript citations when their content has been
checked. Add or change a
manuscript citation only through a verified bibliography record in
`paper/bibliography.bib`, with a citation role supported by the source itself.

## Length and Necessity Discipline

The manuscript has a hard limit of 6,000 words. Do not exceed this limit or
substitute a page-count target for it. Every methodological detail, reporting
item, definition, or caveat added to the text competes for that word budget.
Default to omission.

- Include a detail — a reporting item, a robustness check, a symbol
  definition, a caveat, a limitation statement, a rebuttal of an
  alternative design — only when omitting it would leave a stated claim
  unsupported, a result uninterpretable, or a symbol undefined. Do not
  include it merely because it is true, because it is standard practice in
  the field, or because a rule elsewhere in this document would technically
  permit adding it.
- Do not preemptively defend the validity, realism, or authenticity of an
  experiment beyond what is needed to support the claims actually made in
  the surrounding text. Add further justification, robustness evidence, or
  a caveat only in response to a concrete reviewer objection — one the user
  has actually reported, or one directly implied by a specific weakness in
  the current evidence — not as a default posture applied to every result.
- When a rule elsewhere in this document (for example, in Evidence,
  Terminology, and Reporting or Manuscript Structure and Detail) could be
  satisfied by either a brief statement or an exhaustive one, satisfy it
  with the shortest statement that keeps the claim supported and the result
  interpretable. Those rules define what must not be missing, not a target
  amount of detail to include.
- When trimming a passage under this discipline, remove content rather than
  compress it into denser wording; a shorter sentence that still says the
  same low-value thing is not the goal.
- This discipline does not relax the Evidence, Terminology, and Reporting
  requirements for claims the manuscript actually makes: a stated
  comparison, percentage, or result still needs its baseline, evaluation
  unit, and material conditions. It relaxes only the impulse to add
  supporting detail beyond what a stated claim requires.

## Engineering-Journal Framing

This manuscript targets an engineering journal. Its central question is not
whether a model, training strategy, representation, or prompt produces an
algorithmic gain. Its central question is whether the proposed method produces
credible, feasible, and practically meaningful operational improvement in an
energy system with physical constraints and operating objectives.

- Start from an energy-system task and a concrete operating difficulty, such
  as uncertainty, coupled devices or time scales, delayed consequences,
  cumulative constraints, or real-time decision complexity. Explain why the
  proposed method addresses that difficulty; do not present algorithmic novelty
  as the problem by itself.
- State the method's inputs, outputs, and place in the energy-management chain.
  When an LLM or learning component is used, make its information role and
  connection to control or scheduling traceable. Explain new problem-specific
  interfaces fully; compress standard algorithms, unchanged network structures,
  and software plumbing unless they materially affect reproducibility or
  interpretation.
- Use evidence at the appropriate level. Algorithm-level results explain
  internal differences; interface-level evidence explains how information enters
  the operating process; physical-level evidence establishes balance,
  constraints, and state feasibility; system-level evidence establishes changes
  in cost, emissions, carbon intensity, reliability, or operating behavior;
  deployment-level evidence defines runtime, resource, data, and scenario
  limits.
- Do not let evidence at one level substitute for another. Better prediction
  accuracy does not by itself establish lower operating cost, and faster
  inference does not by itself establish feasible scheduling.
- Make the main results energy-system results. Use aggregate task-relevant
  metrics across the declared evaluation units to establish overall
  performance, and use representative operating trajectories to explain
  behavior or mechanisms rather than to establish generality by themselves.
- Add model, prompt, representation, training-budget, resource, or component
  analyses only when they explain a claimed energy-system result. Do not adopt a
  generic computer-science or LLM experiment package as the paper's main
  contribution.
- For every stated contribution, identify the engineering difficulty, the
  method action, and the resulting capability. Map it to a specific method
  location and direct experimental evidence. If the available evidence does not
  isolate a component's effect, narrow the contribution rather than implying an
  unsupported component-level conclusion.
- Organize each result from the observation in a figure or table, through its
  numerical and operating conditions, to the relevant physical mechanism or
  method component and the system or scenario boundary of that interpretation.
  Separate direct observation from interpretation. Use causal language only
  when the comparison or analysis isolates the claimed cause; otherwise use
  explicitly qualified interpretive language.
- Keep conclusions within the demonstrated physical, system, and deployment
  scope. State limitations or uncovered system conditions when they materially
  limit the conclusion.

## Evidence, Terminology, and Reporting

- Before drafting or revising a material section, identify its core claim, the
  direct evidence that supports it, and the boundary of that claim. Do not turn
  missing evidence or an unresolved boundary into a stronger conclusion.
- Use one canonical form for each recurring system name, method name,
  abbreviation, symbol, metric, unit, and comparison object. Keep those forms
  consistent across the text, equations, figures, tables, and captions.
- Treat the abstract and the main text as separate abbreviation scopes. At the
  first occurrence within each scope, write the full term followed by its
  abbreviation in parentheses. After that definition, use only the abbreviation
  throughout the remainder of that scope: the full term must not appear again,
  either alone or paired with the abbreviation.
- For computational experiments, state the independent evaluation unit, such
  as a scenario, system, representative day, or random seed. Report the
  aggregation method, uncertainty definition, and any result-selection or
  exclusion rule that materially affects a comparison. Do not present repeated
  observations from one evaluation unit as independent evidence.
- For every method comparison, state the common evaluation conditions and any
  material difference in information access, forecast horizon, training or
  optimization budget, constraint handling, or result selection. Every
  percentage or relative improvement must identify its baseline, direction, and
  aggregation level.
- Keep power and energy, instantaneous and cumulative quantities, normalized
  and physical quantities, and per-step and per-episode metrics explicitly
  distinct in notation and prose.
- Do not invent sample counts, uncertainty intervals, statistical tests,
  exclusion rules, or other reporting details that are not supported by the
  available study record.

## Manuscript Structure and Detail

- Place system boundaries, physical assumptions, variables, balance relations,
  device constraints, objectives, and physical dynamics in system modeling.
  Place the RL state, action, transition mapping, reward semantics, discount,
  and horizon in the MDP formulation. Place the proposed algorithm's information
  flow and update mechanisms in methodology. When retained because they
  materially affect reproducibility or interpretation, place data partitions,
  scenarios, initialization protocols, random seeds, training budgets,
  hyperparameters, software or hardware conditions, and baseline procedures in
  experimental setup. Place measured values, comparisons, and evidence-based
  interpretations in results. Move an experiment-specific setting into the
  system model only when it changes the scientific definition of the problem.
- Include a displayed equation only when it defines a load-bearing physical
  relation, decision variable, objective, constraint, state transition,
  proposed update rule, or quantity used to interpret the results. Compress a
  standard relation through a citation or concise definition when its
  derivation is not part of the contribution. Introduce an equation's purpose
  before it appears, define new symbols near their first use, and explain its
  engineering or methodological role afterward. Remove equations that are not
  used by later reasoning, constraints, algorithms, or results.
- Define every manuscript-specific mathematical symbol, index, set, coefficient,
  and nonstandard operator at its first appearance or in the sentence
  immediately following it. Do not leave a symbol to be inferred from code,
  context, a later section, or a table; standard mathematical operators need
  explanation only when they are used with a nonstandard meaning.
- Keep a displayed equation on one line when it fits clearly within the column;
  do not split it merely for visual symmetry or stylistic preference. Break it
  across lines only when a single line would exceed the column width or
  materially reduce readability. Use an appropriate mathematical environment
  such as `aligned`, `split`, or `multline`, place line breaks at semantic
  boundaries, align repeated relation symbols when useful, and preserve the
  punctuation required by the surrounding sentence.

## Academic Prose and LaTeX Paragraphs

- Prefer neutral, descriptive wording. Unless the user explicitly requests
  otherwise, avoid words that convey clear approval, disapproval, or
  preference; describe properties and observed differences in factual terms.
- Keep code implementation, mathematical modeling, and academic exposition as
  three distinct layers. Use code to verify what the system actually computes;
  use the mathematical model to define variables, relations, constraints, and
  assumptions; and use academic prose to explain their engineering meaning,
  rationale, and scope. Do not copy implementation structure, API terminology,
  control flow, schema fields, reset/step language, numerical guards, or
  identifier-style phrasing into the manuscript. Translate implementation
  behavior into domain-level concepts, and include an implementation detail
  only when it materially affects the scientific method, reproducibility, or
  interpretation of the results.
- Do not require the manuscript and the code to correspond one-to-one. The
  code is an implementation and verification record, whereas the manuscript is
  an academic account of the problem, method, evidence, and interpretation.
  They must remain consistent in their core method, assumptions, information
  flow, and experimentally relevant conditions, but the manuscript may
  abstract, reorganize, or explain these elements differently from the code.
  A code module, function, variable, or execution step need not have a matching
  subsection, equation, term, or sentence in the manuscript, unless that detail
  materially affects reproducibility or interpretation.
- Report experimental settings from the actual configuration used for the
  reported runs, not from unrelated defaults or an unverified approximation.
  Preserve exact counts, seeds, data partitions, and discrete settings.
  Use sufficient significant digits or scientific notation for continuous
  parameters; do not impose a fixed number of decimal places. For example,
  report `0.00015` or `1.5 x 10^-4`, never `0.00`. Rounding reported results
  must not materially change their meaning, comparison, or uncertainty.
  Keep the underlying computed values and exact run configurations traceable.
- When a displayed equation ends a sentence with a period, the explanatory text
  that follows must begin a new paragraph and retain first-line indentation.
  Do not use `\noindent` to remove that indentation, and do not attach the
  explanation to the same paragraph as the equation.
- When a recurring model quantity requires an identifier, represent it with
  mathematical notation rather than an implementation identifier. Do not use
  names such as `carbon_cost_increment`, `economic_total_cost`, or other
  camelCase/snake_case field names as notation in prose, equations, tables, or
  figure captions. Use a defined mathematical symbol instead, such as
  $\Delta C_t^{\mathrm{carbon}}$.
- In mathematical notation, use italic type for variables and variable indices,
  and upright roman type for descriptive labels, units, and named operators.
  Choose the font by the symbol's role, not by capitalization or position.
  For example, retain descriptive labels such as `\mathrm{grid}`,
  `\mathrm{ch}`, and `\mathrm{dis}`, while agent and time indices remain
  italic. Apply the convention consistently across text, equations, figures,
  and tables.
- Keep mathematical subscripts and superscripts concise. Use one to five
  letters where possible; replace longer labels with a defined abbreviation or
  a separate symbol definition rather than placing a long word in an index.
- When auditing existing prose for redundancy, defensiveness, or imprecise
  wording — whether reviewing a paragraph the user pastes in or drafting new
  text — check it against the Language Expression Review Checklist below
  before deciding to keep, cut, or reword it.

## Language Expression Review Checklist

Sentence-level problems recur in academic prose independently of the
manuscript's technical content. Before proposing an edit to an existing
passage, or before finalizing new prose, check candidate sentences against
these nine patterns. For any candidate deletion or reword, verify with
Grep/Read against the rest of the document that the phrase is not
load-bearing — i.e., not referenced, reused downstream, or required by an
adjacent equation — before acting.

1. **Redundant restatement** (concrete → concrete → abstract): two concrete
   reasons are given, then a third sentence re-abstracts the same point with
   no new information. Delete the third sentence.
2. **Restated conclusion** (same point, different wording): a sentence states
   a conclusion, and the next sentence restates the identical conclusion in
   different words. Delete the restatement.
3. **Defensive denial** (ruling out an alternative nobody raised): a clause
   pre-emptively denies a design choice that was never proposed, and the
   denial is already implied by an adjacent equation or definition. Delete
   the denial.
4. **Term collision** (reusing an overloaded word): a word is already bound
   elsewhere in the manuscript to one specific, heavily used mechanism;
   reusing it loosely for an unrelated operation creates a false association.
   Reword to a neutral synonym.
5. **Misattributed cause** (borrowed explanation, wrong mechanism): a
   phenomenon is explained with boilerplate language borrowed from a related
   but different context, naming a cause that is not the one actually
   operating in this manuscript. Reword to name the mechanism this manuscript
   actually describes.
6. **Unneeded backward reference**: a phrase such as "as noted above" points
   back to earlier content that the paragraph does not actually depend on.
   Delete the reference rather than merely shortening it.
7. **Unforced admission of a generic flaw**: the text volunteers that the
   method shares some well-known limitation common to a broad method class,
   without connecting it to a paper-specific consequence or mitigation.
   Remove it, or replace it with the paper-specific point actually being
   made.
8. **Unrequested rigor padding**: a reporting detail, robustness check,
   caveat, or justification is added to defend a result's validity or
   authenticity, but no claim actually stated in the surrounding text
   depends on it and no reviewer objection, reported or specifically
   anticipated, calls for it. Delete it; see Length and Necessity
   Discipline above.
9. **Reverse framing** (explaining X by first saying that it is not Y):
   state the positive fact, action, observation, or conclusion first. Avoid
   leading with "not", "does not", "rather than", "instead of", "unlike", or
   a similar contrast when the excluded alternative was never a plausible
   reader assumption. This applies to facts, method descriptions, results,
   and contributions; repeated negative framing makes the prose defensive
   and gives unnecessary attention to the alternative. Keep the contrast
   only when the alternative is the reader's default interpretation or when
   excluding it is necessary to preserve the technical meaning, such as a
   shaded band that would otherwise be read as a confidence interval or a
   training curve that would otherwise be read as test-set performance. In
   those cases, state X first and use the shortest useful clarification, such
   as "X, not Y". Category 3 covers denying an unraised design choice; this
   category covers the broader habit of explaining any fact through its
   opposite.

Categories 1-3 are redundant-expression problems; 4-5 are wording- or
attribution-accuracy problems; 6 is an unnecessary cross-reference; 7 is an
unforced admission of a generic weakness; 8 is unnecessary defensive
padding; 9 is unnecessary reverse framing.

## Figure Evidence Contract

- Define the evidence role or one-sentence conclusion of each figure before
  creating or revising it. Each panel must answer a distinct evidence question
  or serve a distinct explanatory role; merge or remove panels that merely
  redraw the same evidence.
- Keep figure captions concise: identify the figure subject and, for
  multi-panel figures, map each panel key to its meaning. Place the variables,
  units, comparison conditions, operating constraints, aggregation and
  uncertainty definitions, and any necessary data-source explanation in the
  manuscript text that first introduces or interprets the figure. For
  schematics, explain the represented components, information flow, and system
  boundary in that accompanying text.
- In that accompanying text, include only what a reader needs in order to read
  the figure correctly: the evidence the plotted quantity is measured on, the
  meaning of an axis that is not a plain time base or interaction count, what
  the lines and shaded bands aggregate, and any construction that would
  otherwise invite a wrong inference, such as a non-decreasing best-so-far
  series. Keep rendering mechanics out of the manuscript; interpolation and
  smoothing method, window or bin size, marker spacing, and axis ranges the
  figure already displays belong in the asset manifest.
- Place a reading caveat in the paragraph that draws the inference it
  protects, not in the figure's introduction. Grouping each caveat with its
  consequence keeps the same construction from being explained twice.
- Do not select data, scenarios, variables, or result ranges merely because
  they make a figure clearer or more favorable. Any scientifically justified
  exclusion or aggregation must be stated.

## Result-Comparison Prose

Prose that introduces or analyzes a table or figure should guide the reader
to the evidence rather than duplicate it.

- State each comparative fact once, in the paragraph where it most
  naturally belongs. When a later paragraph addresses the same comparison
  from a different angle, such as adding a causal explanation to a fact
  already established, extend the earlier statement with a connective such
  as "also" or "in turn" rather than re-deriving or restating the same
  ordering or magnitude a second time.
- Do not copy a table or figure's own numeric values into the surrounding
  prose merely to name them again. State the comparison qualitatively
  (highest, narrowest, smallest by a wide margin) and cite the table or
  figure with `\ref{}` so the reader can find the exact value without
  reading it twice. Report a number in prose only when the prose does
  something the table or figure cannot by itself, such as combining two
  reported values or naming a threshold the table or figure does not
  display. This does not relax the baseline, direction, and aggregation
  requirements in Evidence, Terminology, and Reporting above; it relaxes
  only the impulse to restate a value the cited evidence already shows.
- Do not restate in prose a full ranking, ordering, or comparison set that
  the table or figure already displays through its own structure, such as
  row order or bolding. State only the conclusion the reader needs from
  that structure, not every entry that supports it.

## Figure and Table Presentation

- Follow the active Elsevier `elsarticle` template. The current manuscript
  uses `\documentclass[preprint,12pt]{elsarticle}`; this is the working
  preprint layout, not a reproduction of the references' published layout.
  A target journal has not yet been confirmed. Do not assume TSTE or IEEE
  requirements or switch document classes without user direction.
  Keep body text and captions controlled by the active class. Size figure
  labels and table contents for readability at their final placement rather
  than imposing the old 10 pt body, 8 pt table, or 5 pt figure-text values.
  Do not shrink fonts or change margins to satisfy the word limit.
- Use black, regular-weight text throughout figures. Reserve color and line
  weight for data marks, curves, bars, uncertainty bands, and other graphical
  encodings rather than textual emphasis.
- In a multi-panel figure, place only sequential lowercase panel keys
  `(a)`, `(b)`, and so on beneath the panels. Put each panel's descriptive
  title in the overall figure caption by explicitly mapping every key to its
  meaning.
- Keep categorical axis labels horizontal. Annotate every bar with its
  numerical value when the chart uses discrete bars. When bars also carry
  uncertainty intervals, place the values in a dedicated annotation band with
  visible clearance from the error bars so the labels cannot be mistaken for
  interval endpoints.
- Use compact figure aspect ratios and small, partially transparent scatter
  markers for dense observations. Add limited deterministic horizontal jitter
  for categorical distributions when points would otherwise overlap. Reduce
  canvas height or whitespace rather than scaling down the required body-size
  figure text.
- Lay out tables at `\footnotesize` without whole-table geometric scaling.
  Use regular-weight headers and row identifiers; adjust columns, wrapping, or
  table structure when the native layout does not fit.
- In a comparison table, bold only the single best numerical value in each
  metric column. State the preferred direction when it is not self-evident;
  do not bold headers, identifiers, or non-comparable entries.
- Preserve and report the computed value of every statistic at a precision
  sufficient to distinguish the compared results.

### General Multi-Panel Figure Design

- Design a figure at its final manuscript placement size instead of drawing a
  large canvas and geometrically shrinking it into a column. Choose a
  single-column or double-column layout first, then set the canvas, panel grid,
  aspect ratios, text sizes, and line widths for that final size.
- Remove repeated information across panels. When panels share the same
  categorical horizontal axis, omit duplicate category labels from upper rows
  and retain the complete labels on the bottom row. Keep categorical labels
  horizontal and aligned on a common baseline; use deliberate line breaks
  before considering staggered labels or smaller text.
- Use a shared legend only when it reduces repetition without creating a large
  empty band. Otherwise, place a compact legend inside a genuinely unused
  region of the relevant panel. A legend must not cover bars, curves, error
  intervals, annotations, or other evidence.
- Match numerical formatting to the plotted scale. Use scientific notation
  for large magnitudes when it shortens tick labels, and apply consistent
  formatting to panels that show comparable quantities.
- Reserve visible headroom for value labels and uncertainty intervals. When
  adjacent value labels would collide, use a small number of consistent
  annotation levels rather than reducing them below the minimum figure-text
  size. Keep annotations visually distinct from error-bar endpoints.
- Use bar width, panel aspect ratio, horizontal and vertical panel spacing, and
  panel-key padding as layout controls. Reduce unused whitespace iteratively,
  but retain visible clearance between panel keys, axis labels, tick labels,
  legends, annotations, and neighboring panels.
- Verify the standalone figure and its final rendered manuscript placement.
  Check that text remains readable, repeated labels are aligned, legends and
  annotations do not overlap data, and no avoidable whitespace remains.

## Build and Verification

All manuscript build outputs must be written in a new directory under
`results/`, including PDFs,
logs, auxiliary files, bibliography outputs, SyncTeX files, converted graphics,
and outputs from audit, review, or temporary compilation. Do not create or
leave generated build artifacts in the repository root or alongside source
files. Preserve previous outputs; do not overwrite or delete existing result
artifacts without explicit user authorization. Keep editable TikZ and other
figure sources in `paper/manuscript/figures/`; generated figure exports belong
under `results/`.

Compile `main.tex` from `paper/manuscript/` so its relative section and figure
inputs resolve correctly. Configure the output directory explicitly for
`pdflatex`, `latexmk`, or any other compilation wrapper. Do not assume that
a `.latexmkrc` exists or routes outputs. For example, replace `<new-build-id>`
with an unused directory name before running:

```powershell
Set-Location C:/Users/xyx/Desktop/GDE-main/paper/manuscript
$out = "C:/Users/xyx/Desktop/GDE-main/results/<new-build-id>"
New-Item -ItemType Directory -Path $out -ErrorAction Stop
pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$out" main.tex
```

The manuscript directly includes its TikZ figure sources; changes to those
sources take effect on compilation without copying a standalone PDF or PNG.
For generated external graphics, point the manuscript to the intended
generated asset under `results/` and verify that the updated asset is used.
Do not introduce a repository-root `figures/` synchronization step.

Batch related manuscript and figure changes before validation; do not run
LaTeX after every small intermediate edit. Compile the coherent change set,
then rerun as needed to resolve cross-references and citations. When
bibliography processing is configured, run the appropriate bibliography tool
with its outputs in the same result directory and complete the required
LaTeX passes. Do not silently change bibliography backends or document
classes to resolve a build error. Cleaning is not validation and must not
delete preserved results. Do not perform rendered-PDF visual checks
by default; check page breaks, figures, tables, citations, references, and
overflow only when the user explicitly asks for verification or when the task
is itself a visual validation or review request.

## Change Rules

Match the surrounding LaTeX style and keep labels, citation keys, figure filenames, and table identifiers stable.

Keep changes narrow and evidence-driven. Alter numbers, references, experimental results, or claims only when their supporting source material is updated. Do not invent citations or implementation details. Treat configurations, logs, result bundles, and archived submissions as sensitive; keep credentials in environment variables or untracked local configuration.

Before handling an old version, legacy implementation, or historical configuration, ask the user for explicit direction. Do not preserve, migrate, remove, archive, or refactor old-version material until the user decides how it should be handled.

## Review and Verification

A manuscript review should state the manuscript impact and validation performed.
For every key claim, record its manuscript location, evidence location, the
most material reviewer challenge, and the condition required to resolve that
challenge. A concern is blocking when the current manuscript cannot establish
its central case until the concern is resolved. Include rendered PDF evidence
when layout changes.

When the review concerns sentence-level wording rather than claims or
evidence, apply the Language Expression Review Checklist above instead of
treating every stylistic observation as a blocking concern.
