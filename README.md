# KinkSol V3

KinkSol solves coupled, nonlinear static field equations on uniform
one-dimensional Cartesian and radial grids. It uses finite differences,
sparse Newton solvers and automatic differentiation to construct the
Jacobian.

There are two ways to define and run a calculation:

- **Python model files** specify equations, boundary conditions, initial
  profiles, parameters, solver controls, observables and plots in code.
- The optional **Model Builder** provides a desktop interface for creating
  the same kind of model. It accepts direct residual equations or derives
  them symbolically from a reduced static energy density. Projects can be
  saved, reopened, exported as Python and run from the interface.

Both workflows use the same numerical backend. Exported Python models
require KinkSol and the numerical dependencies, but do not require the UI,
Qt or SymPy. The separate
[KinkSol V3 User Guide](KinkSol_V3_User_Guide.pdf) provides a numerical reference.

## Contents

- [Installation](#installation)
- [Running Python model files](#running-python-model-files)
- [Using the Model Builder](#using-the-model-builder)
- [Geometry, fields and mathematical input](#geometry-fields-and-mathematical-input)
- [Equations and symbolic derivation](#equations-and-symbolic-derivation)
- [Parameters and scans](#parameters-and-scans)
- [Boundary conditions and special rows](#boundary-conditions-and-special-rows)
- [Solver settings and automatic differentiation](#solver-settings-and-automatic-differentiation)
- [Writing a Python model](#writing-a-python-model)
- [Observables, plots and output files](#observables-plots-and-output-files)
- [Saving and reopening models](#saving-and-reopening-models)
- [Appearance and defaults](#appearance-and-defaults)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

## Installation

Use **Python 3.10 or newer**. Run these commands from the KinkSol project
folder. A virtual environment keeps the dependencies separate from the
operating system's Python. Activation is optional: the commands use the
environment's interpreter explicitly.

| Use | Dependency file | Packages |
|---|---|---|
| Python models and numerical solver | `requirements.txt` | NumPy, SciPy, Matplotlib |
| Desktop Model Builder and numerical solver | `requirements-ui.txt` | The numerical packages, PySide6 Essentials, SymPy |

### Linux and macOS

Install and launch the UI:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ui.txt
.venv/bin/python launch_ui.py
```

For command-line use only, install `requirements.txt` instead:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### Windows PowerShell

Install and launch the UI:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.venv\Scripts\python.exe launch_ui.py
```

For command-line use only, replace `requirements-ui.txt` with
`requirements.txt`. Reuse an existing environment when appropriate;
installing `requirements-ui.txt` into a numerical-only environment adds the
UI dependencies. Normal UI use requires a graphical desktop session.

## Running Python model files

The [models/](models/) directory contains these examples and template:

| Model | File |
|---|---|
| Coupled domain wall with three real fields | [model_template_DOMAIN_WALL.py](models/model_template_DOMAIN_WALL.py) |
| Radial sphaleron | [model_template_SPHALERON.py](models/model_template_SPHALERON.py) |
| Six-field 2HDM domain wall | [model_2HDM_DOMAIN_WALL.py](models/model_2HDM_DOMAIN_WALL.py) |
| Starting point for a new handwritten model | [model_template_BLANK.py](models/model_template_BLANK.py) |

Run a complete model directly. On Linux/macOS:

```bash
.venv/bin/python models/model_template_DOMAIN_WALL.py
```

On Windows:

```powershell
.venv\Scripts\python.exe models/model_template_DOMAIN_WALL.py
```

Edit the model's `get_model_config()` to set the grid, independent
parameters, scan mode, solver options and output controls. Set derived
parameter relations in `finalize_parameters()`. Each supplied model's
`main()` selects a single solve or parameter scan from its configuration.
The handwritten and UI examples have their own saved resolutions.

To write a new model, copy a template to another file under `models/`.
The blank template deliberately raises `NotImplementedError`: complete its
equations and initial guess, and implement or remove unused optional
observable and plotting callbacks. Keep `models/` in the project. The
supplied templates locate the project root from their position there, and
the UI uses the directory for temporary run snapshots.

### Headless runs

Set `show_solution_plots=False` and `show_scan_plot=False` in the model's
`output` configuration. Enable the corresponding `save_*` options if figures
should be written to disk. Select a non-interactive Matplotlib backend:

```bash
MPLBACKEND=Agg .venv/bin/python models/model_template_DOMAIN_WALL.py
```

In Windows PowerShell:

```powershell
$env:MPLBACKEND = 'Agg'
.venv\Scripts\python.exe models/model_template_DOMAIN_WALL.py
```

For an exported UI model, substitute its filename. A JSON builder project
must first be opened in the UI and exported as Python to run this way.

## Using the Model Builder

Launch `launch_ui.py` with the UI dependencies installed. An optional path
opens a JSON project or Python model immediately:

```bash
.venv/bin/python launch_ui.py ui_examples/scalar_kink.json
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

### First calculation

1. Click **Open...** and select `scalar_kink.json` from `ui_examples/`.
2. Click **Validate** and **Preview guess** to inspect the definition and
   starting profile.
3. On **Equations**, use **Derive equations** to inspect the residual and
   its automatically inferred stencil dependencies.
4. Open the **settings cog** to review numerical controls.
5. Click **Run model**. **Results** shows profiles, residual history,
   the observable, convergence status and the run log.
6. Use **Save project** for an editable JSON project or **Export Python**
   for a command-line model.

The solver runs in a child process. **Stop** cancels it; completed points
remain available for inspection. The editor remains responsive, and results
use the model snapshot taken when the run started.

### Editor pages

| Page | What to configure |
|---|---|
| 01 Geometry & grid | Geometry, site count and spacing or extent |
| 02 Fields & guesses | Real field names, plot labels and initial profiles |
| 03 Parameters & scans | Fixed/derived parameters and optional scan ranges |
| 04 Equations | Direct residuals or static energy density; equation/dependency preview |
| 05 Boundaries | Endpoint conditions and optional replacement rows |
| 06 Observables & plots | Scalar observable, profile expressions and output files |
| 07 Python model | Generated source, direct editing and regeneration |
| 08 Results | Profiles, residuals, scan observables, CSV export and log |

**New model** creates an unfinished blank draft with one real field slot,
1001 sites and spacing 0.05. Equations, guesses and boundary expressions
start empty, and the observable is disabled. Added fields start empty too.
Drafts can be saved before completion; validation, export and Run require a
complete model.

### Included builder projects

**Open...** starts in [ui_examples/](ui_examples/):

| Project | Input mode | Default grid |
|---|---|---|
| [scalar_kink.json](ui_examples/scalar_kink.json) | Static energy density | 1001 sites, spacing 0.05, Cartesian `[-25, 25]` |
| [domain_wall_EOM.json](ui_examples/domain_wall_EOM.json) | Direct residuals | 1001 sites, spacing 0.05, Cartesian `[-25, 25]` |
| [domain_wall_ED.json](ui_examples/domain_wall_ED.json) | Static energy density | 1001 sites, spacing 0.05, Cartesian `[-25, 25]` |
| [radial_sphaleron.json](ui_examples/radial_sphaleron.json) | Direct residuals | 501 sites, spacing 0.05, radial `[0, 25]` |

The two domain-wall projects have the same zero-residual solutions and
energy observable. The energy-derived bulk residuals are twice the direct
residuals, so their absolute residual tolerances differ in normalisation.
Examples are defined by their files; edit a saved copy to create a project.

## Geometry, fields and mathematical input

For `N` sites with spacing `h`, the endpoint-to-endpoint extent is
`(N-1)*h`. Cartesian coordinates are centred:

```text
x_i = (i - (N-1)/2) * h
```

Radial coordinates start at the origin:

```text
r_i = i * h
```

An odd Cartesian site count includes a site at zero. At least three sites
are required. The current row compiler supports one-dimensional uniform
grids and several coupled fields. Selecting radial coordinates does not
insert a radial Laplacian or integration measure; include these in the
equations and observable as appropriate.

Each unknown is real. Represent a complex field by separate real fields,
such as `sigma1` and `sigma2`, and enter the real reduced energy or residuals
in those variables. The forms do not support complex literals, the
imaginary unit or automatic complex-field decomposition.

Field and parameter names must be unique identifiers and cannot use
reserved names such as `x`, `r`, `D`, `D2`, `pi` or function names. Labels
are independent of identifiers and accept Matplotlib math text, such as
`$\phi$`.

| Form input | Meaning |
|---|---|
| `phi` | Declared field at the current site |
| `D(phi)` | First coordinate derivative |
| `D2(phi)` | Second coordinate derivative |
| `x` / `r` | Cartesian / radial coordinate |
| `lam`, `v`, etc. | Declared parameters |
| `pi`, `e` | Mathematical constants |
| `+`, `-`, `*`, `/`, `**` | Arithmetic; `^` is also accepted as a power |
| `sin`, `cos`, `exp`, `tanh`, `log`, `sqrt` | Supported single-argument functions |
| `sinh`, `cosh`, `sech` | Supported functions exported using exponential identities |

Derivative calls take a field name, such as `D(phi)`, rather than a general
expression. Initial profiles use the coordinate and parameters, for example
`v*tanh(x/width)`, `sech(x/width)` or `r**2/(1+r**2)`.

The form editor accepts this mathematical grammar rather than arbitrary
Python. Attribute access, indexing, undefined symbols, piecewise expressions
and unsupported function calls are rejected. Use a handwritten Python model
for logic outside the forms' scope.

## Equations and symbolic derivation

### Direct residual equations

Enter one residual per field, understood to be equal to zero. For example:

```text
D2(phi) - lam*phi*(phi**2-v**2)
```

Do not append `= 0`. The builder generates finite-difference callbacks and
infers the referenced field/site dependencies for the sparse Jacobian.

### Static energy density

Choose **Derive from static energy density** and supply a reduced
one-dimensional density and integration weight. For example:

```text
Density:  D(phi)**2/2 + lam*(phi**2-v**2)**2/4
Weight:   1
```

For positive `lam` and `v`, this gives the direct residual above. A kink
interpolates between `-v` and `v`; a suitable initial profile is
`v*tanh(v*sqrt(lam/2)*x)`.

For coordinate `q`, fields `u_a` and weight `w`, the builder varies

$$
I = \int dq\,w(q)\,\mathcal{E}(u_a,u'_a,q)
$$

and constructs

$$
\frac{d}{dq}\left(\frac{\partial(w\mathcal{E})}{\partial u'_a}\right)
-\frac{\partial(w\mathcal{E})}{\partial u_a}=0.
$$

This includes mixed and field-dependent kinetic terms. Continuum equations
are derived first, then discretised with central differences; they are not
obtained by varying the discrete observable quadrature.

| Reduction | Typical weight | Possible angular prefactor for the observable |
|---|---|---|
| Cartesian | `1` | `1` |
| Cylindrical radial | `r` | `2*pi` |
| Spherical radial | `r**2` | `4*pi` |
| Density already includes the measure | `1` | As defined by the reduction |

Include the measure once. Residuals retain their weight and coefficients;
the builder does not divide out a radial weight or potentially vanishing
kinetic factor. The tolerance applies to this residual normalisation.

The symbolic mode accepts a reduced static functional of real fields and
their first spatial derivatives. Supply the ansatz, gauge choice and
one-dimensional reduction yourself. Full spacetime field theories,
time-dependent ansatze and higher-derivative boundary problems are outside
this mode's scope.

## Parameters and scans

In the UI, **fixed** parameters accept constant expressions such as `1` or
`sqrt(3)/2`. **Derived** parameters reference others: for example, name
`lambda_1`, kind `derived`, expression `2*beta`. Dependencies are ordered
automatically, and circular definitions are rejected.

An empty scan table gives one solve. Scan fixed parameters by specifying a
minimum, maximum, point count and linear or logarithmic spacing. Endpoints
are physical values; logarithmic endpoints must be positive. Multiple rows
form the Cartesian product of their ranges.

For a Python model, configure a scan in `get_model_config()`:

```python
"variation_mode": "single",  # "none", "single", or "multi"
"single_variation": {
    "name": "beta", "min": 1.0, "max": 2.0,
    "num": 5, "is_log": False,
},
"multi_variation": [],
```

For `"multi"`, put a list of range dictionaries with those keys in
`multi_variation`. `finalize_parameters(params)` is applied separately to
each point so derived quantities track the varied parameters.

The first point uses the analytic guess. Later points use the most recent
converged solution as a continuation seed when available. Python models can
adapt it through `build_continuation_guess()`. Failed points do not replace
the last converged seed, and their observables are excluded from scan
observable values.

## Boundary conditions and special rows

Set a condition at each end for every field:

| Type | Residual |
|---|---|
| Dirichlet | `field - value` |
| Neumann | `D(field) - value` |
| Robin | `alpha*field + beta*D(field) - value` |
| Custom | A complete residual using the declared fields and parameters |

`D` means differentiation towards increasing coordinate at both ends,
rather than along the outward normal. Generated endpoint derivatives are
first-order one-sided differences and include spacing.

The radial origin row uses its boundary condition instead of the bulk
equation. Set regularity conditions explicitly so singular bulk terms such
as `1/r` are not evaluated at `r=0`.

Optional replacement rows apply a residual to a specified field at `centre`
or an integer site, counted from zero. For example, `phi` pins the field to
zero. A Cartesian centre row requires an odd site count. These replace
existing equations, so they do not add unknowns or equations. The UI infers
custom cross-field boundary dependencies. Conditional constraints, periodic
wrap-around stencils and nonlocal conditions require a Python model.

## Solver settings and automatic differentiation

In the UI, click the **settings cog**. In Python models, edit `solver` in
`get_model_config()`:

| Group | Main controls |
|---|---|
| Final solve | Tolerance, maximum iterations, reporting frequency and plain-Newton step scale |
| Pointwise phase | Optional local pre-phase, its tolerance, iteration limit and relaxation |
| Trust region | Optional trust-region Newton phase, radius bounds, trial count and acceptance/growth controls |
| Recovery | Optional pointwise rescue and retry after a failed global solve |
| Near solution | Optional return of an iterate within a multiple of tolerance; it remains unconverged |
| Advanced | Jacobian differentiation mode: `auto`, `scalar` or `row` |

The UI exposes every option in [`HybridOptions`](strategy/hybrid.py).
Hover over controls for their meaning. **Apply**, **Cancel** and **Restore
Defaults** affect the current model. Blank UI projects default to plain
sparse Newton, tolerance `1e-9`, at most 100 final iterations, and disabled
pointwise, trust-region and rescue phases. Opened projects and handwritten
models retain their own configurations.

The convergence measure is the maximum absolute residual, `max(abs(F))`.
The tolerance therefore depends on equation and boundary normalisation.
Inspect the status and residual: successful process exit alone does not
establish convergence. A small residual also does not establish stability
or select a unique physical branch.

### How the Jacobian is obtained

The numerical backend uses forward-mode automatic differentiation with
dual numbers to differentiate the discrete residuals. It fills a sparse
Jacobian using declared dependencies. Symbolic energy variation in the UI
is a separate step performed before discretisation.

`jacobian_autodiff="auto"` selects scalar or vector-row differentiation and
checks vector compatibility against scalar differentiation. `"scalar"`
forces one seeded unknown at a time. `"row"` selects vector differentiation
for eligible rows and reports incompatibility instead of falling back
automatically.

Handwritten residuals support arithmetic, powers, indexing, and `np.sin`,
`np.cos`, `np.exp`, `np.tanh`, `np.log` and `np.sqrt`. The UI rewrites `sinh`,
`cosh` and `sech` through exponentials. Preserve dual values during
differentiation: avoid converting field-dependent intermediates with
`float()` or forcing them into float arrays. Unsupported NumPy operations
need equivalent supported expressions or compatible callback implementations.

Field-dependent `log` and `sqrt` arguments must be strictly positive during
differentiation. Variable exponents require positive bases; other powers
must remain real and differentiable over the solver's iterates.

## Writing a Python model

Copy [model_template_BLANK.py](models/model_template_BLANK.py) or adapt a
complete example. The model defines the physics through these callbacks:

| Callback | Role |
|---|---|
| `get_model_config()` | Required: grid, fields, parameters, scans, solver and outputs |
| `build_bulk_functions(config, params)` | Required: field index to bulk residual callback |
| `build_boundary_spec(config, params)` | Required: endpoint residuals and any explicit dependencies |
| `build_dependency_spec(config, params)` | Required: bulk field/site references for the sparse Jacobian |
| `build_initial_guess(config, params)` | Required: initial real field vector |
| `finalize_parameters(params)` | Optional: return derived/constrained parameters |
| `build_extra_rows(config, params)` | Optional: replacement rows with callbacks and dependencies |
| `build_continuation_guess(config, params, previous_field)` | Optional: adapt the previous solution to a new point |
| `compute_observable(field, config, params)` | Optional: return one scalar observable |
| `make_solution_plot(field, config, params)` | Optional: return a Matplotlib `Figure` |
| `make_scan_plot(param_points, observable_vals, config)` | Optional: return a custom command-line scan figure |

### Field storage and dependencies

Residual callbacks receive a flat vector in field-major order: all sites
of field 0, then all sites of field 1, and so on. Field `a` at site `j` is
`field[a*n_grid + j]`. The callback index `i` is the flat equation row; its
spatial site is `i % n_grid`. Bind configuration values with
`functools.partial`, as in the supplied models.

Use `build_grid_from_config(config)` for coordinates and
`build_indexer(grid, config["n_fields"])` for field slices and indices.
Initial guesses contain `n_fields*n_grid` real values. Observable and
solution-plot callbacks receive an array shaped `(n_fields, n_grid)`.

Bulk dependencies are `(field_index, relative_site_offset)` pairs. For
example, `(0, -1)`, `(0, 0)`, `(0, 1)` and `(1, 0)` describe a three-point
stencil for field 0 plus a local coupling to field 1. Define these per field
under `bulk_by_field`, or use a conservative default such as this two-field
rule:

```python
def build_dependency_spec(config, params):
    return {
        "default": {"fields": [0, 1], "stencil_radius": 1},
    }
```

Declare every value read by a residual, even when its derivative vanishes
at the initial guess. Missing references remove entries from the sparse
Jacobian. Boundary `left_deps` / `right_deps` and special-row `deps` use
`(field_index, absolute_site)` pairs instead. Supply explicit lists when
boundary callbacks read other fields or sites beyond the default local
endpoint stencil.

### Calling the runner from Python

From the project root, a single-point calculation can also be run as:

```python
from models import model_template_DOMAIN_WALL as model
from run.run_model import run_single

config = model.get_model_config()
params = model.finalize_parameters(dict(config["parameters"]))
field, result, problem, observable = run_single(model, config, params)

print(result.converged, result.residual_inf_norm)
```

For a configured scan, call `run.run_scan.run_scan(model)`, which returns
`param_points, observable_vals, results`. Supplied models already dispatch
between these runners in `main()`.

## Observables, plots and output files

The runner supports one primary scalar observable. Handwritten models
implement its quadrature in `compute_observable()`. In the UI, enter a
density, weight, prefactor and endpoint rule. Static-energy projects can
copy the density and weight from their functional automatically;
direct-residual projects define the observable separately.

| UI quadrature | Convention |
|---|---|
| Interior sum | Centred derivatives, endpoints omitted, `spacing * sum(weight*density)`; matches supplied templates |
| Trapezoid | Full interval, one-sided endpoint derivatives; endpoint densities must be finite |

The prefactor multiplies the integral. **Preview weighted density** shows the initial
guess's weighted density and integral, before solving.

Profile plots can use fields or expressions such as `sigma1**2+sigma2**2`.
Form-based previews and results use Matplotlib's default colours and varied
line styles. Handwritten models supply `make_solution_plot()`; the UI shows
its complete figure, including subplots, labels and styles, as a PNG
preview. The toolbar zooms and pans that image. Normal saved plots retain
their configured format and DPI. If the UI preview cannot be rendered or
read, the log records why and raw field profiles are shown instead.
Numerical data remain available for CSV export.

### Output locations and controls

Choose the base folder and output checkboxes on **Observables & plots**, or
edit the Python model's `output` dictionary. Files are written beneath
`base_dir/run_name/`; UI runs append a timestamp to the name. Command-line
runs reuse the configured name, so change it or the base folder to retain
separate calculations.

Standard outputs, when enabled, are:

| Location within a run folder | Contents |
|---|---|
| `points/summary.json` | Single-point status, residual, iterations, observable and metadata |
| `points/field.txt` | One site per row, one field per column; coordinates follow from the grid |
| `plots/solution_plot.<format>` | Model-defined profile figure |
| `scan/scan_summary.csv` or `.json` | Scan-point status and numerical information |
| `scan/observables.csv` | Point indices, parameters and converged-point observables |
| `scan/scan_plot.<format>` | Observable scan figure |
| `model.py`, `ui_run_config.json` | Source and effective configuration snapshots for UI runs |

Scan-point filenames include suffixes such as `__point_0001`; optional
`file_tag` values also change filenames. `concise` selects compact
summaries. `plot_format` and `plot_dpi` set saved figure format and
resolution. `show_solution_plots` / `show_scan_plot` control interactive
command-line windows; UI workers disable these and display results in-app.

With `save_only_converged=True`, normal point summaries, field files and
profile figures are restricted to converged results. The UI still displays
failed or near-solution iterates and can export them using **Export profiles…**.
This CSV includes coordinate and field columns, with status
and residual in its comment header. Scan observable curves use converged
points. Temporary display data are removed when starting a new run or
closing the window, so export failed iterates you want to retain.

## Saving and reopening models

| File | Behaviour |
|---|---|
| Builder `.json` | Preserves editable forms, expressions and settings, including unfinished drafts |
| Unmodified exported `.py` | Reopens into forms using embedded metadata and checksums |
| Handwritten or manually edited `.py` | Opens in the advanced Python editor, preserving the source |

Opening Python does not execute it. **Run model** executes the displayed
source. Direct edits switch to advanced mode; **Regenerate from forms**
asks before replacing manual edits. Arbitrary Python logic is not
translated back into forms automatically.

For handwritten models, the settings cog can override numerical controls
for a UI run. These overrides do not rewrite the source; saved handwritten
files retain their own configuration. Generated models still need access
to the KinkSol project when run elsewhere.

## Appearance and defaults

The interface uses British racing green (author's favourite colour), warm off-white surfaces and an
installed serif font, with monospace text for expressions and source. No
additional font installation is required. Scientific previews use serif
labels and STIX mathematics.

Appearance settings are collected in [builder/styles.py](builder/styles.py):
`THEME`, font preferences and sizes, `COLORS`, `LINESTYLES`, `curve_style()`
and `PLOT_STYLE`. `curve_style()` controls form-based profile colours, line
styles and widths. Handwritten figures use their own callback styles.

Blank-project defaults are in [builder/schema.py](builder/schema.py).
**Restore Defaults** uses `solver_defaults()` there. Settings are saved per
project; the dialog does not store application-wide personal defaults.
**Help → About KinkSol Model Builder** shows the builder release and
saved-project format.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `externally-managed-environment` during installation | Install into the virtual environment using its interpreter, as above. |
| Missing PySide6, SymPy or another package | Install the appropriate requirements file with the same interpreter used to launch KinkSol. |
| `python` is not found | Use `python3` to create the environment on Linux/macOS, then the explicit `.venv` interpreter paths above. |
| An expression is rejected | Check declared names, supported functions and syntax; enter a residual without `= 0`. |
| Non-finite values or invalid `log`, `sqrt` or powers | Inspect the guess, parameter range, endpoint evaluation and function domains. |
| A solve fails to converge | Check signs and scales, boundary/vacuum values, initial branch, domain size and solver settings. More iterations alone may not fix an inconsistent model. |
| `NotImplementedError` from a model | Complete blank-template placeholders or remove unused optional callbacks. |

Choose resolution and domain size appropriate to the model. A converged
discrete solution still needs model-specific resolution, box-size and
physical-branch assessment.

## Project layout

| Path | Purpose |
|---|---|
| `launch_ui.py`, `builder/` | Desktop Model Builder, symbolic input, export and UI worker |
| `ui_examples/` | Editable builder projects |
| `models/` | Handwritten Python models and template |
| `core/` | Grids, indexing, row compilation, differentiation and sparse operators |
| `solvers/`, `strategy/` | Solvers, phase selection, rescue and continuation |
| `run/`, `diagnostics/` | Runners, outputs, residual history and timings |
| `requirements.txt`, `requirements-ui.txt` | Numerical and optional UI dependencies |
| `outputs/` | Calculation results, created when a model uses the default output location |

Keep the package structure and `models/` directory intact when moving or
sharing the program.
