"""Guided desktop model builder and non-blocking runner."""
from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import shiboken6
from PySide6.QtCore import QProcess, QTimer, Qt, QUrl, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFontMetrics
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from builder import __version__
from builder.expressions import Expression, profile_values
from builder.generator import ModifiedModelError, generate_model, read_builder_file
from builder.schema import SCHEMA_VERSION, check_equation_mode, default_spec, parameter_values, resolved_spec, validate_spec
from builder.settings import SolverSettingsDialog
from builder.styles import (COLORS, ERROR_BANNER_STYLE, curve_style, stylesheet, ui_font,)
from builder.symbolic import equation_latex
from builder.widgets import DomainSketch, ModelComboBox, PlotCanvas, ResultSummary, RowTable, line


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "ui_examples"


def hint(text):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setObjectName("hint")
    return widget


def make_page(title, description):
    page = QWidget()
    page.setObjectName("editorPage")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(24, 20, 24, 18)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("pageTitle")
    layout.addWidget(heading)
    layout.addWidget(hint(description))
    return page, layout


class ModelBuilderWindow(QMainWindow):
    def __init__(self, initial_path=None):
        super().__init__()
        self.setObjectName("modelBuilderWindow")
        self.setFont(ui_font())
        self.resize(1340, 920)
        self.setMinimumSize(960, 680)
        self.spec = default_spec()
        self.file_path = None
        self.baseline = None
        self.legacy_mode = False
        self.manual_source = False
        self.legacy_override = None
        self.loading = True
        self.process = None
        self.job = None
        self.temp_model = None
        self.run_spec = None
        self.results = []
        self.live_history = []
        self.output_dir = None
        self._pipe_buffer = ""
        self.cancelled = False
        self._build_ui()
        self.load_spec(self.spec)
        self.loading = False
        if initial_path:
            self.open_path(initial_path)
        QApplication.instance().focusChanged.connect(self._remember_expression)

    @Slot(QWidget, QWidget)
    def _remember_expression(self, old, new):
        if self.loading or not isinstance(new, (QLineEdit, QPlainTextEdit)):
            return
        if not shiboken6.isValid(new) or new.isReadOnly():
            return
        if new is self.functional_density or any(
            new is self.equation_table.table.cellWidget(k, 1)
            for k in range(self.equation_table.table.rowCount())
        ):
            self.last_expression_focus = new

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QWidget()
        bar = QHBoxLayout(header)
        bar.setContentsMargins(24, 16, 22, 14)
        identity = QVBoxLayout()
        eyebrow = QLabel("MODEL BUILDER")
        eyebrow.setObjectName("eyebrow")
        brand = QLabel("KinkSol")
        brand.setObjectName("brand")
        identity.addWidget(eyebrow)
        identity.addWidget(brand)
        bar.addLayout(identity)
        bar.addStretch()
        new = QPushButton("New model")
        new.setToolTip("Create a blank model")
        new.clicked.connect(self.new_model)
        self.new_button = new
        bar.addWidget(new)
        self.open_button = QPushButton("Open…")
        self.open_button.clicked.connect(self.choose_open)
        bar.addWidget(self.open_button)
        self.save_button = QPushButton("Save project…")
        self.save_button.clicked.connect(self.save_project)
        bar.addWidget(self.save_button)
        self.export_button = QPushButton("Export Python…")
        self.export_button.clicked.connect(self.export_python)
        bar.addWidget(self.export_button)
        self.settings_button = QToolButton()
        self.settings_button.setText("⚙")
        self.settings_button.setObjectName("settingsCog")
        self.settings_button.setAccessibleName("Solver settings")
        self.settings_button.setToolTip("Solver settings")
        self.settings_button.clicked.connect(self.open_settings)
        bar.addWidget(self.settings_button)
        outer.addWidget(header)
        body = QSplitter()
        self.nav = QListWidget()
        self.nav.setObjectName("navigation")
        navigation_titles = ["01  Geometry & grid", "02  Fields & guesses", "03  Parameters & scans", "04  Equations", "05  Boundaries", "06  Observables & plots", "07  Python model", "08  Results"]
        navigation_font = ui_font()
        navigation_font.setBold(True)
        navigation_metrics = QFontMetrics(navigation_font)
        self.nav.setFixedWidth(max(215, max(navigation_metrics.horizontalAdvance(text) for text in navigation_titles) + 56))
        for text in navigation_titles:
            self.nav.addItem(text)
        self.pages = QStackedWidget()
        body.addWidget(self.nav)
        body.addWidget(self.pages)
        body.setStretchFactor(1, 1)
        self._domain_page()
        self._fields_page()
        self._parameters_page()
        self._equations_page()
        self._boundaries_page()
        self._observable_page()
        self._code_page()
        self._results_page()
        self.nav.currentRowChanged.connect(self._page_changed)
        outer.addWidget(body, 1)
        footer = QWidget()
        bottom = QVBoxLayout(footer)
        bottom.setContentsMargins(20, 8, 20, 12)
        self.banner = QLabel("Ready to build a model")
        self.banner.setObjectName("statusBanner")
        self.banner.setWordWrap(True)
        bottom.addWidget(self.banner)
        actions = QHBoxLayout()
        self.solver_summary = QLabel()
        self.solver_summary.setObjectName("hint")
        actions.addWidget(self.solver_summary)
        actions.addStretch()
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.validate_current)
        actions.addWidget(self.validate_button)
        self.guess_button = QPushButton("Preview guess")
        self.guess_button.clicked.connect(self.preview_guess)
        actions.addWidget(self.guess_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_run)
        actions.addWidget(self.stop_button)
        self.run_button = QPushButton("Run model")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run_model)
        actions.addWidget(self.run_button)
        bottom.addLayout(actions)
        outer.addWidget(footer)
        self.nav.setCurrentRow(0)
        self.setStyleSheet(stylesheet())
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        self.addAction(save_action)
        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_open)
        self.addAction(open_action)
        new_action = QAction("New model", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_model)
        self.addAction(new_action)
        help_menu = self.menuBar().addMenu("&Help")
        self.about_action = help_menu.addAction("About KinkSol Model Builder")
        self.about_action.triggered.connect(self.show_about)

    def show_about(self):
        QMessageBox.about(self, "About KinkSol Model Builder",
                          f"KinkSol Model Builder\nVersion {__version__}\n\n"
                          "Build and run static field models on one-dimensional grids.\n"
                          "Uses the KinkSol V3 numerical backend.\n\n"
                          f"Builder project format: {SCHEMA_VERSION}")

    def _add_scroll_page(self, page):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.pages.addWidget(scroll)

    def _domain_page(self):
        page, layout = make_page("Choose the domain", "KinkSol currently solves real fields on a uniform one-dimensional grid.")
        box = QGroupBox("Model and coordinates")
        form = QFormLayout(box)
        self.model_name = line()
        self.model_name.setObjectName("modelName")
        form.addRow("Model name", self.model_name)
        self.geometry = ModelComboBox()
        self.geometry.addItem("Cartesian · two-sided", "cartesian")
        self.geometry.addItem("Radial · one-sided", "radial")
        self.geometry.currentIndexChanged.connect(self.update_domain)
        form.addRow("Geometry", self.geometry)
        self.grid_mode = ModelComboBox()
        self.grid_mode.addItems(["Sites and spacing", "Sites and total extent"])
        self.grid_mode.currentIndexChanged.connect(self._grid_mode_changed)
        form.addRow("Specify grid using", self.grid_mode)
        self.grid_n = QSpinBox()
        self.grid_n.setRange(3, 1000001)
        self.grid_n.setObjectName("gridSites")
        self.grid_n.valueChanged.connect(self._grid_n_changed)
        form.addRow("Number of sites N", self.grid_n)
        self.spacing = QDoubleSpinBox()
        self.spacing.setDecimals(10)
        self.spacing.setRange(1e-9, 1e8)
        self.spacing.setSingleStep(0.005)
        self.spacing.valueChanged.connect(self._spacing_changed)
        form.addRow("Grid spacing", self.spacing)
        self.extent = QDoubleSpinBox()
        self.extent.setDecimals(8)
        self.extent.setRange(1e-8, 1e12)
        self.extent.valueChanged.connect(self._extent_changed)
        form.addRow("Total Cartesian length / radial outer radius", self.extent)
        layout.addWidget(box)
        self.domain_sketch = DomainSketch()
        layout.addWidget(self.domain_sketch)
        self.domain_info = hint("")
        layout.addWidget(self.domain_info)
        layout.addWidget(hint("Radial coordinates do not automatically add a Laplacian or an integration weight. Set these in the equations or functional. The origin receives its own boundary row."))
        layout.addStretch()
        self._add_scroll_page(page)

    def _fields_page(self):
        page, layout = make_page("Name the fields and seed their profiles", "Choose symbolic names for equations and labels for plots. Initial guesses use parameters and x or r.")
        row = QHBoxLayout()
        row.addWidget(QLabel("Number of real fields"))
        self.field_count = QSpinBox()
        self.field_count.setRange(1, 64)
        self.field_count.valueChanged.connect(self._field_count_changed)
        row.addWidget(self.field_count)
        row.addStretch()
        layout.addLayout(row)
        self.field_table = RowTable(["Symbolic name", "Plot label", "Initial guess expression"], editable=False,
                                    expression_columns=(0, 1, 2), column_widths=(145, 160, 350))
        layout.addWidget(self.field_table)
        layout.addWidget(hint("Examples: v*tanh(x/width), sech(x/width), tanh(r), r**2/(1+r**2), or a constant. Complex fields should be split into real components."))
        button = QPushButton("Update profile preview")
        button.clicked.connect(self.preview_guess)
        layout.addWidget(button, alignment=Qt.AlignLeft)
        self.guess_plot = PlotCanvas()
        self.guess_plot.setMinimumHeight(280)
        layout.addWidget(self.guess_plot, 1)
        self._add_scroll_page(page)

    def _parameters_page(self):
        page, layout = make_page("Define parameters and scans", "Fixed parameters may be numbers or constant expressions. Derived parameters can refer to other parameters and are recalculated at every scan point.")
        self.param_table = RowTable(["Name", "Kind", "Value / expression", "Evaluated value"], choices={1: ["fixed", "derived"]},
                                    expression_columns=(0, 2, 3), column_widths=(145, 110, 310, 160))
        self.param_table.add_button.clicked.connect(self._parameter_cells_readonly)
        layout.addWidget(self.param_table)
        evaluate = QPushButton("Evaluate parameters")
        evaluate.clicked.connect(self.preview_parameters)
        layout.addWidget(evaluate, alignment=Qt.AlignLeft)
        self.param_status = hint("")
        layout.addWidget(self.param_status)
        scans = QGroupBox("Optional parameter scan")
        sl = QVBoxLayout(scans)
        sl.addWidget(hint("No rows gives one solve. Multiple rows form a Cartesian-product scan with continuation from converged points."))
        self.scan_table = RowTable(["Parameter", "Minimum", "Maximum", "Points", "Spacing"], choices={4: ["linear", "logarithmic"]},
                                   expression_columns=(0, 1, 2, 3), column_widths=(150, 135, 135, 95, 145),
                                   header_tooltips=("Fixed parameter to scan", "Minimum parameter value", "Maximum parameter value",
                                                    "Number of scan points", "Linear or logarithmic spacing"))
        sl.addWidget(self.scan_table)
        self.scan_count_label = hint("Scan endpoints are physical values, including for logarithmic spacing.")
        sl.addWidget(self.scan_count_label)
        layout.addWidget(scans)
        layout.addStretch()
        self._add_scroll_page(page)

    def _equations_page(self):
        page, layout = make_page("Write equations or derive them", "Every expression is a residual equal to zero. D(field) and D2(field) refer to first and second spatial derivatives.")
        form = QFormLayout()
        self.equation_mode = ModelComboBox()
        self.equation_mode.addItem("Direct residual equations", "direct")
        self.equation_mode.addItem("Derive from static energy density", "energy")
        self.equation_mode.currentIndexChanged.connect(self._equation_mode_changed)
        form.addRow("Input mode", self.equation_mode)
        layout.addLayout(form)
        self.functional_box = QGroupBox("Static energy density")
        fl = QVBoxLayout(self.functional_box)
        self.functional_density = QPlainTextEdit()
        self.functional_density.setMaximumHeight(100)
        self.functional_density.setPlaceholderText("Energy density in your fields and their first derivatives D(field)")
        fl.addWidget(self.functional_density)
        wf = QHBoxLayout()
        wf.addWidget(QLabel("Integration weight"))
        self.functional_weight = ModelComboBox()
        self.functional_weight.setEditable(True)
        self.functional_weight.setProperty("expression", True)
        self.functional_weight.addItems(["1", "r", "r**2"])
        wf.addWidget(self.functional_weight)
        self.derive_button = QPushButton("Derive equations")
        self.derive_button.clicked.connect(self.derive_current)
        wf.addWidget(self.derive_button)
        fl.addLayout(wf)
        fl.addWidget(hint("Vary ∫ dq weight × density. First-derivative functionals only. Derived residuals retain the weight; no field-dependent coefficient is divided out. The origin still needs a boundary condition."))
        layout.addWidget(self.functional_box)
        self.equation_table = RowTable(["Field", "Residual expression = 0"], editable=False,
                                       expression_columns=(0, 1), column_widths=(110, 500))
        layout.addWidget(self.equation_table)
        symbols = QHBoxLayout()
        symbols.addWidget(QLabel("Insert symbol:"))
        self.symbols = ModelComboBox()
        self.symbols.setProperty("expression", True)
        self.symbols.setMinimumWidth(170)
        symbols.addWidget(self.symbols)
        insert = QPushButton("Insert")
        insert.clicked.connect(self.insert_symbol)
        symbols.addWidget(insert)
        render = QPushButton("Preview equations & dependencies")
        render.clicked.connect(self.preview_equations)
        symbols.addWidget(render)
        symbols.addStretch()
        layout.addLayout(symbols)
        self.equation_preview = PlotCanvas(toolbar=False)
        self.equation_preview.setMinimumHeight(110)
        self.equation_preview.setMaximumHeight(160)
        layout.addWidget(self.equation_preview)
        self.dependencies = QPlainTextEdit()
        self.dependencies.setReadOnly(True)
        self.dependencies.setFixedHeight(85)
        self.dependencies.setPlaceholderText("Automatically generated stencil references appear here.")
        layout.addWidget(self.dependencies)
        layout.addWidget(hint("Functions: sin, cos, exp, tanh, log, sqrt, sinh, cosh, sech. Constants: pi, e. Use ** (or ^) for powers. Logarithms and square roots must have valid real arguments."))
        layout.addStretch()
        self._add_scroll_page(page)

    def _boundaries_page(self):
        page, layout = make_page("Set boundary and special rows", "Choose a condition at each end for every field. Boundary rows replace the bulk equation at those sites.")
        layout.addWidget(hint("Dirichlet: field − value = 0. Neumann: D(field) − value = 0. Robin: α·field + β·D(field) − value = 0. Custom: enter the complete residual. D points toward increasing x or r at both ends."))
        types = ["dirichlet", "neumann", "robin", "custom"]
        self.bc_table = RowTable(["Field", "Left type", "Left value / residual", "L α", "L β", "Right type", "Right value / residual", "R α", "R β"],
                                 choices={1: types, 5: types}, editable=False, expression_columns=(0, 2, 3, 4, 6, 7, 8),
                                 column_widths=(85, 115, 175, 55, 55, 115, 175, 55, 55),
                                 stretch_columns=(2, 6),
                                 header_tooltips=("Symbolic field name", "Condition at the left endpoint", "Left value or complete custom residual",
                                                  "Left Robin coefficient α", "Left Robin coefficient β", "Condition at the right endpoint",
                                                  "Right value or complete custom residual", "Right Robin coefficient α", "Right Robin coefficient β"))
        layout.addWidget(self.bc_table)
        self.boundary_note = hint("")
        layout.addWidget(self.boundary_note)
        box = QGroupBox("Optional row overrides")
        bl = QVBoxLayout(box)
        bl.addWidget(hint("Pin a field or impose a centre constraint by replacing one selected equation. Site is 'centre' or an integer grid index. References in the residual are evaluated at that site."))
        self.extra_table = RowTable(["Field", "Site", "Replacement residual = 0"], expression_columns=(0, 1, 2), column_widths=(140, 150, 400),
                                    header_tooltips=("Field whose equation is replaced", "Use centre or an integer grid index", "Complete replacement residual, equal to zero"))
        bl.addWidget(self.extra_table)
        layout.addWidget(box)
        layout.addWidget(hint("First derivatives use centred differences in the bulk and first-order one-sided differences at endpoints. Nonzero derivative conditions include the spacing explicitly."))
        layout.addStretch()
        self._add_scroll_page(page)

    def _observable_page(self):
        page, layout = make_page("Choose observables, plots and output", "Define one primary scalar observable for the existing runner, plus any number of profile or derived-quantity curves.")
        obs = QGroupBox("Integrated observable")
        ol = QVBoxLayout(obs)
        toggles = QHBoxLayout()
        self.observable_enabled = QCheckBox("Compute an observable")
        self.auto_energy = QCheckBox("Use the static energy functional")
        self.auto_energy.toggled.connect(self._observable_enabled_changed)
        self.observable_enabled.toggled.connect(self._observable_enabled_changed)
        toggles.addWidget(self.observable_enabled)
        toggles.addWidget(self.auto_energy)
        toggles.addStretch()
        ol.addLayout(toggles)
        form = QFormLayout()
        self.observable_name = line()
        self.observable_density = QPlainTextEdit()
        self.observable_density.setMaximumHeight(85)
        self.observable_weight, self.observable_prefactor = line(expression=True), line(expression=True)
        self.quadrature = ModelComboBox()
        self.quadrature.addItem("Interior sum · exclude both endpoints", "interior_sum")
        self.quadrature.addItem("Trapezoid · include both endpoints", "trapezoid")
        form.addRow("Observable name", self.observable_name)
        form.addRow("Density", self.observable_density)
        row = QWidget()
        wl = QHBoxLayout(row)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(QLabel("Weight"))
        wl.addWidget(self.observable_weight)
        wl.addWidget(QLabel("Prefactor"))
        wl.addWidget(self.observable_prefactor)
        form.addRow("Integration", row)
        form.addRow("Endpoint treatment", self.quadrature)
        ol.addLayout(form)
        density_preview = QPushButton("Preview weighted density on the initial guess")
        density_preview.clicked.connect(self.preview_density)
        ol.addWidget(density_preview, alignment=Qt.AlignLeft)
        self.density_plot = PlotCanvas(toolbar=False)
        self.density_plot.setMinimumHeight(170)
        ol.addWidget(self.density_plot)
        layout.addWidget(obs)
        self.plot_table = RowTable(["Plot expression", "Legend label"], expression_columns=(0, 1), column_widths=(430, 250))
        layout.addWidget(self.plot_table)
        output = QGroupBox("Output files")
        out = QFormLayout(output)
        self.output_base = line()
        out.addRow("Base folder (relative to KinkSol or absolute)", self.output_base)
        self.output_checks = {}
        choices = [("save_point_summary", "Point summaries"), ("save_scan_summary", "Scan summary"),
                   ("save_observables", "Observable table"), ("save_final_field", "Final field data"),
                   ("save_solution_plots", "Profile figures"), ("save_scan_plot", "Scan figure"),
                   ("save_only_converged", "Save point outputs only when converged"),
                   ("show_solution_plots", "Show profile plots in command-line runs"),
                   ("show_scan_plot", "Show scan plot in command-line runs")]
        checks = QWidget()
        grid = QGridLayout(checks)
        grid.setContentsMargins(0, 0, 0, 0)
        for k, (key, label) in enumerate(choices):
            check = QCheckBox(label)
            self.output_checks[key] = check
            grid.addWidget(check, k // 2, k % 2)
        out.addRow(checks)
        out.addRow(hint("UI runs create a new timestamped run directory. Failed iterates can be inspected in Results and remain explicitly marked unconverged."))
        layout.addWidget(output)
        self._add_scroll_page(page)

    def _code_page(self):
        page, layout = make_page("Inspect the Python model", "Exported models use the existing KinkSol core. Builder metadata allows them to reopen in the form editor.")
        bar = QHBoxLayout()
        self.edit_source = QCheckBox("Edit Python directly")
        self.edit_source.toggled.connect(self._edit_source_changed)
        bar.addWidget(self.edit_source)
        self.regenerate_button = QPushButton("Regenerate from forms")
        self.regenerate_button.clicked.connect(self.regenerate_code)
        bar.addWidget(self.regenerate_button)
        bar.addStretch()
        layout.addLayout(bar)
        self.code_note = hint("Generated code is read-only until direct editing is enabled. Any manual edits are preserved as an advanced Python model.")
        layout.addWidget(self.code_note)
        self.code = QPlainTextEdit()
        self.code.setObjectName("pythonEditor")
        self.code.setReadOnly(True)
        self.code.setPlaceholderText("Complete the model definition to generate Python, or open an existing Python model.")
        self.code.textChanged.connect(self._code_changed)
        layout.addWidget(self.code, 1)
        self.pages.addWidget(page)

    def _results_page(self):
        page, layout = make_page("Run and inspect the result", "The solver runs in a separate process. Numerical convergence is reported independently of whether the process exits successfully.")
        row = QHBoxLayout()
        self.point_selector = ModelComboBox()
        self.point_selector.setMinimumWidth(260)
        self.point_selector.currentIndexChanged.connect(self.draw_result)
        row.addWidget(QLabel("Completed point"))
        row.addWidget(self.point_selector)
        row.addStretch()
        self.export_data_button = QPushButton("Export profiles…")
        self.export_data_button.clicked.connect(self.export_profiles)
        self.export_data_button.setEnabled(False)
        row.addWidget(self.export_data_button)
        self.folder_button = QPushButton("Open output folder")
        self.folder_button.clicked.connect(self.open_output_folder)
        self.folder_button.setEnabled(False)
        row.addWidget(self.folder_button)
        layout.addLayout(row)
        self.result_status = ResultSummary()
        layout.addWidget(self.result_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        layout.addWidget(self.progress)
        self.result_tabs = QTabWidget()
        self.solution_plot, self.residual_plot, self.scan_plot = PlotCanvas(), PlotCanvas(), PlotCanvas()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        for name, widget in [("Profiles", self.solution_plot), ("Residual history", self.residual_plot), ("Observable scan", self.scan_plot), ("Run log", self.log)]:
            self.result_tabs.addTab(widget, name)
        layout.addWidget(self.result_tabs, 1)
        self.pages.addWidget(page)

    def show_message(self, text, error=False):
        self.banner.setText(text)
        self.banner.setStyleSheet(ERROR_BANNER_STYLE if error else "")

    def _page_changed(self, index):
        self.pages.setCurrentIndex(index)
        if self.loading:
            return
        if not self.legacy_mode:
            self.sync_field_names()
        if index == 6 and not self.manual_source:
            self._refresh_code()

    def _grid_mode_changed(self):
        self.spacing.setEnabled(self.grid_mode.currentIndex() == 0)
        self.extent.setEnabled(self.grid_mode.currentIndex() == 1)

    def _grid_n_changed(self):
        if self.grid_mode.currentIndex() == 1:
            self._extent_changed()
        else:
            self._spacing_changed()

    def _spacing_changed(self):
        self.extent.blockSignals(True)
        self.extent.setValue((self.grid_n.value() - 1) * self.spacing.value())
        self.extent.blockSignals(False)
        self.update_domain()

    def _extent_changed(self):
        self.spacing.blockSignals(True)
        self.spacing.setValue(self.extent.value() / (self.grid_n.value() - 1))
        self.spacing.blockSignals(False)
        self.update_domain()

    def update_domain(self):
        if not hasattr(self, "domain_sketch"):
            return
        geometry = self.geometry.currentData()
        n, dx = self.grid_n.value(), self.spacing.value()
        self.domain_sketch.set_grid(geometry, n, dx)
        length = (n - 1) * dx
        info = f"Domain: [0, {length:g}]" if geometry == "radial" else f"Domain: [−{length/2:g}, +{length/2:g}]"
        self.domain_info.setText(info + f"   ·   {self.field_count.value() if hasattr(self, 'field_count') else 1} × {n:,} unknowns")
        if hasattr(self, "boundary_note"):
            self.boundary_note.setText("Left is r = 0; right is the outer radius. Set regularity explicitly at the origin." if geometry == "radial" else "Left and right are the negative and positive ends of the Cartesian domain.")

    def _field_count_changed(self, count):
        if self.loading or not hasattr(self, "equation_table"):
            return
        tables = [self.field_table, self.equation_table, self.bc_table]
        for table in tables:
            while table.table.rowCount() > count:
                table.table.removeRow(table.table.rowCount() - 1)
        while self.field_table.table.rowCount() < count:
            a = self.field_table.table.rowCount()
            name = f"u{a + 1}"
            self.field_table.add_row([name, name, ""])
            self.equation_table.add_row([name, ""])
            self.bc_table.add_row([name, "custom", "", "1", "1", "custom", "", "1", "1"])
        for table in tables:
            table._fit_height()
        self.sync_field_names()
        self._equation_mode_changed()
        self.update_domain()

    def sync_field_names(self):
        names = [row[0] for row in self.field_table.rows()]
        for table in (self.equation_table, self.bc_table):
            for k, name in enumerate(names):
                if k < table.table.rowCount():
                    widget = table.table.cellWidget(k, 0)
                    widget.setText(name)
                    widget.setReadOnly(True)
        self.symbols.clear()
        q = "r" if self.geometry.currentData() == "radial" else "x"
        self.symbols.addItems([q, "pi", *names, *[f"D({n})" for n in names], *[f"D2({n})" for n in names],
                               *[row[0] for row in self.param_table.rows() if row[0]], "sin()", "cos()", "exp()", "tanh()", "sqrt()"])
        self._bc_enabled()

    def _bc_enabled(self):
        for row in range(self.bc_table.table.rowCount()):
            for start in (1, 5):
                combo = self.bc_table.table.cellWidget(row, start)
                if not combo.property("builderConnected"):
                    combo.currentTextChanged.connect(self._bc_enabled)
                    combo.setProperty("builderConnected", True)
                is_robin = combo.currentText() == "robin"
                for col in (start + 2, start + 3):
                    coefficient = self.bc_table.table.cellWidget(row, col)
                    coefficient.setEnabled(is_robin)
                    coefficient.setToolTip("Robin coefficient; used only when this endpoint has a Robin condition.")

    def _parameter_cells_readonly(self):
        for k in range(self.param_table.table.rowCount()):
            self.param_table.table.cellWidget(k, 3).setReadOnly(True)

    def _equation_mode_changed(self):
        if not hasattr(self, "functional_box"):
            return
        mode = self.equation_mode.currentData()
        self.functional_box.setVisible(mode == "energy")
        for k in range(self.equation_table.table.rowCount()):
            self.equation_table.table.cellWidget(k, 1).setReadOnly(mode == "energy")
        if hasattr(self, "auto_energy"):
            self.auto_energy.setEnabled(mode == "energy")
            if mode != "energy":
                self.auto_energy.setChecked(False)
            self._observable_enabled_changed()

    def _observable_enabled_changed(self):
        if not hasattr(self, "observable_density"):
            return
        active = self.observable_enabled.isChecked()
        auto = self.auto_energy.isChecked()
        for widget in (self.observable_name, self.observable_density, self.observable_weight, self.observable_prefactor, self.quadrature):
            widget.setEnabled(active)
        self.observable_density.setReadOnly(auto)
        self.observable_weight.setReadOnly(auto)
        if auto and hasattr(self, "functional_density"):
            self.observable_density.setPlainText(self.functional_density.toPlainText())
            self.observable_weight.setText(self.functional_weight.currentText())

    def collect_spec(self):
        spec = copy.deepcopy(self.spec)
        spec["name"] = self.model_name.text().strip()
        spec["geometry"] = self.geometry.currentData()
        spec["grid"] = {"n": self.grid_n.value(), "spacing": self.spacing.value()}
        eqs, bcs = self.equation_table.rows(), self.bc_table.rows()
        spec["fields"] = []
        for k, (name, label, guess) in enumerate(self.field_table.rows()):
            f = {"name": name, "label": label or name, "guess": guess, "equation": eqs[k][1]}
            for side, start in (("left", 1), ("right", 5)):
                f[side] = {"type": bcs[k][start], "value": bcs[k][start + 1], "alpha": bcs[k][start + 2], "beta": bcs[k][start + 3]}
            spec["fields"].append(f)
        spec["parameters"] = [{"name": name, "kind": kind, "expression": expression} for name, kind, expression, _ in self.param_table.rows()]
        spec["scans"] = [{"name": name, "min": float(lo), "max": float(hi), "num": int(n), "is_log": kind == "logarithmic"}
                         for name, lo, hi, n, kind in self.scan_table.rows()]
        spec["equation_mode"] = self.equation_mode.currentData()
        spec["functional"] = {"density": self.functional_density.toPlainText().strip(), "weight": self.functional_weight.currentText().strip()}
        spec["observable"] = {"enabled": self.observable_enabled.isChecked(), "name": self.observable_name.text().strip(),
                              "from_functional": self.auto_energy.isChecked(), "density": self.observable_density.toPlainText().strip(),
                              "weight": self.observable_weight.text().strip(), "prefactor": self.observable_prefactor.text().strip(), "quadrature": self.quadrature.currentData()}
        spec["plots"] = [{"expression": expr, "label": label or expr} for expr, label in self.plot_table.rows()]
        spec["extra_rows"] = [{"field": field, "site": site, "expression": expr} for field, site, expr in self.extra_table.rows()]
        spec["output"].update({key: widget.isChecked() for key, widget in self.output_checks.items()})
        spec["output"]["base_dir"] = self.output_base.text().strip() or "outputs"
        return spec

    def load_spec(self, spec):
        check_equation_mode(spec)
        self.loading = True
        self.last_expression_focus = None
        self.spec = copy.deepcopy(spec)
        self.legacy_mode = self.manual_source = False
        self.legacy_override = None
        self.model_name.setText(spec["name"])
        self.geometry.setCurrentIndex(self.geometry.findData(spec["geometry"]))
        self.grid_n.setValue(spec["grid"]["n"])
        self.spacing.setValue(spec["grid"]["spacing"])
        self._spacing_changed()
        self.grid_mode.setCurrentIndex(0)
        self.field_count.setValue(len(spec["fields"]))
        self.field_table.set_rows([[f["name"], f["label"], f["guess"]] for f in spec["fields"]])
        self.param_table.set_rows([[p["name"], p["kind"], p["expression"], ""] for p in spec["parameters"]])
        self._parameter_cells_readonly()
        self.scan_table.set_rows([[s["name"], s["min"], s["max"], s["num"], "logarithmic" if s["is_log"] else "linear"] for s in spec["scans"]])
        self.equation_table.set_rows([[f["name"], f["equation"]] for f in spec["fields"]])
        self.equation_mode.setCurrentIndex(self.equation_mode.findData(spec["equation_mode"]))
        self.functional_density.setPlainText(spec["functional"]["density"])
        self.functional_weight.setCurrentText(spec["functional"]["weight"])
        rows = []
        for f in spec["fields"]:
            rows.append([f["name"], *[value for side in ("left", "right") for value in
                         (f[side]["type"], f[side]["value"], f[side].get("alpha", "1"), f[side].get("beta", "1"))]])
        self.bc_table.set_rows(rows)
        self.extra_table.set_rows([[r["field"], r["site"], r["expression"]] for r in spec["extra_rows"]])
        obs = spec["observable"]
        self.observable_enabled.setChecked(obs["enabled"])
        self.observable_name.setText(obs["name"])
        self.observable_density.setPlainText(obs["density"])
        self.observable_weight.setText(obs["weight"])
        self.observable_prefactor.setText(obs["prefactor"])
        self.quadrature.setCurrentIndex(self.quadrature.findData(obs["quadrature"]))
        self.auto_energy.setChecked(obs["from_functional"])
        self.plot_table.set_rows([[p["expression"], p["label"]] for p in spec["plots"]])
        self.output_base.setText(spec["output"]["base_dir"])
        for key, widget in self.output_checks.items():
            widget.setChecked(spec["output"].get(key, False))
        self.edit_source.setChecked(False)
        for k in range(6):
            self.nav.item(k).setFlags(self.nav.item(k).flags() | Qt.ItemIsEnabled)
        self.regenerate_button.setEnabled(True)
        self.guess_button.setEnabled(True)
        self.sync_field_names()
        self._equation_mode_changed()
        self.update_domain()
        self.update_solver_summary()
        self.loading = False
        if (spec["equation_mode"] == "direct" and all(f["equation"].strip() for f in spec["fields"])) or (spec["equation_mode"] == "energy" and spec["functional"]["density"].strip()):
            self._refresh_code()
        else:
            self.code.blockSignals(True)
            self.code.clear()
            self.code.blockSignals(False)
        self.baseline = self._fingerprint()
        self.setWindowTitle(f"KinkSol Model Builder — {spec['name']}")
        if self.code.toPlainText():
            self.show_message("Ready. Configure a model, preview the seed, then run.")
        else:
            self.show_message("New model draft. Set your fields, equations, initial profiles and boundary conditions; you can save the draft at any time.")
        self.nav.setCurrentRow(0)

    def _fingerprint(self):
        if self.manual_source:
            return self.code.toPlainText()
        try:
            return json.dumps(self.collect_spec(), sort_keys=True)
        except (ValueError, KeyError, IndexError):
            # Invalid numeric draft values must still count as unsaved changes.
            texts = [w.text() for w in self.findChildren(QLineEdit)]
            return repr(texts)

    def confirm_discard(self):
        if self.baseline is None or self._fingerprint() == self.baseline:
            return True
        reply = QMessageBox.question(self, "Unsaved model", "Save changes before leaving this model?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return False
        return self.save_project() if reply == QMessageBox.Save else True

    def new_model(self, checked=False):
        if self.process is not None or not self.confirm_discard():
            return
        self.file_path = None
        self.load_spec(default_spec())

    def choose_open(self):
        if self.process is not None:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Open a model or builder project", str(EXAMPLES_DIR), "Models (*.py *.json);;Builder projects (*.json);;Python models (*.py)")
        if filename and self.confirm_discard():
            self.open_path(filename)

    def open_path(self, filename):
        try:
            spec = read_builder_file(filename)
            self.load_spec(spec)
        except ModifiedModelError as exc:
            self.loading = True
            self.legacy_mode = self.manual_source = True
            self.legacy_override = None
            self.code.setPlainText(Path(filename).read_text(encoding="utf-8"))
            self.edit_source.setChecked(True)
            self.code.setReadOnly(False)
            for k in range(6):
                self.nav.item(k).setFlags(self.nav.item(k).flags() & ~Qt.ItemIsEnabled)
            self.regenerate_button.setEnabled(False)
            self.guess_button.setEnabled(False)
            self.nav.setCurrentRow(6)
            self.update_solver_summary()
            self.show_message(str(exc))
            self.loading = False
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.show_message(f"Could not open the model: {exc}", True)
            return False
        source_path = Path(filename)
        is_example = source_path.suffix.lower() == ".json" and EXAMPLES_DIR.resolve() in source_path.resolve().parents
        self.file_path = None if is_example else source_path
        self.baseline = self._fingerprint()
        if is_example:
            self.setWindowTitle(f"KinkSol Model Builder — {self.model_name.text()} (example)")
            self.show_message("Example loaded. Save project creates your own editable copy.")
        else:
            self.setWindowTitle(f"KinkSol Model Builder — {self.file_path.name}")
        return True

    def save_project(self, checked=False, path=None):
        try:
            manual = self.manual_source
            if path is None:
                filters = "Python model (*.py)" if manual else "Builder project (*.json);;Python model (*.py)"
                default = str(self.file_path) if self.file_path else str(PROJECT_ROOT / "models" / (self.model_name.text() + (".py" if manual else ".json")))
                path, selected = QFileDialog.getSaveFileName(self, "Save model", default, filters)
                if not path:
                    return False
                if Path(path).suffix.lower() not in (".py", ".json"):
                    path += ".py" if manual or "Python" in selected else ".json"
            path = Path(path)
            if manual:
                if path.suffix != ".py":
                    raise ValueError("Advanced Python models must be saved as .py files.")
                content = self.code.toPlainText()
            else:
                spec = self.collect_spec()
                content = generate_model(spec) if path.suffix.lower() == ".py" else json.dumps(spec, indent=2)
            path.write_text(content, encoding="utf-8")
            self.file_path = path
            self.baseline = self._fingerprint()
            self.show_message(f"Saved {path.name}")
            return True
        except (ValueError, OSError, KeyError) as exc:
            self.show_message(f"Could not save: {exc}", True)
            return False

    def export_python(self, checked=False, path=None):
        try:
            source = self.code.toPlainText() if self.manual_source else generate_model(self.collect_spec())
            if path is None:
                path, _ = QFileDialog.getSaveFileName(self, "Export runnable Python model", str(PROJECT_ROOT / "models" / (self.model_name.text() + ".py")), "Python model (*.py)")
                if not path:
                    return False
                if not path.endswith(".py"):
                    path += ".py"
            Path(path).write_text(source, encoding="utf-8")
            self.show_message(f"Exported {Path(path).name}. Run it within the KinkSol project.")
            return True
        except (ValueError, OSError, KeyError) as exc:
            self.show_message(f"Could not export: {exc}", True)
            return False

    def update_solver_summary(self):
        if self.manual_source and self.legacy_override is None:
            self.solver_summary.setText("Solver settings from Python file · ⚙ to override for this run")
        else:
            opts = self.legacy_override if self.manual_source else self.spec["solver"]
            method = "Trust-region Newton" if opts["use_trust_region"] else "Sparse Newton"
            self.solver_summary.setText(f"{method}  ·  tolerance {opts['final_tol']:g}  ·  max {opts['final_max_iter']} iterations")

    def open_settings(self):
        options = self.legacy_override or self.spec["solver"]
        dialog = SolverSettingsDialog(options, self.spec["jacobian_autodiff"], self)
        if dialog.exec() == QDialog.Accepted:
            if self.manual_source:
                self.legacy_override = dialog.options
            else:
                self.spec["solver"] = dialog.options
            self.spec["jacobian_autodiff"] = dialog.jacobian
            self.update_solver_summary()
            self.show_message("Solver settings applied." if not self.manual_source else "Solver override applied to UI runs. The source file keeps its own saved settings.")

    def _refresh_code(self):
        try:
            source = generate_model(self.collect_spec())
            self.code.blockSignals(True)
            self.code.setPlainText(source)
            self.code.blockSignals(False)
            return source
        except (ValueError, KeyError, IndexError) as exc:
            self.code.blockSignals(True)
            self.code.clear()
            self.code.blockSignals(False)
            self.show_message(f"Model needs attention: {exc}", True)
            return None

    def _edit_source_changed(self, enabled):
        self.code.setReadOnly(not enabled)
        if not self.loading and enabled:
            self.show_message("Direct editing enabled. Manual Python changes will be preserved when saving or running.")

    def _code_changed(self):
        if not self.loading and self.edit_source.isChecked():
            self.manual_source = True
            self.update_solver_summary()
            self.code_note.setText("This is now an advanced Python model. Save and Run use the exact source shown; regenerate explicitly to return to the forms.")

    def regenerate_code(self):
        if self.manual_source and QMessageBox.question(self, "Regenerate model", "Replace manually edited Python with a new model from the forms?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.manual_source = False
        self.legacy_override = None
        self.edit_source.setChecked(False)
        self.update_solver_summary()
        self._refresh_code()
        self.code_note.setText("Generated code is read-only until direct editing is enabled. Any manual edits are preserved as an advanced Python model.")

    def validate_current(self):
        try:
            if self.manual_source:
                compile(self.code.toPlainText(), "model.py", "exec")
                self.show_message("Python syntax is valid. Numerical compilation occurs in the worker when you run.")
            else:
                spec = validate_spec(self.collect_spec())
                self._guess_data(spec)
                count = np.prod([s["num"] for s in spec["scans"]], dtype=np.int64) if spec["scans"] else 1
                self.show_message(f"Model valid: {len(spec['fields'])} real fields, {spec['grid']['n']:,} sites, {count:,} solve point(s). Dependencies are automatic.")
                self._refresh_code()
            return True
        except (ValueError, SyntaxError, KeyError, IndexError, ArithmeticError) as exc:
            self.show_message(f"Validation: {exc}", True)
            return False

    def _guess_data(self, spec):
        params = parameter_values(spec)
        n, dx = spec["grid"]["n"], spec["grid"]["spacing"]
        q = np.arange(n) * dx
        if spec["geometry"] == "cartesian":
            q -= (n - 1) * dx / 2
        coordinate = "r" if spec["geometry"] == "radial" else "x"
        fields = []
        for f in spec["fields"]:
            value = Expression(f["guess"], parameters=params, coordinate=coordinate).evaluate(parameters=params, coordinate=q)
            values = np.broadcast_to(np.asarray(value, dtype=float), q.shape).copy()
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Initial guess for {f['name']} is not finite.")
            fields.append(values)
        return q, np.asarray(fields), params

    def preview_guess(self):
        try:
            spec = self.collect_spec()
            q, fields, _ = self._guess_data(spec)
            self.guess_plot.clear()
            for k, (f, values) in enumerate(zip(spec["fields"], fields)):
                self.guess_plot.axes.plot(q, values, label=f["label"], **curve_style(k))
            self.guess_plot.finish("r" if spec["geometry"] == "radial" else "x", "initial field")
            self.nav.setCurrentRow(1)
            self.show_message("Initial profiles previewed. These select the starting branch for the solve.")
            return True
        except (ValueError, KeyError, ArithmeticError) as exc:
            self.show_message(f"Initial guess: {exc}", True)
            return False

    def preview_parameters(self):
        try:
            spec = self.collect_spec()
            values = parameter_values(spec)
            for k, row in enumerate(self.param_table.rows()):
                self.param_table.table.cellWidget(k, 3).setText(f"{values[row[0]]:.10g}")
            count = int(np.prod([s["num"] for s in spec["scans"]])) if spec["scans"] else 1
            self.param_status.setText(f"All parameter expressions evaluated. This configuration has {count:,} solve point(s).")
            self.show_message("Parameter expressions evaluated.")
        except (ValueError, KeyError, ArithmeticError) as exc:
            self.show_message(f"Parameters: {exc}", True)

    def derive_current(self):
        try:
            spec = resolved_spec(self.collect_spec())
            for k, f in enumerate(spec["fields"]):
                self.equation_table.table.cellWidget(k, 1).setText(f["equation"])
            self._observable_enabled_changed()
            self.preview_equations(spec=spec)
            self.show_message("Continuum equations derived. Inspect them before running; weighted residuals retain their scale.")
            return True
        except (ValueError, KeyError, ArithmeticError) as exc:
            self.show_message(f"Derivation: {exc}", True)
            return False

    def preview_equations(self, checked=False, spec=None):
        try:
            spec = spec or resolved_spec(self.collect_spec())
            latex = equation_latex(spec)
            fig = self.equation_preview.figure
            fig.clear()
            fig.set_layout_engine(None)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95)
            ax = fig.add_subplot(111)
            self.equation_preview.axes = ax
            ax.axis("off")
            for k, (f, expression) in enumerate(zip(spec["fields"], latex)):
                self.equation_preview.text(0.01, 1 - (k + 0.5) / len(latex), f"${expression}$", fontsize=11, color=COLORS[k % len(COLORS)], va="center", transform=ax.transAxes)
            try:
                self.equation_preview.canvas.draw()
            except ValueError:
                ax.clear()
                ax.axis("off")
                self.equation_preview.text(0, 1, "\n".join(f["equation"] + " = 0" for f in spec["fields"]), fontsize=10, va="top", wrap=True)
                self.equation_preview.canvas.draw()
            names = [f["name"] for f in spec["fields"]]
            params = [p["name"] for p in spec["parameters"]]
            q = "r" if spec["geometry"] == "radial" else "x"
            lines = []
            for f in spec["fields"]:
                deps = Expression(f["equation"], names, params, q).dependencies()
                lines.append(f"{f['name']}: " + ", ".join(f"{names[a]}[i{off:+d}]" for a, off in deps))
            self.dependencies.setPlainText("\n".join(lines))
            self.dependencies.setMaximumHeight(min(120, max(50, len(lines) * 23 + 14)))
            self.show_message("Equations and structural stencil dependencies previewed.")
            return True
        except (ValueError, KeyError, ArithmeticError) as exc:
            self.show_message(f"Equation preview: {exc}", True)
            return False

    def insert_symbol(self):
        # The palette is a convenience; expressions remain fully keyboard-editable.
        widget = QApplication.focusWidget()
        target = getattr(self, "last_expression_focus", None)
        if isinstance(widget, (QLineEdit, QPlainTextEdit)):
            target = widget
        if target is None or not shiboken6.isValid(target):
            target = self.functional_density if self.equation_mode.currentData() != "direct" else self.equation_table.table.cellWidget(0, 1)
        text = self.symbols.currentText()
        if isinstance(target, QLineEdit):
            target.insert(text)
        else:
            target.insertPlainText(text)
        target.setFocus()

    def preview_density(self):
        try:
            spec = resolved_spec(self.collect_spec())
            q, fields, params = self._guess_data(spec)
            obs = spec["observable"]
            interior = obs["quadrature"] == "interior_sum"
            selected_q = q[1:-1] if interior else q
            values = profile_values(obs["density"], spec, params, fields, q, interior)
            coordinate = "r" if spec["geometry"] == "radial" else "x"
            weight = Expression(obs["weight"], parameters=params, coordinate=coordinate).evaluate(parameters=params, coordinate=selected_q)
            prefactor = Expression(obs["prefactor"], parameters=params).evaluate(parameters=params)
            integrand = values * weight * prefactor
            integrate = getattr(np, "trapezoid", None) or np.trapz
            value = spec["grid"]["spacing"] * np.sum(integrand) if interior else integrate(integrand, selected_q)
            self.density_plot.clear()
            self.density_plot.axes.plot(selected_q, integrand, color=COLORS[0])
            self.density_plot.finish(coordinate, "weighted density", legend=False)
            self.show_message(f"Initial-guess {obs['name']} = {value:.10g}. This is a seed observable, not a solved result.")
            return True
        except (ValueError, KeyError, ArithmeticError) as exc:
            self.show_message(f"Observable preview: {exc}", True)
            return False

    def run_model(self):
        if self.process is not None:
            return False
        try:
            if self.manual_source:
                source = self.code.toPlainText()
                compile(source, "model.py", "exec")
                self.run_spec = None
            else:
                raw_spec = self.collect_spec()
                self.run_spec = validate_spec(raw_spec)
                self._guess_data(self.run_spec)
                source = generate_model(raw_spec)
            self._cleanup_job()
            self.job = tempfile.TemporaryDirectory(prefix="kinksol_ui_")
            job_path = Path(self.job.name)
            # The supplied handwritten templates use parents[1] as PROJECT_ROOT.
            model_dir = PROJECT_ROOT / "models"
            with tempfile.NamedTemporaryFile(prefix="_ui_run_", suffix=".py", dir=model_dir, delete=False) as stream:
                self.temp_model = Path(stream.name)
                stream.write(source.encode("utf-8"))
            request = {"model_path": str(self.temp_model), "job_dir": str(job_path),
                       "solver_override": self.legacy_override if self.manual_source else None,
                       "jacobian_autodiff": self.spec["jacobian_autodiff"]}
            request["use_model_plot"] = bool(self.manual_source)
            request_path = job_path / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            self.results, self.live_history, self._pipe_buffer = [], [], ""
            self.cancelled = False
            self.output_dir = None
            self.point_selector.clear()
            self.log.clear()
            self.solution_plot.clear("Solving… completed profiles will appear here")
            self.residual_plot.clear("Waiting for the first reported residual")
            self.scan_plot.clear("Scan observables appear after points complete")
            self.export_data_button.setEnabled(False)
            self.folder_button.setEnabled(False)
            self.progress.setRange(0, 0)
            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.setWorkingDirectory(str(PROJECT_ROOT))
            self.process.readyReadStandardOutput.connect(self._read_output)
            self.process.finished.connect(self._process_finished)
            self.process.errorOccurred.connect(self._process_error)
            self._set_running(True)
            self.nav.setCurrentRow(7)
            self.result_status.set_status("Starting solver…", "running")
            self.show_message("Solver running. Use Stop to cancel; the editor remains responsive.")
            self.process.start(sys.executable, ["-u", "-m", "builder.worker", str(request_path)])
            return True
        except (ValueError, OSError, KeyError, SyntaxError, ArithmeticError) as exc:
            self.show_message(f"Cannot start: {exc}", True)
            self._cleanup_model()
            return False

    def _set_running(self, active):
        for widget in (self.run_button, self.open_button, self.save_button, self.export_button, self.settings_button, self.new_button):
            widget.setEnabled(not active)
        self.stop_button.setEnabled(active)

    def _read_output(self):
        if self.process is None:
            return
        self._pipe_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._pipe_buffer:
            text, self._pipe_buffer = self._pipe_buffer.split("\n", 1)
            if text.startswith("[KINK_UI] "):
                try:
                    self._handle_event(json.loads(text[len("[KINK_UI] "):]))
                except (ValueError, KeyError, OSError) as exc:
                    self.log.appendPlainText(f"[UI] Could not display event: {exc}")
            else:
                self.log.appendPlainText(text)
                match = re.search(r"\|\|F\|\|_inf[=:]([0-9.eE+\-]+)", text)
                if match:
                    residual = float(match.group(1))
                    if np.isfinite(residual):
                        self.live_history.append(residual)
                        self._draw_history(self.live_history, "reported residual")

    def _handle_event(self, event):
        kind = event["event"]
        if kind == "started":
            self.output_dir = event["output_dir"]
            self.run_field_names = event["field_names"]
            self.folder_button.setEnabled(True)
            self.progress.setRange(0, event["points"])
            self.progress.setValue(0)
        elif kind == "point_started":
            self.live_history = []
            self.result_status.set_status(f"Solving point {event['index'] + 1} of {event['total']}…", "running")
        elif kind == "point_result":
            self.results.append(event)
            self.point_selector.addItem(f"Point {event['index'] + 1} · {event['status']}")
            self.point_selector.setCurrentIndex(len(self.results) - 1)
            self.progress.setValue(len(self.results))
            self.export_data_button.setEnabled(True)
            self.draw_result(len(self.results) - 1)
            self._draw_scan()
        elif kind == "finished":
            self.draw_result(self.point_selector.currentIndex())
            self.folder_button.setToolTip(self.output_dir or "")
            self.show_message(f"Run complete: {event['converged']} of {event['total']} points converged.", event["converged"] != event["total"])
        elif kind == "error":
            self.show_message(f"Solver error: {event['message']}", True)
            self.result_status.set_status(event["message"], "error")

    def _draw_history(self, history, ylabel="maximum absolute residual"):
        self.residual_plot.clear()
        values = np.maximum(np.asarray(history, dtype=float), np.finfo(float).tiny)
        self.residual_plot.axes.semilogy(np.arange(len(values)), values, color=COLORS[0])
        if self.run_spec:
            self.residual_plot.axes.axhline(self.run_spec["solver"]["final_tol"], color=COLORS[1], linestyle="--", label="requested tolerance")
        self.residual_plot.finish("history entry", ylabel)

    def draw_result(self, index):
        if not 0 <= index < len(self.results):
            return
        event = self.results[index]
        with np.load(event["data_path"], allow_pickle=False) as data:
            fields, q, history = data["field"].copy(), data["coordinate"].copy(), data["history"].copy()
        self.solution_plot.clear()
        model_plot_shown = False
        if self.run_spec is None and event.get("model_plot_path"):
            try:
                self.solution_plot.show_model_plot(event["model_plot_path"])
                model_plot_shown = True
            except (OSError, ValueError, TypeError) as exc:
                self.log.appendPlainText(f"[UI plot] Could not display the model's figure: {exc}. Displaying raw fields instead.")
        coordinate = "r" if event["geometry"] == "radial" else "x"
        if self.run_spec:
            curves = self.run_spec["plots"] or [{"expression": f["name"], "label": f["label"]} for f in self.run_spec["fields"]]
            for k, curve in enumerate(curves):
                try:
                    values = profile_values(curve["expression"], self.run_spec, event["params"], fields, q)
                    self.solution_plot.axes.plot(q, values, label=curve["label"], **curve_style(k))
                except ValueError as exc:
                    self.log.appendPlainText(f"[Plot] {exc}")
        elif not model_plot_shown:
            for k, values in enumerate(fields):
                name = self.run_field_names[k] if k < len(self.run_field_names) else f"field {k}"
                self.solution_plot.axes.plot(q, values, label=name, **curve_style(k))
        self.solution_plot.finish(coordinate, "field / derived quantity")
        self._draw_history(history)
        observable_name = self.run_spec["observable"]["name"] if self.run_spec and self.run_spec["observable"]["enabled"] else None
        self.result_status.set_result(event, observable_name)

    def _draw_scan(self):
        self.scan_plot.clear()
        if len(self.results) < 2:
            self.scan_plot.clear("Run a parameter scan to compare observables")
            return
        varied = [s["name"] for s in self.run_spec["scans"]] if self.run_spec else []
        if not varied:
            varied = [name for name in self.results[0]["params"] if len({p["params"].get(name) for p in self.results}) > 1]
        values = np.asarray([p["observable"] if p["converged"] and p["observable"] is not None else np.nan for p in self.results])
        ax = self.scan_plot.axes
        if len(varied) == 2:
            x, y = [[p["params"][name] for p in self.results] for name in varied]
            valid = np.isfinite(values)
            points = ax.scatter(np.asarray(x)[valid], np.asarray(y)[valid], c=values[valid], cmap="viridis", s=75)
            if np.any(valid):
                self.scan_plot.figure.colorbar(points, ax=ax, label="observable")
            self.scan_plot.finish(varied[0], varied[1], legend=False)
        else:
            x = [p["params"][varied[0]] for p in self.results] if len(varied) == 1 else np.arange(1, len(self.results) + 1)
            ax.plot(x, values, marker="o", color=COLORS[0])
            self.scan_plot.finish(varied[0] if len(varied) == 1 else "point", "observable", legend=False)

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self.show_message("The solver process could not start. Check the Python environment.", True)
            self._process_finished(1, QProcess.CrashExit)

    def _process_finished(self, exit_code, status):
        self._read_output()
        if self._pipe_buffer:
            self.log.appendPlainText(self._pipe_buffer)
        self._set_running(False)
        if self.cancelled:
            self.show_message(f"Run cancelled. {len(self.results)} completed point(s) remain available.")
            self.result_status.set_status("Completed points are retained.", "cancelled")
        elif exit_code != 0:
            self.show_message("Run failed. Details are available in the Run log.", True)
            self.result_status.set_status("Details are available in the Run log.", "error")
            self.result_tabs.setCurrentIndex(3)
        process = self.process
        self.process = None
        if process is not None:
            process.deleteLater()
        self.progress.setRange(0, max(1, len(self.results)))
        self.progress.setValue(len(self.results))
        self._cleanup_model()

    def _cleanup_job(self):
        if self.job is not None:
            self.job.cleanup()
            self.job = None

    def stop_run(self):
        if self.process is None:
            return
        self.cancelled = True
        process = self.process
        process.terminate()
        QTimer.singleShot(1500, lambda: process.kill() if self.process is process and process.state() != QProcess.NotRunning else None)
        self.show_message("Stopping the solver…")

    def _cleanup_model(self):
        if self.temp_model is not None:
            self.temp_model.unlink(missing_ok=True)
            self.temp_model = None

    def export_profiles(self, checked=False, path=None):
        index = self.point_selector.currentIndex()
        if not 0 <= index < len(self.results):
            return False
        event = self.results[index]
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export selected profiles", f"profiles_point_{index+1}_{event['status'].replace(' ', '_')}.csv", "CSV (*.csv)")
            if not path:
                return False
        with np.load(event["data_path"], allow_pickle=False) as data:
            fields, q = data["field"], data["coordinate"]
            names = [f["name"] for f in self.run_spec["fields"]] if self.run_spec else self.run_field_names
            header = f"status={event['status']}; residual={event['residual']}\n" + ",".join(["r" if event["geometry"] == "radial" else "x", *names])
            np.savetxt(path, np.column_stack([q, fields.T]), delimiter=",", header=header)
        self.show_message(f"Exported profiles ({event['status']}) to {Path(path).name}")
        return True

    def open_output_folder(self):
        if self.output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir))

    def closeEvent(self, event):
        if self.process is not None:
            choice = QMessageBox.question(self, "Solver running", "Stop the current solve and close?", QMessageBox.Yes | QMessageBox.No)
            if choice != QMessageBox.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(2000)
        if not self.confirm_discard():
            event.ignore()
            return
        self._cleanup_model()
        self._cleanup_job()
        event.accept()


def main(path=None):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("KinkSol Model Builder")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("KinkSol")
    window = ModelBuilderWindow(path)
    window.show()
    return app.exec()
