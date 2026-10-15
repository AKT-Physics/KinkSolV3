"""The separate cog-accessed dialog for every current HybridOptions control."""
from __future__ import annotations

import copy

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QLineEdit, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from builder.schema import solver_defaults, validate_solver
from builder.styles import ERROR_TEXT_STYLE, stylesheet, ui_font
from builder.widgets import ModelComboBox


GROUPS = [
    ("Main solver", [
        ("use_pointwise_phase", "Use pointwise pre-phase", "Relax each local equation before the global Newton phase."),
        ("use_trust_region", "Use trust-region final phase", "Use KinkSol's step-limiting trust-region solver."),
        ("final_tol", "Residual tolerance", "Convergence requires the maximum absolute residual to be at most this value."),
        ("final_max_iter", "Maximum final iterations", "Maximum Newton iterations in each final-phase attempt."),
        ("final_freq_out", "Report every N iterations", "Controls final-phase console reporting and live residual updates."),
        ("final_step_scale", "Plain Newton step scale", "Multiplier applied to a plain sparse Newton update."),
    ]),
    ("Pointwise", [
        ("pointwise_tol", "Pointwise tolerance", "Stopping residual for the pointwise pre-phase and rescue."),
        ("pointwise_max_iter", "Maximum pre-phase iterations", "Iteration limit before switching to the global phase."),
        ("pointwise_relaxation", "Pre-phase relaxation", "Multiplier on pointwise Newton updates."),
        ("pointwise_freq_out", "Report every N iterations", "Console reporting interval for pointwise phases."),
    ]),
    ("Trust region", [
        ("trust_initial_radius", "Initial radius", "Initial limit on the norm of a proposed step."),
        ("trust_min_radius", "Minimum radius", "Smallest permitted trust radius."),
        ("trust_max_radius", "Maximum radius", "Largest permitted trust radius."),
        ("trust_shrink", "Shrink factor", "Radius multiplier after a rejected trial; between zero and one."),
        ("trust_grow", "Growth factor", "Radius multiplier after a sufficiently good trial; greater than one."),
        ("trust_max_trials", "Maximum trials per step", "Maximum trial radii at each Newton iteration."),
        ("trust_accept_growth", "Permitted residual growth", "Existing KinkSol acceptance threshold relative to the current residual."),
        ("trust_good_improve", "Good improvement threshold", "Existing KinkSol residual-improvement threshold for radius growth."),
    ]),
    ("Recovery", [
        ("enable_rescue", "Enable pointwise rescue", "If the global solve fails, relax the best retained phase endpoint and retry."),
        ("rescue_pointwise_max_iter", "Maximum rescue iterations", "Iteration limit for the rescue pointwise phase."),
        ("rescue_pointwise_relaxation", "Rescue relaxation", "Multiplier on rescue pointwise updates."),
        ("accept_best_near_tol", "Return a near solution", "Retain an iterate within the tolerance multiplier; it is still marked unconverged."),
        ("near_tol_factor", "Near-solution tolerance factor", "Maximum allowed residual relative to the main tolerance."),
    ]),
]


class SolverSettingsDialog(QDialog):
    def __init__(self, options, jacobian="auto", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Solver settings")
        self.setObjectName("solverSettingsDialog")
        self.setFont(ui_font())
        self.setStyleSheet(stylesheet())
        self.resize(650, 580)
        self.options = copy.deepcopy(options)
        self.jacobian = jacobian
        self.widgets = {}
        layout = QVBoxLayout(self)
        title = QLabel("Configure the numerical solve")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        explanation = QLabel("These settings are saved with your model. Hover over a control for its meaning.")
        explanation.setObjectName("hint")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.tabs = QTabWidget()
        for label, entries in GROUPS:
            body, form = QWidget(), QFormLayout()
            body.setObjectName("settingsPage")
            body.setLayout(form)
            form.setVerticalSpacing(16)
            for key, description, tooltip in entries:
                default = solver_defaults()[key]
                if type(default) is bool:
                    widget = QCheckBox()
                    widget.toggled.connect(self.update_enabled)
                elif type(default) is int:
                    widget = QSpinBox()
                    widget.setRange(1, 10000000)
                else:
                    widget = QLineEdit()
                    widget.setPlaceholderText("e.g. 1e-9")
                    widget.setMaximumWidth(210)
                widget.setObjectName(key)
                widget.setProperty("expression", type(default) is not bool)
                widget.setToolTip(tooltip)
                self.widgets[key] = widget
                field_label = QLabel(description)
                field_label.setToolTip(tooltip)
                form.addRow(field_label, widget)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setWidget(body)
            self.tabs.addTab(scroll, label)
        advanced = QWidget()
        advanced.setObjectName("settingsPage")
        form = QFormLayout(advanced)
        self.jacobian_widget = ModelComboBox()
        self.jacobian_widget.addItems(["auto", "scalar", "row"])
        self.jacobian_widget.setToolTip("Automatic mode checks vector dual support and falls back to scalar AD when needed.")
        form.addRow("Jacobian automatic differentiation", self.jacobian_widget)
        note = QLabel("The UI uses the existing KinkSol solvers and their acceptance rules.\nA near solution remains explicitly marked unconverged.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        form.addRow(note)
        self.tabs.addTab(advanced, "Advanced")
        layout.addWidget(self.tabs)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet(ERROR_TEXT_STYLE)
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout.addWidget(self.buttons)
        self.populate(options, jacobian)

    def populate(self, options, jacobian):
        for key, widget in self.widgets.items():
            if isinstance(widget, QCheckBox):
                widget.setChecked(options[key])
            elif isinstance(widget, QSpinBox):
                widget.setValue(options[key])
            else:
                widget.setText(f"{options[key]:.12g}")
        self.jacobian_widget.setCurrentText(jacobian)
        self.update_enabled()

    def update_enabled(self, *_):
        if not all(key in self.widgets for key in ("use_pointwise_phase", "use_trust_region", "enable_rescue", "accept_best_near_tol")):
            return
        pointwise = self.widgets["use_pointwise_phase"].isChecked()
        trust = self.widgets["use_trust_region"].isChecked()
        rescue = self.widgets["enable_rescue"].isChecked()
        for key, widget in self.widgets.items():
            if key.startswith("pointwise_"):
                widget.setEnabled(pointwise or rescue if key in ("pointwise_tol", "pointwise_freq_out") else pointwise)
            elif key.startswith("trust_"):
                widget.setEnabled(trust)
            elif key.startswith("rescue_"):
                widget.setEnabled(rescue)
        self.widgets["final_step_scale"].setEnabled(not trust)
        self.widgets["near_tol_factor"].setEnabled(self.widgets["accept_best_near_tol"].isChecked())

    def restore_defaults(self):
        self.populate(solver_defaults(), "auto")
        self.error.clear()

    def apply(self):
        try:
            options = {}
            for key, widget in self.widgets.items():
                options[key] = widget.isChecked() if isinstance(widget, QCheckBox) else widget.value() if isinstance(widget, QSpinBox) else float(widget.text())
            validate_solver(options)
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        self.options = options
        self.jacobian = self.jacobian_widget.currentText()
        self.accept()
