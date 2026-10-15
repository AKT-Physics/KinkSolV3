"""Lightweight previews and editable row tables."""
from __future__ import annotations

from matplotlib import rc_context
from matplotlib.image import imread
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.text import Text
from PySide6.QtCore import QSize, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListView, QPushButton,
    QSizePolicy, QStyledItemDelegate, QTableWidget, QVBoxLayout, QWidget,
)

from builder.styles import PLOT_STYLE, THEME


class ModelComboBox(QComboBox):
    """Close committed selections after the current Qt event has finished."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Use a plain list and standard item delegate consistently across desktops.
        self.setView(QListView(self))
        self.setItemDelegate(QStyledItemDelegate(self.view()))
        QApplication.setEffectEnabled(Qt.UI_AnimateCombo, False)
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.hidePopup)
        self.activated.connect(self._selection_committed)

    @Slot(int)
    def _selection_committed(self, index):
        # Index/text handlers update the forms synchronously; close once they return.
        # Parenting the timer to the control cancels it if a row is rebuilt.
        self._close_timer.start(0)

    def showPopup(self):
        # A pending close from an earlier selection must not close a new popup.
        self._close_timer.stop()
        super().showPopup()


class PlotCanvas(QWidget):
    def __init__(self, toolbar=True, parent=None):
        super().__init__(parent)
        with rc_context(PLOT_STYLE):
            self.figure = Figure(figsize=(6, 3.5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setObjectName("plotToolbar")
            self.toolbar.setIconSize(QSize(18, 18))
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.clear("A preview will appear here")

    def clear(self, message=None):
        self.model_plot_path = None
        with rc_context(PLOT_STYLE):
            self.figure.clear()
            self.figure.set_layout_engine("constrained")
            self.axes = self.figure.add_subplot(111)
        self.axes.spines[["top", "right"]].set_visible(False)
        self.axes.grid(True, color=THEME["grid"], linewidth=0.7)
        if message:
            self.text(0.5, 0.5, message, ha="center", va="center", transform=self.axes.transAxes, color=THEME["muted"])
            self.axes.set_xticks([])
            self.axes.set_yticks([])
        self.canvas.draw_idle()

    def show_model_plot(self, path):
        """Show the worker's complete figure, including custom axes and legends."""
        pixels = imread(path)
        self.figure.clear()
        self.figure.set_layout_engine(None)
        self.axes = self.figure.add_axes((0, 0, 1, 1))
        self.axes.imshow(pixels, interpolation="antialiased")
        self.axes.set_axis_off()
        self.axes.format_coord = lambda x, y: ""
        self.model_plot_path = str(path)
        if hasattr(self, "toolbar"):
            self.toolbar.update()
        self.canvas.draw_idle()

    def finish(self, xlabel, ylabel, legend=True):
        with rc_context(PLOT_STYLE):
            self.axes.set(xlabel=xlabel, ylabel=ylabel)
            handles, labels = self.axes.get_legend_handles_labels()
            if legend and handles:
                self.axes.legend(handles, labels, frameon=False)
        # Include labels created by a colour bar or other caller outside rc_context.
        for text in self.figure.findobj(Text):
            text.set_fontfamily(PLOT_STYLE["font.serif"])
            text.set_math_fontfamily(PLOT_STYLE["mathtext.fontset"])
        self.canvas.draw_idle()

    def text(self, *args, **kwargs):
        with rc_context(PLOT_STYLE):
            return self.axes.text(*args, **kwargs)


class ResultSummary(QFrame):
    """A convergence badge with separately readable numerical results."""
    def __init__(self):
        super().__init__()
        self.setObjectName("resultSummary")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        top = QHBoxLayout()
        self.badge = QLabel()
        self.badge.setObjectName("resultBadge")
        top.addWidget(self.badge, alignment=Qt.AlignTop)
        self.message = QLabel()
        self.message.setObjectName("hint")
        self.message.setWordWrap(True)
        top.addWidget(self.message, 1)
        layout.addLayout(top)
        metrics = QHBoxLayout()
        self.values, self.captions = {}, {}
        for name, caption in (("residual", "Maximum residual"), ("iterations", "Iterations"), ("observable", "Observable")):
            column = QVBoxLayout()
            label, value = QLabel(caption), QLabel("—")
            label.setObjectName("hint")
            value.setObjectName("resultValue")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.captions[name], self.values[name] = label, value
            column.addWidget(label)
            column.addWidget(value)
            metrics.addLayout(column, 1)
        layout.addLayout(metrics)
        self.set_status("Complete a model and choose Run model.")

    def _badge(self, text, state):
        self.badge.setText(text)
        self.badge.setProperty("state", state)
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.badge.update()

    def set_status(self, message, state="idle"):
        titles = {"idle": "Ready", "running": "Running", "cancelled": "Cancelled", "error": "Run failed"}
        self._badge(titles[state], state)
        self.message.setText(message)
        for value in self.values.values():
            value.setText("—")
        self.captions["observable"].setText("Observable")
        self._text = message

    def set_result(self, event, observable_name=None):
        converged = event["converged"]
        self._badge("Converged" if converged else "Unconverged", "success" if converged else "warning")
        self.message.setText(f"Point {event['index'] + 1}" + ("" if converged else " · values describe the retained unconverged iterate"))
        residual = "non-finite" if event["residual"] is None else f"{event['residual']:.6g}"
        observable = "—" if event["observable"] is None else f"{event['observable']:.10g}"
        self.values["residual"].setText(residual)
        self.values["iterations"].setText(str(event["iterations"]))
        self.values["observable"].setText(observable)
        self.captions["observable"].setText(f"Observable · {observable_name}" if observable_name else "Observable")
        self._text = f"Point {event['index'] + 1}: {event['status']} · residual {residual} · iterations {event['iterations']} · observable {observable}"

    def text(self):
        return self._text


class DomainSketch(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.geometry_name, self.n, self.dx = "cartesian", 401, 0.05

    def set_grid(self, geometry, n, dx):
        self.geometry_name, self.n, self.dx = geometry, n, dx
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setFont(self.font())
        p.fillRect(self.rect(), QColor(THEME["highlight"]))
        left, right, y = 40, w - 40, h // 2
        p.setPen(QPen(QColor(THEME["domain_line"]), 2))
        p.drawLine(left, y, right, y)
        for i in range(17):
            x = left + (right - left) * i / 16
            p.setBrush(QColor(THEME["accent"]) if i in (0, 16) else QColor(THEME["domain_site"]))
            p.setPen(Qt.NoPen)
            p.drawEllipse(int(x) - 3, y - 3, 6, 6)
        length = (self.n - 1) * self.dx
        p.setPen(QColor(THEME["text"]))
        labels = ("0", f"R = {length:g}") if self.geometry_name == "radial" else (f"−{length / 2:g}", f"+{length / 2:g}")
        p.drawText(left - 20, y + 18, 100, 24, Qt.AlignLeft, labels[0])
        p.drawText(right - 80, y + 18, 100, 24, Qt.AlignRight, labels[1])
        title = "One-sided radial coordinate" if self.geometry_name == "radial" else "Two-sided Cartesian coordinate"
        p.drawText(0, 15, w, 30, Qt.AlignCenter, title)
        if self.geometry_name == "cartesian":
            p.drawText(w // 2 - 70, y + 18, 140, 24, Qt.AlignCenter, "0 · site" if self.n % 2 else "0 · between sites")
        p.drawText(0, h - 34, w, 24, Qt.AlignCenter, f"{self.n:,} sites     •     spacing {self.dx:g}")


def line(text="", tooltip=None, expression=False):
    widget = QLineEdit(str(text))
    widget.setProperty("expression", expression)
    if tooltip:
        widget.setToolTip(tooltip)
    return widget


class RowTable(QWidget):
    def __init__(self, headers, choices=None, editable=True, parent=None, *,
                 expression_columns=(), column_widths=None, header_tooltips=None, stretch_columns=()):
        super().__init__(parent)
        self.headers = headers
        self.choices = choices or {}
        self.expression_columns = set(expression_columns)
        self.column_widths = column_widths or [100] * len(headers)
        self.stretch_columns = tuple(stretch_columns)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for col, text in enumerate(header_tooltips or headers):
            self.table.horizontalHeaderItem(col).setToolTip(text)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(not self.stretch_columns)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMinimumHeight(70)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        if editable:
            bar = QHBoxLayout()
            self.add_button, self.remove_button = QPushButton("+ Add row"), QPushButton("Remove selected")
            self.add_button.clicked.connect(lambda: self.add_row())
            self.remove_button.clicked.connect(self.remove_selected)
            bar.addWidget(self.add_button)
            bar.addWidget(self.remove_button)
            bar.addStretch()
            layout.addLayout(bar)

    def add_row(self, values=None):
        values = values or [""] * len(self.headers)
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, text in enumerate(values):
            if col in self.choices:
                widget = ModelComboBox()
                widget.addItems(self.choices[col])
                if text:
                    widget.setCurrentText(str(text))
            else:
                widget = line(text, expression=col in self.expression_columns)
            self.table.setCellWidget(row, col, widget)
        self._fit_height()

    def set_rows(self, rows):
        self.table.setRowCount(0)
        for row in rows:
            self.add_row(row)
        self._fit_height()

    def _fit_height(self):
        self.table.ensurePolished()
        self._fit_columns()
        header = self.table.horizontalHeader()
        row_height = max(40, self.table.fontMetrics().height() + 18)
        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, row_height)
        # Reserve scrollbar space so larger fonts and narrow windows do not clip a row.
        height = header.sizeHint().height() + row_height * self.table.rowCount()
        height += self.table.horizontalScrollBar().sizeHint().height() + 2 * self.table.frameWidth()
        self.table.setFixedHeight(min(320, max(80, height)))

    def _fit_columns(self):
        header = self.table.horizontalHeader()
        widths = [max(width, header.fontMetrics().horizontalAdvance(title) + 22)
                  for width, title in zip(self.column_widths, self.headers)]
        if self.stretch_columns:
            extra = max(0, self.table.viewport().width() - sum(widths))
            share, remainder = divmod(extra, len(self.stretch_columns))
            for index, col in enumerate(self.stretch_columns):
                widths[col] += share + (index < remainder)
        for col, width in enumerate(widths):
            self.table.setColumnWidth(col, width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_columns()

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_height()

    def rows(self):
        rows = []
        for row in range(self.table.rowCount()):
            values = []
            for col in range(self.table.columnCount()):
                widget = self.table.cellWidget(row, col)
                values.append(widget.currentText() if isinstance(widget, QComboBox) else widget.text().strip())
            rows.append(values)
        return rows

    def remove_selected(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for row in rows:
            self.table.removeRow(row)
        self._fit_height()
