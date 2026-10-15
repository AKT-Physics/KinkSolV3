"""Shared desktop theme, font selection and scientific preview styling."""
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo
from matplotlib import rcParamsDefault


THEME = {
    "accent": "#004225",
    "accent_hover": "#0B5735",
    "background": "#F6F4EF",
    "surface": "#FFFFFF",
    "sidebar": "#E8EEE7",
    "highlight": "#EDF3EC",
    "text": "#26312B",
    "muted": "#5F6C63",
    "border": "#D5DDD3",
    "grid": "#E9EEE7",
    "disabled_surface": "#F0F2ED",
    "disabled_text": "#7B867F",
    "disabled_accent": "#8DA697",
    "domain_line": "#92A697",
    "domain_site": "#A9C2B0",
    "error": "#A6291E",
    "error_surface": "#FFF0EE",
    "error_border": "#E8B8B0",
    "warning": "#765015",
    "warning_surface": "#FFF5E2",
    "warning_border": "#E7D0A5",
}

# Prefer an installed serif; no downloaded or bundled UI font is required.
SERIF_FAMILIES = ("Georgia", "DejaVu Serif", "Liberation Serif", "Times New Roman", "Times", "Noto Serif")
MONOSPACE_FAMILIES = ("DejaVu Sans Mono", "Consolas", "Liberation Mono", "Menlo", "Courier New")
UI_FONT_SIZE = 11
EXPRESSION_FONT_SIZE = 10
COLORS = list(rcParamsDefault["axes.prop_cycle"].by_key()["color"])
LINESTYLES = ["-", "--", "-.", ":"]

def curve_style(index):
    return {
        "color": COLORS[index % len(COLORS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        "linewidth": 1.8,
    }
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "text.color": THEME["text"],
    "axes.labelcolor": THEME["text"],
    "axes.edgecolor": THEME["muted"],
    "xtick.color": THEME["text"],
    "ytick.color": THEME["text"],
    "figure.facecolor": THEME["surface"],
    "axes.facecolor": THEME["surface"],
    "savefig.facecolor": THEME["surface"],
}


def _installed_font(families, style_hint, size):
    available = {family.casefold(): family for family in QFontDatabase.families()}
    font = QFont()
    font.setStyleHint(style_hint)
    for candidate in families:
        if candidate.casefold() in available:
            font.setFamily(available[candidate.casefold()])
            break
    else:
        font.setFamily(font.defaultFamily())
    font.setPointSize(size)
    return font


def ui_font():
    return _installed_font(SERIF_FAMILIES, QFont.Serif, UI_FONT_SIZE)


def monospace_font():
    return _installed_font(MONOSPACE_FAMILIES, QFont.Monospace, EXPRESSION_FONT_SIZE)


# Colour tokens and the two resolved font families keep all theme choices here.
STYLES = """
QWidget { color: @text@; font-family: '@ui_font@'; font-size: @ui_size@pt; }
QMainWindow, QDialog, QWidget#editorPage { background: @background@; }
QWidget#settingsPage { background: @surface@; }
QLabel#brand { color: @accent@; font-size: 22pt; font-weight: 700; }
QLabel#eyebrow { color: @muted@; font-size: 8pt; font-weight: 700; }
QLabel#pageTitle { color: @accent@; font-size: 17pt; font-weight: 700; padding-bottom: 3px; }
QLabel#hint { color: @muted@; }
QLabel#statusBanner { border: 1px solid @border@; background: @highlight@; border-radius: 7px; padding: 9px 12px; }
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: @surface@; border: 1px solid @border@; border-radius: 5px; padding: 6px; selection-background-color: @accent@; selection-color: @surface@; }
QComboBox QAbstractItemView { background: @surface@; color: @text@; selection-background-color: @accent@; selection-color: @surface@; border: 1px solid @border@; outline: none; }
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid @accent@; }
QLineEdit:read-only { background: @disabled_surface@; color: @muted@; }
QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { background: @disabled_surface@; color: @disabled_text@; border-color: @grid@; }
QPlainTextEdit, QLineEdit[expression="true"], QComboBox[expression="true"], QSpinBox[expression="true"], QDoubleSpinBox[expression="true"] { font-family: '@mono_font@'; font-size: @expression_size@pt; }
QPushButton, QToolButton { background: @surface@; border: 1px solid @border@; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover, QToolButton:hover { background: @highlight@; border-color: @domain_line@; }
QPushButton:disabled, QToolButton:disabled { color: @disabled_text@; background: @disabled_surface@; }
QPushButton#primary { background: @accent@; color: @surface@; border: none; font-weight: 700; padding: 10px 24px; }
QPushButton#primary:hover { background: @accent_hover@; }
QPushButton#primary:disabled { background: @disabled_accent@; }
QPushButton#stopButton { color: @error@; }
QPushButton#stopButton:disabled { color: @disabled_text@; }
QToolButton#settingsCog { font-size: 20pt; padding: 3px 10px; }
QListWidget#navigation { background: @sidebar@; border: none; padding: 12px 8px; outline: none; }
QListWidget#navigation::item { padding: 12px 10px; margin: 2px 0; border-radius: 6px; color: @muted@; }
QListWidget#navigation::item:hover { background: @highlight@; color: @accent@; }
QListWidget#navigation::item:selected { background: @accent@; color: @surface@; font-weight: 700; }
QGroupBox { background: @surface@; border: 1px solid @border@; border-radius: 7px; margin-top: 16px; padding: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; font-weight: 700; }
QTableWidget { background: @surface@; alternate-background-color: @disabled_surface@; border: 1px solid @border@; border-radius: 5px; gridline-color: @grid@; }
QHeaderView::section { background: @disabled_surface@; color: @muted@; font-size: 10pt; border: none; border-bottom: 1px solid @border@; padding: 7px; }
QTabWidget::pane { background: @surface@; border: 1px solid @border@; border-radius: 5px; }
QTabBar::tab { padding: 9px 12px; color: @muted@; }
QTabBar::tab:selected { color: @accent@; background: @surface@; font-weight: 700; }
QProgressBar { border: none; background: @sidebar@; border-radius: 4px; height: 8px; }
QProgressBar::chunk { background: @accent@; border-radius: 4px; }
QScrollArea { border: none; background: transparent; }
QToolBar#plotToolbar { background: @surface@; border: none; padding: 2px; spacing: 2px; }
QToolBar#plotToolbar QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 4px; }
QToolBar#plotToolbar QToolButton:hover, QToolBar#plotToolbar QToolButton:checked { background: @highlight@; border-color: @domain_line@; }
QFrame#resultSummary { background: @surface@; border: 1px solid @border@; border-radius: 7px; }
QLabel#resultBadge { background: @disabled_surface@; color: @muted@; border-radius: 5px; padding: 5px 10px; font-weight: 700; }
QLabel#resultBadge[state="success"] { background: @highlight@; color: @accent@; }
QLabel#resultBadge[state="running"] { background: @sidebar@; color: @accent@; }
QLabel#resultBadge[state="warning"] { background: @warning_surface@; color: @warning@; }
QLabel#resultBadge[state="error"] { background: @error_surface@; color: @error@; }
QLabel#resultValue { font-family: '@mono_font@'; font-size: 12pt; }
QToolTip { background: @surface@; color: @text@; border: 1px solid @border@; padding: 5px; }
"""


def stylesheet():
    tokens = dict(THEME, ui_font=QFontInfo(ui_font()).family(), mono_font=QFontInfo(monospace_font()).family(),
                  ui_size=str(UI_FONT_SIZE), expression_size=str(EXPRESSION_FONT_SIZE))
    result = STYLES
    for name, value in tokens.items():
        result = result.replace(f"@{name}@", value)
    return result


ERROR_BANNER_STYLE = f"background: {THEME['error_surface']}; border-color: {THEME['error_border']}; color: {THEME['error']};"
ERROR_TEXT_STYLE = f"color: {THEME['error']};"
