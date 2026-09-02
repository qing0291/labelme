from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest
from PySide6 import QtGui
from PySide6.QtCore import QPoint
from PySide6.QtCore import QPointF
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from labelme._app import MainWindow
from labelme._widgets._canvas_interaction import CursorRole
from labelme._widgets.canvas import Canvas
from labelme._widgets.canvas import _CanvasMode

from ..conftest import close_or_pause
from .conftest import click_canvas_fraction
from .conftest import drag_canvas

_DRAG_OFFSET_PX: Final[int] = 40


def _diagonal_drag_endpoints(*, canvas: Canvas) -> tuple[QPoint, QPoint]:
    start = QPoint(canvas.width() // 2, canvas.height() // 2)
    end = QPoint(start.x() + _DRAG_OFFSET_PX, start.y() + _DRAG_OFFSET_PX)
    return start, end


def _zoom_until_overflow(*, canvas: Canvas) -> None:
    # Mirrors middle_drag_scroll_test: force the image past the viewport so
    # panning has somewhere to go, without routing through the zoom pipeline.
    viewport = canvas._scroll_viewport()
    assert viewport is not None
    while not (
        canvas.pixmap.width() * canvas.scale > viewport.width()
        or canvas.pixmap.height() * canvas.scale > viewport.height()
    ):
        canvas.scale *= 1.5
        canvas.adjustSize()
        canvas.update()


def _shrink_below_viewport(*, canvas: Canvas) -> None:
    viewport = canvas._scroll_viewport()
    assert viewport is not None
    canvas.scale = (
        min(
            viewport.width() / canvas.pixmap.width(),
            viewport.height() / canvas.pixmap.height(),
        )
        * 0.5
    )
    canvas.adjustSize()
    canvas.update()


@pytest.mark.gui
def test_hand_drag_pans_without_disturbing_annotations(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    _zoom_until_overflow(canvas=canvas)
    annotated_win.set_hand_mode(True)  # noqa: FBT003 -- positional-only slot signature
    assert canvas.is_hand_mode()

    points_before = [shape.points.copy() for shape in canvas.shapes]
    labels_before = [shape.label for shape in canvas.shapes]
    backups_before = len(canvas.shape_backups)
    assert points_before, "fixture is expected to carry annotations"

    deltas: list[QPoint] = []
    canvas.pan_request.connect(deltas.append)

    start, end = _diagonal_drag_endpoints(canvas=canvas)
    drag_canvas(
        qtbot=qtbot,
        canvas=canvas,
        button=Qt.MouseButton.LeftButton,
        start=start,
        end=end,
    )

    assert deltas, "pan_request was not emitted during a hand drag"
    assert sum(p.x() for p in deltas) == _DRAG_OFFSET_PX
    assert sum(p.y() for p in deltas) == _DRAG_OFFSET_PX

    # The left button moved the view and nothing else.
    assert [shape.label for shape in canvas.shapes] == labels_before
    for shape, before in zip(canvas.shapes, points_before, strict=True):
        assert np.array_equal(shape.points, before)
    assert canvas.selected_shapes == []
    assert len(canvas.shape_backups) == backups_before
    assert not annotated_win._is_changed

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_hand_drag_does_not_create_a_shape(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    _zoom_until_overflow(canvas=canvas)
    annotated_win._switch_canvas_mode(edit=False, create_mode="polygon")
    annotated_win.set_hand_mode(True)  # noqa: FBT003 -- positional-only slot signature
    shapes_before = len(canvas.shapes)

    start, end = _diagonal_drag_endpoints(canvas=canvas)
    drag_canvas(
        qtbot=qtbot,
        canvas=canvas,
        button=Qt.MouseButton.LeftButton,
        start=start,
        end=end,
    )

    assert len(canvas.shapes) == shapes_before
    assert not canvas.is_drawing

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_hand_drag_does_not_pan_when_image_fits_viewport(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    _shrink_below_viewport(canvas=canvas)
    annotated_win.set_hand_mode(True)  # noqa: FBT003 -- positional-only slot signature

    start, end = _diagonal_drag_endpoints(canvas=canvas)
    with qtbot.assertNotEmitted(canvas.pan_request):
        drag_canvas(
            qtbot=qtbot,
            canvas=canvas,
            button=Qt.MouseButton.LeftButton,
            start=start,
            end=end,
        )

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_hand_mode_keeps_the_open_hand_after_release(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    _zoom_until_overflow(canvas=canvas)
    annotated_win.set_hand_mode(True)  # noqa: FBT003 -- positional-only slot signature

    start, end = _diagonal_drag_endpoints(canvas=canvas)
    drag_canvas(
        qtbot=qtbot,
        canvas=canvas,
        button=Qt.MouseButton.LeftButton,
        start=start,
        end=end,
    )

    assert canvas._cursor == CursorRole.GRAB

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
@pytest.mark.parametrize("exit_via", ["escape", "action"])
def test_leaving_hand_mode_restores_the_previous_tool(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
    exit_via: str,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    annotated_win._switch_canvas_mode(edit=False, create_mode="rectangle")
    annotated_win.set_hand_mode(True)  # noqa: FBT003 -- positional-only slot signature
    assert canvas.is_hand_mode()
    assert annotated_win._actions.hand.isChecked()

    if exit_via == "escape":
        qtbot.keyPress(canvas, Qt.Key.Key_Escape)
    else:
        annotated_win.set_hand_mode(False)  # noqa: FBT003 -- positional-only slot signature
    qtbot.wait(50)

    assert not canvas.is_hand_mode()
    assert not annotated_win._actions.hand.isChecked()
    # The overlay never replaced the tool underneath it.
    assert canvas.mode == _CanvasMode.CREATE
    assert canvas.create_mode == "rectangle"

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_the_h_key_toggles_hand_mode(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    # Hand mode swallows key events on the canvas, so the shortcut has to keep
    # working from inside it to get back out.
    canvas = annotated_win._canvas_widgets.canvas
    canvas.setFocus()
    qtbot.waitUntil(lambda: canvas.hasFocus())

    qtbot.keyClick(canvas, Qt.Key.Key_H)
    qtbot.waitUntil(canvas.is_hand_mode)
    assert annotated_win._actions.hand.isChecked()

    qtbot.keyClick(canvas, Qt.Key.Key_H)
    qtbot.waitUntil(lambda: not canvas.is_hand_mode())
    assert not annotated_win._actions.hand.isChecked()

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_hand_mode_is_refused_while_a_shape_is_unfinished(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    annotated_win._switch_canvas_mode(edit=False, create_mode="polygon")
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=(0.3, 0.3))
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=(0.5, 0.3))
    assert canvas.is_drawing

    # Both the toolbar button and the slot refuse: an unfinished shape owns
    # the left button and would otherwise be stranded.
    assert not annotated_win._actions.hand.isEnabled()
    annotated_win.set_hand_mode(True)  # noqa: FBT003 -- positional-only slot signature

    assert not canvas.is_hand_mode()
    assert not annotated_win._actions.hand.isChecked()
    assert canvas.is_drawing

    qtbot.keyPress(canvas, Qt.Key.Key_Escape)
    qtbot.wait(50)
    assert not canvas.is_drawing

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_switching_tools_leaves_hand_mode(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    annotated_win.set_hand_mode(True)  # noqa: FBT003 -- positional-only slot signature
    assert canvas.is_hand_mode()

    annotated_win._switch_canvas_mode(edit=False, create_mode="polygon")

    assert not canvas.is_hand_mode()
    assert not annotated_win._actions.hand.isChecked()

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_navigating_leaves_the_label_file_byte_identical(
    *,
    qtbot: QtBot,
    annotated_win: MainWindow,
    data_path: Path,
    tmp_path: Path,
    pause: bool,
) -> None:
    # Panning and zooming move the viewport only. Nothing they do may reach
    # the annotations, so the file on disk must come back bit for bit.
    label_path = data_path / "annotated/2011_000003.json"
    before = label_path.read_bytes()

    canvas = annotated_win._canvas_widgets.canvas
    _zoom_until_overflow(canvas=canvas)
    pos = QPointF(canvas.width() / 2, canvas.height() / 2)

    for delta in (QPoint(0, 120), QPoint(0, -120), QPoint(120, 0)):
        for modifiers in (
            Qt.KeyboardModifier.NoModifier,
            Qt.KeyboardModifier.ShiftModifier,
            Qt.KeyboardModifier.ControlModifier,
        ):
            canvas.wheelEvent(
                QtGui.QWheelEvent(
                    pos,
                    pos,
                    QPoint(0, 0),
                    delta,
                    Qt.MouseButton.NoButton,
                    modifiers,
                    Qt.ScrollPhase.NoScrollPhase,
                    False,  # noqa: FBT003 -- QWheelEvent takes inverted positionally
                )
            )
    qtbot.wait(50)

    start, end = _diagonal_drag_endpoints(canvas=canvas)
    drag_canvas(
        qtbot=qtbot,
        canvas=canvas,
        button=Qt.MouseButton.MiddleButton,
        start=start,
        end=end,
    )
    annotated_win.set_hand_mode(True)  # noqa: FBT003 -- positional-only slot signature
    drag_canvas(
        qtbot=qtbot,
        canvas=canvas,
        button=Qt.MouseButton.LeftButton,
        start=start,
        end=end,
    )
    qtbot.wait(100)

    assert label_path.read_bytes() == before
    # Nothing was dirty, so auto-save had no reason to write an output either.
    # tmp_path doubles as the fixture's scratch space, so look for saved
    # annotations specifically rather than requiring an empty directory.
    assert list(tmp_path.glob("*.json")) == []
    assert not annotated_win._is_changed

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)
