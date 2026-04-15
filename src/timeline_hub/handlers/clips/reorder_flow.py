from collections.abc import Sequence
from typing import Any

from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.formatting import Bold, Text

from timeline_hub.handlers.clips.common import (
    MenuAction,
    back_button,
    create_padding_line,
    dummy_button,
    set_flow_context,
)
from timeline_hub.settings import Settings

REORDER_FLOW_MODE = 'reorder'
REORDER_RESET_CALLBACK_VALUE = 'reset'
_REORDER_MAX_CLIPS = 16
_REORDER_SELECTION_PROMPT = 'Select new order:'


class ReorderClipFlow(StatesGroup):
    selecting = State()


class ReorderCallbackData(CallbackData, prefix='clip_reorder'):
    action: MenuAction
    value: str


def reorder_validation_error(total_clips: int) -> str | None:
    if total_clips == 1:
        return 'Unexpected number of clips'
    if total_clips > _REORDER_MAX_CLIPS:
        return 'Too many clips'
    return None


async def show_reorder_selection_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    total_clips: int,
    buffer_version: int,
) -> None:
    await set_flow_context(
        state=state,
        mode=REORDER_FLOW_MODE,
        menu_message_id=message.message_id,
        fsm_state=ReorderClipFlow.selecting,
    )
    await state.update_data(
        selected_order=[],
        total_clips=total_clips,
        buffer_version=buffer_version,
    )
    await message.edit_text(
        **reorder_selection_kwargs(
            selected_order=[],
            message_width=settings.message_width,
        ),
        reply_markup=reorder_selection_keyboard(
            total_clips=total_clips,
            selected_order=[],
        ),
    )


def reorder_selection_keyboard(
    *,
    total_clips: int,
    selected_order: Sequence[int],
) -> InlineKeyboardMarkup:
    buttons = [
        _create_reorder_select_button(
            index=index,
            selected=index in set(selected_order),
        )
        for index in range(1, total_clips + 1)
    ]
    top_row: list[InlineKeyboardButton] = []
    middle_row: list[InlineKeyboardButton] = []
    for index, button in enumerate(reversed(buttons)):
        if index % 2 == 0:
            top_row.append(button)
        else:
            middle_row.append(button)
    if total_clips % 2 != 0:
        middle_row.insert(0, dummy_button())

    return InlineKeyboardMarkup(
        inline_keyboard=[
            top_row,
            middle_row,
            [_reorder_navigation_button(selected_order=selected_order)],
        ]
    )


def reorder_selection_kwargs(
    *,
    selected_order: Sequence[int],
    message_width: int,
) -> dict[str, Any]:
    return Text(
        _reorder_selected_content(selected_order),
        '\n',
        create_padding_line(message_width),
        '\n',
        _REORDER_SELECTION_PROMPT,
    ).as_kwargs()


def reorder_final_kwargs(selected_order: Sequence[int]) -> dict[str, Any]:
    return _reorder_selected_content(selected_order).as_kwargs()


def reorder_selected_order_from_state(data: dict[str, object]) -> list[int] | None:
    raw_selected_order = data.get('selected_order')
    if not isinstance(raw_selected_order, list):
        return None
    selected_order: list[int] = []
    for value in raw_selected_order:
        if not isinstance(value, int):
            return None
        selected_order.append(value)
    return selected_order


def reorder_total_clips_from_state(data: dict[str, object]) -> int | None:
    total_clips = data.get('total_clips')
    if isinstance(total_clips, int):
        return total_clips
    return None


def parse_reorder_index(value: str) -> int | None:
    if not value.isdigit():
        return None
    return int(value)


def reordered_video_messages(
    video_messages: Sequence[Message],
    *,
    selected_order: Sequence[int],
    total_clips: int,
) -> list[Message]:
    # Fail fast if the flushed video set no longer matches the validated entry
    # count; interactive reorder should never execute against drifted input.
    if len(video_messages) != total_clips:
        raise RuntimeError('Reorder buffer changed unexpectedly before completion')
    return [video_messages[index - 1] for index in selected_order]


def _reorder_navigation_button(*, selected_order: Sequence[int]) -> InlineKeyboardButton:
    if not selected_order:
        return back_button(
            callback_data=ReorderCallbackData(
                action=MenuAction.BACK,
                value='back',
            ).pack()
        )
    return InlineKeyboardButton(
        text='Reset',
        callback_data=ReorderCallbackData(
            action=MenuAction.BACK,
            value=REORDER_RESET_CALLBACK_VALUE,
        ).pack(),
    )


def _create_reorder_select_button(*, index: int, selected: bool) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=str(index),
        style='primary' if selected else None,
        callback_data=ReorderCallbackData(
            action=MenuAction.SELECT,
            value=str(index),
        ).pack(),
    )


def _reorder_selected_content(selected_order: Sequence[int]) -> Text:
    parts: list[object] = ['Selected: ', Bold('Reorder')]
    if selected_order:
        parts.extend([' -> '])
        for index, value in enumerate(selected_order):
            if index > 0:
                parts.append(' ')
            parts.append(Bold(str(value)))
    return Text(*parts)
