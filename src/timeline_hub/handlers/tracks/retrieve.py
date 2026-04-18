from enum import StrEnum, auto
from typing import Any

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.formatting import Bold, Text

from timeline_hub.handlers.menu import (
    callback_message,
    create_padding_line,
    dummy_button,
    handle_stale_selection,
    selected_text,
    stacked_keyboard,
    width_reserved_text,
)
from timeline_hub.services.container import Services
from timeline_hub.settings import Settings

router = Router()


class RetrieveEntryAction(StrEnum):
    CANCEL = auto()


class RetrieveEntryCallbackData(CallbackData, prefix='track_retrieve_entry'):
    action: RetrieveEntryAction


class TrackIntakeAction(StrEnum):
    CANCEL = auto()


class TrackIntakeActionCallbackData(CallbackData, prefix='track_intake'):
    action: TrackIntakeAction
    buffer_version: int


@router.message(F.text == 'Tracks')
async def on_tracks(message: Message, state: FSMContext, settings: Settings) -> None:
    await state.clear()
    await message.answer(
        **width_reserved_text(
            text='Select action:',
            message_width=settings.message_width,
        ),
        reply_markup=_track_entry_reply_markup(),
    )


@router.callback_query(
    RetrieveEntryCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_retrieve_entry(
    callback: CallbackQuery,
    callback_data: RetrieveEntryCallbackData,
    state: FSMContext,
) -> None:
    await callback.answer()
    message = callback_message(callback)
    if message is None:
        await state.clear()
        return

    if callback_data.action is RetrieveEntryAction.CANCEL:
        await state.clear()
        await message.edit_text(
            **selected_text(selected='Cancel'),
            reply_markup=None,
        )


@router.callback_query(
    TrackIntakeActionCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_track_intake_action(
    callback: CallbackQuery,
    callback_data: TrackIntakeActionCallbackData,
    state: FSMContext,
    services: Services,
) -> None:
    await callback.answer()
    message = callback_message(callback)
    if message is None:
        await state.clear()
        return

    if callback_data.action is TrackIntakeAction.CANCEL:
        if callback_data.buffer_version != services.chat_message_buffer.version(message.chat.id):
            await handle_stale_selection(message=message, state=state)
            return
        await state.clear()
        await message.edit_text(
            **selected_text(selected='Cancel'),
            reply_markup=None,
        )
        services.chat_message_buffer.flush(message.chat.id)


async def try_dispatch_track_intake(
    *,
    message: Message,
    services: Services,
    settings: Settings,
) -> bool:
    track_count = len(
        [
            buffered_message
            for buffered_message in services.chat_message_buffer.peek(message.chat.id)
            if buffered_message.audio is not None
        ]
    )
    if track_count == 0:
        return False

    await message.answer(
        **_track_intake_menu_kwargs(
            track_count=track_count,
            message_width=settings.message_width,
            buffer_version=services.chat_message_buffer.version(message.chat.id),
        )
    )
    return True


def _track_entry_reply_markup():
    return stacked_keyboard(
        buttons=[
            dummy_button(),
            dummy_button(),
            InlineKeyboardButton(
                text='Cancel',
                callback_data=RetrieveEntryCallbackData(action=RetrieveEntryAction.CANCEL).pack(),
            ),
        ]
    )


def _track_intake_menu_kwargs(
    *,
    track_count: int,
    message_width: int,
    buffer_version: int,
) -> dict[str, Any]:
    return {
        **Text(
            'Tracks: ',
            Bold(str(track_count)),
            '\n',
            create_padding_line(message_width),
            '\n',
            'Select action:',
        ).as_kwargs(),
        'reply_markup': stacked_keyboard(
            buttons=[
                dummy_button(),
                dummy_button(),
                InlineKeyboardButton(
                    text='Cancel',
                    callback_data=TrackIntakeActionCallbackData(
                        action=TrackIntakeAction.CANCEL,
                        buffer_version=buffer_version,
                    ).pack(),
                ),
            ]
        ),
    }
