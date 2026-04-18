from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.types import InlineKeyboardButton, Message
from aiogram.utils.formatting import Bold, Text

from timeline_hub.handlers.clips.common import create_padding_line, dummy_button, stacked_keyboard
from timeline_hub.handlers.clips.ingest import try_dispatch_clip_intake
from timeline_hub.handlers.tracks.retrieve import (
    TrackIntakeAction,
    TrackIntakeActionCallbackData,
    try_dispatch_track_intake,
)
from timeline_hub.services.container import Services
from timeline_hub.settings import Settings

router = Router()


@router.message(F.chat.type == ChatType.PRIVATE, F.text | F.photo | F.audio | F.video)
async def on_buffered_relevant_message(
    message: Message,
    services: Services,
    settings: Settings,
) -> None:
    chat_id = message.chat.id
    services.chat_message_buffer.append(message, chat_id=chat_id)

    async def send_action_selection() -> None:
        buffered_messages = services.chat_message_buffer.peek(chat_id)
        video_count = len(
            [buffered_message for buffered_message in buffered_messages if buffered_message.video is not None]
        )
        audio_count = len(
            [buffered_message for buffered_message in buffered_messages if buffered_message.audio is not None]
        )

        if video_count >= 1 and audio_count == 0:
            handled = await try_dispatch_clip_intake(
                message=message,
                services=services,
                settings=settings,
            )
        elif audio_count >= 1 and video_count == 0:
            handled = await try_dispatch_track_intake(
                message=message,
                services=services,
                settings=settings,
            )
        elif video_count >= 1 and audio_count >= 1:
            services.chat_message_buffer.flush(chat_id)
            await message.answer(text="Can't dispatch mixed input")
            return
        else:
            handled = False

        if not handled:
            await _show_fallback_menu(
                message=message,
                message_count=len(buffered_messages),
                settings=settings,
                buffer_version=services.chat_message_buffer.version(chat_id),
            )

    services.task_scheduler.schedule(
        send_action_selection,
        key=chat_id,
        delay=settings.forward_batch_timeout,
    )


async def _show_fallback_menu(
    *,
    message: Message,
    message_count: int,
    settings: Settings,
    buffer_version: int,
) -> None:
    await message.answer(
        **Text(
            'Messages: ',
            Bold(str(message_count)),
            '\n',
            create_padding_line(settings.message_width),
            '\n',
            'Select action:',
        ).as_kwargs(),
        reply_markup=stacked_keyboard(
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
    )
