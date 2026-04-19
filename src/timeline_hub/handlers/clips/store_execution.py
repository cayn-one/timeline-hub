from aiogram import Bot
from aiogram.types import Message
from loguru import logger

from timeline_hub.handlers.clips.common import download_video_bytes, store_summary_kwargs
from timeline_hub.handlers.clips.delivery import audio_normalization_from_settings, send_fetched_clip_batches
from timeline_hub.services.clip_store import ClipGroup, ClipSubGroup, Scope, StoreResult
from timeline_hub.services.container import Services
from timeline_hub.services.message_buffer import MessageGroup
from timeline_hub.settings import Settings
from timeline_hub.types import ChatId, Extension, FileBytes

_TELEGRAM_MEDIA_GROUP_LIMIT = 10


async def execute_store_or_produce(
    *,
    bot: Bot,
    message: Message,
    services: Services,
    settings: Settings,
    clip_group: ClipGroup,
    clip_sub_group: ClipSubGroup,
    produce: bool,
) -> StoreResult:
    result = await _store_buffered_clips(
        bot=bot,
        chat_id=message.chat.id,
        services=services,
        clip_group=clip_group,
        clip_sub_group=clip_sub_group,
    )

    await message.answer(**store_summary_kwargs(result))

    if result.stored_count > 0 and _should_compact_after_store(clip_sub_group.scope):
        try:
            await services.clip_store.compact(
                clip_group,
                clip_sub_group,
                batch_size=_TELEGRAM_MEDIA_GROUP_LIMIT,
            )
        except Exception:
            logger.exception(
                'Post-store clip compaction failed for {} {}',
                clip_group,
                clip_sub_group,
            )
            raise

    if produce and result.stored_count > 0:
        await send_fetched_clip_batches(
            bot=bot,
            chat_id=message.chat.id,
            group=clip_group,
            sub_group=clip_sub_group,
            clip_batches=services.clip_store.fetch(
                clip_group,
                clip_sub_group,
                clip_ids=result.clip_ids,
                audio_normalization=audio_normalization_from_settings(settings=settings),
            ),
        )
        await bot.send_message(chat_id=message.chat.id, text='Done')

    return result


async def _store_buffered_clips(
    *,
    bot: Bot,
    chat_id: ChatId,
    services: Services,
    clip_group: ClipGroup,
    clip_sub_group: ClipSubGroup,
) -> StoreResult:
    result = StoreResult(stored_count=0, duplicate_count=0)
    message_groups = services.chat_message_buffer.peek_grouped(chat_id)
    services.chat_message_buffer.flush(chat_id)

    for message_group in message_groups:
        clip_file_batch = await _message_group_to_clip_files(bot=bot, message_group=message_group)
        if not clip_file_batch:
            continue
        result += await services.clip_store.store(
            clip_group,
            clip_sub_group,
            clips=clip_file_batch,
        )

    return result


async def _message_group_to_clip_files(
    *,
    bot: Bot,
    message_group: MessageGroup,
) -> list[FileBytes]:
    clips: list[FileBytes] = []

    for message in message_group:
        if message.video is None:
            continue
        clips.append(
            FileBytes(
                data=await download_video_bytes(bot, file_id=message.video.file_id),
                extension=Extension.MP4,
            )
        )

    return clips


def _should_compact_after_store(scope: Scope) -> bool:
    """Return the handler-level post-store compaction policy for intake flows.

    Intake decides whether to compact after storing. `Scope.COLLECTION` keeps
    its original stored grouping, while `Scope.EXTRA` and `Scope.SOURCE`
    compact after store. `Produce` must follow the same post-store compaction
    policy as `Store`.

    `ClipStore.fetch()` only reflects the current manifest layout; it does not
    decide whether compaction should happen.
    """
    return scope in {Scope.EXTRA, Scope.SOURCE}
