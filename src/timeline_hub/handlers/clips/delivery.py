from collections.abc import AsyncIterator, Callable, Sequence

from aiogram import Bot
from aiogram.types import BufferedInputFile, InputMediaVideo

from timeline_hub.services.clip_store import (
    AudioNormalization,
    ClipGroup,
    ClipId,
    ClipStore,
    ClipSubGroup,
    FetchedClip,
    Universe,
)
from timeline_hub.settings import Settings
from timeline_hub.types import ChatId, Extension


async def send_fetched_clip_batches(
    *,
    bot: Bot,
    chat_id: ChatId,
    group: ClipGroup,
    sub_group: ClipSubGroup,
    clip_batches: AsyncIterator[tuple[FetchedClip, ...]],
) -> None:
    await _send_fetched_clip_batches(
        bot=bot,
        chat_id=chat_id,
        clip_batches=clip_batches,
        filename_for_clip=lambda clip_id: _fetched_clip_filename(group, sub_group, clip_id),
    )


async def send_fetched_inbox_clip_batches(
    *,
    bot: Bot,
    chat_id: ChatId,
    universe: Universe,
    clip_batches: AsyncIterator[tuple[FetchedClip, ...]],
) -> None:
    """Send streamed inbox batches without prefetching later batches."""
    await _send_fetched_clip_batches(
        bot=bot,
        chat_id=chat_id,
        clip_batches=clip_batches,
        filename_for_clip=lambda clip_id: _inbox_fetched_clip_filename(universe, clip_id),
    )


async def send_fetched_clip_batch(
    *,
    bot: Bot,
    chat_id: ChatId,
    group: ClipGroup,
    sub_group: ClipSubGroup,
    clips: Sequence[FetchedClip],
) -> None:
    await _send_fetched_clip_batch(
        bot=bot,
        chat_id=chat_id,
        clips=clips,
        filename_for_clip=lambda clip_id: _fetched_clip_filename(group, sub_group, clip_id),
    )


def audio_normalization_from_settings(*, settings: Settings) -> AudioNormalization:
    return AudioNormalization(
        loudness=settings.normalization_loudness,
        bitrate=settings.normalization_bitrate,
    )


async def _send_fetched_clip_batches(
    *,
    bot: Bot,
    chat_id: ChatId,
    clip_batches: AsyncIterator[tuple[FetchedClip, ...]],
    filename_for_clip: Callable[[ClipId], str],
) -> None:
    async for batch in clip_batches:
        await _send_fetched_clip_batch(
            bot=bot,
            chat_id=chat_id,
            clips=batch,
            filename_for_clip=filename_for_clip,
        )


async def _send_fetched_clip_batch(
    *,
    bot: Bot,
    chat_id: ChatId,
    clips: Sequence[FetchedClip],
    filename_for_clip: Callable[[ClipId], str],
) -> None:
    if not clips:
        raise ValueError('`clips` must not be empty')

    if len(clips) == 1:
        clip = clips[0]
        await bot.send_video(
            chat_id=chat_id,
            video=BufferedInputFile(clip.file.data, filename=filename_for_clip(clip.id)),
        )
        return

    await bot.send_media_group(
        chat_id=chat_id,
        media=[
            InputMediaVideo(
                media=BufferedInputFile(clip.file.data, filename=filename_for_clip(clip.id)),
            )
            for clip in clips
        ],
    )


def _fetched_clip_filename(group: ClipGroup, sub_group: ClipSubGroup, clip_id: str) -> str:
    identity = ClipStore.clip_identity_to_string(group, clip_id)
    return f'{identity}{Extension.MP4.suffix}'


def _inbox_fetched_clip_filename(universe: Universe, clip_id: ClipId) -> str:
    identity = ClipStore.inbox_clip_identity_to_string(universe, clip_id)
    return f'{identity}{Extension.MP4.suffix}'
