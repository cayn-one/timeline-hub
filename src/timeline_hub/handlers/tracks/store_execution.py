from collections.abc import Sequence

from aiogram import Bot
from aiogram.types import Message

from timeline_hub.infra.ffmpeg import to_opus
from timeline_hub.infra.images import to_jpg
from timeline_hub.services.track_store import Track
from timeline_hub.types import Extension, FileBytes


class TrackInputError(ValueError):
    pass


def validate_track_batch(messages: Sequence[Message]) -> list[tuple[tuple[str, ...], str]]:
    if len(messages) < 2 or len(messages) % 2 != 0:
        raise TrackInputError("Can't dispatch input")

    parsed_tracks: list[tuple[tuple[str, ...], str]] = []
    for index in range(0, len(messages), 2):
        photo_message = messages[index]
        audio_message = messages[index + 1]
        if photo_message.photo is None or audio_message.audio is None:
            raise TrackInputError("Can't dispatch input")
        if photo_message.caption is None or not photo_message.caption.strip():
            raise TrackInputError("Can't dispatch input")

        parsed_tracks.append(_caption_to_artists_and_title(photo_message.caption))

    return parsed_tracks


async def prepare_tracks_from_buffer(*, bot: Bot, messages: Sequence[Message]) -> list[Track]:
    parsed_tracks = validate_track_batch(messages)
    prepared_tracks: list[Track] = []
    for parsed_track, index in zip(parsed_tracks, range(0, len(messages), 2), strict=True):
        photo_message = messages[index]
        audio_message = messages[index + 1]
        photo = photo_message.photo
        audio = audio_message.audio
        if photo is None or audio is None:
            raise TrackInputError("Can't dispatch input")

        artists, title = parsed_track
        cover_bytes = await _download_file_bytes(
            bot=bot,
            file_id=photo[-1].file_id,
        )
        audio_bytes = await _download_file_bytes(
            bot=bot,
            file_id=audio.file_id,
        )

        try:
            cover_jpg = to_jpg(cover_bytes)
        except Exception as error:
            raise TrackInputError("Can't process cover image") from error

        try:
            audio_opus = await to_opus(audio_bytes)
        except Exception as error:
            raise TrackInputError("Can't process audio") from error

        prepared_tracks.append(
            Track(
                artists=artists,
                title=title,
                cover=FileBytes(data=cover_jpg, extension=Extension.JPG),
                audio=FileBytes(data=audio_opus, extension=Extension.OPUS),
            )
        )

    return prepared_tracks


def _caption_to_artists_and_title(caption: str | None) -> tuple[tuple[str, ...], str]:
    lines = [line.strip() for line in (caption or '').splitlines() if line.strip()]
    if len(lines) < 2:
        raise TrackInputError('Not enough lines to extract artists and title')
    return tuple(lines[:-1]), lines[-1]


async def _download_file_bytes(*, bot: Bot, file_id: str) -> bytes:
    telegram_file = await bot.get_file(file_id)
    if telegram_file.file_path is None:
        raise TrackInputError("Can't dispatch input")

    downloaded = await bot.download_file(telegram_file.file_path)
    if downloaded is None:
        raise TrackInputError("Can't dispatch input")

    return downloaded.read()
