import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from aiogram import Bot
from aiogram.types import Message

from timeline_hub.infra.ffmpeg import probe_audio_sample_rate, to_opus
from timeline_hub.infra.images import normalize_cover_to_jpg
from timeline_hub.infra.ytdlp import TrackMetadata, YtDlpMetadataError, download_audio_as_opus
from timeline_hub.services.track_store import (
    Track,
    TrackGroup,
    TrackId,
    TrackInvalidAudioFormatError,
    TrackStore,
    UploadedVariant,
)
from timeline_hub.types import Extension, FileBytes, InvalidExtensionError


class TrackInputError(ValueError):
    pass


class TrackLinkDownloadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LinkOnlyTrackInput:
    url: str
    artists: tuple[str, ...] | None
    title: str | None

    @property
    def requires_metadata(self) -> bool:
        return self.artists is None or self.title is None


@dataclass(frozen=True, slots=True)
class DownloadedLinkAudioCover:
    audio: FileBytes
    cover: FileBytes
    metadata: TrackMetadata | None


@dataclass(frozen=True, slots=True)
class CoverLinkTrackInput:
    url: str
    artists: tuple[str, ...] | None
    title: str | None

    @property
    def requires_metadata(self) -> bool:
        return self.artists is None or self.title is None


@dataclass(frozen=True, slots=True)
class TrackAudioAttachment:
    file_id: str
    file_name: str | None


@dataclass(frozen=True, slots=True)
class UploadedMp3Attachment:
    file_id: str
    file_name: str


@dataclass(frozen=True, slots=True)
class ParsedUploadedVariant:
    attachment: UploadedMp3Attachment
    speed: float
    reverb: float


_SUPPORTED_DOCUMENT_AUDIO_EXTENSIONS = frozenset({'.wav', '.flac', '.opus'})


def extract_track_audio_attachment(message: Message) -> TrackAudioAttachment | None:
    audio = message.audio
    if audio is not None:
        return TrackAudioAttachment(file_id=audio.file_id, file_name=audio.file_name)

    document = getattr(message, 'document', None)
    if document is None:
        return None
    file_name = getattr(document, 'file_name', None)
    if not _is_supported_document_audio_filename(file_name):
        return None
    file_id = getattr(document, 'file_id', None)
    if not isinstance(file_id, str) or not file_id:
        return None
    return TrackAudioAttachment(file_id=file_id, file_name=file_name)


def _is_supported_document_audio_filename(file_name: str | None) -> bool:
    if file_name is None:
        return False
    return any(file_name.lower().endswith(suffix) for suffix in _SUPPORTED_DOCUMENT_AUDIO_EXTENSIONS)


def extract_uploaded_mp3_attachment(message: Message) -> UploadedMp3Attachment | None:
    audio = message.audio
    if audio is not None:
        file_name = getattr(audio, 'file_name', None)
        file_id = getattr(audio, 'file_id', None)
        if isinstance(file_name, str) and _is_uploaded_mp3_filename(file_name) and isinstance(file_id, str) and file_id:
            return UploadedMp3Attachment(file_id=file_id, file_name=file_name)

    document = getattr(message, 'document', None)
    if document is None:
        return None
    file_name = getattr(document, 'file_name', None)
    file_id = getattr(document, 'file_id', None)
    if not isinstance(file_name, str) or not _is_uploaded_mp3_filename(file_name):
        return None
    if not isinstance(file_id, str) or not file_id:
        return None
    return UploadedMp3Attachment(file_id=file_id, file_name=file_name)


def _is_uploaded_mp3_filename(file_name: str | None) -> bool:
    if not isinstance(file_name, str):
        return False
    if not file_name.lower().endswith(Extension.MP3.suffix):
        return False

    stem_parts = Path(file_name).stem.split('_')
    if len(stem_parts) != 2:
        return False
    try:
        int(stem_parts[0])
        int(stem_parts[1])
    except ValueError:
        return False
    return True


def extract_single_photo_audio_messages(messages: Sequence[Message]) -> tuple[Message, Message]:
    """Return exactly one photo message and one audio message, order-independent."""
    if len(messages) != 2:
        raise TrackInputError('Invalid input')

    photo_messages = [message for message in messages if message.photo is not None]
    audio_messages = [message for message in messages if extract_track_audio_attachment(message) is not None]
    if len(photo_messages) != 1 or len(audio_messages) != 1:
        raise TrackInputError('Invalid input')
    return photo_messages[0], audio_messages[0]


def extract_photo_messages_for_remove(messages: Sequence[Message]) -> tuple[Message, ...]:
    """Return one or more photo messages for remove actions."""
    if len(messages) < 1:
        raise TrackInputError('Invalid input')
    if any(message.photo is None for message in messages):
        raise TrackInputError('Invalid input')
    return tuple(messages)


def _validate_and_order_uploaded_variant_messages(
    uploaded_messages: Sequence[Message],
) -> tuple[ParsedUploadedVariant, ...]:
    """Validate uploaded variant messages fully before any file download."""
    parsed_variants: list[ParsedUploadedVariant] = []
    for uploaded_message in uploaded_messages:
        attachment = extract_uploaded_mp3_attachment(uploaded_message)
        if attachment is None:
            raise TrackInputError('Invalid input')
        speed, reverb = parse_uploaded_variant_filename(attachment.file_name)
        if not math.isfinite(speed) or speed <= 0.0:
            raise TrackInputError('Invalid input')
        if not math.isfinite(reverb) or reverb < 0.0:
            raise TrackInputError('Invalid input')
        parsed_variants.append(
            ParsedUploadedVariant(
                attachment=attachment,
                speed=speed,
                reverb=reverb,
            )
        )

    if not parsed_variants or len(parsed_variants) > 10:
        raise TrackInputError('Invalid input')

    ordered_variants = sorted(parsed_variants, key=lambda variant: variant.speed)
    previous_speed: float | None = None
    for variant in ordered_variants:
        if previous_speed is not None and variant.speed <= previous_speed:
            raise TrackInputError('Invalid input')
        previous_speed = variant.speed

    return tuple(ordered_variants)


def validate_uploaded_source_variant_batch(
    messages: Sequence[Message],
) -> tuple[Message, Message, tuple[ParsedUploadedVariant, ...]]:
    """Validate one ordered uploaded action batch before any file download.

    Expected shape:
        1. identity-bearing cover photo
        2. authoritative source Telegram audio
        3..N uploaded MP3 variants
    """
    if len(messages) < 3:
        raise TrackInputError('Invalid input')
    if any(message.text is not None for message in messages):
        raise TrackInputError('Invalid input')
    if any(message.video is not None or getattr(message, 'animation', None) is not None for message in messages):
        raise TrackInputError('Invalid input')

    photo_message = messages[0]
    if photo_message.photo is None:
        raise TrackInputError('Invalid input')
    if any(message.photo is not None for message in messages[1:]):
        raise TrackInputError('Invalid input')

    source_audio_message = messages[1]
    if source_audio_message.audio is None:
        raise TrackInputError('Invalid input')

    uploaded_messages = tuple(messages[2:])
    if any(extract_uploaded_mp3_attachment(message) is None for message in uploaded_messages):
        raise TrackInputError('Invalid input')

    return photo_message, source_audio_message, _validate_and_order_uploaded_variant_messages(uploaded_messages)


def extract_track_identity_from_photo_message(photo_message: Message) -> tuple[TrackGroup, TrackId]:
    """Decode linked-dot cover caption identity into `(group, track_id)`."""
    caption = photo_message.caption
    if caption is None or not caption or caption[0] != '·':
        raise TrackInputError('Invalid input')

    entities = photo_message.caption_entities
    if not entities:
        raise TrackInputError('Invalid input')

    link_url: str | None = None
    for entity in entities:
        entity_type = getattr(entity, 'type', None)
        if getattr(entity_type, 'value', entity_type) != 'text_link':
            continue
        if getattr(entity, 'offset', None) != 0 or getattr(entity, 'length', None) != 1:
            continue
        link_url = getattr(entity, 'url', None)
        break

    if not link_url or not isinstance(link_url, str):
        raise TrackInputError('Invalid input')
    if not link_url.startswith('https://'):
        raise TrackInputError('Invalid input')

    identity = link_url.removeprefix('https://')
    if identity.endswith('.com/'):
        identity = identity.removesuffix('.com/')
    elif identity.endswith('.com'):
        identity = identity.removesuffix('.com')
    else:
        raise TrackInputError('Invalid input')

    if not identity:
        raise TrackInputError('Invalid input')
    return TrackStore.string_to_track_identity(identity)


async def prepare_audio_from_message(*, bot: Bot, audio_message: Message) -> FileBytes:
    """Download one audio message and normalize to OPUS `FileBytes`."""
    attachment = extract_track_audio_attachment(audio_message)
    if attachment is None:
        raise TrackInputError('Invalid input')

    audio_bytes = await _download_file_bytes(bot=bot, file_id=attachment.file_id)
    try:
        audio_extension = Extension.try_from_filename(attachment.file_name)
        if audio_extension is Extension.OPUS:
            audio_opus = await _validate_raw_opus_audio_bytes(audio_bytes)
        else:
            audio_opus = await to_opus(audio_bytes)
    except TrackInvalidAudioFormatError:
        raise
    except Exception as error:
        raise TrackInputError("Can't process audio") from error

    return FileBytes(data=audio_opus, extension=Extension.OPUS)


def parse_uploaded_variant_filename(file_name: str) -> tuple[float, float]:
    """Parse exact `<int>_<int>.mp3` metadata from an uploaded variant filename."""
    if not _is_uploaded_mp3_filename(file_name):
        raise TrackInputError('Invalid input')

    stem_parts = Path(file_name).stem.split('_')
    if len(stem_parts) != 2:
        raise TrackInputError('Invalid input')

    try:
        speed_part = int(stem_parts[0])
        reverb_part = int(stem_parts[1])
    except ValueError as error:
        raise TrackInputError('Invalid input') from error
    return speed_part / 100, reverb_part / 10


async def prepare_uploaded_variant_from_parsed(
    *,
    bot: Bot,
    parsed_variant: ParsedUploadedVariant,
) -> UploadedVariant:
    """Download one validated uploaded MP3 message without transcoding it."""
    raw_bytes = await _download_file_bytes(bot=bot, file_id=parsed_variant.attachment.file_id)
    try:
        return UploadedVariant(
            speed=parsed_variant.speed,
            reverb=parsed_variant.reverb,
            audio=FileBytes(data=raw_bytes, extension=Extension.MP3),
        )
    except (InvalidExtensionError, ValueError) as error:
        raise TrackInputError('Invalid input') from error


def extract_store_messages(messages: Sequence[Message]) -> list[Message]:
    """Return store-relevant messages in original order."""
    return [
        message
        for message in messages
        if message.photo is not None or extract_track_audio_attachment(message) is not None
    ]


def track_count_from_store_messages(messages: Sequence[Message]) -> int:
    return len(extract_store_messages(messages)) // 2


def extract_audio_only_store_messages(messages: Sequence[Message]) -> tuple[Message | None, Message]:
    """Return validated audio-only store inputs as optional text + audio."""
    if len(messages) == 0 or len(messages) > 2:
        raise TrackInputError('Invalid input')
    if any(
        message.photo is not None or message.video is not None or getattr(message, 'animation', None) is not None
        for message in messages
    ):
        raise TrackInputError('Invalid input')
    text_messages = [message for message in messages if message.text is not None]
    audio_messages = [message for message in messages if extract_track_audio_attachment(message) is not None]
    if len(audio_messages) != 1 or len(text_messages) > 1:
        raise TrackInputError('Invalid input')
    audio_message = audio_messages[0]
    text_message = text_messages[0] if text_messages else None
    if len(messages) == 1:
        if text_message is not None:
            raise TrackInputError('Invalid input')
        if audio_message.caption is None:
            raise TrackInputError('Invalid input')
        return None, audio_message
    if len(messages) != 2 or text_message is None:
        raise TrackInputError('Invalid input')
    if audio_message.caption is not None:
        raise TrackInputError('Invalid input')
    return text_message, audio_message


def parse_audio_only_track_metadata(
    *, text_message: Message | None, audio_message: Message
) -> tuple[tuple[str, ...], str]:
    """Parse artists/title for audio-only store from a plain text message."""
    try:
        source = text_message.text if text_message is not None else audio_message.caption
        return _caption_to_artists_and_title(source)
    except TrackInputError as error:
        raise TrackInputError('Invalid input') from error


def validate_audio_only_store_input(messages: Sequence[Message]) -> tuple[tuple[str, ...], str]:
    """Validate audio-only store shape and metadata without downloading files."""
    text_message, audio_message = extract_audio_only_store_messages(messages)
    return parse_audio_only_track_metadata(text_message=text_message, audio_message=audio_message)


def is_supported_youtube_store_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    normalized_url = url.strip()
    if not normalized_url:
        return False
    parsed = urlparse(normalized_url)
    if parsed.scheme != 'https':
        return False
    if parsed.netloc not in ('www.youtube.com', 'music.youtube.com', 'youtu.be'):
        return False
    if parsed.netloc == 'youtu.be':
        path_segments = [segment for segment in parsed.path.split('/') if segment]
        return len(path_segments) == 1
    if parsed.path != '/watch':
        return False
    video_ids = parse_qs(parsed.query).get('v', [])
    return any(video_id.strip() for video_id in video_ids)


def parse_link_only_store_input(text: str) -> LinkOnlyTrackInput:
    if not isinstance(text, str):
        raise TrackInputError('Invalid input')

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise TrackInputError('Invalid input')
    url = lines[0]
    if not is_supported_youtube_store_url(url):
        raise TrackInputError('Invalid input')
    if len(lines) == 1:
        return LinkOnlyTrackInput(url=url, artists=None, title=None)
    if len(lines) == 2:
        raise TrackInputError('Invalid input')
    artists = tuple(lines[1:-1])
    title = lines[-1]
    if not artists or any(not artist for artist in artists) or not title:
        raise TrackInputError('Invalid input')
    return LinkOnlyTrackInput(url=url, artists=artists, title=title)


def validate_link_only_store_input(messages: Sequence[Message]) -> LinkOnlyTrackInput:
    if len(messages) != 1:
        raise TrackInputError('Invalid input')
    message = messages[0]
    if message.photo is not None or extract_track_audio_attachment(message) is not None:
        raise TrackInputError('Invalid input')
    if message.video is not None or getattr(message, 'animation', None) is not None:
        raise TrackInputError('Invalid input')
    if message.text is None:
        raise TrackInputError('Invalid input')
    return parse_link_only_store_input(message.text)


def extract_audio_link_store_messages(messages: Sequence[Message]) -> tuple[Message, Message]:
    """Return validated URL text + audio messages, order-independent."""
    if len(messages) != 2:
        raise TrackInputError('Invalid input')
    if any(
        message.photo is not None or message.video is not None or getattr(message, 'animation', None) is not None
        for message in messages
    ):
        raise TrackInputError('Invalid input')

    text_messages = [message for message in messages if message.text is not None]
    audio_messages = [message for message in messages if extract_track_audio_attachment(message) is not None]
    if len(text_messages) != 1 or len(audio_messages) != 1:
        raise TrackInputError('Invalid input')
    text_message = text_messages[0]
    audio_message = audio_messages[0]
    if audio_message.caption is not None:
        raise TrackInputError('Invalid input')
    return text_message, audio_message


def validate_audio_link_store_input(messages: Sequence[Message]) -> tuple[Message, LinkOnlyTrackInput]:
    """Validate one URL text + audio input and parse URL metadata lines."""
    text_message, audio_message = extract_audio_link_store_messages(messages)
    if text_message.text is None:
        raise TrackInputError('Invalid input')
    return audio_message, parse_link_only_store_input(text_message.text)


def parse_cover_link_store_input(messages: Sequence[Message]) -> CoverLinkTrackInput:
    if len(messages) != 1:
        raise TrackInputError('Invalid input')

    message = messages[0]
    if message.photo is None:
        raise TrackInputError('Invalid input')
    if (
        extract_track_audio_attachment(message) is not None
        or message.video is not None
        or getattr(message, 'animation', None) is not None
    ):
        raise TrackInputError('Invalid input')
    if message.caption is None:
        raise TrackInputError('Invalid input')
    parsed_link_input = parse_link_only_store_input(message.caption)

    return CoverLinkTrackInput(
        url=parsed_link_input.url,
        artists=parsed_link_input.artists,
        title=parsed_link_input.title,
    )


async def download_link_audio(url: str, *, max_duration: timedelta | None = None) -> FileBytes:
    try:
        result = await download_audio_as_opus(url, max_duration=max_duration)
    except Exception as error:
        raise TrackLinkDownloadError(str(error)) from error
    return FileBytes(data=result.audio, extension=Extension.OPUS)


async def download_link_audio_and_cover(
    url: str,
    *,
    with_metadata: bool = False,
    max_duration: timedelta | None = None,
) -> DownloadedLinkAudioCover:
    try:
        result = await download_audio_as_opus(
            url,
            with_cover=True,
            with_metadata=with_metadata,
            max_duration=max_duration,
        )
    except YtDlpMetadataError:
        raise
    except Exception as error:
        raise TrackLinkDownloadError(str(error)) from error
    if result.cover is None:
        raise TrackLinkDownloadError('yt-dlp did not produce cover output')
    return DownloadedLinkAudioCover(
        audio=FileBytes(data=result.audio, extension=Extension.OPUS),
        cover=FileBytes(data=result.cover, extension=Extension.JPG),
        metadata=result.metadata,
    )


def validate_track_batch(messages: Sequence[Message]) -> list[tuple[tuple[str, ...], str]]:
    if len(messages) < 2 or len(messages) % 2 != 0:
        raise TrackInputError("Can't dispatch input")

    parsed_tracks: list[tuple[tuple[str, ...], str]] = []
    for index in range(0, len(messages), 2):
        photo_message = messages[index]
        audio_message = messages[index + 1]
        if photo_message.photo is None or extract_track_audio_attachment(audio_message) is None:
            raise TrackInputError("Can't dispatch input")
        if photo_message.caption is None or not photo_message.caption.strip():
            raise TrackInputError("Can't dispatch input")

        try:
            parsed_tracks.append(_caption_to_artists_and_title(photo_message.caption))
        except TrackInputError as error:
            raise TrackInputError("Can't dispatch input") from error

    return parsed_tracks


async def prepare_tracks_from_buffer(*, bot: Bot, messages: Sequence[Message]) -> list[Track]:
    store_messages = extract_store_messages(messages)
    parsed_tracks = validate_track_batch(store_messages)
    prepared_tracks: list[Track] = []
    for parsed_track, index in zip(parsed_tracks, range(0, len(store_messages), 2), strict=True):
        photo_message = store_messages[index]
        audio_message = store_messages[index + 1]
        photo = photo_message.photo
        attachment = extract_track_audio_attachment(audio_message)
        if photo is None or attachment is None:
            raise TrackInputError("Can't dispatch input")

        artists, title = parsed_track
        cover_bytes = await _download_file_bytes(
            bot=bot,
            file_id=photo[-1].file_id,
        )
        audio_bytes = await _download_file_bytes(
            bot=bot,
            file_id=attachment.file_id,
        )

        try:
            cover_jpg = normalize_cover_to_jpg(cover_bytes)
        except Exception as error:
            raise TrackInputError("Can't process cover image") from error

        try:
            # Best-effort extension parse (filename may be missing or invalid).
            audio_extension = Extension.try_from_filename(attachment.file_name)
            if audio_extension is Extension.OPUS:
                # Fast-path: avoid re-encoding already-Opus input.
                audio_opus = await _validate_raw_opus_audio_bytes(audio_bytes)
            else:
                audio_opus = await to_opus(audio_bytes)
        except TrackInvalidAudioFormatError:
            raise
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


async def _validate_raw_opus_audio_bytes(audio_bytes: bytes) -> bytes:
    """Validate raw Opus source sample rate without re-encoding valid input."""
    sample_rate = await probe_audio_sample_rate(audio_bytes)
    if sample_rate != 48_000:
        raise TrackInvalidAudioFormatError(f'Audio sample rate must be 48000 Hz, got {sample_rate}')
    return audio_bytes


async def prepare_audio_only_track_from_buffer(
    *,
    bot: Bot,
    messages: Sequence[Message],
    album_id: str,
) -> Track:
    """Prepare one store-ready track for audio-only + text metadata flows."""
    text_message, audio_message = extract_audio_only_store_messages(messages)
    artists, title = parse_audio_only_track_metadata(text_message=text_message, audio_message=audio_message)
    audio = await prepare_audio_from_message(bot=bot, audio_message=audio_message)
    return Track(
        artists=artists,
        title=title,
        audio=audio,
        cover=None,
        album_id=album_id,
    )


def prepare_link_only_track_from_buffer(
    *,
    messages: Sequence[Message],
) -> LinkOnlyTrackInput:
    return validate_link_only_store_input(messages)


def prepare_audio_link_track_from_buffer(
    *,
    messages: Sequence[Message],
) -> tuple[Message, LinkOnlyTrackInput]:
    return validate_audio_link_store_input(messages)


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
