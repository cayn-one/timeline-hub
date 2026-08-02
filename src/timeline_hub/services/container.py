from dataclasses import dataclass

from timeline_hub.infra.tasks import TaskScheduler
from timeline_hub.services.clip_store import ClipStore
from timeline_hub.services.message_buffer import ChatMessageBuffer
from timeline_hub.services.track_store import TrackStore
from timeline_hub.services.youtube_cookies import YoutubeCookieStore


@dataclass(frozen=True, slots=True)
class Services:
    chat_message_buffer: ChatMessageBuffer
    task_scheduler: TaskScheduler
    clip_store: ClipStore
    track_store: TrackStore
    youtube_cookie_store: YoutubeCookieStore
