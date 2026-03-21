from dataclasses import dataclass

from general_bot.infra.tasks import TaskScheduler
from general_bot.services.clip_store import ClipStore
from general_bot.services.message_buffer import ChatMessageBuffer


@dataclass(frozen=True, slots=True)
class Services:
    chat_message_buffer: ChatMessageBuffer
    task_scheduler: TaskScheduler
    clip_store: ClipStore
