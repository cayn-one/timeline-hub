import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta

from aiogram.types import Message, User

from general_bot.task_supervisor import TaskSupervisor
from general_bot.types import UserId

# Function that returns a coroutine when called.
# Example: lambda: send_message(user_id)
type Job = Callable[[], Awaitable[None]]

type Messages = list[Message]
type MessageGroup = tuple[Message, ...]
type MessageGroups = list[MessageGroup]


class TaskScheduler:
    """Per-user delayed task scheduler with debounce semantics.

    Each user may have at most one pending timer. Calling `schedule()` cancels
    the previous timer and schedules `job()` to run after `delay`.

    If scheduling occurs again before the delay elapses, the previous timer is
    discarded and only the most recent job will run.

    Once the job starts executing it is shielded from cancellation and allowed
    to run to completion.
    """

    def __init__(self, task_supervisor: TaskSupervisor) -> None:
        self._tasks: dict[UserId, asyncio.Task[None]] = {}
        self._generation: dict[UserId, int] = {}
        self._task_supervisor = task_supervisor

    def schedule(self, job: Job, *, user: User, delay: timedelta) -> None:
        self.cancel(user)
        self._generation[user.id] = self._generation.get(user.id, 0) + 1
        self._tasks[user.id] = self._task_supervisor.spawn(
            self._delayed(user.id, job, self._generation[user.id], delay),
        )

    def cancel(self, user: User) -> None:
        if task := self._tasks.pop(user.id, None):
            task.cancel()

    async def _delayed(self, user_id: UserId, job: Job, generation: int, delay: timedelta) -> None:
        try:
            await asyncio.sleep(delay.total_seconds())
        except asyncio.CancelledError:
            return
        if self._generation.get(user_id) != generation:
            return

        # Once real task started, it can't be canceled. So remove it from scheduler
        _ = self._tasks.pop(user_id, None)
        try:
            await asyncio.shield(job())
        except asyncio.CancelledError:
            return


class MessageBuffer:
    def __init__(self) -> None:
        self._messages: dict[UserId, Messages] = {}

    def append(self, message: Message, *, user: User) -> None:
        self._messages.setdefault(user.id, []).append(message)

    def peek(self, user: User) -> Messages:
        return list(self._messages.get(user.id, []))

    def flush(self, user: User) -> Messages:
        return self._messages.pop(user.id, [])

    def flush_grouped(self, user: User) -> MessageGroups:
        return self._group(self.flush(user))

    @staticmethod
    def _group(messages: Messages) -> MessageGroups:
        groups: list[Messages] = []
        ordered_messages = sorted(messages, key=lambda m: m.message_id)

        for message in ordered_messages:
            if not groups:
                groups.append([message])
                continue
            if message.media_group_id is not None and message.media_group_id == groups[-1][-1].media_group_id:
                groups[-1].append(message)
            else:
                groups.append([message])

        return [tuple(group) for group in groups]


@dataclass(frozen=True, slots=True)
class Services:
    task_scheduler: TaskScheduler
    message_buffer: MessageBuffer
