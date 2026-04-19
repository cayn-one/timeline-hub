from aiogram.types import Message

from timeline_hub.types import ChatId

type Messages = list[Message]
type MessageGroup = tuple[Message, ...]
type MessageGroups = list[MessageGroup]


class ChatMessageBuffer:
    """Chat-scoped buffer for incoming Telegram messages.

    Messages are stored by `chat_id`. `peek_raw()` exposes a non-destructive
    append-order view, `peek_flat()` exposes authoritative `message_id` order,
    and `peek_grouped()` exposes a grouped view derived from that authoritative
    order. `flush()` is the only destructive consume operation and returns no
    buffered data.

    Note:
        In Telegram private chats, `chat_id` is equal to the sender's
        `user_id`. Therefore either identifier may be used as the key
        when the bot operates exclusively in personal chats.
    """

    def __init__(self) -> None:
        self._messages: dict[ChatId, Messages] = {}
        self._versions: dict[ChatId, int] = {}

    def append(self, message: Message, *, chat_id: ChatId) -> None:
        self._messages.setdefault(chat_id, []).append(message)
        self._bump_version(chat_id)

    def peek_raw(self, chat_id: ChatId) -> Messages:
        """Return buffered messages in non-destructive append order."""
        return list(self._messages.get(chat_id, []))

    def peek_flat(self, chat_id: ChatId) -> Messages:
        """Return buffered messages flattened and sorted by `message_id` (authoritative order)."""
        messages = self.peek_raw(chat_id)
        return sorted(messages, key=lambda message: message.message_id)

    def peek_grouped(self, chat_id: ChatId) -> MessageGroups:
        """Return grouped buffered messages derived from authoritative `message_id` order."""
        return self._group(self.peek_raw(chat_id))

    def version(self, chat_id: ChatId) -> int:
        return self._versions.get(chat_id, 0)

    def flush(self, chat_id: ChatId) -> None:
        messages = self._messages.pop(chat_id, [])
        if messages:
            self._bump_version(chat_id)

    def _bump_version(self, chat_id: ChatId) -> None:
        self._versions[chat_id] = self.version(chat_id) + 1

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
