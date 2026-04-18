from collections.abc import Sequence
from enum import StrEnum, auto
from typing import Any

from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.formatting import Bold, Text

from timeline_hub.handlers.menu import ensure_three_rows, three_row_keyboard
from timeline_hub.services.clip_store import Scope, Season, StoreResult, SubSeason, Universe

FLOW_GET = 'get'
FLOW_PULL = 'pull'
FLOW_RECONCILE = 'reconcile'
FLOW_STORE = 'store'
BACK_CALLBACK_VALUE = 'back'
NONE_CALLBACK_VALUE = SubSeason.NONE.value
ALL_SCOPES_CALLBACK_VALUE = 'all'
UNSET = object()


class MenuAction(StrEnum):
    SELECT = auto()
    BACK = auto()


class MenuStep(StrEnum):
    UNIVERSE = auto()
    YEAR = auto()
    SEASON = auto()
    SUB_SEASON = auto()
    SCOPE = auto()


class RetrieveClipFlow(StatesGroup):
    universe = State()
    year = State()
    season = State()
    sub_season = State()
    scope = State()


class StoreClipFlow(StatesGroup):
    universe = State()
    year = State()
    season = State()
    sub_season = State()
    scope = State()


class ReconcileClipFlow(StatesGroup):
    sub_season = State()
    scope = State()


RETRIEVE_STATE_BY_STEP = {
    MenuStep.UNIVERSE: RetrieveClipFlow.universe,
    MenuStep.YEAR: RetrieveClipFlow.year,
    MenuStep.SEASON: RetrieveClipFlow.season,
    MenuStep.SUB_SEASON: RetrieveClipFlow.sub_season,
    MenuStep.SCOPE: RetrieveClipFlow.scope,
}
STORE_STATE_BY_STEP = {
    MenuStep.UNIVERSE: StoreClipFlow.universe,
    MenuStep.YEAR: StoreClipFlow.year,
    MenuStep.SEASON: StoreClipFlow.season,
    MenuStep.SUB_SEASON: StoreClipFlow.sub_season,
    MenuStep.SCOPE: StoreClipFlow.scope,
}
RECONCILE_STATE_BY_STEP = {
    MenuStep.SUB_SEASON: ReconcileClipFlow.sub_season,
    MenuStep.SCOPE: ReconcileClipFlow.scope,
}


async def download_video_bytes(bot: Bot, *, file_id: str) -> bytes:
    file = await bot.get_file(file_id)
    if file.file_path is None:
        raise ValueError(f'Telegram file {file_id} has no downloadable path')
    buffer = await bot.download_file(file.file_path)
    if buffer is None:
        raise RuntimeError(f'Telegram file download returned no content for {file_id}')
    return buffer.read()


def selection_labels(
    *,
    universe: Universe | object = UNSET,
    year: int | object = UNSET,
    season: Season | object = UNSET,
    sub_season: SubSeason | object = UNSET,
    scope: Scope | str | object = UNSET,
) -> list[str]:
    labels: list[str] = []

    if universe is not UNSET:
        labels.append(format_selection_value(universe))
    if year is not UNSET:
        labels.append(format_selection_value(year))
    if season is not UNSET:
        labels.append(format_selection_value(season))
    if sub_season is not UNSET and sub_season != SubSeason.NONE:
        labels.append(format_selection_value(sub_season))
    if scope is not UNSET:
        labels.append(format_selection_value(scope))

    return labels


def format_selection_value(value: int | Season | Universe | SubSeason | Scope | str | object) -> str:
    if isinstance(value, Season):
        return str(int(value))
    if isinstance(value, (Universe, Scope, SubSeason)):
        return value.value.title()
    return str(value)


def special_top_selection_keyboard(
    *,
    buttons: Sequence[InlineKeyboardButton],
    back_button: InlineKeyboardButton,
    special_top_button: InlineKeyboardButton,
) -> InlineKeyboardMarkup:
    top_row, middle_row, bottom_row = ensure_three_rows(
        top_row=[special_top_button],
        middle_row=list(buttons),
        bottom_row=[back_button],
    )
    return three_row_keyboard(
        top_row=top_row,
        middle_row=middle_row,
        bottom_row=bottom_row,
    )


def year_selection_keyboard(
    *,
    buttons: Sequence[InlineKeyboardButton],
    back_button: InlineKeyboardButton,
) -> InlineKeyboardMarkup:
    year_rows = _snake_rows(buttons)
    top_row, middle_row, bottom_row = ensure_three_rows(
        top_row=year_rows.top_row,
        middle_row=year_rows.bottom_row,
        bottom_row=[back_button],
    )
    return three_row_keyboard(
        top_row=top_row,
        middle_row=middle_row,
        bottom_row=bottom_row,
    )


def single_button_keyboard(*, button: InlineKeyboardButton) -> InlineKeyboardMarkup:
    top_row, middle_row, bottom_row = ensure_three_rows(
        top_row=[button],
        middle_row=[],
        bottom_row=[],
    )
    return three_row_keyboard(
        top_row=top_row,
        middle_row=middle_row,
        bottom_row=bottom_row,
    )


def split_sub_season_buttons(sub_seasons: Sequence[SubSeason]) -> tuple[list[SubSeason], SubSeason | None]:
    special_top = SubSeason.NONE if SubSeason.NONE in sub_seasons else None
    regular_buttons = [
        sub_season
        for sub_season in reversed(tuple(SubSeason))
        if sub_season in sub_seasons and sub_season != SubSeason.NONE
    ]
    return regular_buttons, special_top


def encode_sub_season(sub_season: SubSeason) -> str:
    return sub_season.value


def parse_year(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def parse_season(value: str) -> Season | None:
    try:
        return Season(int(value))
    except ValueError:
        return None


def parse_universe(value: str) -> Universe | None:
    try:
        return Universe(value)
    except ValueError:
        return None


def parse_sub_season(value: str) -> SubSeason | object:
    try:
        return SubSeason(value)
    except ValueError:
        return UNSET


def parse_scope(value: str) -> Scope | None:
    try:
        return Scope(value)
    except ValueError:
        return None


def format_store_summary(result: StoreResult) -> str:
    lines: list[str] = []
    if result.stored_count > 0:
        lines.append(f'Stored: {result.stored_count}')
    if result.duplicate_count > 0:
        lines.append(f'Deduplicated: {result.duplicate_count}')
    if not lines:
        return 'Nothing changed'
    return '\n'.join(lines)


def store_summary_kwargs(result: StoreResult) -> dict[str, Any]:
    summary = format_store_summary(result)
    if summary == 'Nothing changed':
        return {'text': summary}

    parts: list[Any] = []
    for index, line in enumerate(summary.splitlines()):
        if index > 0:
            parts.append('\n')
        label, value = line.split(': ', maxsplit=1)
        parts.extend([f'{label}: ', Bold(value)])
    return Text(*parts).as_kwargs()


async def set_flow_context(
    *,
    state: FSMContext,
    mode: str,
    menu_message_id: int,
    fsm_state: State,
    universe: Universe | object = UNSET,
    year: int | object = UNSET,
    season: Season | object = UNSET,
    sub_season: SubSeason | object = UNSET,
) -> None:
    existing_data = await state.get_data()
    groups = existing_data.get('groups')

    await state.clear()
    await state.set_state(fsm_state)

    data: dict[str, object] = {
        'mode': mode,
        'menu_message_id': menu_message_id,
    }
    if isinstance(groups, list):
        data['groups'] = groups
    if universe is not UNSET:
        data['universe'] = universe
    if year is not UNSET:
        data['year'] = year
    if season is not UNSET:
        data['season'] = season
    if sub_season is not UNSET:
        data['sub_season'] = sub_season

    await state.update_data(data)


class _TwoRowButtons:
    def __init__(
        self,
        *,
        top_row: Sequence[InlineKeyboardButton],
        bottom_row: Sequence[InlineKeyboardButton],
    ) -> None:
        self.top_row = list(top_row)
        self.bottom_row = list(bottom_row)


def _snake_rows(buttons: Sequence[InlineKeyboardButton]) -> _TwoRowButtons:
    button_list = list(buttons)
    top_row_size, bottom_row_size = _two_row_sizes(len(button_list))
    top_row: list[InlineKeyboardButton | None] = [None] * top_row_size
    bottom_row: list[InlineKeyboardButton | None] = [None] * bottom_row_size

    for button, (row_name, index) in zip(
        button_list,
        _snake_positions(top_row_size=top_row_size, bottom_row_size=bottom_row_size),
        strict=True,
    ):
        if row_name == 'top':
            top_row[index] = button
        else:
            bottom_row[index] = button

    return _TwoRowButtons(
        top_row=[button for button in top_row if button is not None],
        bottom_row=[button for button in bottom_row if button is not None],
    )


def _two_row_sizes(button_count: int) -> tuple[int, int]:
    if button_count < 0:
        raise ValueError('`button_count` must be >= 0')
    if button_count == 1:
        return 0, 1
    top_row_size = button_count // 2
    return top_row_size, button_count - top_row_size


def _snake_positions(*, top_row_size: int, bottom_row_size: int) -> list[tuple[str, int]]:
    positions: list[tuple[str, int]] = []
    for offset in range(max(top_row_size, bottom_row_size)):
        top_index = top_row_size - 1 - offset
        bottom_index = bottom_row_size - 1 - offset
        if offset % 2 == 0:
            if top_index >= 0:
                positions.append(('top', top_index))
            if bottom_index >= 0:
                positions.append(('bottom', bottom_index))
        else:
            if bottom_index >= 0:
                positions.append(('bottom', bottom_index))
            if top_index >= 0:
                positions.append(('top', top_index))
    return positions
