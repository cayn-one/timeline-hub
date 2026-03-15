import asyncio
from collections.abc import Iterator

import pytest
from loguru import logger

from general_bot.infra.tasks import TaskFailure, TaskSupervisor


@pytest.fixture
def error_records() -> Iterator[list[dict]]:
    records: list[dict] = []
    logger.remove()

    def append_slim_record(message) -> None:
        record = message.record
        records.append(
            {
                'message': record['message'],
                'level': record['level'].name,
                'extra': dict(record['extra']),
                'has_exception': record['exception'] is not None,
            }
        )

    sink_id = logger.add(append_slim_record, level='ERROR')
    try:
        yield records
    finally:
        logger.remove(sink_id)


@pytest.mark.asyncio
async def test_failure_logs_and_calls_on_failure_once(error_records: list[dict]) -> None:
    calls = 0
    received: TaskFailure | None = None
    on_failure_called = asyncio.Event()

    async def on_failure(failure: TaskFailure) -> None:
        nonlocal calls, received
        calls += 1
        received = failure
        on_failure_called.set()

    supervisor = TaskSupervisor(on_failure=on_failure)

    async def boom() -> None:
        raise ValueError('boom')

    supervisor.spawn(boom(), name='clip-job', context={'user_id': 1})

    await asyncio.wait_for(on_failure_called.wait(), timeout=1)
    await supervisor.wait()

    assert calls == 1
    assert received is not None
    assert received.name == 'clip-job'
    assert isinstance(received.exception, ValueError)
    assert received.context == {'user_id': 1}
    failure_logs = [record for record in error_records if record['message'] == 'Detached task failed']
    assert len(failure_logs) == 1
    assert failure_logs[0]['extra']['task'] == 'clip-job'
    assert failure_logs[0]['has_exception'] is True


@pytest.mark.asyncio
async def test_multiple_failures_trigger_failure_hook_once(error_records: list[dict]) -> None:
    calls = 0
    on_failure_called = asyncio.Event()

    async def on_failure(_: TaskFailure) -> None:
        nonlocal calls
        calls += 1
        on_failure_called.set()

    supervisor = TaskSupervisor(on_failure=on_failure)

    async def boom() -> None:
        raise RuntimeError('boom')

    supervisor.spawn(boom(), name='first')
    supervisor.spawn(boom(), name='second')

    await asyncio.wait_for(on_failure_called.wait(), timeout=1)
    await asyncio.sleep(0)
    await supervisor.wait()

    assert calls == 1
    failure_logs = [record for record in error_records if record['message'] == 'Detached task failed']
    assert len(failure_logs) == 2
    assert {record['extra']['task'] for record in failure_logs} == {'first', 'second'}
    assert all(record['has_exception'] for record in failure_logs)


@pytest.mark.asyncio
async def test_cancelled_task_does_not_log_failure_or_call_hook(error_records: list[dict]) -> None:
    calls = 0
    started = asyncio.Event()

    async def on_failure(_: TaskFailure) -> None:
        nonlocal calls
        calls += 1

    async def blocker() -> None:
        started.set()
        await asyncio.Event().wait()

    supervisor = TaskSupervisor(on_failure=on_failure)
    task = supervisor.spawn(blocker(), name='blocked')

    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    await supervisor.wait()

    assert task.cancelled()
    assert calls == 0
    assert len([record for record in error_records if record['message'] == 'Detached task failed']) == 0


@pytest.mark.asyncio
async def test_wait_does_not_disable_fail_fast(error_records: list[dict]) -> None:
    calls = 0
    on_failure_called = asyncio.Event()

    async def on_failure(_: TaskFailure) -> None:
        nonlocal calls
        calls += 1
        on_failure_called.set()

    supervisor = TaskSupervisor(on_failure=on_failure)

    async def boom() -> None:
        raise ValueError('fail-fast')

    supervisor.spawn(boom(), name='wait-case')
    await supervisor.wait()
    await asyncio.wait_for(on_failure_called.wait(), timeout=1)

    assert calls == 1
    failure_logs = [record for record in error_records if record['message'] == 'Detached task failed']
    assert len(failure_logs) == 1
    assert failure_logs[0]['extra']['task'] == 'wait-case'
    assert failure_logs[0]['has_exception'] is True


@pytest.mark.asyncio
async def test_cancel_all_cancels_pending_tasks() -> None:
    supervisor = TaskSupervisor()
    task_a = supervisor.spawn(asyncio.sleep(60), name='sleep-a')
    task_b = supervisor.spawn(asyncio.sleep(60), name='sleep-b')

    supervisor.cancel_all()
    await supervisor.wait()

    assert task_a.cancelled()
    assert task_b.cancelled()
