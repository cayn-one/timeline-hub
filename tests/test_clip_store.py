import asyncio
import json
import re
import uuid

import pytest
from async_s3 import S3Client, S3ObjectNotFoundError

import timeline_hub.services.clip_store as clip_store_module
from timeline_hub.infra.ffmpeg import PerceptualMetadataUnavailableError, UnsupportedVideoCodecError
from timeline_hub.services.clip_store import (
    AudioNormalization,
    ClipGroup,
    ClipGroupNotFoundError,
    ClipIdsNotInSubGroupError,
    ClipInfo,
    ClipManifestSyncError,
    ClipRemoveManifestSyncError,
    ClipStore,
    ClipSubGroup,
    DuplicateClipIdsError,
    FetchedClip,
    InvalidClipIdentityError,
    Manifest,
    ManifestCorruptedError,
    ManifestEntry,
    NormalizedClipManifestSyncError,
    ReconcileDeleteError,
    ReconcileResult,
    Scope,
    Season,
    StoreResult,
    SubSeason,
    Universe,
    UnknownClipsError,
)
from timeline_hub.types import Extension, FileBytes, InvalidExtensionError

_UUID_1 = uuid.UUID('018f05c1-f1a3-7b34-8d29-1f53a1c9d0e1').hex
_UUID_2 = uuid.UUID('018f05c1-f1a3-7b34-8d29-1f53a1c9d0e2').hex
_UUID_3 = uuid.UUID('018f05c1-f1a3-7b34-8d29-1f53a1c9d0e3').hex
_UUID_4 = uuid.UUID('018f05c1-f1a3-7b34-8d29-1f53a1c9d0e4').hex
_UUID_5 = uuid.UUID('018f05c1-f1a3-7b34-8d29-1f53a1c9d0e5').hex
_HASH_A = 'a' * 64
_HASH_B = 'b' * 64
_HASH_C = 'c' * 64
_HASH_D = 'd' * 64
_HASH_E = 'e' * 64
_CLIP_NAMESPACE = 'clips'
_FRAME_COUNT = 3
_SAMPLED_PHASHES = (11, 22, 33)


def test_store_result_adds_counts() -> None:
    assert StoreResult(stored_count=1, duplicate_count=2, clip_ids=(_UUID_1, _UUID_2)) + StoreResult(
        stored_count=3,
        duplicate_count=4,
        clip_ids=(_UUID_3,),
    ) == StoreResult(
        stored_count=4,
        duplicate_count=6,
        clip_ids=(_UUID_1, _UUID_2, _UUID_3),
    )


@pytest.mark.parametrize(
    ('kwargs', 'expected_message'),
    [
        ({'loudness': True, 'bitrate': 128}, '`loudness` must be a numeric value'),
        ({'loudness': 'loud', 'bitrate': 128}, '`loudness` must be a numeric value'),
        ({'loudness': float('inf'), 'bitrate': 128}, '`loudness` must be finite'),
        ({'loudness': float('nan'), 'bitrate': 128}, '`loudness` must be finite'),
        ({'loudness': -14, 'bitrate': True}, '`bitrate` must be an integer'),
        ({'loudness': -14, 'bitrate': 128.0}, '`bitrate` must be an integer'),
        ({'loudness': -14, 'bitrate': 0}, '`bitrate` must be >= 1'),
    ],
)
def test_audio_normalization_rejects_invalid_values(
    kwargs: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        AudioNormalization(**kwargs)


def test_season_from_month_uses_exact_mapping() -> None:
    assert Season.from_month(2) is Season.S1
    assert Season.from_month(3) is Season.S2
    assert Season.from_month(6) is Season.S3
    assert Season.from_month(9) is Season.S4
    assert Season.from_month(12) is Season.S5


def test_previous_clip_group_rolls_over_to_prior_year_s5() -> None:
    store = _store(_FakeS3Client())

    assert store._previous_clip_group(ClipGroup(universe=Universe.WEST, year=2026, season=Season.S1)) == ClipGroup(
        universe=Universe.WEST,
        year=2025,
        season=Season.S5,
    )


def test_next_clip_group_rolls_over_to_next_year_s1() -> None:
    store = _store(_FakeS3Client())

    assert store._next_clip_group(ClipGroup(universe=Universe.EAST, year=2025, season=Season.S5)) == ClipGroup(
        universe=Universe.EAST,
        year=2026,
        season=Season.S1,
    )


def test_sub_season_exists_property() -> None:
    assert SubSeason.NONE.exists is False
    assert SubSeason.A.exists is True
    assert SubSeason.B.exists is True


def test_clip_identity_to_string_returns_expected_identity_string() -> None:
    group = ClipGroup(universe=Universe.WEST, year=2026, season=Season.S1)
    assert ClipStore.clip_identity_to_string(group, _UUID_1) == f'west-2026-1--{_UUID_1}'


def test_clip_identity_to_string_canonicalizes_valid_uuid7_input() -> None:
    group = ClipGroup(universe=Universe.WEST, year=2026, season=Season.S1)
    non_canonical_uuid = str(uuid.UUID(_UUID_1)).upper()

    assert ClipStore.clip_identity_to_string(group, non_canonical_uuid) == f'west-2026-1--{_UUID_1}'


@pytest.mark.parametrize(
    'clip_id',
    [
        'not-a-uuid',
        uuid.UUID(int=0x1234).hex,
    ],
)
def test_clip_identity_to_string_rejects_invalid_clip_ids(clip_id: str) -> None:
    group = ClipGroup(universe=Universe.WEST, year=2026, season=Season.S1)

    with pytest.raises(ValueError, match='must be a valid UUID|must be a UUIDv7'):
        ClipStore.clip_identity_to_string(group, clip_id)


def test_clip_identity_string_round_trips() -> None:
    group = ClipGroup(universe=Universe.EAST, year=2027, season=Season.S4)
    identity = ClipStore.clip_identity_to_string(group, _UUID_1)

    assert ClipStore.string_to_clip_identity(identity) == (group, _UUID_1)


@pytest.mark.parametrize(
    ('identity', 'message'),
    [
        ('west-2026-1-018f05c1f1a37b348d291f53a1c9d0e1', "exactly one '--' separator"),
        ('west-2026-1', "exactly one '--' separator"),
        (f'west-2026-1--{_UUID_1}--extra', "exactly one '--' separator"),
        (f'west-2026--{_UUID_1}', 'malformed group segment'),
        (f'west-2026-1-extra--{_UUID_1}', 'malformed group segment'),
        (f'north-2026-1--{_UUID_1}', 'unsupported universe'),
        (f'west-year-1--{_UUID_1}', 'invalid year'),
        (f'west-2026-9--{_UUID_1}', 'invalid season'),
        ('west-2026-1--not-a-uuid', 'must be a valid UUID'),
        (f'west-2026-1--{uuid.UUID(int=0x1234).hex}', 'must be a UUIDv7'),
        (f'west-2026-1--{_UUID_1}/clip', 'must not contain path separators'),
        (f'west-2026-1--{_UUID_1}.mp4', 'must not contain extensions'),
    ],
)
def test_string_to_clip_identity_rejects_malformed_values(identity: str, message: str) -> None:
    with pytest.raises(InvalidClipIdentityError, match=re.escape(message)):
        ClipStore.string_to_clip_identity(identity)


class _FakeS3Client:
    def __init__(
        self,
        objects: dict[str, bytes] | None = None,
        *,
        delete_failures: set[str] | None = None,
        put_failures: set[str] | None = None,
        prefixes: list[str] | None = None,
    ) -> None:
        self.objects = dict(objects or {})
        self.delete_failures = set(delete_failures or set())
        self.put_failures = set(put_failures or set())
        self.prefixes = list(prefixes or [])
        self.get_calls: list[str] = []
        self.put_calls: list[tuple[str, bytes, str | None]] = []
        self.deleted_keys: list[str] = []

    async def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        if key in self.put_failures:
            raise RuntimeError(f'boom putting {key}')
        self.objects[key] = data
        self.put_calls.append((key, data, content_type))

    async def get_bytes(self, key: str) -> bytes:
        self.get_calls.append(key)
        try:
            return self.objects[key]
        except KeyError as error:
            raise S3ObjectNotFoundError(key) from error

    async def list_prefixes(self, prefix: str = '') -> list[str]:
        if prefix == '':
            return [candidate.removesuffix('/') for candidate in self.prefixes]
        expected_parts = S3Client.split(prefix)
        return [
            candidate.removesuffix('/')
            for candidate in self.prefixes
            if S3Client.split(candidate)[: len(expected_parts)] == expected_parts
        ]

    async def delete_key(self, key: str) -> None:
        if key in self.delete_failures:
            raise RuntimeError(f'boom deleting {key}')
        self.deleted_keys.append(key)
        self.objects.pop(key, None)


def _clip_key(*, year: int, season: Season, universe: Universe, clip_id: str) -> str:
    return S3Client.join(_CLIP_NAMESPACE, f'{universe}-{year}-{season}', clip_id + Extension.MP4.suffix)


def _manifest_key(*, year: int, season: Season, universe: Universe) -> str:
    return S3Client.join(_CLIP_NAMESPACE, f'{universe}-{year}-{season}', 'manifest.json')


def _manifest_bytes(entries: list[ManifestEntry]) -> bytes:
    return json.dumps(Manifest(entries).to_dict(), separators=(',', ':')).encode('utf-8')


def _manifest_payload(entries: list[ManifestEntry]) -> dict[str, list[dict[str, object]]]:
    return Manifest(entries).to_dict()


def _normalized_clip_key(*, year: int, season: Season, universe: Universe, clip_id: str) -> str:
    return S3Client.join(_CLIP_NAMESPACE, f'{universe}-{year}-{season}', clip_id + '-normalized' + Extension.MP4.suffix)


def _patch_hashes(monkeypatch: pytest.MonkeyPatch, hashes: dict[bytes, str]) -> None:
    async def _fake_hash(video_bytes: bytes) -> str:
        return hashes[video_bytes]

    monkeypatch.setattr(clip_store_module, 'hash_video_content', _fake_hash)


def _patch_perceptual_metadata(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[bytes, tuple[int, tuple[int, ...]]],
) -> None:
    async def _fake_frame_count(video_bytes: bytes) -> int:
        return metadata[video_bytes][0]

    async def _fake_sampled_phashes(video_bytes: bytes, *, frame_count: int) -> tuple[int, ...]:
        expected_frame_count, sampled_phashes = metadata[video_bytes]
        assert frame_count == expected_frame_count
        return sampled_phashes

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _fake_frame_count)
    monkeypatch.setattr(clip_store_module, 'compute_video_sampled_phashes', _fake_sampled_phashes)


def _sampled_phashes_for(video_bytes: bytes) -> tuple[int, ...]:
    base = int.from_bytes(video_bytes[:8].ljust(8, b'\0'), 'big') % (2**63)
    return (base, (base + 1) % (2**63), (base + 2) % (2**63))


async def _default_compute_video_frame_count(video_bytes: bytes) -> int:
    del video_bytes
    return _FRAME_COUNT


async def _default_compute_video_sampled_phashes(video_bytes: bytes, *, frame_count: int) -> tuple[int, ...]:
    assert frame_count == _FRAME_COUNT
    return _sampled_phashes_for(video_bytes)


clip_store_module.compute_video_frame_count = _default_compute_video_frame_count
clip_store_module.compute_video_sampled_phashes = _default_compute_video_sampled_phashes


def _patch_uuid7(monkeypatch: pytest.MonkeyPatch, *clip_ids: str) -> None:
    uuids = iter(uuid.UUID(clip_id) for clip_id in clip_ids)
    monkeypatch.setattr(clip_store_module, '_uuid7', lambda: next(uuids))


def _mp4_file(data: bytes) -> FileBytes:
    return FileBytes(data=data, extension=Extension.MP4)


def _store(s3_client: _FakeS3Client) -> ClipStore:
    return ClipStore(s3_client, namespace=_CLIP_NAMESPACE, sampled_phash_mean_threshold=1.5)


def _entry(**kwargs: object) -> ManifestEntry:
    payload = {
        'frame_count': _FRAME_COUNT,
        'sampled_phashes': _SAMPLED_PHASHES,
    }
    payload.update(kwargs)
    return ManifestEntry(
        **payload,
    )


def test_extension_mp4_supports_string_and_filename_parsing() -> None:
    assert Extension.from_string('mp4') is Extension.MP4
    assert Extension.from_string('.MP4') is Extension.MP4
    assert Extension.from_filename('clip.mp4') is Extension.MP4
    assert Extension.MP4.suffix == '.mp4'


@pytest.mark.asyncio
async def test_manifest_uses_data_root_with_preferred_field_order() -> None:
    entry = _entry(
        id=_UUID_1,
        video_hash=_HASH_A,
        sub_season=SubSeason.A,
        scope=Scope.COLLECTION,
        batch=1,
        order=1,
        audio_normalization=AudioNormalization(loudness=-14, bitrate=128),
    )

    payload = Manifest([entry]).to_dict()

    assert payload == {
        'data': [
            {
                'id': _UUID_1,
                'video_hash': _HASH_A,
                'frame_count': _FRAME_COUNT,
                'sampled_phashes': list(_SAMPLED_PHASHES),
                'audio_normalization': {'loudness': -14, 'bitrate': 128},
                'sub_season': 'A',
                'scope': 'collection',
                'batch': 1,
                'order': 1,
            }
        ]
    }
    assert list(payload['data'][0]) == [
        'id',
        'video_hash',
        'frame_count',
        'sampled_phashes',
        'audio_normalization',
        'sub_season',
        'scope',
        'batch',
        'order',
    ]
    assert list(Manifest.from_dict(payload)) == [entry]


def test_manifest_rejects_old_object_wrapper_shape() -> None:
    with pytest.raises(ValueError, match="manifest root must be an object with only 'data'"):
        Manifest.from_dict({'clips': []})


def test_manifest_rejects_old_list_root_shape() -> None:
    with pytest.raises(ValueError, match="manifest root must be an object with only 'data'"):
        Manifest.from_dict([])


def test_manifest_accepts_null_perceptual_metadata_pair() -> None:
    entry = _entry(
        id=_UUID_1,
        video_hash=_HASH_A,
        frame_count=None,
        sampled_phashes=None,
        sub_season=SubSeason.A,
        scope=Scope.COLLECTION,
        batch=1,
        order=1,
    )

    payload = Manifest([entry]).to_dict()

    assert payload['data'][0]['frame_count'] is None
    assert payload['data'][0]['sampled_phashes'] is None
    assert list(Manifest.from_dict(payload)) == [entry]


def test_manifest_accepts_frame_count_without_sampled_phashes() -> None:
    entry = _entry(
        id=_UUID_1,
        video_hash=_HASH_A,
        frame_count=_FRAME_COUNT,
        sampled_phashes=None,
        sub_season=SubSeason.A,
        scope=Scope.COLLECTION,
        batch=1,
        order=1,
    )

    payload = Manifest([entry]).to_dict()

    assert payload['data'][0]['frame_count'] == _FRAME_COUNT
    assert payload['data'][0]['sampled_phashes'] is None
    assert list(Manifest.from_dict(payload)) == [entry]


def test_manifest_rejects_legacy_null_sub_season() -> None:
    with pytest.raises(ValueError, match='manifest `sub_season` must be a string'):
        Manifest.from_dict(
            {
                'data': [
                    {
                        'id': _UUID_1,
                        'video_hash': _HASH_A,
                        'frame_count': _FRAME_COUNT,
                        'sampled_phashes': list(_SAMPLED_PHASHES),
                        'audio_normalization': None,
                        'sub_season': None,
                        'scope': 'extra',
                        'batch': 1,
                        'order': 1,
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    ('frame_count', 'sampled_phashes'),
    [
        (None, [1]),
    ],
)
def test_manifest_rejects_mixed_null_perceptual_metadata(
    frame_count: int | None,
    sampled_phashes: list[int] | None,
) -> None:
    payload = {
        'id': _UUID_1,
        'video_hash': _HASH_A,
        'frame_count': frame_count,
        'sampled_phashes': sampled_phashes,
        'audio_normalization': None,
        'sub_season': 'A',
        'scope': 'collection',
        'batch': 1,
        'order': 1,
    }

    with pytest.raises(ValueError, match='`frame_count` must be present when `sampled_phashes` are present'):
        Manifest.from_dict({'data': [payload]})


@pytest.mark.parametrize(
    ('field', 'value', 'expected_message'),
    [
        ('batch', 0, 'manifest `batch` must be >= 1'),
        ('order', 0, 'manifest `order` must be >= 1'),
    ],
)
def test_manifest_rejects_non_positive_batch_and_order(
    field: str,
    value: int,
    expected_message: str,
) -> None:
    payload = {
        'id': _UUID_1,
        'video_hash': _HASH_A,
        'frame_count': _FRAME_COUNT,
        'sampled_phashes': list(_SAMPLED_PHASHES),
        'audio_normalization': None,
        'sub_season': 'A',
        'scope': 'collection',
        'batch': 1,
        'order': 1,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=expected_message):
        Manifest.from_dict({'data': [payload]})


@pytest.mark.parametrize(
    ('field', 'expected_message'),
    [
        ('frame_count', 'manifest clip entry has unexpected fields'),
        ('sampled_phashes', 'manifest clip entry has unexpected fields'),
    ],
)
def test_manifest_rejects_missing_required_perceptual_fields(field: str, expected_message: str) -> None:
    payload = Manifest(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            )
        ]
    ).to_dict()['data'][0]
    del payload[field]

    with pytest.raises(ValueError, match=expected_message):
        Manifest.from_dict({'data': [payload]})


@pytest.mark.parametrize(
    ('field', 'value', 'expected_message'),
    [
        ('frame_count', 0, 'manifest `frame_count` must be >= 1'),
        ('sampled_phashes', [], 'manifest `sampled_phashes` must not be empty'),
        ('sampled_phashes', [-1], 'manifest `sampled_phashes` entries must be >= 0'),
        ('sampled_phashes', [2**63], 'manifest `sampled_phashes` entries must be < 2\\*\\*63'),
    ],
)
def test_manifest_rejects_invalid_perceptual_fields(
    field: str,
    value: object,
    expected_message: str,
) -> None:
    payload = Manifest(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            )
        ]
    ).to_dict()['data'][0]
    payload[field] = value

    with pytest.raises(ValueError, match=expected_message):
        Manifest.from_dict({'data': [payload]})


@pytest.mark.parametrize(
    ('frame_count', 'sampled_phashes'),
    [
        (2, [1]),
        (2, [1, 2, 3]),
    ],
)
def test_manifest_rejects_sampled_phash_length_mismatch(
    frame_count: int,
    sampled_phashes: list[int],
) -> None:
    payload = Manifest(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            )
        ]
    ).to_dict()['data'][0]
    payload['frame_count'] = frame_count
    payload['sampled_phashes'] = sampled_phashes

    with pytest.raises(ValueError, match=r'manifest `sampled_phashes` length must equal min\(frame_count, 25\)'):
        Manifest.from_dict({'data': [payload]})


def test_manifest_rejects_duplicate_batch_order_position() -> None:
    with pytest.raises(
        ValueError,
        match='duplicate manifest position for sub_season=A scope=collection batch=2 order=1',
    ):
        Manifest.from_dict(
            {
                'data': [
                    {
                        'id': _UUID_1,
                        'video_hash': _HASH_A,
                        'frame_count': _FRAME_COUNT,
                        'sampled_phashes': list(_SAMPLED_PHASHES),
                        'audio_normalization': None,
                        'sub_season': 'A',
                        'scope': 'collection',
                        'batch': 2,
                        'order': 1,
                    },
                    {
                        'id': _UUID_2,
                        'video_hash': _HASH_B,
                        'frame_count': _FRAME_COUNT,
                        'sampled_phashes': list(_SAMPLED_PHASHES),
                        'audio_normalization': None,
                        'sub_season': 'A',
                        'scope': 'collection',
                        'batch': 2,
                        'order': 1,
                    },
                ]
            }
        )


@pytest.mark.asyncio
async def test_fetch_returns_grouped_clips_with_portable_filenames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    clip_key_4 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_4,
                        video_hash=_HASH_D,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=2,
                        order=2,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=2,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                ]
            ),
            clip_key_1: b'batch-1-first',
            clip_key_2: b'batch-1-second',
            clip_key_3: b'batch-2-first',
            clip_key_4: b'batch-2-second',
        }
    )
    store = _store(s3_client)

    async def _unexpected_normalize(video_bytes: bytes, *, loudness: float, bitrate: int) -> bytes:
        raise AssertionError('raw fetch must not normalize audio')

    monkeypatch.setattr(clip_store_module, 'normalize_video_audio_loudness', _unexpected_normalize)

    batches = [
        batch
        async for batch in store.fetch(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            audio_normalization=None,
        )
    ]

    assert batches == [
        (
            FetchedClip(id=_UUID_1, file=_mp4_file(b'batch-1-first')),
            FetchedClip(id=_UUID_2, file=_mp4_file(b'batch-1-second')),
        ),
        (
            FetchedClip(id=_UUID_3, file=_mp4_file(b'batch-2-first')),
            FetchedClip(id=_UUID_4, file=_mp4_file(b'batch-2-second')),
        ),
    ]
    assert all(clip.file.extension is Extension.MP4 for batch in batches for clip in batch)


@pytest.mark.asyncio
async def test_fetch_raw_preserves_manifest_order_with_concurrent_reads() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)

    class _DelayedGetS3Client(_FakeS3Client):
        async def get_bytes(self, key: str) -> bytes:
            delay_by_key = {
                clip_key_1: 0.02,
                clip_key_2: 0.0,
                clip_key_3: 0.01,
            }
            await asyncio.sleep(delay_by_key.get(key, 0.0))
            return await super().get_bytes(key)

    store = _store(
        _DelayedGetS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=2,
                        ),
                        _entry(
                            id=_UUID_3,
                            video_hash=_HASH_C,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=3,
                        ),
                    ]
                ),
                clip_key_1: b'first',
                clip_key_2: b'second',
                clip_key_3: b'third',
            }
        )
    )

    batches = [
        batch
        async for batch in store.fetch(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            audio_normalization=None,
        )
    ]

    assert batches == [
        (
            FetchedClip(id=_UUID_1, file=_mp4_file(b'first')),
            FetchedClip(id=_UUID_2, file=_mp4_file(b'second')),
            FetchedClip(id=_UUID_3, file=_mp4_file(b'third')),
        )
    ]


@pytest.mark.asyncio
async def test_fetch_with_audio_normalization_generates_normalized_twins_and_updates_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    clip_key_4 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalized_key_2 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    normalized_key_3 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    normalized_key_4 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)
    normalization = AudioNormalization(loudness=-14, bitrate=128)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=2,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_4,
                        video_hash=_HASH_D,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=2,
                        order=2,
                    ),
                ]
            ),
            clip_key_1: b'batch-1-first',
            clip_key_2: b'batch-1-second',
            clip_key_3: b'batch-2-first',
            clip_key_4: b'batch-2-second',
        }
    )
    store = _store(s3_client)
    calls: list[tuple[bytes, float, int]] = []

    async def _fake_normalize(video_bytes: bytes, *, loudness: float, bitrate: int) -> bytes:
        calls.append((video_bytes, loudness, bitrate))
        return b'normalized:' + video_bytes

    monkeypatch.setattr(clip_store_module, 'normalize_video_audio_loudness', _fake_normalize)

    batches = [
        batch
        async for batch in store.fetch(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            audio_normalization=normalization,
        )
    ]

    assert calls == [
        (b'batch-1-first', -14, 128),
        (b'batch-1-second', -14, 128),
        (b'batch-2-first', -14, 128),
        (b'batch-2-second', -14, 128),
    ]
    assert batches == [
        (
            FetchedClip(id=_UUID_1, file=_mp4_file(b'normalized:batch-1-first')),
            FetchedClip(id=_UUID_2, file=_mp4_file(b'normalized:batch-1-second')),
        ),
        (
            FetchedClip(id=_UUID_3, file=_mp4_file(b'normalized:batch-2-first')),
            FetchedClip(id=_UUID_4, file=_mp4_file(b'normalized:batch-2-second')),
        ),
    ]
    assert all(clip.file.extension is Extension.MP4 for batch in batches for clip in batch)
    assert s3_client.objects[normalized_key_1] == b'normalized:batch-1-first'
    assert s3_client.objects[normalized_key_2] == b'normalized:batch-1-second'
    assert s3_client.objects[normalized_key_3] == b'normalized:batch-2-first'
    assert s3_client.objects[normalized_key_4] == b'normalized:batch-2-second'
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
                audio_normalization=normalization,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=2,
                audio_normalization=normalization,
            ),
            _entry(
                id=_UUID_3,
                video_hash=_HASH_C,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=2,
                order=1,
                audio_normalization=normalization,
            ),
            _entry(
                id=_UUID_4,
                video_hash=_HASH_D,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=2,
                order=2,
                audio_normalization=normalization,
            ),
        ]
    )
    assert [key for key, _, _ in s3_client.put_calls] == [
        normalized_key_1,
        normalized_key_2,
        manifest_key,
        normalized_key_3,
        normalized_key_4,
        manifest_key,
    ]


@pytest.mark.asyncio
async def test_fetch_with_same_audio_normalization_reuses_existing_normalized_twins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalized_key_2 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    normalization = AudioNormalization(loudness=-14, bitrate=128)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                        audio_normalization=normalization,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                        audio_normalization=normalization,
                    ),
                ]
            ),
            clip_key_1: b'raw-first',
            clip_key_2: b'raw-second',
            normalized_key_1: b'normalized-first',
            normalized_key_2: b'normalized-second',
        }
    )
    store = _store(s3_client)

    async def _unexpected_normalize(video_bytes: bytes, *, loudness: float, bitrate: int) -> bytes:
        raise AssertionError('existing normalized twins should be reused')

    monkeypatch.setattr(clip_store_module, 'normalize_video_audio_loudness', _unexpected_normalize)

    batches = [
        batch
        async for batch in store.fetch(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            audio_normalization=normalization,
        )
    ]

    assert batches == [
        (
            FetchedClip(id=_UUID_1, file=_mp4_file(b'normalized-first')),
            FetchedClip(id=_UUID_2, file=_mp4_file(b'normalized-second')),
        )
    ]
    assert s3_client.put_calls == []
    assert s3_client.get_calls == [manifest_key, normalized_key_1, normalized_key_2]


@pytest.mark.asyncio
async def test_fetch_with_changed_audio_normalization_overwrites_stable_normalized_twins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalized_key_2 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    old_normalization = AudioNormalization(loudness=-14, bitrate=128)
    new_normalization = AudioNormalization(loudness=-18, bitrate=192)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                        audio_normalization=old_normalization,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                        audio_normalization=old_normalization,
                    ),
                ]
            ),
            clip_key_1: b'raw-first',
            clip_key_2: b'raw-second',
            normalized_key_1: b'old-normalized-first',
            normalized_key_2: b'old-normalized-second',
        }
    )
    store = _store(s3_client)
    calls: list[tuple[bytes, float, int]] = []

    async def _fake_normalize(video_bytes: bytes, *, loudness: float, bitrate: int) -> bytes:
        calls.append((video_bytes, loudness, bitrate))
        return b'new:' + video_bytes

    monkeypatch.setattr(clip_store_module, 'normalize_video_audio_loudness', _fake_normalize)

    batches = [
        batch
        async for batch in store.fetch(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            audio_normalization=new_normalization,
        )
    ]

    assert calls == [
        (b'raw-first', -18, 192),
        (b'raw-second', -18, 192),
    ]
    assert batches == [
        (
            FetchedClip(id=_UUID_1, file=_mp4_file(b'new:raw-first')),
            FetchedClip(id=_UUID_2, file=_mp4_file(b'new:raw-second')),
        )
    ]
    assert s3_client.objects[normalized_key_1] == b'new:raw-first'
    assert s3_client.objects[normalized_key_2] == b'new:raw-second'
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
                audio_normalization=new_normalization,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=2,
                audio_normalization=new_normalization,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_fetch_audio_normalization_runs_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=3,
                    ),
                ]
            ),
            clip_key_1: b'one',
            clip_key_2: b'two',
            clip_key_3: b'three',
        }
    )
    store = _store(s3_client)
    active_calls = 0
    max_active_calls = 0
    call_order: list[bytes] = []

    async def _fake_normalize(video_bytes: bytes, *, loudness: float, bitrate: int) -> bytes:
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        call_order.append(video_bytes)
        await asyncio.sleep(0)
        active_calls -= 1
        return b'n:' + video_bytes

    monkeypatch.setattr(clip_store_module, 'normalize_video_audio_loudness', _fake_normalize)

    [
        batch
        async for batch in store.fetch(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            audio_normalization=AudioNormalization(loudness=-14, bitrate=128),
        )
    ]

    assert max_active_calls == 1
    assert call_order == [b'one', b'two', b'three']


@pytest.mark.asyncio
async def test_fetch_raises_explicit_error_when_manifest_write_fails_after_normalized_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalized_key_2 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                    ),
                ]
            ),
            clip_key_1: b'raw-first',
            clip_key_2: b'raw-second',
        },
        put_failures={manifest_key},
    )
    store = _store(s3_client)

    async def _fake_normalize(video_bytes: bytes, *, loudness: float, bitrate: int) -> bytes:
        return b'n:' + video_bytes

    monkeypatch.setattr(clip_store_module, 'normalize_video_audio_loudness', _fake_normalize)

    with pytest.raises(NormalizedClipManifestSyncError, match='manifest synchronization failed') as excinfo:
        [
            batch
            async for batch in store.fetch(
                ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
                ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
                audio_normalization=AudioNormalization(loudness=-14, bitrate=128),
            )
        ]

    assert excinfo.value.written_keys == (normalized_key_1, normalized_key_2)
    assert excinfo.value.affected_clip_ids == (_UUID_1, _UUID_2)
    assert excinfo.value.stage == 'manifest_write'
    assert s3_client.objects[normalized_key_1] == b'n:raw-first'
    assert s3_client.objects[normalized_key_2] == b'n:raw-second'
    assert s3_client.deleted_keys == []


@pytest.mark.asyncio
async def test_fetch_raises_explicit_error_when_normalized_write_path_fails_before_manifest_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalized_key_2 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                    ),
                ]
            ),
            clip_key_1: b'raw-first',
            clip_key_2: b'raw-second',
        },
        put_failures={normalized_key_2},
    )
    store = _store(s3_client)

    async def _fake_normalize(video_bytes: bytes, *, loudness: float, bitrate: int) -> bytes:
        return b'n:' + video_bytes

    monkeypatch.setattr(clip_store_module, 'normalize_video_audio_loudness', _fake_normalize)

    with pytest.raises(NormalizedClipManifestSyncError, match='stage=before_manifest_write') as excinfo:
        [
            batch
            async for batch in store.fetch(
                ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
                ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
                audio_normalization=AudioNormalization(loudness=-14, bitrate=128),
            )
        ]

    assert excinfo.value.written_keys == (normalized_key_1,)
    assert excinfo.value.affected_clip_ids == (_UUID_1,)
    assert excinfo.value.stage == 'before_manifest_write'
    assert s3_client.objects[normalized_key_1] == b'n:raw-first'
    assert manifest_key in s3_client.objects
    assert s3_client.deleted_keys == []


@pytest.mark.asyncio
async def test_fetch_regenerates_missing_normalized_twin_when_manifest_says_it_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalization = AudioNormalization(loudness=-14, bitrate=128)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                        audio_normalization=normalization,
                    )
                ]
            ),
            clip_key_1: b'raw-first',
        }
    )
    store = _store(s3_client)
    calls: list[tuple[bytes, float, int]] = []

    async def _fake_normalize(video_bytes: bytes, *, loudness: float, bitrate: int) -> bytes:
        calls.append((video_bytes, loudness, bitrate))
        return b'regenerated:' + video_bytes

    monkeypatch.setattr(clip_store_module, 'normalize_video_audio_loudness', _fake_normalize)

    batches = [
        batch
        async for batch in store.fetch(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            audio_normalization=normalization,
        )
    ]

    assert calls == [(b'raw-first', -14, 128)]
    assert batches == [(FetchedClip(id=_UUID_1, file=_mp4_file(b'regenerated:raw-first')),)]
    assert s3_client.objects[normalized_key_1] == b'regenerated:raw-first'
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
                audio_normalization=normalization,
            )
        ]
    )


@pytest.mark.asyncio
async def test_fetch_with_clip_ids_returns_only_requested_sub_group_subset_in_manifest_order() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    clip_key_4 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=2,
                        ),
                        _entry(
                            id=_UUID_3,
                            video_hash=_HASH_C,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=2,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_4,
                            video_hash=_HASH_D,
                            sub_season=SubSeason.B,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        ),
                    ]
                ),
                clip_key_1: b'batch-1-first',
                clip_key_2: b'batch-1-second',
                clip_key_3: b'batch-2-first',
                clip_key_4: b'other-sub-group',
            }
        )
    )

    batches = [
        batch
        async for batch in store.fetch(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_ids=[_UUID_3, _UUID_1],
        )
    ]

    assert batches == [
        (FetchedClip(id=_UUID_1, file=_mp4_file(b'batch-1-first')),),
        (FetchedClip(id=_UUID_3, file=_mp4_file(b'batch-2-first')),),
    ]


@pytest.mark.asyncio
async def test_fetch_with_duplicate_clip_ids_raises() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(DuplicateClipIdsError, match=_UUID_1):
        [
            batch
            async for batch in store.fetch(
                ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
                ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
                clip_ids=[_UUID_1, _UUID_1],
            )
        ]


@pytest.mark.asyncio
async def test_fetch_with_unknown_clip_ids_raises() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(UnknownClipsError, match='not present in manifest'):
        [
            batch
            async for batch in store.fetch(
                ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
                ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
                clip_ids=[_UUID_2],
            )
        ]


@pytest.mark.asyncio
async def test_fetch_with_clip_ids_from_other_sub_group_raises() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.B,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        ),
                    ]
                )
            }
        )
    )

    with pytest.raises(ClipIdsNotInSubGroupError, match=_UUID_2):
        [
            batch
            async for batch in store.fetch(
                ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
                ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
                clip_ids=[_UUID_2],
            )
        ]


@pytest.mark.asyncio
async def test_fetch_fails_with_empty_sub_group_fields_when_group_is_missing() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ClipGroupNotFoundError) as excinfo:
        [
            batch
            async for batch in store.fetch(
                ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
                ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            )
        ]

    assert excinfo.value.year == 2024
    assert excinfo.value.season is Season.S1
    assert excinfo.value.universe is Universe.WEST
    assert excinfo.value.sub_season is None
    assert excinfo.value.scope is None


@pytest.mark.asyncio
async def test_fetch_fails_with_requested_sub_group_fields_when_sub_group_is_missing() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.B,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(ClipGroupNotFoundError) as excinfo:
        [
            batch
            async for batch in store.fetch(
                ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
                ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            )
        ]

    assert excinfo.value.year == 2024
    assert excinfo.value.season is Season.S1
    assert excinfo.value.universe is Universe.WEST
    assert excinfo.value.sub_season is SubSeason.A
    assert excinfo.value.scope is Scope.COLLECTION


@pytest.mark.asyncio
async def test_list_groups_returns_parsed_groups() -> None:
    store = _store(
        _FakeS3Client(
            prefixes=[
                'clips/west-2024-1/',
                'clips/east-2025-2',
            ]
        )
    )

    assert await store.list_groups() == [
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipGroup(universe=Universe.EAST, year=2025, season=Season.S2),
    ]


@pytest.mark.asyncio
async def test_list_groups_returns_sorted_groups() -> None:
    store = _store(
        _FakeS3Client(
            prefixes=[
                'clips/west-2025-2',
                'clips/east-2024-2',
                'clips/west-2024-1',
                'clips/east-2024-1',
            ]
        )
    )

    assert await store.list_groups() == [
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipGroup(universe=Universe.WEST, year=2025, season=Season.S2),
        ClipGroup(universe=Universe.EAST, year=2024, season=Season.S1),
        ClipGroup(universe=Universe.EAST, year=2024, season=Season.S2),
    ]


@pytest.mark.asyncio
async def test_list_groups_fails_on_malformed_prefix() -> None:
    store = _store(_FakeS3Client(prefixes=['clips/west-2024-1/extra']))

    with pytest.raises(ValueError, match=r"'clips/west-2024-1/extra'"):
        await store.list_groups()


@pytest.mark.asyncio
async def test_list_clips_returns_unique_sub_groups_as_keys() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=2,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_4,
                            video_hash=_HASH_C,
                            sub_season=SubSeason.NONE,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        ),
                    ]
                )
            }
        )
    )

    result = await store.list_clips(ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1))

    assert list(result.keys()) == [
        ClipSubGroup(sub_season=SubSeason.NONE, scope=Scope.EXTRA),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
    ]
    assert result[ClipSubGroup(sub_season=SubSeason.NONE, scope=Scope.EXTRA)] == [
        (ClipInfo(id=_UUID_4),),
    ]
    assert result[ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)] == [
        (ClipInfo(id=_UUID_1),),
        (ClipInfo(id=_UUID_2),),
    ]


@pytest.mark.asyncio
async def test_list_clips_returns_sorted_sub_groups() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.B,
                            scope=Scope.SOURCE,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.NONE,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_4,
                            video_hash=_HASH_C,
                            sub_season=SubSeason.B,
                            scope=Scope.COLLECTION,
                            batch=2,
                            order=1,
                        ),
                    ]
                )
            }
        )
    )

    result = await store.list_clips(ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1))

    assert list(result.keys()) == [
        ClipSubGroup(sub_season=SubSeason.NONE, scope=Scope.EXTRA),
        ClipSubGroup(sub_season=SubSeason.B, scope=Scope.COLLECTION),
        ClipSubGroup(sub_season=SubSeason.B, scope=Scope.SOURCE),
    ]


@pytest.mark.asyncio
async def test_list_clips_returns_batches_sorted_within_sub_group() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_4,
                            video_hash=_HASH_C,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=2,
                            order=2,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=2,
                        ),
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_3,
                            video_hash=_HASH_D,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=2,
                            order=1,
                        ),
                    ]
                )
            }
        )
    )

    result = await store.list_clips(ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1))

    assert result[ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)] == [
        (
            ClipInfo(id=_UUID_1),
            ClipInfo(id=_UUID_2),
        ),
        (
            ClipInfo(id=_UUID_3),
            ClipInfo(id=_UUID_4),
        ),
    ]


@pytest.mark.asyncio
async def test_list_clips_fails_on_missing_manifest() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ClipGroupNotFoundError) as excinfo:
        await store.list_clips(ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1))

    assert excinfo.value.year == 2024
    assert excinfo.value.season is Season.S1
    assert excinfo.value.universe is Universe.WEST
    assert excinfo.value.sub_season is None
    assert excinfo.value.scope is None


@pytest.mark.asyncio
async def test_list_clips_fails_on_corrupted_manifest() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(_FakeS3Client({manifest_key: b'{"clips": []}'}))

    with pytest.raises(ManifestCorruptedError):
        await store.list_clips(ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1))


@pytest.mark.asyncio
async def test_store_logs_and_raises_for_corrupt_current_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[tuple[object, ...]] = []

    def log_error(*args: object) -> None:
        errors.append(args)

    monkeypatch.setattr(clip_store_module.logger, 'error', log_error)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(_FakeS3Client({manifest_key: b'{"clips": []}'}))

    with pytest.raises(ManifestCorruptedError):
        await store.store(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clips=[_mp4_file(b'clip')],
        )

    assert len(errors) == 1
    assert 'failed to load current clip-group manifest for store' in str(errors[0][0])
    assert manifest_key in str(errors[0])


@pytest.mark.asyncio
async def test_store_treats_existing_video_hash_as_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})

    async def _unexpected_frame_count(video_bytes: bytes) -> int:
        raise AssertionError(f'frame count must not be computed for exact duplicate: {video_bytes!r}')

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _unexpected_frame_count)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            )
        }
    )
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result.stored_count == 0
    assert result.duplicate_count == 1
    assert s3_client.put_calls == []


@pytest.mark.asyncio
async def test_store_treats_previous_group_video_hash_as_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})

    async def _unexpected_frame_count(video_bytes: bytes) -> int:
        raise AssertionError(f'frame count must not be computed for exact duplicate: {video_bytes!r}')

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _unexpected_frame_count)
    previous_manifest_key = _manifest_key(year=2023, season=Season.S5, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                previous_manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=0, duplicate_count=1)


@pytest.mark.asyncio
async def test_store_treats_next_group_video_hash_as_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})

    async def _unexpected_frame_count(video_bytes: bytes) -> int:
        raise AssertionError(f'frame count must not be computed for exact duplicate: {video_bytes!r}')

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _unexpected_frame_count)
    next_manifest_key = _manifest_key(year=2024, season=Season.S2, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                next_manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=0, duplicate_count=1)


@pytest.mark.asyncio
async def test_store_deduplicates_perceptual_duplicate_with_different_video_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    _patch_perceptual_metadata(
        monkeypatch,
        {
            b'clip': (_FRAME_COUNT, _SAMPLED_PHASHES),
        },
    )
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            )
        }
    )
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=0, duplicate_count=1)
    assert s3_client.put_calls == []


@pytest.mark.asyncio
async def test_store_deduplicates_perceptual_duplicate_in_previous_group(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    _patch_perceptual_metadata(monkeypatch, {b'clip': (_FRAME_COUNT, _SAMPLED_PHASHES)})
    previous_manifest_key = _manifest_key(year=2023, season=Season.S5, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                previous_manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=0, duplicate_count=1)


@pytest.mark.asyncio
async def test_store_deduplicates_perceptual_duplicate_in_next_group(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    _patch_perceptual_metadata(monkeypatch, {b'clip': (_FRAME_COUNT, _SAMPLED_PHASHES)})
    next_manifest_key = _manifest_key(year=2024, season=Season.S2, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                next_manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=0, duplicate_count=1)


@pytest.mark.asyncio
async def test_store_accepts_clip_when_perceptual_metadata_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})

    async def _raise_unavailable(video_bytes: bytes) -> int:
        del video_bytes
        raise PerceptualMetadataUnavailableError('missing frame count')

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _raise_unavailable)
    _patch_uuid7(monkeypatch, _UUID_1)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client()
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_1,))
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                frame_count=None,
                sampled_phashes=None,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            )
        ]
    )


@pytest.mark.asyncio
async def test_store_accepts_clip_when_sampled_phashes_are_unavailable_after_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})

    async def _compute_frame_count(video_bytes: bytes) -> int:
        del video_bytes
        return _FRAME_COUNT

    async def _raise_unavailable(video_bytes: bytes, *, frame_count: int) -> tuple[int, ...]:
        del video_bytes, frame_count
        raise PerceptualMetadataUnavailableError('failed to extract sampled frames')

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _compute_frame_count)
    monkeypatch.setattr(clip_store_module, 'compute_video_sampled_phashes', _raise_unavailable)
    _patch_uuid7(monkeypatch, _UUID_1)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client()
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_1,))
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                frame_count=_FRAME_COUNT,
                sampled_phashes=None,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            )
        ]
    )


@pytest.mark.asyncio
async def test_store_perceptual_comparison_skips_persisted_null_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    _patch_perceptual_metadata(monkeypatch, {b'clip': (_FRAME_COUNT, _SAMPLED_PHASHES)})
    _patch_uuid7(monkeypatch, _UUID_2)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        frame_count=None,
                        sampled_phashes=None,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            )
        }
    )
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_2,))


@pytest.mark.asyncio
async def test_store_ignores_missing_previous_neighbor_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})
    _patch_uuid7(monkeypatch, _UUID_1)
    next_manifest_key = _manifest_key(year=2024, season=Season.S2, universe=Universe.WEST)
    target_manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                next_manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_1,))
    assert target_manifest_key in store._s3_client.objects


@pytest.mark.asyncio
async def test_store_ignores_missing_next_neighbor_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})
    _patch_uuid7(monkeypatch, _UUID_1)
    previous_manifest_key = _manifest_key(year=2023, season=Season.S5, universe=Universe.WEST)
    target_manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                previous_manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_1,))
    assert target_manifest_key in store._s3_client.objects


@pytest.mark.asyncio
async def test_store_skips_corrupt_previous_neighbor_manifest_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})
    _patch_uuid7(monkeypatch, _UUID_1)
    warnings: list[tuple[object, ...]] = []

    def log_warning(*args: object) -> None:
        warnings.append(args)

    monkeypatch.setattr(clip_store_module.logger, 'warning', log_warning)
    previous_manifest_key = _manifest_key(year=2023, season=Season.S5, universe=Universe.WEST)
    target_manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(_FakeS3Client({previous_manifest_key: b'{"clips": []}'}))

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_1,))
    assert target_manifest_key in store._s3_client.objects
    assert len(warnings) == 1
    assert 'skipping neighbor manifest during dedup' in str(warnings[0][0])
    assert '2023' in str(warnings[0])
    assert previous_manifest_key in str(warnings[0])


@pytest.mark.asyncio
async def test_store_skips_corrupt_next_neighbor_manifest_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})
    _patch_uuid7(monkeypatch, _UUID_1)
    warnings: list[tuple[object, ...]] = []

    def log_warning(*args: object) -> None:
        warnings.append(args)

    monkeypatch.setattr(clip_store_module.logger, 'warning', log_warning)
    next_manifest_key = _manifest_key(year=2024, season=Season.S2, universe=Universe.WEST)
    target_manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(_FakeS3Client({next_manifest_key: b'{"clips": []}'}))

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_1,))
    assert target_manifest_key in store._s3_client.objects
    assert len(warnings) == 1
    assert 'skipping neighbor manifest during dedup' in str(warnings[0][0])
    assert '2024' in str(warnings[0])
    assert next_manifest_key in str(warnings[0])


@pytest.mark.asyncio
async def test_store_computes_sampled_phashes_before_storing_without_same_frame_count_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})
    observed_calls: list[tuple[bytes, int]] = []

    async def _compute_frame_count(video_bytes: bytes) -> int:
        del video_bytes
        return _FRAME_COUNT

    async def _compute_sampled_phashes(video_bytes: bytes, *, frame_count: int) -> tuple[int, ...]:
        observed_calls.append((video_bytes, frame_count))
        return _SAMPLED_PHASHES

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _compute_frame_count)
    monkeypatch.setattr(clip_store_module, 'compute_video_sampled_phashes', _compute_sampled_phashes)
    _patch_uuid7(monkeypatch, _UUID_1)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client()
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_1,))
    assert observed_calls == [(b'clip', _FRAME_COUNT)]
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                frame_count=_FRAME_COUNT,
                sampled_phashes=_SAMPLED_PHASHES,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            )
        ]
    )


@pytest.mark.asyncio
async def test_store_exact_video_hash_duplicate_still_dedupes_when_manifest_perceptual_fields_are_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})

    async def _unexpected_frame_count(video_bytes: bytes) -> int:
        raise AssertionError(f'frame count must not be computed for exact duplicate: {video_bytes!r}')

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _unexpected_frame_count)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        frame_count=None,
                        sampled_phashes=None,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            )
        }
    )
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=0, duplicate_count=1)


@pytest.mark.asyncio
async def test_store_same_call_perceptual_comparison_skips_null_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_A, b'second': _HASH_B})

    async def _compute_frame_count(video_bytes: bytes) -> int:
        if video_bytes == b'first':
            raise PerceptualMetadataUnavailableError('missing frame count')
        return _FRAME_COUNT

    async def _compute_sampled_phashes(video_bytes: bytes, *, frame_count: int) -> tuple[int, ...]:
        del frame_count
        return _SAMPLED_PHASHES

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _compute_frame_count)
    monkeypatch.setattr(clip_store_module, 'compute_video_sampled_phashes', _compute_sampled_phashes)
    _patch_uuid7(monkeypatch, _UUID_1, _UUID_2)
    store = _store(_FakeS3Client())

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'first'), _mp4_file(b'second')],
    )

    assert result == StoreResult(stored_count=2, duplicate_count=0, clip_ids=(_UUID_1, _UUID_2))


@pytest.mark.asyncio
async def test_store_perceptual_comparison_skips_same_frame_count_rows_without_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    observed_calls: list[tuple[bytes, int]] = []

    async def _compute_frame_count(video_bytes: bytes) -> int:
        del video_bytes
        return _FRAME_COUNT

    async def _compute_sampled_phashes(video_bytes: bytes, *, frame_count: int) -> tuple[int, ...]:
        observed_calls.append((video_bytes, frame_count))
        return _SAMPLED_PHASHES

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _compute_frame_count)
    monkeypatch.setattr(clip_store_module, 'compute_video_sampled_phashes', _compute_sampled_phashes)
    _patch_uuid7(monkeypatch, _UUID_2)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        frame_count=_FRAME_COUNT,
                        sampled_phashes=None,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            )
        }
    )
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_2,))
    assert observed_calls == [(b'clip', _FRAME_COUNT)]


@pytest.mark.asyncio
async def test_store_neighbor_same_frame_count_without_hashes_does_not_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    observed_calls: list[tuple[bytes, int]] = []

    async def _compute_frame_count(video_bytes: bytes) -> int:
        del video_bytes
        return _FRAME_COUNT

    async def _compute_sampled_phashes(video_bytes: bytes, *, frame_count: int) -> tuple[int, ...]:
        observed_calls.append((video_bytes, frame_count))
        return _SAMPLED_PHASHES

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _compute_frame_count)
    monkeypatch.setattr(clip_store_module, 'compute_video_sampled_phashes', _compute_sampled_phashes)
    _patch_uuid7(monkeypatch, _UUID_2)
    previous_manifest_key = _manifest_key(year=2023, season=Season.S5, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                previous_manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            frame_count=_FRAME_COUNT,
                            sampled_phashes=None,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_2,))
    assert observed_calls == [(b'clip', _FRAME_COUNT)]


@pytest.mark.asyncio
async def test_store_skips_perceptual_duplicate_when_frame_count_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    observed_calls: list[tuple[bytes, int]] = []

    async def _compute_frame_count(video_bytes: bytes) -> int:
        del video_bytes
        return _FRAME_COUNT + 1

    async def _compute_sampled_phashes(video_bytes: bytes, *, frame_count: int) -> tuple[int, ...]:
        observed_calls.append((video_bytes, frame_count))
        return _SAMPLED_PHASHES

    monkeypatch.setattr(clip_store_module, 'compute_video_frame_count', _compute_frame_count)
    monkeypatch.setattr(clip_store_module, 'compute_video_sampled_phashes', _compute_sampled_phashes)
    _patch_uuid7(monkeypatch, _UUID_2)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            )
        }
    )
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_2,))
    assert observed_calls == [(b'clip', _FRAME_COUNT + 1)]


@pytest.mark.asyncio
async def test_store_accepts_perceptual_non_duplicate_with_same_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    _patch_perceptual_metadata(
        monkeypatch,
        {
            b'clip': (_FRAME_COUNT, (1000, 2000, 3000)),
        },
    )
    _patch_uuid7(monkeypatch, _UUID_2)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            )
        }
    )
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_2,))


@pytest.mark.asyncio
async def test_store_deduplicates_same_call_perceptual_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_A, b'second': _HASH_B})
    _patch_perceptual_metadata(
        monkeypatch,
        {
            b'first': (_FRAME_COUNT, (1000, 2000, 3000)),
            b'second': (_FRAME_COUNT, (1000, 2000, 3000)),
        },
    )
    _patch_uuid7(monkeypatch, _UUID_4)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_C,
                            sampled_phashes=(4000, 5000, 6000),
                            sub_season=SubSeason.C,
                            scope=Scope.SOURCE,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.C, scope=Scope.SOURCE),
        clips=[_mp4_file(b'first'), _mp4_file(b'second')],
    )

    assert result == StoreResult(stored_count=1, duplicate_count=1, clip_ids=(_UUID_4,))


@pytest.mark.asyncio
async def test_store_rejects_non_mp4_filebytes() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(InvalidExtensionError, match='clips entries must use Extension.MP4'):
        await store.store(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clips=[FileBytes(data=b'clip', extension=Extension.JPG)],
        )


@pytest.mark.asyncio
async def test_store_propagates_unsupported_codec_before_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _raise_unsupported(video_bytes: bytes) -> str:
        del video_bytes
        raise UnsupportedVideoCodecError(codec='vp9', supported_codecs=('h264', 'hevc'))

    monkeypatch.setattr(clip_store_module, 'hash_video_content', _raise_unsupported)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client({manifest_key: _manifest_bytes([])})
    store = _store(s3_client)

    with pytest.raises(UnsupportedVideoCodecError) as excinfo:
        await store.store(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clips=[_mp4_file(b'clip')],
        )

    assert excinfo.value.codec == 'vp9'
    assert s3_client.put_calls == []
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload([])


@pytest.mark.asyncio
async def test_store_generates_new_ids_for_same_call_distinct_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_A, b'second': _HASH_B})
    _patch_uuid7(monkeypatch, _UUID_4, _UUID_2)
    target_manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.EAST)
    s3_client = _FakeS3Client()
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.EAST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[
            _mp4_file(b'first'),
            _mp4_file(b'second'),
        ],
    )

    assert result.stored_count == 2
    assert result.duplicate_count == 0
    assert result.clip_ids == (_UUID_4, _UUID_2)
    assert json.loads(s3_client.objects[target_manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_4,
                video_hash=_HASH_A,
                sampled_phashes=_sampled_phashes_for(b'first'),
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sampled_phashes=_sampled_phashes_for(b'second'),
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=2,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_store_deduplicates_same_call_by_video_hash_and_keeps_dense_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_C, b'second': _HASH_C, b'third': _HASH_D})
    _patch_uuid7(monkeypatch, _UUID_4, _UUID_5)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client()
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.C, scope=Scope.SOURCE),
        clips=[
            _mp4_file(b'first'),
            _mp4_file(b'second'),
            _mp4_file(b'third'),
        ],
    )

    assert result == StoreResult(
        stored_count=2,
        duplicate_count=1,
        clip_ids=(_UUID_4, _UUID_5),
    )
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_4,
                video_hash=_HASH_C,
                sampled_phashes=_sampled_phashes_for(b'first'),
                sub_season=SubSeason.C,
                scope=Scope.SOURCE,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_5,
                video_hash=_HASH_D,
                sampled_phashes=_sampled_phashes_for(b'third'),
                sub_season=SubSeason.C,
                scope=Scope.SOURCE,
                batch=1,
                order=2,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_store_creates_new_batch_per_call_and_resets_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(
        monkeypatch,
        {
            b'first': _HASH_A,
            b'second': _HASH_B,
            b'third': _HASH_C,
        },
    )
    _patch_uuid7(monkeypatch, _UUID_1, _UUID_2, _UUID_3)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client()
    store = _store(s3_client)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)

    first_result = await store.store(
        clip_group,
        clip_sub_group,
        clips=[
            _mp4_file(b'first'),
            _mp4_file(b'second'),
        ],
    )
    second_result = await store.store(
        clip_group,
        clip_sub_group,
        clips=[_mp4_file(b'third')],
    )

    assert first_result == StoreResult(
        stored_count=2,
        duplicate_count=0,
        clip_ids=(_UUID_1, _UUID_2),
    )
    assert second_result == StoreResult(
        stored_count=1,
        duplicate_count=0,
        clip_ids=(_UUID_3,),
    )
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sampled_phashes=_sampled_phashes_for(b'first'),
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sampled_phashes=_sampled_phashes_for(b'second'),
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=2,
            ),
            _entry(
                id=_UUID_3,
                video_hash=_HASH_C,
                sampled_phashes=_sampled_phashes_for(b'third'),
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=2,
                order=1,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_store_reuses_cached_neighbor_manifests_across_repeated_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_A, b'second': _HASH_B})
    _patch_uuid7(monkeypatch, _UUID_1, _UUID_2)
    target_manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    previous_manifest_key = _manifest_key(year=2023, season=Season.S5, universe=Universe.WEST)
    next_manifest_key = _manifest_key(year=2024, season=Season.S2, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            previous_manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            ),
            next_manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_4,
                        video_hash=_HASH_D,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            ),
        }
    )
    store = _store(s3_client)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)

    first_result = await store.store(clip_group, clip_sub_group, clips=[_mp4_file(b'first')])
    second_result = await store.store(clip_group, clip_sub_group, clips=[_mp4_file(b'second')])

    assert first_result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_1,))
    assert second_result == StoreResult(stored_count=1, duplicate_count=0, clip_ids=(_UUID_2,))
    assert s3_client.get_calls == [target_manifest_key, previous_manifest_key, next_manifest_key]


@pytest.mark.asyncio
async def test_store_all_duplicates_do_not_create_new_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_A})
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    original_manifest = [
        _entry(
            id=_UUID_1,
            video_hash=_HASH_A,
            sub_season=SubSeason.A,
            scope=Scope.COLLECTION,
            batch=1,
            order=1,
        )
    ]
    s3_client = _FakeS3Client({manifest_key: _manifest_bytes(original_manifest)})
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[_mp4_file(b'clip')],
    )

    assert result == StoreResult(stored_count=0, duplicate_count=1)
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == Manifest(original_manifest).to_dict()
    assert s3_client.put_calls == []


@pytest.mark.asyncio
async def test_store_propagates_first_clip_upload_failure_without_sync_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'clip': _HASH_B})
    _patch_uuid7(monkeypatch, _UUID_2)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    original_manifest = [
        _entry(
            id=_UUID_1,
            video_hash=_HASH_A,
            sub_season=SubSeason.A,
            scope=Scope.COLLECTION,
            batch=1,
            order=1,
        )
    ]
    s3_client = _FakeS3Client(
        {manifest_key: _manifest_bytes(original_manifest)},
        put_failures={clip_key},
    )
    store = _store(s3_client)
    clip_group_prefix = store._clip_group_prefix(
        universe=clip_group.universe,
        year=clip_group.year,
        season=clip_group.season,
    )

    with pytest.raises(RuntimeError, match=f'boom putting {clip_key}'):
        await store.store(
            clip_group,
            clip_sub_group,
            clips=[_mp4_file(b'clip')],
        )

    assert manifest_key not in [call[0] for call in s3_client.put_calls]
    assert clip_key not in s3_client.objects
    assert s3_client.deleted_keys == []
    assert store._manifest_cache[clip_group_prefix].to_dict() == Manifest(original_manifest).to_dict()


@pytest.mark.asyncio
async def test_store_raises_sync_error_when_later_clip_upload_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_B, b'second': _HASH_C})
    _patch_uuid7(monkeypatch, _UUID_2, _UUID_3)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    first_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    second_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    original_manifest = [
        _entry(
            id=_UUID_1,
            video_hash=_HASH_A,
            sub_season=SubSeason.A,
            scope=Scope.COLLECTION,
            batch=1,
            order=1,
        )
    ]
    s3_client = _FakeS3Client(
        {manifest_key: _manifest_bytes(original_manifest)},
        put_failures={second_clip_key},
    )
    store = _store(s3_client)
    clip_group_prefix = store._clip_group_prefix(
        universe=clip_group.universe,
        year=clip_group.year,
        season=clip_group.season,
    )

    with pytest.raises(ClipManifestSyncError, match='Staged clip store failed at clip_upload') as excinfo:
        await store.store(
            clip_group,
            clip_sub_group,
            clips=[
                _mp4_file(b'first'),
                _mp4_file(b'second'),
            ],
        )

    assert excinfo.value.stage == 'clip_upload'
    assert excinfo.value.written_keys == (first_clip_key,)
    assert excinfo.value.affected_clip_ids == (_UUID_2,)
    assert excinfo.value.manifest_key == manifest_key
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == f'boom putting {second_clip_key}'
    assert excinfo.value.__notes__ == [f"Original clip upload error: RuntimeError('boom putting {second_clip_key}')"]
    assert first_clip_key in s3_client.objects
    assert second_clip_key not in s3_client.objects
    assert manifest_key in s3_client.objects
    assert s3_client.objects[manifest_key] == _manifest_bytes(original_manifest)
    assert manifest_key not in [call[0] for call in s3_client.put_calls]
    assert s3_client.deleted_keys == []
    assert store._manifest_cache[clip_group_prefix].to_dict() == Manifest(original_manifest).to_dict()


@pytest.mark.asyncio
async def test_store_raises_sync_error_with_concurrent_partial_success_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_B, b'second': _HASH_C, b'third': _HASH_D})
    _patch_uuid7(monkeypatch, _UUID_2, _UUID_3, _UUID_4)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    first_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    second_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    third_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)
    s3_client = _FakeS3Client(put_failures={second_clip_key})
    store = _store(s3_client)

    with pytest.raises(ClipManifestSyncError, match='Staged clip store failed at clip_upload') as excinfo:
        await store.store(
            clip_group,
            clip_sub_group,
            clips=[
                _mp4_file(b'first'),
                _mp4_file(b'second'),
                _mp4_file(b'third'),
            ],
        )

    assert excinfo.value.stage == 'clip_upload'
    assert excinfo.value.written_keys == (first_clip_key, third_clip_key)
    assert excinfo.value.affected_clip_ids == (_UUID_2, _UUID_4)
    assert excinfo.value.manifest_key == manifest_key
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == f'boom putting {second_clip_key}'
    assert first_clip_key in s3_client.objects
    assert second_clip_key not in s3_client.objects
    assert third_clip_key in s3_client.objects
    assert manifest_key not in s3_client.objects
    assert s3_client.deleted_keys == []


@pytest.mark.asyncio
async def test_store_treats_cancelled_upload_result_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_B, b'second': _HASH_C, b'third': _HASH_D})
    _patch_uuid7(monkeypatch, _UUID_2, _UUID_3, _UUID_4)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    first_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    second_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    third_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)

    class _CancelledPutS3Client(_FakeS3Client):
        async def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
            if key == second_clip_key:
                raise asyncio.CancelledError()
            await super().put_bytes(key, data, content_type=content_type)

    s3_client = _CancelledPutS3Client()
    store = _store(s3_client)

    with pytest.raises(ClipManifestSyncError, match='Staged clip store failed at clip_upload') as excinfo:
        await store.store(
            clip_group,
            clip_sub_group,
            clips=[
                _mp4_file(b'first'),
                _mp4_file(b'second'),
                _mp4_file(b'third'),
            ],
        )

    assert excinfo.value.stage == 'clip_upload'
    assert excinfo.value.written_keys == (first_clip_key, third_clip_key)
    assert excinfo.value.affected_clip_ids == (_UUID_2, _UUID_4)
    assert excinfo.value.manifest_key == manifest_key
    assert isinstance(excinfo.value.__cause__, asyncio.CancelledError)
    assert second_clip_key not in excinfo.value.written_keys
    assert second_clip_key not in s3_client.objects
    assert first_clip_key in s3_client.objects
    assert third_clip_key in s3_client.objects
    assert manifest_key not in s3_client.objects


@pytest.mark.asyncio
async def test_store_raises_first_exception_when_all_concurrent_uploads_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_B, b'second': _HASH_C})
    _patch_uuid7(monkeypatch, _UUID_2, _UUID_3)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)
    first_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    second_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    s3_client = _FakeS3Client(put_failures={first_clip_key, second_clip_key})
    store = _store(s3_client)

    with pytest.raises(RuntimeError, match=f'boom putting {first_clip_key}'):
        await store.store(
            clip_group,
            clip_sub_group,
            clips=[
                _mp4_file(b'first'),
                _mp4_file(b'second'),
            ],
        )

    assert first_clip_key not in s3_client.objects
    assert second_clip_key not in s3_client.objects


@pytest.mark.asyncio
async def test_store_uploads_all_clips_successfully_with_concurrent_uploads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_A, b'second': _HASH_B, b'third': _HASH_C})
    _patch_uuid7(monkeypatch, _UUID_1, _UUID_2, _UUID_3)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client()
    store = _store(s3_client)

    result = await store.store(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clips=[
            _mp4_file(b'first'),
            _mp4_file(b'second'),
            _mp4_file(b'third'),
        ],
    )

    assert result == StoreResult(
        stored_count=3,
        duplicate_count=0,
        clip_ids=(_UUID_1, _UUID_2, _UUID_3),
    )
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sampled_phashes=_sampled_phashes_for(b'first'),
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sampled_phashes=_sampled_phashes_for(b'second'),
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=2,
            ),
            _entry(
                id=_UUID_3,
                video_hash=_HASH_C,
                sampled_phashes=_sampled_phashes_for(b'third'),
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=3,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_store_raises_sync_error_when_manifest_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_B, b'second': _HASH_C})
    _patch_uuid7(monkeypatch, _UUID_2, _UUID_3)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    first_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    second_clip_key = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    original_manifest = [
        _entry(
            id=_UUID_1,
            video_hash=_HASH_A,
            sub_season=SubSeason.A,
            scope=Scope.COLLECTION,
            batch=1,
            order=1,
        )
    ]
    s3_client = _FakeS3Client(
        {manifest_key: _manifest_bytes(original_manifest)},
        put_failures={manifest_key},
    )
    store = _store(s3_client)
    clip_group_prefix = store._clip_group_prefix(
        universe=clip_group.universe,
        year=clip_group.year,
        season=clip_group.season,
    )

    with pytest.raises(ClipManifestSyncError, match='Staged clip store failed at manifest_write') as excinfo:
        await store.store(
            clip_group,
            clip_sub_group,
            clips=[
                _mp4_file(b'first'),
                _mp4_file(b'second'),
            ],
        )

    assert excinfo.value.stage == 'manifest_write'
    assert excinfo.value.written_keys == (first_clip_key, second_clip_key)
    assert excinfo.value.affected_clip_ids == (_UUID_2, _UUID_3)
    assert excinfo.value.manifest_key == manifest_key
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == f'boom putting {manifest_key}'
    assert excinfo.value.__notes__ == [f"Original manifest write error: RuntimeError('boom putting {manifest_key}')"]
    assert s3_client.objects[first_clip_key] == b'first'
    assert s3_client.objects[second_clip_key] == b'second'
    assert s3_client.objects[manifest_key] == _manifest_bytes(original_manifest)
    assert s3_client.deleted_keys == []
    assert store._manifest_cache[clip_group_prefix].to_dict() == Manifest(original_manifest).to_dict()


@pytest.mark.asyncio
async def test_reorder_rewrites_only_target_sub_group_batches() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=3,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=3,
                        order=2,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.NONE,
                        scope=Scope.EXTRA,
                        batch=5,
                        order=1,
                    ),
                ]
            ),
            clip_key_1: b'one',
            clip_key_2: b'two',
            clip_key_3: b'three',
        }
    )
    store = _store(s3_client)

    await store.reorder(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clip_id_batches=[[_UUID_2], [_UUID_1]],
    )

    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=2,
                order=1,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_3,
                video_hash=_HASH_C,
                sub_season=SubSeason.NONE,
                scope=Scope.EXTRA,
                batch=5,
                order=1,
            ),
        ]
    )
    assert [key for key, _, _ in s3_client.put_calls] == [manifest_key]
    assert s3_client.deleted_keys == []


@pytest.mark.asyncio
async def test_reorder_rejects_empty_clip_id_batches() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ValueError, match='reorder\\(\\) clip_id_batches must not contain empty batches'):
        await store.reorder(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[]],
        )


@pytest.mark.asyncio
async def test_reorder_rejects_empty_inner_batch_before_duplicates() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ValueError, match='reorder\\(\\) clip_id_batches must not contain empty batches'):
        await store.reorder(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_1], [], [_UUID_1]],
        )


@pytest.mark.asyncio
async def test_reorder_rejects_duplicate_clip_ids() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(DuplicateClipIdsError, match=_UUID_1):
        await store.reorder(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_1, _UUID_1]],
        )


@pytest.mark.asyncio
async def test_reorder_rejects_unknown_clip_ids() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(UnknownClipsError, match=_UUID_4):
        await store.reorder(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_4]],
        )


@pytest.mark.asyncio
async def test_reorder_rejects_clip_ids_outside_target_sub_group() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.NONE,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        ),
                    ]
                )
            }
        )
    )

    with pytest.raises(ClipIdsNotInSubGroupError, match=_UUID_2):
        await store.reorder(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_1, _UUID_2]],
        )


@pytest.mark.asyncio
async def test_reorder_rejects_non_exact_target_sub_group_coverage() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=2,
                            order=1,
                        ),
                    ]
                )
            }
        )
    )

    with pytest.raises(
        ValueError,
        match='reorder\\(\\) clip_id_batches must match exactly the full set of clip ids in the sub-group',
    ):
        await store.reorder(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_1]],
        )


@pytest.mark.asyncio
async def test_move_appends_batches_and_compacts_source_sub_groups() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    clip_key_4 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)
    clip_key_5 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_5)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=2,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.NONE,
                        scope=Scope.EXTRA,
                        batch=3,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.NONE,
                        scope=Scope.EXTRA,
                        batch=5,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_4,
                        video_hash=_HASH_D,
                        sub_season=SubSeason.B,
                        scope=Scope.SOURCE,
                        batch=4,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_5,
                        video_hash=_HASH_E,
                        sub_season=SubSeason.B,
                        scope=Scope.SOURCE,
                        batch=7,
                        order=1,
                    ),
                ]
            ),
            clip_key_1: b'one',
            clip_key_2: b'two',
            clip_key_3: b'three',
            clip_key_4: b'four',
            clip_key_5: b'five',
        }
    )
    store = _store(s3_client)

    await store.move(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        target_sub_group=ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clip_id_batches=[[_UUID_2, _UUID_4], [_UUID_3]],
    )

    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=2,
                order=1,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=3,
                order=1,
            ),
            _entry(
                id=_UUID_3,
                video_hash=_HASH_C,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=4,
                order=1,
            ),
            _entry(
                id=_UUID_4,
                video_hash=_HASH_D,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=3,
                order=2,
            ),
            _entry(
                id=_UUID_5,
                video_hash=_HASH_E,
                sub_season=SubSeason.B,
                scope=Scope.SOURCE,
                batch=1,
                order=1,
            ),
        ]
    )
    assert [key for key, _, _ in s3_client.put_calls] == [manifest_key]
    assert s3_client.deleted_keys == []


@pytest.mark.asyncio
async def test_move_rejects_empty_clip_id_batches() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ValueError, match='move\\(\\) clip_id_batches must not contain empty batches'):
        await store.move(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            target_sub_group=ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[]],
        )


@pytest.mark.asyncio
async def test_move_rejects_empty_inner_batch_before_duplicates() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ValueError, match='move\\(\\) clip_id_batches must not contain empty batches'):
        await store.move(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            target_sub_group=ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_1], [], [_UUID_1]],
        )


@pytest.mark.asyncio
async def test_move_rejects_duplicate_clip_ids() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.NONE,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(DuplicateClipIdsError, match=_UUID_1):
        await store.move(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            target_sub_group=ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_1, _UUID_1]],
        )


@pytest.mark.asyncio
async def test_move_rejects_unknown_clip_ids() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(_FakeS3Client({manifest_key: _manifest_bytes([])}))

    with pytest.raises(UnknownClipsError, match=_UUID_4):
        await store.move(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            target_sub_group=ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_4]],
        )


@pytest.mark.asyncio
async def test_move_rejects_clip_ids_already_in_target_sub_group() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        ),
                        _entry(
                            id=_UUID_2,
                            video_hash=_HASH_B,
                            sub_season=SubSeason.NONE,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        ),
                    ]
                )
            }
        )
    )

    with pytest.raises(
        ValueError,
        match='move\\(\\) only supports actual cross-sub-group moves; target sub-group clips must not be included',
    ):
        await store.move(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            target_sub_group=ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_2, _UUID_1]],
        )


@pytest.mark.asyncio
async def test_remove_deletes_authoritative_objects_and_compacts_affected_sub_groups() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    clip_key_4 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalization = AudioNormalization(loudness=-14, bitrate=128)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=3,
                        order=1,
                        audio_normalization=normalization,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=7,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.NONE,
                        scope=Scope.EXTRA,
                        batch=2,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_4,
                        video_hash=_HASH_D,
                        sub_season=SubSeason.NONE,
                        scope=Scope.EXTRA,
                        batch=4,
                        order=1,
                    ),
                ]
            ),
            clip_key_1: b'one',
            clip_key_2: b'two',
            clip_key_3: b'three',
            clip_key_4: b'four',
            normalized_key_1: b'normalized-one',
        }
    )
    store = _store(s3_client)

    affected_sub_groups = await store.remove(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        clip_ids=[_UUID_1, _UUID_3],
    )

    assert set(affected_sub_groups) == {
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        ClipSubGroup(sub_season=SubSeason.NONE, scope=Scope.EXTRA),
    }
    assert clip_key_1 not in s3_client.objects
    assert normalized_key_1 not in s3_client.objects
    assert clip_key_3 not in s3_client.objects
    assert clip_key_2 in s3_client.objects
    assert clip_key_4 in s3_client.objects
    assert s3_client.deleted_keys == [clip_key_1, normalized_key_1, clip_key_3]
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_4,
                video_hash=_HASH_D,
                sub_season=SubSeason.NONE,
                scope=Scope.EXTRA,
                batch=1,
                order=1,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_remove_rejects_empty_clip_ids() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ValueError, match='remove\\(\\) requires at least one clip id'):
        await store.remove(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            clip_ids=[],
        )


@pytest.mark.asyncio
async def test_remove_rejects_duplicate_clip_ids() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(DuplicateClipIdsError, match=_UUID_1):
        await store.remove(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            clip_ids=[_UUID_1, _UUID_1],
        )


@pytest.mark.asyncio
async def test_remove_rejects_unknown_clip_ids() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(_FakeS3Client({manifest_key: _manifest_bytes([])}))

    with pytest.raises(UnknownClipsError, match=_UUID_4):
        await store.remove(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            clip_ids=[_UUID_4],
        )


@pytest.mark.asyncio
async def test_remove_raises_cleanup_error_when_raw_delete_fails_after_manifest_commit() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    original_manifest = _manifest_bytes(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            ),
        ]
    )
    s3_client = _FakeS3Client(
        {
            manifest_key: original_manifest,
            clip_key_1: b'one',
        },
        delete_failures={clip_key_1},
    )
    store = _store(s3_client)

    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_group_prefix = store._clip_group_prefix(
        universe=clip_group.universe,
        year=clip_group.year,
        season=clip_group.season,
    )

    with pytest.raises(ClipRemoveManifestSyncError, match='raw_clip_delete') as excinfo:
        await store.remove(
            clip_group,
            clip_ids=[_UUID_1],
        )

    assert excinfo.value.stage == 'raw_clip_delete'
    assert excinfo.value.clip_ids == (_UUID_1,)
    assert excinfo.value.touched_keys == (clip_key_1,)
    assert excinfo.value.manifest_key == manifest_key
    assert excinfo.value.manifest_committed is True
    assert 'manifest has already been committed' in str(excinfo.value)
    assert _UUID_1 in str(excinfo.value)
    assert clip_key_1 in str(excinfo.value)
    assert 'removed logically' in str(excinfo.value)
    assert 'Manual cleanup or inspection may be required' in str(excinfo.value)
    assert f"RuntimeError('boom deleting {clip_key_1}')" in str(excinfo.value)
    assert clip_key_1 in s3_client.objects
    assert manifest_key not in s3_client.objects
    assert clip_group_prefix not in store._manifest_cache


@pytest.mark.asyncio
async def test_remove_raises_cleanup_error_when_normalized_delete_fails_after_manifest_commit() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalization = AudioNormalization(loudness=-14, bitrate=128)
    original_manifest = _manifest_bytes(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
                audio_normalization=normalization,
            )
        ]
    )
    s3_client = _FakeS3Client(
        {
            manifest_key: original_manifest,
            clip_key_1: b'one',
            normalized_key_1: b'normalized-one',
        },
        delete_failures={normalized_key_1},
    )
    store = _store(s3_client)

    with pytest.raises(ClipRemoveManifestSyncError, match='normalized_clip_delete') as excinfo:
        await store.remove(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            clip_ids=[_UUID_1],
        )

    assert excinfo.value.stage == 'normalized_clip_delete'
    assert excinfo.value.clip_ids == (_UUID_1,)
    assert excinfo.value.touched_keys == (clip_key_1, normalized_key_1)
    assert excinfo.value.manifest_key == manifest_key
    assert excinfo.value.manifest_committed is True
    assert 'manifest has already been committed' in str(excinfo.value)
    assert normalized_key_1 in str(excinfo.value)
    assert 'removed logically' in str(excinfo.value)
    assert f"RuntimeError('boom deleting {normalized_key_1}')" in str(excinfo.value)
    assert clip_key_1 not in s3_client.objects
    assert normalized_key_1 in s3_client.objects
    assert manifest_key not in s3_client.objects


@pytest.mark.asyncio
async def test_remove_deletes_manifest_instead_of_writing_empty_manifest_for_last_clip() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    )
                ]
            ),
            clip_key_1: b'one',
        }
    )
    store = _store(s3_client)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_group_prefix = store._clip_group_prefix(
        universe=clip_group.universe,
        year=clip_group.year,
        season=clip_group.season,
    )

    await store.remove(
        clip_group,
        clip_ids=[_UUID_1],
    )

    await store.compact(
        clip_group,
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        batch_size=2,
        require_exists=False,
    )

    assert clip_key_1 not in s3_client.objects
    assert manifest_key not in s3_client.objects
    assert s3_client.deleted_keys == [manifest_key, clip_key_1]
    assert manifest_key not in [key for key, _, _ in s3_client.put_calls]
    assert clip_group_prefix not in store._manifest_cache


@pytest.mark.asyncio
async def test_remove_raises_sync_error_when_manifest_delete_fails_before_cleanup_starts() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    original_manifest = _manifest_bytes(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            )
        ]
    )
    s3_client = _FakeS3Client(
        {
            manifest_key: original_manifest,
            clip_key_1: b'one',
        },
        delete_failures={manifest_key},
    )
    store = _store(s3_client)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_group_prefix = store._clip_group_prefix(
        universe=clip_group.universe,
        year=clip_group.year,
        season=clip_group.season,
    )

    with pytest.raises(ClipRemoveManifestSyncError, match='manifest_delete') as excinfo:
        await store.remove(
            clip_group,
            clip_ids=[_UUID_1],
        )

    assert excinfo.value.stage == 'manifest_delete'
    assert excinfo.value.clip_ids == (_UUID_1,)
    assert excinfo.value.touched_keys == (manifest_key,)
    assert excinfo.value.manifest_key == manifest_key
    assert excinfo.value.manifest_committed is False
    assert 'manifest commit failed before cleanup started' in str(excinfo.value)
    assert 'Logical state remains unchanged' in str(excinfo.value)
    assert clip_key_1 in s3_client.objects
    assert manifest_key in s3_client.objects
    assert s3_client.objects[manifest_key] == original_manifest
    assert clip_group_prefix in store._manifest_cache


@pytest.mark.asyncio
async def test_remove_raises_sync_error_when_manifest_write_fails_before_cleanup_starts() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalization = AudioNormalization(loudness=-14, bitrate=128)
    original_manifest = _manifest_bytes(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
                audio_normalization=normalization,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=2,
                order=1,
            ),
        ]
    )
    s3_client = _FakeS3Client(
        {
            manifest_key: original_manifest,
            clip_key_1: b'one',
            clip_key_2: b'two',
            normalized_key_1: b'normalized-one',
        },
        put_failures={manifest_key},
    )
    store = _store(s3_client)

    with pytest.raises(ClipRemoveManifestSyncError, match='manifest_write') as excinfo:
        await store.remove(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            clip_ids=[_UUID_1],
        )

    assert excinfo.value.stage == 'manifest_write'
    assert excinfo.value.clip_ids == (_UUID_1,)
    assert excinfo.value.touched_keys == (manifest_key,)
    assert excinfo.value.manifest_key == manifest_key
    assert excinfo.value.manifest_committed is False
    assert 'manifest commit failed before cleanup started' in str(excinfo.value)
    assert 'Logical state remains unchanged' in str(excinfo.value)
    assert clip_key_1 in s3_client.objects
    assert normalized_key_1 in s3_client.objects
    assert s3_client.objects[manifest_key] == original_manifest


@pytest.mark.asyncio
async def test_reconcile_reorders_and_rebatches_target_sub_group() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    clip_key_4 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_4)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=3,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=3,
                        order=2,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=10,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_4,
                        video_hash=_HASH_D,
                        sub_season=SubSeason.NONE,
                        scope=Scope.EXTRA,
                        batch=1,
                        order=1,
                    ),
                ]
            ),
            clip_key_1: b'one',
            clip_key_2: b'two',
            clip_key_3: b'three',
            clip_key_4: b'four',
        }
    )
    store = _store(s3_client)

    result = await store.reconcile(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clip_id_batches=[
            [_UUID_3, _UUID_1],
            [_UUID_2],
        ],
    )

    assert result == ReconcileResult(updated=3, removed=0)
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_4,
                video_hash=_HASH_D,
                sub_season=SubSeason.NONE,
                scope=Scope.EXTRA,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_3,
                video_hash=_HASH_C,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=2,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=2,
                order=1,
            ),
        ]
    )
    assert [key for key, _, _ in s3_client.put_calls] == [manifest_key]
    assert s3_client.deleted_keys == []


@pytest.mark.asyncio
async def test_reconcile_moves_from_other_sub_group_and_deletes_omitted_clip() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    clip_key_3 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_3)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    prior_normalization = AudioNormalization(loudness=-14, bitrate=128)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                        audio_normalization=prior_normalization,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.NONE,
                        scope=Scope.EXTRA,
                        batch=2,
                        order=1,
                    ),
                ]
            ),
            clip_key_1: b'one',
            clip_key_2: b'two',
            clip_key_3: b'three',
            normalized_key_1: b'normalized-one',
        }
    )
    store = _store(s3_client)

    result = await store.reconcile(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        clip_id_batches=[
            [_UUID_3, _UUID_2],
        ],
    )

    assert result == ReconcileResult(updated=2, removed=1)
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_3,
                video_hash=_HASH_C,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=2,
            ),
        ]
    )
    assert s3_client.deleted_keys == [clip_key_1, normalized_key_1]
    assert clip_key_1 not in s3_client.objects
    assert normalized_key_1 not in s3_client.objects


@pytest.mark.asyncio
async def test_reconcile_rejects_duplicate_clip_ids() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(DuplicateClipIdsError):
        await store.reconcile(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_1, _UUID_1]],
        )


@pytest.mark.asyncio
async def test_reconcile_rejects_empty_clip_id_batches() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ValueError, match='reconcile\\(\\) clip_id_batches must not contain empty batches'):
        await store.reconcile(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[]],
        )


@pytest.mark.asyncio
async def test_reconcile_rejects_empty_inner_batch_in_mixed_input() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ValueError, match='reconcile\\(\\) clip_id_batches must not contain empty batches'):
        await store.reconcile(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_1], [], [_UUID_2]],
        )


@pytest.mark.asyncio
async def test_reconcile_rejects_unknown_clip_ids() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(UnknownClipsError, match=_UUID_4):
        await store.reconcile(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_4]],
        )


@pytest.mark.asyncio
async def test_reconcile_rejects_clip_ids_missing_from_provided_group_manifest() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.A,
                            scope=Scope.COLLECTION,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(UnknownClipsError, match=_UUID_4):
        await store.reconcile(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_4]],
        )


@pytest.mark.asyncio
async def test_reconcile_raises_cleanup_error_after_manifest_commit_and_updates_cache() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                    ),
                ]
            ),
            clip_key_1: b'one',
            clip_key_2: b'two',
        },
        delete_failures={clip_key_1},
    )
    store = _store(s3_client)

    with pytest.raises(ReconcileDeleteError, match='raw_clip_delete') as excinfo:
        await store.reconcile(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            clip_id_batches=[[_UUID_2]],
        )

    assert excinfo.value.stage == 'raw_clip_delete'
    assert excinfo.value.clip_ids == (_UUID_1,)
    assert excinfo.value.failed_keys == (clip_key_1,)
    assert excinfo.value.manifest_key == manifest_key
    assert excinfo.value.manifest_committed is True
    assert 'manifest has already been committed' in str(excinfo.value)
    assert _UUID_1 in str(excinfo.value)
    assert clip_key_1 in str(excinfo.value)
    assert 'logical state now follows that manifest' in str(excinfo.value)
    assert f"RuntimeError('boom deleting {clip_key_1}')" in str(excinfo.value)
    expected_manifest = _manifest_payload(
        [
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sub_season=SubSeason.A,
                scope=Scope.COLLECTION,
                batch=1,
                order=1,
            )
        ]
    )
    assert list(store._manifest_cache.values())[0].to_dict() == expected_manifest
    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == expected_manifest


@pytest.mark.asyncio
async def test_remove_stops_before_normalized_cleanup_when_raw_delete_fails() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    normalized_key_1 = _normalized_clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                        audio_normalization=AudioNormalization(loudness=-14, bitrate=128),
                    )
                ]
            ),
            clip_key_1: b'one',
            normalized_key_1: b'normalized-one',
        },
        delete_failures={clip_key_1},
    )
    store = _store(s3_client)

    with pytest.raises(ClipRemoveManifestSyncError, match='raw_clip_delete') as excinfo:
        await store.remove(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            clip_ids=[_UUID_1],
        )

    assert excinfo.value.touched_keys == (clip_key_1,)
    assert normalized_key_1 in s3_client.objects
    assert s3_client.deleted_keys == [manifest_key]


@pytest.mark.asyncio
async def test_compact_rejects_batch_size_below_one() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ValueError, match='`batch_size` must be >= 1'):
        await store.compact(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            batch_size=0,
        )


@pytest.mark.asyncio
async def test_compact_fails_with_empty_sub_group_fields_when_group_is_missing() -> None:
    store = _store(_FakeS3Client())

    with pytest.raises(ClipGroupNotFoundError) as excinfo:
        await store.compact(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            batch_size=2,
            require_exists=True,
        )

    assert excinfo.value.year == 2024
    assert excinfo.value.season is Season.S1
    assert excinfo.value.universe is Universe.WEST
    assert excinfo.value.sub_season is None
    assert excinfo.value.scope is None


@pytest.mark.asyncio
async def test_compact_fails_with_requested_sub_group_fields_when_sub_group_is_missing() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    store = _store(
        _FakeS3Client(
            {
                manifest_key: _manifest_bytes(
                    [
                        _entry(
                            id=_UUID_1,
                            video_hash=_HASH_A,
                            sub_season=SubSeason.B,
                            scope=Scope.EXTRA,
                            batch=1,
                            order=1,
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(ClipGroupNotFoundError) as excinfo:
        await store.compact(
            ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
            ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
            batch_size=2,
            require_exists=True,
        )

    assert excinfo.value.year == 2024
    assert excinfo.value.season is Season.S1
    assert excinfo.value.universe is Universe.WEST
    assert excinfo.value.sub_season is SubSeason.A
    assert excinfo.value.scope is Scope.COLLECTION


@pytest.mark.asyncio
async def test_compact_ignores_missing_sub_group_when_require_exists_is_false() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.B,
                        scope=Scope.EXTRA,
                        batch=1,
                        order=1,
                    )
                ]
            )
        }
    )
    store = _store(s3_client)

    await store.compact(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        batch_size=2,
        require_exists=False,
    )

    assert s3_client.put_calls == []


@pytest.mark.asyncio
async def test_compact_preserves_relative_order_while_rewriting_positions() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    normalization = AudioNormalization(loudness=-14, bitrate=128)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_4,
                        video_hash=_HASH_D,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=10,
                        order=1,
                        audio_normalization=normalization,
                    ),
                    _entry(
                        id=_UUID_5,
                        video_hash='e' * 64,
                        sub_season=SubSeason.NONE,
                        scope=Scope.EXTRA,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=3,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_3,
                        video_hash=_HASH_C,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=3,
                        order=2,
                    ),
                ]
            )
        }
    )
    store = _store(s3_client)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)

    await store.compact(
        clip_group,
        clip_sub_group,
        batch_size=2,
    )

    rewritten_manifest = Manifest.from_dict(json.loads(s3_client.objects[manifest_key].decode('utf-8')))
    compacted_entries = sorted(
        (entry for entry in rewritten_manifest if entry.sub_season is SubSeason.A and entry.scope is Scope.COLLECTION),
        key=lambda entry: (entry.batch, entry.order),
    )

    assert [entry.id for entry in compacted_entries] == [_UUID_1, _UUID_2, _UUID_3, _UUID_4]
    assert [(entry.batch, entry.order) for entry in compacted_entries] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert [entry.audio_normalization for entry in compacted_entries] == [None, None, None, normalization]


@pytest.mark.asyncio
async def test_compact_only_affects_specified_sub_group_and_leaves_others_unchanged() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    original_other_entries = [
        _entry(
            id=_UUID_4,
            video_hash=_HASH_D,
            sub_season=SubSeason.NONE,
            scope=Scope.EXTRA,
            batch=7,
            order=1,
        ),
        _entry(
            id=_UUID_5,
            video_hash='e' * 64,
            sub_season=SubSeason.A,
            scope=Scope.SOURCE,
            batch=3,
            order=2,
        ),
    ]
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    original_other_entries[0],
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    original_other_entries[1],
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=4,
                        order=1,
                    ),
                ]
            )
        }
    )
    store = _store(s3_client)

    await store.compact(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        batch_size=2,
    )

    rewritten_manifest = Manifest.from_dict(json.loads(s3_client.objects[manifest_key].decode('utf-8')))
    rewritten_other_entries = [
        entry
        for entry in rewritten_manifest
        if not (entry.sub_season is SubSeason.A and entry.scope is Scope.COLLECTION)
    ]

    assert rewritten_other_entries == original_other_entries


@pytest.mark.asyncio
async def test_compact_does_not_upload_manifest_when_positions_do_not_change() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=2,
                    ),
                ]
            )
        }
    )
    store = _store(s3_client)

    await store.compact(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        batch_size=2,
    )

    assert s3_client.put_calls == []


@pytest.mark.asyncio
async def test_compact_updates_manifest_cache_consistently_after_rewrite() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    original_manifest = [
        _entry(
            id=_UUID_1,
            video_hash=_HASH_A,
            sub_season=SubSeason.A,
            scope=Scope.COLLECTION,
            batch=1,
            order=1,
        ),
        _entry(
            id=_UUID_2,
            video_hash=_HASH_B,
            sub_season=SubSeason.A,
            scope=Scope.COLLECTION,
            batch=3,
            order=1,
        ),
    ]
    s3_client = _FakeS3Client({manifest_key: _manifest_bytes(original_manifest)})
    store = _store(s3_client)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION)

    await store.compact(
        clip_group,
        clip_sub_group,
        batch_size=2,
    )

    clip_group_prefix = store._clip_group_prefix(
        year=clip_group.year,
        season=clip_group.season,
        universe=clip_group.universe,
    )
    cached_entries = list(store._manifest_cache[clip_group_prefix])
    assert [(entry.id, entry.batch, entry.order) for entry in cached_entries] == [
        (_UUID_1, 1, 1),
        (_UUID_2, 1, 2),
    ]

    s3_client.objects[manifest_key] = _manifest_bytes(original_manifest)
    await store.compact(
        clip_group,
        clip_sub_group,
        batch_size=2,
    )

    assert len(s3_client.put_calls) == 1


@pytest.mark.asyncio
async def test_compact_is_manifest_only_and_does_not_touch_clip_objects() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    clip_key_1 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_1)
    clip_key_2 = _clip_key(year=2024, season=Season.S1, universe=Universe.WEST, clip_id=_UUID_2)
    s3_client = _FakeS3Client(
        {
            manifest_key: _manifest_bytes(
                [
                    _entry(
                        id=_UUID_1,
                        video_hash=_HASH_A,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=1,
                        order=1,
                    ),
                    _entry(
                        id=_UUID_2,
                        video_hash=_HASH_B,
                        sub_season=SubSeason.A,
                        scope=Scope.COLLECTION,
                        batch=2,
                        order=1,
                    ),
                ]
            ),
            clip_key_1: b'clip-1',
            clip_key_2: b'clip-2',
        }
    )
    store = _store(s3_client)

    await store.compact(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        batch_size=2,
    )

    assert s3_client.get_calls == [manifest_key]
    assert [call[0] for call in s3_client.put_calls] == [manifest_key]
    assert s3_client.objects[clip_key_1] == b'clip-1'
    assert s3_client.objects[clip_key_2] == b'clip-2'
    assert s3_client.deleted_keys == []


@pytest.mark.asyncio
async def test_compact_can_pull_newly_stored_single_clip_into_previous_batch_when_there_is_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hashes(monkeypatch, {b'first': _HASH_A, b'second': _HASH_B})
    _patch_uuid7(monkeypatch, _UUID_1, _UUID_2)
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    s3_client = _FakeS3Client()
    store = _store(s3_client)
    clip_group = ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1)
    clip_sub_group = ClipSubGroup(sub_season=SubSeason.NONE, scope=Scope.EXTRA)

    await store.store(
        clip_group,
        clip_sub_group,
        clips=[_mp4_file(b'first')],
    )
    await store.store(
        clip_group,
        clip_sub_group,
        clips=[_mp4_file(b'second')],
    )

    await store.compact(
        clip_group,
        clip_sub_group,
        batch_size=2,
    )

    assert json.loads(s3_client.objects[manifest_key].decode('utf-8')) == _manifest_payload(
        [
            _entry(
                id=_UUID_1,
                video_hash=_HASH_A,
                sampled_phashes=_sampled_phashes_for(b'first'),
                sub_season=SubSeason.NONE,
                scope=Scope.EXTRA,
                batch=1,
                order=1,
            ),
            _entry(
                id=_UUID_2,
                video_hash=_HASH_B,
                sampled_phashes=_sampled_phashes_for(b'second'),
                sub_season=SubSeason.NONE,
                scope=Scope.EXTRA,
                batch=1,
                order=2,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_compact_with_batch_size_ten_creates_dense_batches_with_final_partial() -> None:
    manifest_key = _manifest_key(year=2024, season=Season.S1, universe=Universe.WEST)
    entries = [
        _entry(
            id=uuid.uuid7().hex,
            video_hash=f'{index + 1:064x}',
            sub_season=SubSeason.A,
            scope=Scope.COLLECTION,
            batch=(index * 3) + 1,
            order=1,
        )
        for index in range(12)
    ]
    s3_client = _FakeS3Client({manifest_key: _manifest_bytes(entries)})
    store = _store(s3_client)

    await store.compact(
        ClipGroup(universe=Universe.WEST, year=2024, season=Season.S1),
        ClipSubGroup(sub_season=SubSeason.A, scope=Scope.COLLECTION),
        batch_size=10,
    )

    rewritten_manifest = Manifest.from_dict(json.loads(s3_client.objects[manifest_key].decode('utf-8')))
    compacted_entries = sorted(
        (entry for entry in rewritten_manifest if entry.sub_season is SubSeason.A and entry.scope is Scope.COLLECTION),
        key=lambda entry: (entry.batch, entry.order),
    )

    assert [(entry.batch, entry.order) for entry in compacted_entries] == [
        *[(1, order) for order in range(1, 11)],
        (2, 1),
        (2, 2),
    ]
