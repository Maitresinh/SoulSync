"""Album-bundle dispatch for torrent / usenet single-source downloads.

Lifted from ``run_full_missing_tracks_process`` so the master
worker doesn't carry a 90-line inline branch and so the gate logic
can be unit-tested in isolation.

The gate fires only when ALL conditions hold:

- Batch is an album-context download (``is_album_download`` flag).
- Active download source is ``torrent``, ``usenet``, or ``soulseek``.
  In hybrid mode the caller may pass the first configured source as a
  source override; later hybrid sources stay per-track to preserve fallback.
- Both album-name and artist-name are populated in batch context.
- The resolved plugin exposes ``download_album_to_staging``.

When the gate engages it runs the plugin synchronously (the master
worker is already on a thread-pool executor) and mirrors the
plugin's lifecycle payloads into the batch state so the Downloads
page can render meaningful progress before per-track tasks exist.

Return semantics: ``True`` means the gate handled the batch — the
master worker should stop and not run per-track analysis. ``False``
means the gate didn't engage (or engaged-and-fell-back) — caller
continues the normal per-track flow.

The ``BatchStateAccess`` Protocol exists so this module doesn't
import ``download_batches`` from runtime_state directly. The
caller (master worker) injects accessors so this module stays
testable without touching live runtime state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Protocol
import os
import re
import shutil
from collections import Counter
from datetime import datetime

from utils.logging_config import get_logger

# Use the project logger factory so these lines land in app.log under the
# ``soulsync.*`` namespace the file handler captures. Plain
# ``logging.getLogger(__name__)`` logs to the console only (the file
# handler is attached to the ``soulsync`` logger), which is why
# ``[Album Bundle] flow failed`` showed up in the terminal but never in
# app.log during the #721 triage.
logger = get_logger("downloads.album_bundle_dispatch")


class BatchStateAccess(Protocol):
    """Narrow shim around the batch-state dict ops the dispatch needs.

    Two methods to keep the surface small:
    - ``update_fields(batch_id, fields)`` — atomic merge into the
      batch dict under tasks_lock.
    - ``mark_failed(batch_id, error)`` — convenience for the failure
      path (sets phase + error + album_bundle_state in one shot).
    """

    def update_fields(self, batch_id: str, fields: dict) -> None: ...

    def mark_failed(self, batch_id: str, error: str) -> None: ...


# Fields the album-bundle progress callback may carry. Anything in
# this set gets mirrored onto the batch row as ``album_bundle_<key>``
# so the Downloads page can render it without coupling to the
# specific payload shape.
_MIRRORED_KEYS = ('progress', 'release', 'speed', 'downloaded',
                  'size', 'seeders', 'grabs', 'count', 'failed')


def is_eligible(
    *,
    mode: str,
    is_album: bool,
    album_name: str,
    artist_name: str,
) -> bool:
    """Pure predicate: does this batch even qualify for the album
    flow? Separate from the resolution+run step so tests can pin
    the gate logic without standing up a plugin."""
    if not is_album:
        return False
    if (mode or '').lower() not in ('torrent', 'usenet', 'soulseek'):
        return False
    if not (album_name or '').strip():
        return False
    if not (artist_name or '').strip():
        return False
    return True


def try_dispatch(
    *,
    batch_id: str,
    is_album: bool,
    album_context: Optional[dict],
    artist_context: Optional[dict],
    config_get: Callable[..., Any],
    plugin_resolver: Callable[[str], Optional[Any]],
    state: BatchStateAccess,
    source_override: Optional[str] = None,
    plugin_kwargs: Optional[dict] = None,
) -> bool:
    """Attempt the album-bundle flow. Returns ``True`` iff the
    master worker should return early (gate engaged and completed
    — success OR failure). ``False`` means fall through to the
    normal per-track flow.

    ``config_get`` is a callable shaped like ``config_manager.get``;
    ``plugin_resolver`` resolves a source-name string to an
    initialised plugin instance (or None); ``state`` is the
    BatchStateAccess shim. Injecting these keeps the module
    dependency-light + unit-testable.
    """
    mode = (source_override or config_get('download_source.mode', 'soulseek') or 'soulseek').lower()
    album_name = (album_context or {}).get('name') or ''
    artist_name = (artist_context or {}).get('name') or ''

    if not is_eligible(mode=mode, is_album=is_album,
                       album_name=album_name, artist_name=artist_name):
        return False

    album_name = album_name.strip()
    artist_name = artist_name.strip()

    plugin = None
    try:
        plugin = plugin_resolver(mode)
    except Exception as exc:
        logger.warning("[Album Bundle] Could not resolve %s plugin: %s", mode, exc)

    if plugin is None or not hasattr(plugin, 'download_album_to_staging'):
        logger.warning(
            "[Album Bundle] Gate matched but plugin / context unavailable "
            "(mode=%s album=%r artist=%r plugin=%s) — falling back to per-track flow",
            mode, album_name, artist_name,
            type(plugin).__name__ if plugin else None,
        )
        return False

    staging_root = config_get(
        'download_source.album_bundle_staging_path',
        'storage/album_bundle_staging',
    ) or 'storage/album_bundle_staging'
    staging_dir = str(Path(staging_root) / _safe_batch_dirname(batch_id))
    logger.info(
        "[Album Bundle] Engaging %s album flow for '%s' by '%s' -> %s",
        mode, album_name, artist_name, staging_dir,
    )
    state.update_fields(batch_id, {
        'phase': 'album_downloading',
        'album_bundle_state': 'searching',
        'album_bundle_source': mode,
        'album_bundle_staging_path': staging_dir,
        'album_bundle_private_staging': True,
    })

    def _emit(payload):
        """Mirror plugin lifecycle into batch state for UI rendering."""
        try:
            fields = {'album_bundle_state': payload.get('state', '')}
            for key in _MIRRORED_KEYS:
                if key in payload:
                    fields[f'album_bundle_{key}'] = payload[key]
            state.update_fields(batch_id, fields)
        except Exception as exc:
            logger.debug("[Album Bundle] emit failed: %s", exc)

    try:
        source_plugin_kwargs = dict(plugin_kwargs or {})
        source_plugin_kwargs.pop('expected_track_count', None)
        outcome = plugin.download_album_to_staging(
            album_name, artist_name, staging_dir, _emit,
            **source_plugin_kwargs,
        )
    except Exception as exc:
        logger.exception("[Album Bundle] %s plugin raised: %s", mode, exc)
        # An OSError means an I/O step failed after the source already had the
        # album — most importantly the staging dir not being writable (#760),
        # but also any transient filesystem error. Treat it as fallback-eligible
        # so we return to the per-track flow instead of hard-failing the whole
        # batch (the #715 symptom: files download, then the batch fails).
        # Programming errors (TypeError, KeyError, …) are NOT OSError and stay
        # terminal, so genuine bugs still fail loudly. (requests' network
        # exceptions also subclass OSError, but plugins normally catch those
        # internally and return an outcome rather than raising; if one does
        # surface here, falling back to per-track is still the safe choice.)
        is_io_failure = isinstance(exc, OSError)
        outcome = {
            'success': False,
            'error': f'Plugin error: {exc}',
            'fallback': is_io_failure,
        }

    if not outcome.get('success'):
        err = outcome.get('error', 'Album bundle download failed')
        if outcome.get('fallback'):
            if mode == 'soulseek' and config_get('album_downloads.atomic_publish', False):
                logger.warning(
                    "[Codex Guard] Soulseek album flow could not commit for '%s': %s; "
                    "atomic album mode will not fall back to per-track downloads",
                    album_name, err,
                )
                _discard_atomic_wishlist_album(album_name, err)
                state.mark_failed(batch_id, err)
                return True
            logger.warning(
                "[Album Bundle] %s flow could not commit for '%s': %s — falling back to per-track flow",
                mode, album_name, err,
            )
            state.update_fields(batch_id, {
                'phase': 'analysis',
                'album_bundle_state': 'fallback',
                'album_bundle_error': err,
                'album_bundle_private_staging': False,
                'album_bundle_staging_path': None,
            })
            return False
        logger.error("[Album Bundle] %s flow failed for '%s': %s",
                     mode, album_name, err)
        if mode == 'soulseek' and config_get('album_downloads.atomic_publish', False):
            _discard_atomic_wishlist_album(album_name, err)
        state.mark_failed(batch_id, err)
        return True

    completed_count = outcome.get('completed_count', len(outcome.get('files', [])))
    source_expected_count = _positive_int(outcome.get('expected_count'))
    requested_expected_count = _positive_int((plugin_kwargs or {}).get('expected_track_count'))
    if not requested_expected_count:
        requested_expected_count = _positive_int((album_context or {}).get('total_tracks'))
    expected_count = max(source_expected_count, requested_expected_count)
    if mode == 'soulseek' and config_get('album_downloads.atomic_publish', False):
        if requested_expected_count and completed_count < requested_expected_count:
            err = (
                f"Soulseek album staged only {completed_count}/"
                f"{requested_expected_count} requested tracks"
            )
            logger.warning("[Codex Guard] %s; atomic album mode will not publish a partial album", err)
            _discard_atomic_wishlist_album(album_name, err)
            state.mark_failed(batch_id, err)
            return True
        if outcome.get('partial') or (expected_count and completed_count < expected_count):
            err = f"Soulseek album staged only {completed_count}/{expected_count or '?'} files"
            logger.warning("[Codex Guard] %s; atomic album mode will not publish a partial album", err)
            _discard_atomic_wishlist_album(album_name, err)
            state.mark_failed(batch_id, err)
            return True
        published = _publish_completed_soulseek_album_to_library(
            staging_dir, album_name, artist_name, config_get,
            expected_count=expected_count,
        )
        if published:
            if published.get('already_present'):
                logger.info(
                    "[Codex Guard] Complete Soulseek album '%s' already exists at %s (%d file(s)); skipped duplicate publish",
                    album_name, published['path'], published['count'],
                )
            else:
                logger.info(
                    "[Codex Guard] Published complete Soulseek album '%s' to %s (%d file(s))",
                    album_name, published['path'], published['count'],
                )
            state.update_fields(batch_id, {
                'phase': 'complete',
                'album_bundle_state': 'published',
                'album_bundle_partial': False,
                'album_bundle_expected_count': expected_count,
                'album_bundle_completed_count': completed_count,
                'album_bundle_published_path': published['path'],
                'album_bundle_already_present': bool(published.get('already_present')),
                'completed_tracks': published['count'],
                'total_tracks': published['count'],
            })
            _remove_atomic_wishlist_album(album_name, 'downloaded_complete_album')
            return True
        err = 'Soulseek album staged but could not be published to library'
        state.mark_failed(batch_id, err)
        return True

    logger.info(
        "[Album Bundle] %s staged %d files for '%s' — handing off to per-track staging matcher",
        mode, len(outcome.get('files', [])), album_name,
    )
    state.update_fields(batch_id, {
        'phase': 'analysis',
        'album_bundle_state': 'staged',
        'album_bundle_partial': bool(outcome.get('partial')),
        'album_bundle_expected_count': expected_count,
        'album_bundle_completed_count': completed_count,
    })
    # Engaged-and-succeeded: we DON'T early-return because the
    # per-track flow needs to run to create + complete the per-track
    # task rows. Those tasks will hit try_staging_match and pull the
    # files we just staged.
    return False


def _discard_atomic_wishlist_album(album_name: str, reason: str) -> None:
    """Defer a wishlist album when all currently visible Soulseek sources failed."""
    if not (album_name or '').strip():
        return
    try:
        from core.wishlist.processing import _defer_wishlist_album_rows

        short_reason = re.sub(r'\s+', ' ', str(reason or 'unavailable')).strip()[:120]
        deferred = _defer_wishlist_album_rows(
            album_name,
            f'soulseek_atomic_unavailable:{short_reason}',
            logger=logger,
        )
        logger.info(
            "[Codex Guard] Deferred atomic album to bottom of wishlist: '%s' (%d row(s), reason=%s)",
            album_name,
            deferred,
            short_reason,
        )
    except Exception as exc:
        logger.warning(
            "[Codex Guard] Could not defer atomic album '%s' in wishlist: %s",
            album_name,
            exc,
        )


def _remove_atomic_wishlist_album(album_name: str, reason: str) -> None:
    """Remove an album from wishlist after an atomic Soulseek terminal result."""
    if not (album_name or '').strip():
        return
    try:
        from core.wishlist.processing import _remove_wishlist_album_rows

        short_reason = re.sub(r'\s+', ' ', str(reason or 'unavailable')).strip()[:120]
        removed = _remove_wishlist_album_rows(
            album_name,
            short_reason,
            logger=logger,
        )
        logger.info(
            "[Codex Guard] Removed atomic album from wishlist: '%s' (%d row(s), reason=%s)",
            album_name,
            removed,
            short_reason,
        )
    except Exception as exc:
        logger.warning(
            "[Codex Guard] Could not remove atomic album '%s' from wishlist: %s",
            album_name,
            exc,
        )


def _safe_batch_dirname(batch_id: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(batch_id or 'batch'))
    return safe or 'batch'


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


_AUDIO_EXTS = {'.flac', '.mp3', '.m4a', '.ogg', '.opus', '.wav', '.aiff', '.aif'}
_VAR_ARTISTS = {'various', 'various artists', 'compilation', 'compilations', 'v/a', 'va'}


def _safe_component(value: Any) -> str:
    text = str(value or '').strip() or 'Unknown'
    text = re.sub(r'[\\/:*?"<>|]+', '_', text)
    text = re.sub(r'\s+', ' ', text).strip(' .')
    return text[:180] or 'Unknown'


def _tag_first(tags: Any, names: tuple[str, ...]) -> str:
    if not tags:
        return ''
    for name in names:
        val = tags.get(name) if hasattr(tags, 'get') else None
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            val = val[0] if val else ''
        text = str(val).strip()
        if text:
            return text
    return ''


def _read_audio_tags(path: Path) -> dict[str, str]:
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(str(path), easy=True)
    except Exception:
        audio = None
    tags = getattr(audio, 'tags', None) if audio is not None else None
    return {
        'albumartist': _tag_first(tags, ('albumartist', 'albumartistsort')),
        'artist': _tag_first(tags, ('artist', 'artistsort')),
        'album': _tag_first(tags, ('album',)),
        'discnumber': _tag_first(tags, ('discnumber',)),
    }


def _normalise_disc_number(raw_disc: str, filename: str) -> int:
    """Return a sane disc number for library folders.

    Soulseek shares sometimes carry polluted ``discnumber`` tags where the
    track number is stored as the disc number: 10, 14, 101, 211, etc.  Using
    those values verbatim creates Plex folders such as ``Disc 101`` and splits
    a single album into many one-track albums.  Keep normal disc values, infer
    101/211-style combined numbers as disc 1/2, and otherwise fall back to 1.
    """
    text = str(raw_disc or '').strip()
    first = re.split(r'[/\\]', text)[0].strip() if text else ''
    if first.isdigit():
        value = int(first)
        if 1 <= value <= 9:
            return value
        if value >= 100:
            inferred = int(str(value)[0])
            if 1 <= inferred <= 9:
                logger.info(
                    "[Codex Guard] Corrected polluted discnumber=%s -> disc=%s for %s",
                    value,
                    inferred,
                    filename,
                )
                return inferred
        logger.info(
            "[Codex Guard] Ignoring implausible discnumber=%s for %s; using Disc 1",
            value,
            filename,
        )
        return 1

    m = re.match(r'^(\d{3})[.\-\s_]', filename)
    if m:
        inferred = int(m.group(1)[0])
        if 1 <= inferred <= 9:
            return inferred
    return 1


def _publish_completed_soulseek_album_to_library(
    staging_dir: str,
    album_name: str,
    artist_name: str,
    config_get: Callable[[str, Any], Any],
    *,
    expected_count: int = 0,
) -> Optional[dict[str, Any]]:
    staging = Path(staging_dir)
    if not staging.is_dir():
        logger.warning('[Codex Guard] Soulseek staging dir missing: %s', staging_dir)
        return None
    files = sorted(p for p in staging.rglob('*') if p.is_file() and p.suffix.lower() in _AUDIO_EXTS)
    if not files:
        logger.warning('[Codex Guard] Soulseek staging dir has no audio files: %s', staging_dir)
        return None

    tag_rows = [_read_audio_tags(p) for p in files[:80]]
    album_title_counts = Counter(t['album'] for t in tag_rows if t.get('album'))
    album_title = album_title_counts.most_common(1)[0][0] if album_title_counts else album_name
    album_artists = [t['albumartist'] for t in tag_rows if t.get('albumartist')]
    artists = [t['artist'] for t in tag_rows if t.get('artist')]
    album_artist = Counter(album_artists).most_common(1)[0][0] if album_artists else artist_name
    artist_variety = len({a.lower() for a in artists if a})
    folder_artist = 'Compilations' if album_artist.lower() in _VAR_ARTISTS or artist_variety >= 4 else album_artist

    transfer_dir = Path(str(config_get('soulseek.transfer_path', '/host/music') or '/host/music'))
    if folder_artist == 'Compilations':
        dest = transfer_dir / 'Compilations' / _safe_component(album_title)
    else:
        safe_artist = _safe_component(folder_artist)
        dest = transfer_dir / safe_artist / _safe_component(f'{safe_artist} - {album_title}')

    if dest.exists() and any(p.is_file() and p.suffix.lower() in _AUDIO_EXTS for p in dest.rglob('*')):
        existing_count = sum(1 for p in dest.rglob('*') if p.is_file() and p.suffix.lower() in _AUDIO_EXTS)
        min_expected = _positive_int(expected_count) or len(files)
        if existing_count >= min_expected:
            try:
                shutil.rmtree(staging)
            except Exception as exc:
                logger.debug('[Codex Guard] Could not remove duplicate staging dir %s: %s', staging, exc)
            return {'path': str(dest), 'count': existing_count, 'already_present': True}
        dest = dest.with_name(dest.name + ' [SoulSync ' + datetime.now().strftime('%Y%m%d_%H%M%S') + ']')

    count = 0
    for src in files:
        rel_name = src.name
        tag_disc = _read_audio_tags(src).get('discnumber', '')
        disc = _normalise_disc_number(tag_disc, rel_name)
        target_dir = dest / f'Disc {disc}'
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / rel_name
        if target.exists():
            target = target_dir / (target.stem + ' [SoulSync duplicate]' + target.suffix)
        shutil.copy2(src, target)
        count += 1
    return {'path': str(dest), 'count': count}
