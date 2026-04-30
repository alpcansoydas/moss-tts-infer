from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import queue
import tempfile
import threading
import time
import urllib.parse
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, TypeVar

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from moss_tts_nano_runtime import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    NanoTTSService,
)
from text_normalization_pipeline import (
    TextNormalizationSnapshot as SharedTextNormalizationSnapshot,
    WeTextProcessingManager as SharedWeTextProcessingManager,
    prepare_tts_request_texts as shared_prepare_tts_request_texts,
)


APP_DIR = Path(__file__).resolve().parent
DEMO_METADATA_PATH = APP_DIR / "assets" / "demo.jsonl"
PROMPT_UPLOAD_DIR = APP_DIR / ".app_prompt_uploads"


@dataclass(frozen=True)
class DemoEntry:
    demo_id: str
    name: str
    prompt_audio_path: Path
    prompt_audio_relative_path: str
    text: str


def _load_demo_entries() -> list[DemoEntry]:
    if not DEMO_METADATA_PATH.is_file():
        logging.warning("demo metadata file not found: %s", DEMO_METADATA_PATH)
        return []

    demo_entries: list[DemoEntry] = []
    for line_index, raw_line in enumerate(DEMO_METADATA_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            logging.warning("failed to parse demo metadata line=%s path=%s", line_index, DEMO_METADATA_PATH, exc_info=True)
            continue

        prompt_audio_relative_path = str(payload.get("role", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not prompt_audio_relative_path or not text:
            logging.warning("skip invalid demo metadata line=%s role/text missing", line_index)
            continue

        prompt_audio_path = (APP_DIR / prompt_audio_relative_path).resolve()
        if not prompt_audio_path.is_file():
            logging.warning(
                "skip demo metadata line=%s prompt speech missing: %s",
                line_index,
                prompt_audio_path,
            )
            continue

        try:
            prompt_audio_relative_path = str(prompt_audio_path.relative_to(APP_DIR))
        except ValueError:
            logging.warning(
                "skip demo metadata line=%s prompt speech escaped app dir: %s",
                line_index,
                prompt_audio_path,
            )
            continue

        demo_index = len(demo_entries) + 1
        name = str(payload.get("name", "")).strip() or f"Demo {demo_index}: {prompt_audio_path.stem}"
        demo_entries.append(
            DemoEntry(
                demo_id=f"demo-{demo_index}",
                name=name,
                prompt_audio_path=prompt_audio_path,
                prompt_audio_relative_path=prompt_audio_relative_path,
                text=text,
            )
        )
    return demo_entries


def _resolve_vscode_root_path(vscode_proxy_uri: Optional[str], server_port: int) -> Optional[str]:
    if not vscode_proxy_uri:
        return None
    raw = vscode_proxy_uri.strip()
    if not raw or raw == "/":
        return None

    port_str = str(server_port)
    replacements = (
        "{{port}}",
        "{port}",
        "%7B%7Bport%7D%7D",
        "%7b%7bport%7d%7d",
        "%7Bport%7D",
        "%7bport%7d",
    )
    resolved = raw
    for token in replacements:
        resolved = resolved.replace(token, port_str)

    parsed = urllib.parse.urlsplit(resolved)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
    else:
        path = resolved

    if not path.startswith("/"):
        path = "/" + path
    normalized = path.rstrip("/")
    return normalized or None


@dataclass(frozen=True)
class WarmupSnapshot:
    state: str
    progress: float
    message: str
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def failed(self) -> bool:
        return self.state == "failed"


class WarmupManager:
    def __init__(self, runtime: NanoTTSService, text_normalizer_manager: "WeTextProcessingManager | None" = None) -> None:
        self.runtime = runtime
        self.text_normalizer_manager = text_normalizer_manager
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._state = "pending"
        self._progress = 0.0
        self._message = "Waiting for startup warmup."
        self._error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target=self._run, name="nano-tts-warmup", daemon=True)
            self._thread.start()

    def snapshot(self) -> WarmupSnapshot:
        with self._lock:
            return WarmupSnapshot(
                state=self._state,
                progress=self._progress,
                message=self._message,
                error=self._error,
            )

    def ensure_ready(self) -> WarmupSnapshot:
        with self._lock:
            if not self._started:
                self._started = True
                self._thread = threading.Thread(target=self._run, name="nano-tts-warmup", daemon=True)
                self._thread.start()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join()
        return self.snapshot()

    def _set_state(
        self,
        *,
        state: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            if progress is not None:
                self._progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                self._message = message
            self._error = error

    def _run(self) -> None:
        try:
            self._set_state(state="running", progress=0.1, message="Loading Nano-TTS model.", error=None)
            self.runtime.get_model()
            self._set_state(state="running", progress=0.6, message="Running startup warmup synthesis.", error=None)
            result = self.runtime.warmup()
            _maybe_delete_file(result["audio_path"])
            if self.text_normalizer_manager is not None:
                self._set_state(
                    state="running",
                    progress=0.85,
                    message="Loading WeTextProcessing text normalization.",
                    error=None,
                )
                normalization_snapshot = self.text_normalizer_manager.ensure_ready()
                if normalization_snapshot.failed:
                    raise RuntimeError(normalization_snapshot.error or normalization_snapshot.message)
            self._set_state(
                state="ready",
                progress=1.0,
                message=(
                    f"Warmup complete. device={self.runtime.device} "
                    f"elapsed={result['elapsed_seconds']:.2f}s"
                    + (" | WeTextProcessing ready." if self.text_normalizer_manager is not None else "")
                ),
                error=None,
            )
        except Exception as exc:
            logging.exception("Nano-TTS warmup failed")
            self._set_state(state="failed", progress=1.0, message="Warmup failed.", error=str(exc))


T = TypeVar("T")


class RequestRuntimeManager:
    def __init__(self, default_runtime: NanoTTSService) -> None:
        self.default_runtime = default_runtime
        self.default_cpu_threads = max(1, int(os.cpu_count() or 1))
        self._lock = threading.Lock()
        self._cpu_execution_lock = threading.Lock()
        self._cpu_runtime: NanoTTSService | None = None

    @staticmethod
    def normalize_requested_execution_device(requested: str | None) -> str:
        normalized = str(requested or "default").strip().lower()
        if normalized not in {"default", "cpu"}:
            return "default"
        return normalized

    def is_dedicated_cpu_request(self, requested: str | None) -> bool:
        normalized = self.normalize_requested_execution_device(requested)
        return normalized == "cpu" and self.default_runtime.device.type != "cpu"

    def is_cpu_runtime_loaded(self) -> bool:
        with self._lock:
            return self._cpu_runtime is not None

    def _build_cpu_runtime_locked(self) -> NanoTTSService:
        if self._cpu_runtime is not None:
            return self._cpu_runtime
        self._cpu_runtime = NanoTTSService(
            checkpoint_path=self.default_runtime.checkpoint_path,
            audio_tokenizer_path=self.default_runtime.audio_tokenizer_path,
            device="cpu",
            dtype="float32",
            attn_implementation=self.default_runtime.attn_implementation or "auto",
            output_dir=self.default_runtime.output_dir,
            voice_presets=self.default_runtime.voice_presets,
        )
        return self._cpu_runtime

    def resolve_runtime(self, requested: str | None) -> tuple[NanoTTSService, str]:
        normalized = self.normalize_requested_execution_device(requested)
        if normalized != "cpu":
            return self.default_runtime, str(self.default_runtime.device.type)
        if self.default_runtime.device.type == "cpu":
            return self.default_runtime, "cpu"
        with self._lock:
            return self._build_cpu_runtime_locked(), "cpu"

    def _resolve_cpu_threads(self, cpu_threads: int | None) -> int:
        if cpu_threads is None:
            return self.default_cpu_threads
        try:
            normalized_threads = int(cpu_threads)
        except Exception:
            return self.default_cpu_threads
        if normalized_threads <= 0:
            return self.default_cpu_threads
        return max(1, normalized_threads)

    def call_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        callback: Callable[[NanoTTSService], T],
    ) -> tuple[T, str, int | None]:
        runtime, execution_device = self.resolve_runtime(requested_execution_device)
        if runtime.device.type != "cpu":
            return callback(runtime), execution_device, None

        resolved_cpu_threads = self._resolve_cpu_threads(cpu_threads)
        with self._cpu_execution_lock:
            previous_threads = torch.get_num_threads()
            threads_changed = previous_threads != resolved_cpu_threads
            if threads_changed:
                torch.set_num_threads(resolved_cpu_threads)
            try:
                return callback(runtime), execution_device, resolved_cpu_threads
            finally:
                if threads_changed:
                    torch.set_num_threads(previous_threads)

    def iter_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        factory: Callable[[NanoTTSService], Iterator[T]],
    ) -> Iterator[tuple[T, str, int | None]]:
        runtime, execution_device = self.resolve_runtime(requested_execution_device)
        if runtime.device.type != "cpu":
            for item in factory(runtime):
                yield item, execution_device, None
            return

        resolved_cpu_threads = self._resolve_cpu_threads(cpu_threads)
        with self._cpu_execution_lock:
            previous_threads = torch.get_num_threads()
            threads_changed = previous_threads != resolved_cpu_threads
            if threads_changed:
                torch.set_num_threads(resolved_cpu_threads)
            try:
                for item in factory(runtime):
                    yield item, execution_device, resolved_cpu_threads
            finally:
                if threads_changed:
                    torch.set_num_threads(previous_threads)


@dataclass
class StreamingJob:
    stream_id: str
    audio_queue: "queue.Queue[bytes | None]" = field(default_factory=lambda: queue.Queue(maxsize=64))
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    first_audio_at: float | None = None
    completed_at: float | None = None
    state: str = "starting"
    run_status: str = "Starting realtime synthesis..."
    error: str | None = None
    prompt_audio_path: str | None = None
    sample_rate: int = 48000
    channels: int = 2
    emitted_audio_seconds: float = 0.0
    lead_seconds: float = 0.0
    current_chunk_index: int | None = None
    text_chunks: list[str] = field(default_factory=list)
    chunk_index_base: int | None = None
    audio_chunk_ranges: list[tuple[float, float, int]] = field(default_factory=list)
    is_closed: bool = False
    final_result: dict[str, object] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _resolve_playback_chunk_index_locked(self) -> int | None:
        if not self.audio_chunk_ranges:
            return self.current_chunk_index

        playback_audio_seconds = max(0.0, float(self.emitted_audio_seconds) - float(self.lead_seconds))
        for start_seconds, end_seconds, chunk_index in self.audio_chunk_ranges:
            if playback_audio_seconds <= end_seconds + 1e-6:
                return chunk_index
        return self.audio_chunk_ranges[-1][2]

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "stream_id": self.stream_id,
                "state": self.state,
                "run_status": self.run_status,
                "error": self.error,
                "prompt_audio_path": self.prompt_audio_path,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "emitted_audio_seconds": self.emitted_audio_seconds,
                "lead_seconds": self.lead_seconds,
                "current_chunk_index": self.current_chunk_index,
                "playback_chunk_index": self._resolve_playback_chunk_index_locked(),
                "text_chunks": list(self.text_chunks),
                "first_audio_latency_seconds": (
                    None
                    if self.started_at is None or self.first_audio_at is None
                    else max(0.0, self.first_audio_at - self.started_at)
                ),
                "completed_at": self.completed_at,
                "ready": self.state == "done",
                "failed": self.state == "failed",
                "closed": self.is_closed,
            }


class StreamingJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, StreamingJob] = {}

    def create(self) -> StreamingJob:
        stream_id = f"stream-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        job = StreamingJob(stream_id=stream_id)
        with self._lock:
            self._jobs[stream_id] = job
        return job

    def get(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            return self._jobs.get(stream_id)

    def close(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            job = self._jobs.get(stream_id)
        if job is None:
            return None
        with job.lock:
            job.is_closed = True
            job.state = "closed" if job.state not in {"done", "failed"} else job.state
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass
        return job

    def delete(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            return self._jobs.pop(stream_id, None)


def _warmup_status_text(snapshot: WarmupSnapshot) -> str:
    progress_pct = int(round(snapshot.progress * 100.0))
    if snapshot.failed:
        return f"Warmup failed: {snapshot.error or snapshot.message}"
    if snapshot.ready:
        return snapshot.message
    return f"Warmup in progress ({progress_pct}%): {snapshot.message}"


def _format_run_status(result: dict[str, object]) -> str:
    waveform_numpy = np.asarray(result["waveform_numpy"])
    sample_count = int(waveform_numpy.shape[0]) if waveform_numpy.ndim >= 1 else 0
    sample_rate = int(result["sample_rate"])
    audio_seconds = sample_count / sample_rate if sample_rate > 0 else 0.0
    global_attn = str(result.get("effective_global_attn_implementation", "unknown"))
    local_attn = str(result.get("effective_local_attn_implementation", global_attn))
    attn_summary = global_attn if global_attn == local_attn else f"{global_attn}/{local_attn}"
    tts_batch_size = result.get("voice_clone_chunk_batch_size")
    codec_batch_size = result.get("voice_clone_codec_batch_size")
    batch_summary = ""
    if tts_batch_size is not None or codec_batch_size is not None:
        batch_summary = f" | tts_batch={int(tts_batch_size or 1)} | codec_batch={int(codec_batch_size or 1)}"
    execution_summary = ""
    execution_device = result.get("execution_device")
    cpu_threads = result.get("cpu_threads")
    if execution_device:
        execution_summary = f" | exec={execution_device}"
        if cpu_threads is not None:
            execution_summary += f" | cpu_threads={int(cpu_threads)}"
    prompt_audio_display_path = str(result.get("prompt_audio_display_path") or "").strip()
    prompt_audio_path = str(result.get("prompt_audio_path") or "").strip()
    speaker_summary = f"voice={result['voice']}"
    if prompt_audio_display_path:
        if prompt_audio_display_path.lower().startswith("uploaded:"):
            speaker_summary = f"prompt={prompt_audio_display_path.split(':', 1)[1].strip()}"
        else:
            speaker_summary = f"prompt={Path(prompt_audio_display_path).stem}"
    elif prompt_audio_path:
        speaker_summary = f"prompt={Path(prompt_audio_path).stem}"
    return (
        f"Done | mode={result['mode']} | {speaker_summary} | "
        f"attn={attn_summary}{batch_summary}{execution_summary} | audio={audio_seconds:.2f}s | elapsed={float(result['elapsed_seconds']):.2f}s"
    )


def _format_stream_status(snapshot: dict[str, object]) -> str:
    if bool(snapshot.get("failed")):
        return f"Stream failed: {snapshot.get('error') or snapshot.get('run_status') or 'Unknown error'}"
    if bool(snapshot.get("ready")):
        return str(snapshot.get("run_status") or "Stream complete.")
    if bool(snapshot.get("closed")):
        return "Stream closed."
    return str(snapshot.get("run_status") or "Streaming...")


def _normalize_stream_chunk_index(
    raw_chunk_index: object,
    *,
    chunk_count: int,
    current_base: int | None,
) -> tuple[int | None, int | None]:
    try:
        numeric_chunk_index = int(raw_chunk_index)
    except Exception:
        return None, current_base

    if chunk_count <= 0:
        return max(0, numeric_chunk_index), current_base

    normalized_base = current_base
    if normalized_base is None:
        if numeric_chunk_index == 0:
            normalized_base = 0
        elif numeric_chunk_index == chunk_count:
            normalized_base = 1
        elif numeric_chunk_index == 1:
            normalized_base = 1
        else:
            normalized_base = 0

    normalized_chunk_index = numeric_chunk_index - normalized_base
    if 0 <= normalized_chunk_index < chunk_count:
        return normalized_chunk_index, normalized_base
    if 0 <= numeric_chunk_index < chunk_count:
        return numeric_chunk_index, 0
    if 1 <= numeric_chunk_index <= chunk_count:
        return numeric_chunk_index - 1, 1
    return None, normalized_base


def _audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    elif audio_np.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(int(audio_int16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


def _audio_to_pcm16le_bytes(audio_array) -> bytes:
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    elif audio_np.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    return audio_int16.tobytes()


def _read_audio_file_base64(path_value: str | None) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_file():
        return ""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        logging.warning("failed to read audio file for base64 response: %s", path, exc_info=True)
        return ""


def _maybe_delete_file(path_value: str | None) -> None:
    if not path_value:
        return
    try:
        Path(path_value).unlink(missing_ok=True)
    except Exception:
        logging.warning("failed to remove temporary file: %s", path_value, exc_info=True)


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _sanitize_uploaded_prompt_filename(filename: str | None) -> str:
    base_name = Path(str(filename or "")).name.strip()
    if not base_name:
        return "prompt_speech.wav"
    return base_name


def _format_uploaded_prompt_display_name(filename: str | None) -> str:
    return f"Uploaded: {_sanitize_uploaded_prompt_filename(filename)}"


async def _persist_uploaded_prompt_audio(upload: UploadFile | None) -> tuple[str | None, str | None]:
    if upload is None:
        return None, None

    original_filename = _sanitize_uploaded_prompt_filename(upload.filename)
    suffix = Path(original_filename).suffix
    if not suffix or len(suffix) > 16:
        suffix = ".wav"

    PROMPT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    bytes_written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix="prompt-speech-",
            suffix=suffix,
            dir=str(PROMPT_UPLOAD_DIR),
        ) as handle:
            temp_path = handle.name
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    finally:
        await upload.close()

    if not temp_path or bytes_written <= 0:
        _maybe_delete_file(temp_path)
        raise ValueError("Uploaded prompt speech is empty.")

    return temp_path, _format_uploaded_prompt_display_name(original_filename)


def _render_index_html(
    *,
    request: Request,
    runtime: NanoTTSService,
    demo_entries: list[DemoEntry],
    warmup_status: str,
    text_normalization_status: str,
) -> str:
    base_path = request.scope.get("root_path", "").rstrip("/")
    _template_path = Path(__file__).resolve().parent / "templates" / "index.html"
    template = _template_path.read_text(encoding="utf-8")
    demos_payload = [
        {
            "id": demo_entry.demo_id,
            "name": demo_entry.name,
            "prompt_speech": demo_entry.prompt_audio_relative_path,
            "text": demo_entry.text,
        }
        for demo_entry in demo_entries
    ]
    replacements = {
        "__APP_BASE__": json.dumps(base_path),
        "__DEMOS__": json.dumps(demos_payload, ensure_ascii=False),
        "__DEFAULT_DEMO_ID__": json.dumps(demo_entries[0].demo_id if demo_entries else ""),
        "__DEFAULT_ATTN_IMPLEMENTATION__": json.dumps(runtime.attn_implementation or "model_default"),
        "__DEFAULT_CPU_THREADS__": json.dumps(max(1, int(os.cpu_count() or 1))),
        "__WARMUP_STATUS__": warmup_status,
        "__TEXT_NORMALIZATION_STATUS__": text_normalization_status,
        "__CHECKPOINT__": str(runtime.checkpoint_path),
        "__AUDIO_TOKENIZER__": str(runtime.audio_tokenizer_path),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _build_app(
    runtime: NanoTTSService,
    warmup_manager: WarmupManager,
    text_normalizer_manager: WeTextProcessingManager | None,
    root_path: str | None,
) -> FastAPI:
    app = FastAPI(title="MOSS-TTS-Nano Demo", root_path=root_path or "")
    stream_jobs = StreamingJobManager()
    runtime_manager = RequestRuntimeManager(runtime)
    demo_entries = _load_demo_entries()
    demo_entries_by_id = {demo_entry.demo_id: demo_entry for demo_entry in demo_entries}

    def _resolve_voice_clone_text_chunks(
        *,
        text: str,
        voice_clone_max_text_tokens: int,
        cpu_threads: int,
    ) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []

        try:
            chunks, _, _ = runtime_manager.call_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                callback=lambda selected_runtime: selected_runtime.split_voice_clone_text(
                    text=normalized_text,
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                ),
            )
        except Exception:
            logging.warning("failed to resolve playback text chunks", exc_info=True)
            return [normalized_text]

        normalized_chunks = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
        return normalized_chunks or [normalized_text]

    def _resolve_demo_entry(demo_id: str) -> DemoEntry:
        normalized_demo_id = str(demo_id or "").strip()
        if not normalized_demo_id:
            raise ValueError("demo_id is required.")
        demo_entry = demo_entries_by_id.get(normalized_demo_id)
        if demo_entry is None:
            raise ValueError(f"Unknown demo_id: {normalized_demo_id}")
        return demo_entry

    async def _resolve_prompt_audio_request(
        *,
        demo_id: str,
        prompt_audio: UploadFile | None,
    ) -> tuple[DemoEntry | None, str, str, str | None]:
        normalized_demo_id = str(demo_id or "").strip()
        demo_entry = _resolve_demo_entry(normalized_demo_id) if normalized_demo_id else None

        uploaded_prompt_audio_path, uploaded_prompt_audio_display_path = await _persist_uploaded_prompt_audio(prompt_audio)
        if uploaded_prompt_audio_path is not None and uploaded_prompt_audio_display_path is not None:
            return (
                demo_entry,
                uploaded_prompt_audio_path,
                uploaded_prompt_audio_display_path,
                uploaded_prompt_audio_path,
            )

        if demo_entry is None:
            raise ValueError("demo_id is required unless prompt speech is uploaded.")

        return (
            demo_entry,
            str(demo_entry.prompt_audio_path),
            demo_entry.prompt_audio_relative_path,
            None,
        )

    def _stream_metrics_text(snapshot: dict[str, object]) -> str:
        metrics = [
            f"state={snapshot['state']}",
            f"emitted={float(snapshot['emitted_audio_seconds']):.2f}s",
            f"lead={float(snapshot['lead_seconds']):.2f}s",
        ]
        first_audio_latency = snapshot.get("first_audio_latency_seconds")
        if first_audio_latency is not None:
            metrics.append(f"first_audio={float(first_audio_latency):.2f}s")
        return " | ".join(metrics)

    def _text_normalization_status_text(snapshot: SharedTextNormalizationSnapshot | None) -> str:
        if snapshot is None:
            return "WeTextProcessing disabled."
        if snapshot.failed:
            return f"{snapshot.message} error={snapshot.error}"
        return snapshot.message

    def _resolve_attn_for_runtime(selected_runtime: NanoTTSService, requested_attn: str) -> str:
        normalized = str(requested_attn or "model_default").strip().lower()
        if selected_runtime.device.type != "cpu":
            return requested_attn
        if normalized in {"", "auto", "default", "model_default", "flash_attention_2"}:
            return "eager"
        return requested_attn

    def _put_stream_audio(job: StreamingJob, pcm_bytes: bytes) -> None:
        while True:
            with job.lock:
                if job.is_closed:
                    return
            try:
                job.audio_queue.put(pcm_bytes, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run_streaming_job(
        job: StreamingJob,
        *,
        text: str,
        prompt_audio_path: str,
        prompt_audio_display_path: str,
        prompt_audio_cleanup_path: str | None,
        max_new_frames: int,
        voice_clone_max_text_tokens: int,
        tts_max_batch_size: int,
        codec_max_batch_size: int,
        cpu_threads: int,
        attn_implementation: str,
        do_sample: bool,
        text_temperature: float,
        text_top_p: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        seed: int | None,
    ) -> None:
        try:
            initial_execution_label = "cpu"
            with job.lock:
                job.started_at = time.monotonic()
                job.state = "running"
                job.run_status = f"Streaming realtime audio... exec={initial_execution_label}"

            def _stream_factory(selected_runtime: NanoTTSService):
                return selected_runtime.synthesize_stream(
                    text=text,
                    mode="voice_clone",
                    voice=None,
                    prompt_audio_path=prompt_audio_path,
                    max_new_frames=int(max_new_frames),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    tts_max_batch_size=int(tts_max_batch_size),
                    codec_max_batch_size=int(codec_max_batch_size),
                    attn_implementation=_resolve_attn_for_runtime(selected_runtime, attn_implementation),
                    do_sample=bool(do_sample),
                    text_temperature=float(text_temperature),
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                    audio_repetition_penalty=float(audio_repetition_penalty),
                    seed=seed,
                )

            for event, resolved_execution_device, resolved_cpu_threads in runtime_manager.iter_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                factory=_stream_factory,
            ):
                event_type = str(event.get("type", ""))
                with job.lock:
                    if job.is_closed:
                        break

                if event_type == "audio":
                    waveform_numpy = np.asarray(event["waveform_numpy"], dtype=np.float32)
                    pcm_bytes = _audio_to_pcm16le_bytes(waveform_numpy)
                    if not pcm_bytes:
                        continue
                    sample_rate = int(event["sample_rate"])
                    channels = 1 if waveform_numpy.ndim == 1 else int(waveform_numpy.shape[1])
                    is_pause = bool(event.get("is_pause", False))
                    event_duration_seconds = (
                        float(waveform_numpy.shape[0]) / float(sample_rate)
                        if sample_rate > 0 and waveform_numpy.ndim >= 1
                        else 0.0
                    )
                    with job.lock:
                        job.sample_rate = sample_rate
                        job.channels = channels
                        job.emitted_audio_seconds = float(event.get("emitted_audio_seconds", 0.0))
                        job.lead_seconds = float(event.get("lead_seconds", 0.0))
                        normalized_chunk_index, job.chunk_index_base = _normalize_stream_chunk_index(
                            event.get("chunk_index"),
                            chunk_count=len(job.text_chunks),
                            current_base=job.chunk_index_base,
                        )
                        if normalized_chunk_index is not None:
                            job.current_chunk_index = normalized_chunk_index
                            if not is_pause and event_duration_seconds > 0.0:
                                chunk_end_seconds = job.emitted_audio_seconds
                                chunk_start_seconds = max(0.0, chunk_end_seconds - event_duration_seconds)
                                job.audio_chunk_ranges.append(
                                    (chunk_start_seconds, chunk_end_seconds, normalized_chunk_index)
                                )
                        if job.first_audio_at is None and not is_pause:
                            job.first_audio_at = time.monotonic()
                        job.run_status = (
                            f"Streaming | emitted={job.emitted_audio_seconds:.2f}s | lead={job.lead_seconds:.2f}s"
                        )
                    _put_stream_audio(job, pcm_bytes)
                    continue

                if event_type == "result":
                    formatted_result = dict(event)
                    formatted_result["execution_device"] = resolved_execution_device
                    formatted_result["prompt_audio_display_path"] = prompt_audio_display_path
                    if resolved_cpu_threads is not None:
                        formatted_result["cpu_threads"] = resolved_cpu_threads
                    formatted_run_status = _format_run_status(formatted_result)
                    with job.lock:
                        job.final_result = {
                            "audio_path": event.get("audio_path"),
                            "prompt_audio_path": prompt_audio_display_path,
                            "run_status": formatted_run_status,
                            "text_chunks": list(job.text_chunks),
                        }
                        job.prompt_audio_path = prompt_audio_display_path
                        job.state = "done"
                        job.completed_at = time.monotonic()
                        job.run_status = formatted_run_status
        except Exception as exc:
            logging.exception("Nano-TTS realtime streaming job failed")
            with job.lock:
                job.state = "failed"
                job.error = str(exc)
                job.completed_at = time.monotonic()
                job.run_status = f"Stream failed: {exc}"
        finally:
            _maybe_delete_file(prompt_audio_cleanup_path)
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return HTMLResponse(
            _render_index_html(
                request=request,
                runtime=runtime,
                demo_entries=demo_entries,
                warmup_status=_warmup_status_text(warmup_manager.snapshot()),
                text_normalization_status=_text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
            )
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "cpu_runtime_loaded": runtime_manager.is_cpu_runtime_loaded(),
            "default_cpu_threads": runtime_manager.default_cpu_threads,
            "attn_implementation": runtime.attn_implementation or "model_default",
            "checkpoint_default_attn_implementation": runtime._checkpoint_global_attn_implementation or "unknown",
            "checkpoint_default_local_attn_implementation": runtime._checkpoint_local_attn_implementation or "unknown",
            "configured_attn_implementation": runtime._configured_global_attn_implementation or "unknown",
            "configured_local_attn_implementation": runtime._configured_local_attn_implementation or "unknown",
            "checkpoint_path": str(runtime.checkpoint_path),
            "audio_tokenizer_path": str(runtime.audio_tokenizer_path),
            "text_normalization_status": _text_normalization_status_text(
                text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
            ),
        }

    @app.get("/api/warmup-status")
    async def warmup_status():
        snapshot = warmup_manager.snapshot()
        return {
            "state": snapshot.state,
            "progress": snapshot.progress,
            "message": snapshot.message,
            "error": snapshot.error,
            "ready": snapshot.ready,
            "failed": snapshot.failed,
            "status_text": _warmup_status_text(snapshot),
        }

    @app.get("/api/text-normalization-status")
    async def text_normalization_status():
        snapshot = text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
        if snapshot is None:
            return {
                "state": "disabled",
                "message": "WeTextProcessing disabled.",
                "error": None,
                "ready": False,
                "failed": False,
                "available": False,
                "status_text": "WeTextProcessing disabled.",
            }
        return {
            "state": snapshot.state,
            "message": snapshot.message,
            "error": snapshot.error,
            "ready": snapshot.ready,
            "failed": snapshot.failed,
            "available": snapshot.available,
            "status_text": _text_normalization_status_text(snapshot),
        }

    @app.get("/api/demo-prompt-audio/{demo_id}")
    async def demo_prompt_audio(demo_id: str):
        try:
            demo_entry = _resolve_demo_entry(demo_id)
        except ValueError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})

        media_type = "audio/wav" if demo_entry.prompt_audio_path.suffix.lower() == ".wav" else "application/octet-stream"
        return FileResponse(
            path=str(demo_entry.prompt_audio_path),
            media_type=media_type,
            filename=demo_entry.prompt_audio_path.name,
        )

    @app.post("/api/generate-stream/start")
    async def generate_stream_start(
        text: str = Form(...),
        demo_id: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
        max_new_frames: int = Form(375),
        voice_clone_max_text_tokens: int = Form(75),
        tts_max_batch_size: int = Form(0),
        codec_max_batch_size: int = Form(0),
        enable_text_normalization: str = Form("1"),
        enable_normalize_tts_text: str = Form("1"),
        cpu_threads: int = Form(0),
        attn_implementation: str = Form("model_default"),
        do_sample: str = Form("1"),
        text_temperature: float = Form(1.0),
        text_top_p: float = Form(1.0),
        text_top_k: int = Form(50),
        audio_temperature: float = Form(0.8),
        audio_top_p: float = Form(0.95),
        audio_top_k: int = Form(25),
        audio_repetition_penalty: float = Form(1.2),
        seed: str = Form("0"),
    ):
        try:
            demo_entry, prompt_audio_path, prompt_audio_display_path, prompt_audio_cleanup_path = (
                await _resolve_prompt_audio_request(demo_id=demo_id, prompt_audio=prompt_audio)
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        resolved_text = str(text or "").strip() or (demo_entry.text if demo_entry is not None else "")
        if not resolved_text:
            _maybe_delete_file(prompt_audio_cleanup_path)
            return JSONResponse(status_code=400, content={"error": "text is required."})

        try:
            prepared_texts = shared_prepare_tts_request_texts(
                text=resolved_text,
                enable_wetext=_coerce_bool(enable_text_normalization, False),
                enable_normalize_tts_text=_coerce_bool(enable_normalize_tts_text, True),
                text_normalizer_manager=text_normalizer_manager,
            )
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise
        warmup_snapshot = warmup_manager.snapshot()
        if not warmup_snapshot.ready:
            warmup_snapshot = warmup_manager.ensure_ready()
            if not warmup_snapshot.ready:
                _maybe_delete_file(prompt_audio_cleanup_path)
                return JSONResponse(
                    status_code=500,
                    content={"error": _warmup_status_text(warmup_snapshot)},
                )

        try:
            normalized_seed = None if seed in {"", "0"} else int(seed)
            text_chunks = _resolve_voice_clone_text_chunks(
                text=str(prepared_texts["text"]),
                voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                cpu_threads=int(cpu_threads),
            )
            job = stream_jobs.create()
            with job.lock:
                job.prompt_audio_path = prompt_audio_display_path
                job.text_chunks = list(text_chunks)
            thread = threading.Thread(
                target=_run_streaming_job,
                kwargs={
                    "job": job,
                    "text": str(prepared_texts["text"]),
                    "prompt_audio_path": prompt_audio_path,
                    "prompt_audio_display_path": prompt_audio_display_path,
                    "prompt_audio_cleanup_path": prompt_audio_cleanup_path,
                    "max_new_frames": int(max_new_frames),
                    "voice_clone_max_text_tokens": int(voice_clone_max_text_tokens),
                    "tts_max_batch_size": int(tts_max_batch_size),
                    "codec_max_batch_size": int(codec_max_batch_size),
                    "cpu_threads": int(cpu_threads),
                    "attn_implementation": attn_implementation,
                    "do_sample": _coerce_bool(do_sample, True),
                    "text_temperature": float(text_temperature),
                    "text_top_p": float(text_top_p),
                    "text_top_k": int(text_top_k),
                    "audio_temperature": float(audio_temperature),
                    "audio_top_p": float(audio_top_p),
                    "audio_top_k": int(audio_top_k),
                    "audio_repetition_penalty": float(audio_repetition_penalty),
                    "seed": normalized_seed,
                },
                name=f"nano-tts-stream-{job.stream_id}",
                daemon=True,
            )
            thread.start()
            prompt_audio_cleanup_path = None

            initial_execution_label = "cpu"

            return {
                "stream_id": job.stream_id,
                "audio_url": f"{app.root_path}/api/generate-stream/{job.stream_id}/audio",
                "status_url": f"{app.root_path}/api/generate-stream/{job.stream_id}/status",
                "result_url": f"{app.root_path}/api/generate-stream/{job.stream_id}/result",
                "sample_rate": job.sample_rate,
                "channels": job.channels,
                "run_status": f"Streaming realtime audio... exec={initial_execution_label}",
                "prompt_audio_path": prompt_audio_display_path,
                "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
                "text_normalization_status_text": _text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
                "text_chunks": text_chunks,
                "normalized_text": str(prepared_texts["normalized_text"]),
                "normalization_method": str(prepared_texts["normalization_method"]),
                "text_normalization_language": str(prepared_texts["text_normalization_language"]),
            }
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise

    @app.get("/api/generate-stream/{stream_id}/status")
    async def generate_stream_status(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        snapshot = job.snapshot()
        snapshot["status_text"] = _format_stream_status(snapshot)
        snapshot["stream_metrics"] = _stream_metrics_text(snapshot)
        return snapshot

    @app.get("/api/generate-stream/{stream_id}/audio")
    async def generate_stream_audio(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})

        def _iter_audio():
            while True:
                item = job.audio_queue.get()
                if item is None:
                    break
                yield item

        return StreamingResponse(
            _iter_audio(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Codec": "pcm_s16le",
                "X-Audio-Sample-Rate": str(job.sample_rate),
                "X-Audio-Channels": str(job.channels),
                "X-Stream-Id": stream_id,
            },
        )

    @app.get("/api/generate-stream/{stream_id}/result")
    async def generate_stream_result(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        snapshot = job.snapshot()
        if snapshot["failed"]:
            return JSONResponse(status_code=500, content={"error": snapshot["error"], **snapshot})
        if not snapshot["ready"] or job.final_result is None:
            return JSONResponse(status_code=202, content=snapshot)

        result = dict(job.final_result)
        audio_chunk_ranges: list[list[float | int]] = []
        with job.lock:
            audio_chunk_ranges = [
                [float(start_seconds), float(end_seconds), int(chunk_index)]
                for start_seconds, end_seconds, chunk_index in job.audio_chunk_ranges
            ]
        audio_base64_payload = str(result.get("audio_base64") or "")
        audio_path_for_response = str(result.get("audio_path") or "").strip()
        if not audio_base64_payload and audio_path_for_response:
            audio_base64_payload = _read_audio_file_base64(audio_path_for_response)
            if audio_base64_payload:
                with job.lock:
                    if job.final_result is not None:
                        job.final_result["audio_base64"] = audio_base64_payload
                        job.final_result["audio_path"] = ""
                _maybe_delete_file(audio_path_for_response)

        return {
            "stream_id": stream_id,
            "ready": True,
            "state": snapshot["state"],
            "prompt_audio_path": result.get("prompt_audio_path") or snapshot.get("prompt_audio_path") or "",
            "run_status": result.get("run_status") or snapshot["run_status"],
            "stream_metrics": _stream_metrics_text(snapshot),
            "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
            "text_chunks": result.get("text_chunks") or snapshot.get("text_chunks") or [],
            "audio_chunk_ranges": audio_chunk_ranges,
            "audio_base64": audio_base64_payload,
        }
    
    @app.post("/api/generate-stream/{stream_id}/close")
    async def generate_stream_close(stream_id: str):
        job = stream_jobs.close(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        audio_cleanup_path = ""
        with job.lock:
            if job.final_result is not None:
                audio_cleanup_path = str(job.final_result.get("audio_path") or "").strip()
        snapshot = job.snapshot()
        snapshot["status_text"] = _format_stream_status(snapshot)
        stream_jobs.delete(stream_id)
        _maybe_delete_file(audio_cleanup_path)
        return snapshot

    @app.post("/api/generate")
    async def generate(
        text: str = Form(...),
        demo_id: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
        max_new_frames: int = Form(375),
        voice_clone_max_text_tokens: int = Form(75),
        tts_max_batch_size: int = Form(0),
        codec_max_batch_size: int = Form(0),
        enable_text_normalization: str = Form("1"),
        enable_normalize_tts_text: str = Form("1"),
        cpu_threads: int = Form(0),
        attn_implementation: str = Form("model_default"),
        do_sample: str = Form("1"),
        text_temperature: float = Form(1.0),
        text_top_p: float = Form(1.0),
        text_top_k: int = Form(50),
        audio_temperature: float = Form(0.8),
        audio_top_p: float = Form(0.95),
        audio_top_k: int = Form(25),
        audio_repetition_penalty: float = Form(1.2),
        seed: str = Form("0"),
    ):
        try:
            demo_entry, prompt_audio_path, prompt_audio_display_path, prompt_audio_cleanup_path = (
                await _resolve_prompt_audio_request(demo_id=demo_id, prompt_audio=prompt_audio)
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        resolved_text = str(text or "").strip() or (demo_entry.text if demo_entry is not None else "")
        if not resolved_text:
            _maybe_delete_file(prompt_audio_cleanup_path)
            return JSONResponse(status_code=400, content={"error": "text is required."})

        try:
            prepared_texts = shared_prepare_tts_request_texts(
                text=resolved_text,
                enable_wetext=_coerce_bool(enable_text_normalization, False),
                enable_normalize_tts_text=_coerce_bool(enable_normalize_tts_text, True),
                text_normalizer_manager=text_normalizer_manager,
            )
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise
        warmup_snapshot = warmup_manager.snapshot()
        if not warmup_snapshot.ready:
            warmup_snapshot = warmup_manager.ensure_ready()
            if not warmup_snapshot.ready:
                _maybe_delete_file(prompt_audio_cleanup_path)
                return JSONResponse(
                    status_code=500,
                    content={"error": _warmup_status_text(warmup_snapshot)},
                )

        generated_audio_path: str | None = None
        try:
            normalized_seed = None if seed in {"", "0"} else int(seed)

            def _synthesize(selected_runtime: NanoTTSService):
                return selected_runtime.synthesize(
                    text=str(prepared_texts["text"]),
                    mode="voice_clone",
                    voice=None,
                    prompt_audio_path=prompt_audio_path,
                    max_new_frames=int(max_new_frames),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    tts_max_batch_size=int(tts_max_batch_size),
                    codec_max_batch_size=int(codec_max_batch_size),
                    attn_implementation=_resolve_attn_for_runtime(selected_runtime, attn_implementation),
                    do_sample=_coerce_bool(do_sample, True),
                    text_temperature=float(text_temperature),
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                    audio_repetition_penalty=float(audio_repetition_penalty),
                    seed=normalized_seed,
                )

            result, resolved_execution_device, resolved_cpu_threads = runtime_manager.call_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                callback=_synthesize,
            )
            result["execution_device"] = resolved_execution_device
            result["prompt_audio_display_path"] = prompt_audio_display_path
            if resolved_cpu_threads is not None:
                result["cpu_threads"] = resolved_cpu_threads
            text_chunks = [
                str(chunk).strip()
                for chunk in (result.get("voice_clone_text_chunks") or [])
                if str(chunk).strip()
            ]
            if not text_chunks:
                text_chunks = _resolve_voice_clone_text_chunks(
                    text=str(prepared_texts["text"]),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    cpu_threads=int(cpu_threads),
                )
            generated_audio_path = str(result["audio_path"])
            wav_bytes = _audio_to_wav_bytes(result["waveform_numpy"], int(result["sample_rate"]))
            return {
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "sample_rate": int(result["sample_rate"]),
                "run_status": _format_run_status(result),
                "prompt_audio_path": prompt_audio_display_path,
                "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
                "text_normalization_status_text": _text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
                "text_chunks": text_chunks,
                "normalized_text": str(prepared_texts["normalized_text"]),
                "normalization_method": str(prepared_texts["normalization_method"]),
                "text_normalization_language": str(prepared_texts["text_normalization_language"]),
            }
        except Exception as exc:
            logging.exception("Nano-TTS generation failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})
        finally:
            _maybe_delete_file(generated_audio_path)
            _maybe_delete_file(prompt_audio_cleanup_path)

    return app


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano web demo")
    parser.add_argument("--checkpoint-path", "--checkpoint_path", dest="checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument(
        "--audio-tokenizer-path",
        "--audio_tokenizer_path",
        dest="audio_tokenizer_path",
        type=str,
        default=str(DEFAULT_AUDIO_TOKENIZER_PATH),
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--attn-implementation",
        "--attn_implementation",
        dest="attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    resolved_runtime_device = args.device

    runtime = NanoTTSService(
        checkpoint_path=args.checkpoint_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        device=resolved_runtime_device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        output_dir=args.output_dir,
    )
    text_normalizer_manager = SharedWeTextProcessingManager()
    text_normalizer_manager.start()
    warmup_manager = WarmupManager(runtime, text_normalizer_manager=text_normalizer_manager)
    warmup_manager.start()

    vscode_proxy_uri = os.getenv("VSCODE_PROXY_URI", "")
    root_path = _resolve_vscode_root_path(vscode_proxy_uri, args.port)
    logging.info("root_path=%s", root_path)
    if args.share:
        logging.warning("--share is ignored by the FastAPI-based Nano-TTS app.")

    app = _build_app(runtime, warmup_manager, text_normalizer_manager, root_path)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        root_path=root_path or "",
    )


if __name__ == "__main__":
    main()
