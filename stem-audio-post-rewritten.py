"""
Stem Audio Post — автоматизация сведения и экспорта стемов.

Что делает:
1. При необходимости разделяет готовый трек на drums / bass / other / vocals через Demucs.
2. Обрабатывает стемы умеренными, отключаемыми DSP-операциями.
3. Собирает стереомикс, нормализует громкость и ограничивает sample peak.
4. Сохраняет WAV float и отчёт JSON; из интерфейса можно экспортировать FLAC PCM_24.

Важно:
- Это автоматизированный стартовый микс, а не универсальный мастеринг.
- Все значения обработки — исходные настройки, их нужно проверять на слух.
- Demucs — предобученная модель разделения источников; остальная обработка не является ИИ.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

# Проверяем зависимости без скрытой установки пакетов при запуске.
def check_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "soundfile": "soundfile",
        "scipy": "scipy",
        "pyloudnorm": "pyloudnorm",
        "pedalboard": "pedalboard",
        "tkinterdnd2": "tkinterdnd2",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Не найдены зависимости:", ", ".join(missing))
        print(f'Установите их командой:\n"{sys.executable}" -m pip install ' + " ".join(missing))
        raise SystemExit(1)

check_dependencies()

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import resample_poly
from pedalboard import (
    Pedalboard,
    Compressor,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
    HighShelfFilter,
    Limiter,
    Reverb,
)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD


APP_NAME = "Stem Audio Post"
MODEL_NAME = "htdemucs_ft"
PARTS = ("drums", "bass", "other", "vocals")
TARGET_LUFS = -14.0
PEAK_CEILING_DBFS = -1.0
DEFAULT_SR = 44100
SUPPORTED_INPUTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}


@dataclass
class Settings:
    """Умеренные стартовые параметры; не заменяют настройку под конкретный трек."""
    target_lufs: float = TARGET_LUFS
    peak_ceiling_dbfs: float = PEAK_CEILING_DBFS
    drums_gain_db: float = 0.0
    bass_gain_db: float = 0.0
    other_gain_db: float = -1.0
    vocals_gain_db: float = 0.0
    vocal_duck_db: float = -1.5
    kick_duck_db: float = -1.5
    drum_width: float = 1.05
    use_vocal_ducking: bool = True
    use_kick_ducking: bool = True
    use_gentle_eq: bool = True
    use_stereo_widening: bool = True


def log_exception(log: Callable[[str], None], prefix: str, exc: BaseException) -> None:
    log(f"{prefix}: {type(exc).__name__}: {exc}")


def sha256_file(path: Path, log: Callable[[str], None] | None = None) -> str:
    """Хеш содержимого нужен, чтобы кэш стемов никогда не смешивал разные песни."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_demucs_python() -> str | None:
    """Ищет Demucs в текущем Python, затем в установках Windows через py launcher."""
    candidates: list[str] = [sys.executable]
    if os.name == "nt":
        for version in ("3.12", "3.11", "3.10"):
            try:
                result = subprocess.run(
                    ["py", f"-{version}", "-c", "import sys; print(sys.executable)"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    candidates.append(result.stdout.strip())
            except (OSError, subprocess.SubprocessError):
                pass

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not Path(candidate).exists():
            continue
        seen.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-c", "import demucs"],
                capture_output=True, text=True, timeout=20
            )
            if result.returncode == 0:
                return candidate
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def install_demucs(log: Callable[[str], None]) -> str | None:
    """Устанавливает Demucs в текущий Python и показывает диагностику при ошибке."""
    python = sys.executable
    log("Demucs не найден в текущем Python. Пробую установить его в это окружение.")
    log("Установка может скачать модель и зависимости; потребуется интернет.")
    try:
        result = subprocess.run(
            [python, "-m", "pip", "install", "demucs"],
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "Нет подробностей")[-2500:]
            log("Установка Demucs не удалась. Последний вывод:")
            for line in details.splitlines():
                log(line)
            return None
        check = subprocess.run(
            [python, "-c", "import demucs"],
            capture_output=True, text=True, timeout=30
        )
        if check.returncode == 0:
            log("Demucs установлен.")
            return python
        log("Пакет установился, но импорт Demucs не прошёл.")
        log((check.stderr or check.stdout)[-1500:])
    except subprocess.TimeoutExpired:
        log("Время ожидания установки Demucs истекло.")
    except OSError as exc:
        log(f"Не удалось запустить pip: {exc}")
    return None


def ensure_demucs(log: Callable[[str], None]) -> str | None:
    python = find_demucs_python()
    if python:
        return python
    return install_demucs(log)


def valid_stem_set(directory: Path, expected_duration: float | None = None) -> bool:
    """Кэш считается пригодным только при наличии и читаемости всех четырёх стемов."""
    try:
        infos = []
        for part in PARTS:
            path = directory / f"{part}.wav"
            if not path.is_file() or path.stat().st_size < 44:
                return False
            info = sf.info(str(path))
            if info.frames <= 0 or info.samplerate <= 0 or info.channels < 1:
                return False
            infos.append(info.frames / info.samplerate)
        if max(infos) - min(infos) > 0.25:
            return False
        if expected_duration is not None and abs(min(infos) - expected_duration) > max(1.0, expected_duration * 0.01):
            return False
        return True
    except Exception:
        return False


def separate_track(input_path: Path, log: Callable[[str], None]) -> Path | None:
    """Разделяет трек. Кэш уникален по SHA-256 исходного аудио."""
    try:
        log("Проверяю исходный файл и кэш стемов…")
        digest = sha256_file(input_path)
        cache_root = input_path.parent / ".stem-audio-post-cache"
        source_cache = cache_root / digest
        stem_dir = source_cache / "stems"
        demucs_output_root = source_cache / "demucs"
        try:
            input_info = sf.info(str(input_path))
            input_duration = input_info.frames / input_info.samplerate
        except Exception:
            input_duration = None

        if valid_stem_set(stem_dir, input_duration):
            log(f"Найден полный кэш стемов для этого файла: {stem_dir}")
            return stem_dir

        if stem_dir.exists():
            log("Найден неполный или некорректный кэш; создаю его заново.")
            shutil.rmtree(stem_dir.parent, ignore_errors=True)

        python = ensure_demucs(log)
        if not python:
            log("Demucs недоступен. Проверьте Python, интернет и вывод установки выше.")
            return None

        cache_root.mkdir(parents=True, exist_ok=True)
        log(f"Разделяю «{input_path.name}» моделью {MODEL_NAME}. Это может занять несколько минут.")
        command = [
            python, "-m", "demucs", "-n", MODEL_NAME,
            "--out", str(demucs_output_root), str(input_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "Нет подробностей")[-3000:]
            log("Demucs завершился с ошибкой:")
            for line in details.splitlines():
                log(line)
            return None

        demucs_dir = demucs_output_root / MODEL_NAME / input_path.stem
        if not demucs_dir.is_dir():
            # Некоторые версии Demucs могут нормализовать имя папки иначе.
            model_root = demucs_output_root / MODEL_NAME
            matches = list(model_root.glob("*")) if model_root.exists() else []
            if len(matches) == 1 and matches[0].is_dir():
                demucs_dir = matches[0]
            else:
                log(f"Не найдена папка результата Demucs: {demucs_dir}")
                return None

        stem_dir.mkdir(parents=True, exist_ok=True)
        for part in PARTS:
            source = demucs_dir / f"{part}.wav"
            if not source.is_file():
                log(f"Demucs не создал обязательный стем: {part}.wav")
                return None
            shutil.copy2(source, stem_dir / f"{part}.wav")

        if not valid_stem_set(stem_dir, input_duration):
            log("Стемы созданы, но не прошли проверку целостности.")
            return None
        log("Разделение завершено; кэш проверен.")
        return stem_dir
    except subprocess.TimeoutExpired:
        log("Demucs работает дольше двух часов; операция остановлена по тайм-ауту.")
        return None
    except Exception as exc:
        log_exception(log, "Ошибка разделения", exc)
        return None


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] == 2:
        return x
    # Не отбрасываем каналы молча: downmix всех каналов в стерео.
    left = x[:, 0::2].mean(axis=1)
    right = x[:, 1::2].mean(axis=1) if x.shape[1] > 1 else left
    return np.stack([left, right], axis=1)


def load_audio(path: Path, target_sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if x.size == 0:
        raise ValueError(f"Файл пустой: {path.name}")
    if not np.isfinite(x).all():
        raise ValueError(f"В файле обнаружены NaN/Inf: {path.name}")
    x = to_stereo(x)
    if sr != target_sr:
        divisor = math.gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // divisor, sr // divisor, axis=0).astype(np.float32)
    return np.asarray(x, dtype=np.float32), target_sr


def fit_length(x: np.ndarray, frames: int) -> np.ndarray:
    if len(x) < frames:
        return np.pad(x, ((0, frames - len(x)), (0, 0)))
    return x[:frames]


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def safe_lufs(x: np.ndarray, sr: int) -> float | None:
    if len(x) < int(sr * 0.4):
        return None
    try:
        value = float(pyln.Meter(sr).integrated_loudness(x))
        return value if math.isfinite(value) else None
    except Exception:
        return None


def sample_peak_dbfs(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return -120.0
    return 20.0 * math.log10(peak)


def stereo_correlation(x: np.ndarray) -> float | None:
    if x.ndim != 2 or x.shape[1] < 2 or len(x) < 2:
        return None
    left, right = x[:, 0], x[:, 1]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def widen(x: np.ndarray, amount: float = 1.05, low_hz: float = 120.0) -> np.ndarray:
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    side = Pedalboard([HighpassFilter(cutoff_frequency_hz=low_hz)])(side[None, :], DEFAULT_SR)[0]
    side *= amount
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def gentle_eq(part: str, x: np.ndarray) -> np.ndarray:
    """Лёгкие корректировки. Не пытаются автоматически «исправить» любой исходник."""
    if part == "drums":
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=28.0),
            PeakFilter(cutoff_frequency_hz=250.0, gain_db=-1.0, q=1.0),
        ])
    elif part == "bass":
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=25.0),
            PeakFilter(cutoff_frequency_hz=250.0, gain_db=-1.0, q=1.0),
        ])
    elif part == "vocals":
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80.0),
            PeakFilter(cutoff_frequency_hz=350.0, gain_db=-1.0, q=1.0),
        ])
    else:
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=35.0)])
    return board(x.T, DEFAULT_SR).T.astype(np.float32)


def block_envelope(mono: np.ndarray, block_size: int) -> np.ndarray:
    usable = (len(mono) // block_size) * block_size
    if usable < block_size * 4:
        return np.zeros(0, dtype=np.float32)
    blocks = mono[:usable].reshape(-1, block_size)
    return np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-12)


def duck_from_trigger(trigger: np.ndarray, target: np.ndarray, depth_db: float,
                      block_ms: float = 5.0, threshold_percentile: float = 55.0,
                      attack_ms: float = 5.0, release_ms: float = 140.0) -> np.ndarray:
    block = max(1, int(DEFAULT_SR * block_ms / 1000.0))
    env = block_envelope(trigger, block)
    if len(env) < 4:
        return target
    db = 20.0 * np.log10(env + 1e-12)
    threshold = float(np.percentile(db, threshold_percentile)) - 6.0
    drive = np.clip((db - threshold) / 12.0, 0.0, 1.0)
    target_curve = depth_db * drive
    attack = math.exp(-block / (DEFAULT_SR * attack_ms / 1000.0))
    release = math.exp(-block / (DEFAULT_SR * release_ms / 1000.0))
    gain_db = np.zeros_like(target_curve)
    state = 0.0
    for i, desired in enumerate(target_curve):
        # Ослабление должно быстро включаться и плавнее отпускаться.
        coefficient = attack if desired < state else release
        state = coefficient * state + (1.0 - coefficient) * desired
        gain_db[i] = state
    positions = np.arange(len(gain_db)) * block
    curve = np.interp(np.arange(len(target)), positions, gain_db,
                      left=float(gain_db[0]), right=float(gain_db[-1]))
    return (target * db_to_gain(0.0) * (10.0 ** (curve / 20.0))[:, None]).astype(np.float32)


def measure_audio(x: np.ndarray, sr: int) -> dict:
    lufs = safe_lufs(x, sr)
    return {
        "duration_seconds": round(len(x) / sr, 3),
        "sample_rate_hz": sr,
        "channels": int(x.shape[1]) if x.ndim == 2 else 1,
        "integrated_lufs": round(lufs, 2) if lufs is not None else None,
        "sample_peak_dbfs": round(sample_peak_dbfs(x), 2),
        "stereo_correlation": round(stereo_correlation(x), 4) if stereo_correlation(x) is not None else None,
        "finite_samples": bool(np.isfinite(x).all()),
    }


def process_track(input_path: Path, output_path: Path, settings: Settings,
                  log: Callable[[str], None]) -> dict:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Исходный файл не найден: {input_path}")
    if input_path == output_path:
        raise ValueError("Нельзя перезаписывать исходный файл.")

    stems_dir = separate_track(input_path, log)
    if stems_dir is None:
        raise RuntimeError("Не удалось получить полный набор стемов.")

    log("Загружаю и проверяю стемы…")
    buffers: dict[str, np.ndarray] = {}
    sample_rates = []
    for part in PARTS:
        path = stems_dir / f"{part}.wav"
        x, sr = load_audio(path, DEFAULT_SR)
        buffers[part] = x
        sample_rates.append(sr)

    # Длительность выбирается по самому короткому стему, чтобы не подмешивать тишину
    # и не обрезать случайно второй по длине файл.
    frames = min(len(x) for x in buffers.values())
    if frames < DEFAULT_SR:
        raise ValueError("Стемы короче одной секунды: проверьте результат Demucs.")
    for part in PARTS:
        buffers[part] = fit_length(buffers[part], frames)

    original_metrics = {part: measure_audio(buffers[part], DEFAULT_SR) for part in PARTS}
    kick_mono = buffers["drums"].mean(axis=1).copy()

    log("Применяю настройки шин…")
    if settings.use_gentle_eq:
        for part in PARTS:
            buffers[part] = gentle_eq(part, buffers[part])

    if settings.use_stereo_widening and settings.drum_width > 1.0:
        buffers["drums"] = widen(buffers["drums"], settings.drum_width, 120.0)

    if settings.use_vocal_ducking:
        buffers["other"] = duck_from_trigger(
            buffers["vocals"].mean(axis=1), buffers["other"],
            settings.vocal_duck_db, block_ms=5.0, threshold_percentile=55.0,
            attack_ms=5.0, release_ms=140.0
        )

    if settings.use_kick_ducking:
        buffers["bass"] = duck_from_trigger(
            kick_mono, buffers["bass"], settings.kick_duck_db,
            block_ms=2.0, threshold_percentile=90.0,
            attack_ms=3.0, release_ms=80.0
        )

    gains = {
        "drums": settings.drums_gain_db,
        "bass": settings.bass_gain_db,
        "other": settings.other_gain_db,
        "vocals": settings.vocals_gain_db,
    }
    for part, gain_db in gains.items():
        buffers[part] = (buffers[part] * db_to_gain(gain_db)).astype(np.float32)

    log("Собираю микс…")
    mix = np.zeros((frames, 2), dtype=np.float32)
    for part in PARTS:
        mix += buffers[part]

    before_master = measure_audio(mix, DEFAULT_SR)
    loudness_before = safe_lufs(mix, DEFAULT_SR)
    if loudness_before is None:
        raise ValueError("Не удалось измерить LUFS микса; проверьте длительность и аудиоданные.")

    # Нормализация до цели, затем лимитер. Если лимитер сработает сильно,
    # итоговый LUFS окажется ниже целевого — это отражается в отчёте.
    mix *= db_to_gain(settings.target_lufs - loudness_before)
    limiter = Pedalboard([Limiter(threshold_db=settings.peak_ceiling_dbfs)])
    mix = limiter(mix.T, DEFAULT_SR).T.astype(np.float32)

    # Защита от нечисловых данных и жёсткая страховка от sample peak выше потолка.
    if not np.isfinite(mix).all():
        raise ValueError("После обработки появились NaN/Inf.")
    peak_db = sample_peak_dbfs(mix)
    if peak_db > settings.peak_ceiling_dbfs + 0.05:
        mix *= db_to_gain(settings.peak_ceiling_dbfs - peak_db)

    final_metrics = measure_audio(mix, DEFAULT_SR)
    report = {
        "application": APP_NAME,
        "model": MODEL_NAME,
        "source_file": str(input_path),
        "output_file": str(output_path),
        "settings": asdict(settings),
        "source_info": {
            "duration_seconds": round(sf.info(str(input_path)).duration, 3),
            "sample_rate_hz": sf.info(str(input_path)).samplerate,
            "channels": sf.info(str(input_path)).channels,
        },
        "stems_before_processing": original_metrics,
        "mix_before_loudness_normalization": before_master,
        "mix_after_processing": final_metrics,
        "notes": [
            "LUFS измерен после обработки; целевое значение может не достигаться из-за лимитера.",
            "sample_peak_dbfs — пиковый уровень сэмплов, не измерение true peak.",
            "Стереокорреляция — диагностическая метрика, не универсальная оценка качества.",
            "Автоматические настройки необходимо проверить прослушиванием и сравнением с исходником."
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mix, DEFAULT_SR, subtype="FLOAT", format="WAV")
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Готово: {output_path.name}")
    log(f"Итог: {final_metrics['integrated_lufs']} LUFS; sample peak {final_metrics['sample_peak_dbfs']} dBFS")
    log(f"Стереокорреляция: {final_metrics['stereo_correlation']}; отчёт: {report_path.name}")
    return report


class MixerApp:
    def __init__(self, root: TkinterDnD.Tk):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("760x720")
        self.root.minsize(680, 620)
        self.input_file: Path | None = None
        self.output_file: Path | None = None
        self.processing = False

        self.settings = Settings()
        self._build_ui()
        self.log("Выберите аудиофайл. Исходник не изменяется.")
        self.log("Первый запуск может потребовать установки Demucs и скачивания модели.")

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Stem Audio Post", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Автоматизированный стартовый микс из четырёх стемов. Результат нужно проверить на слух.",
            wraplength=700
        ).pack(anchor="w", pady=(2, 10))

        self.drop_zone = tk.Label(
            frame, text="Перетащите аудиофайл сюда\nили нажмите «Выбрать файл»",
            bg="#30343b", fg="white", font=("Segoe UI", 12), height=4, relief="groove", bd=2
        )
        self.drop_zone.pack(fill="x", pady=5)
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind("<<Drop>>", self.on_drop)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=5)
        ttk.Button(buttons, text="Выбрать файл", command=self.choose_file).pack(side="left")
        self.run_button = ttk.Button(buttons, text="Обработать", command=self.start_processing, state="disabled")
        self.run_button.pack(side="left", padx=8)
        self.export_button = ttk.Button(buttons, text="Экспорт WAV / FLAC", command=self.export_file, state="disabled")
        self.export_button.pack(side="left")

        self.info_label = ttk.Label(frame, text="Файл не выбран")
        self.info_label.pack(anchor="w", pady=5)

        settings_box = ttk.LabelFrame(frame, text="Настройки микса (дБ)", padding=8)
        settings_box.pack(fill="x", pady=8)
        self.gain_vars = {}
        gain_defs = [
            ("drums_gain_db", "Барабаны", -12, 12),
            ("bass_gain_db", "Бас", -12, 12),
            ("other_gain_db", "Остальная музыка", -12, 12),
            ("vocals_gain_db", "Вокал", -12, 12),
        ]
        for row, (key, label, low, high) in enumerate(gain_defs):
            ttk.Label(settings_box, text=label, width=19).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.DoubleVar(value=getattr(self.settings, key))
            self.gain_vars[key] = var
            ttk.Scale(settings_box, from_=low, to=high, variable=var, orient="horizontal").grid(
                row=row, column=1, sticky="ew", padx=6, pady=2
            )
            value_label = ttk.Label(settings_box, textvariable=var, width=6)
            value_label.grid(row=row, column=2, sticky="e", padx=4)
        settings_box.columnconfigure(1, weight=1)

        options = ttk.Frame(frame)
        options.pack(fill="x", pady=3)
        self.eq_var = tk.BooleanVar(value=self.settings.use_gentle_eq)
        self.vocal_duck_var = tk.BooleanVar(value=self.settings.use_vocal_ducking)
        self.kick_duck_var = tk.BooleanVar(value=self.settings.use_kick_ducking)
        self.width_var = tk.BooleanVar(value=self.settings.use_stereo_widening)
        ttk.Checkbutton(options, text="Мягкий EQ", variable=self.eq_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(options, text="Освободить место вокалу", variable=self.vocal_duck_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(options, text="Сайдчейн баса от бочки", variable=self.kick_duck_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(options, text="Немного расширить барабаны", variable=self.width_var).pack(side="left")

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 4))

        log_frame = ttk.LabelFrame(frame, text="Журнал", padding=5)
        log_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.log_box = tk.Text(log_frame, height=9, state="disabled", wrap="word", font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=scroll.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def log(self, message: str):
        def update():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, update)

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Аудиофайлы", "*.wav *.flac *.mp3 *.m4a *.ogg *.aiff *.aif"), ("Все файлы", "*.*")]
        )
        if path:
            self.set_input(Path(path))

    def on_drop(self, event):
        try:
            paths = self.root.tk.splitlist(event.data)
            if not paths:
                return
            self.set_input(Path(paths[0]))
        except Exception as exc:
            self.log(f"Не удалось прочитать перетащенный путь: {exc}")

    def set_input(self, path: Path):
        path = path.expanduser().resolve()
        if path.suffix.lower() not in SUPPORTED_INPUTS:
            messagebox.showwarning(APP_NAME, "Этот формат не поддерживается.")
            return
        if not path.is_file():
            messagebox.showerror(APP_NAME, "Файл не найден.")
            return
        try:
            info = sf.info(str(path))
            self.input_file = path
            self.output_file = None
            self.info_label.configure(
                text=f"{path.name} | {info.duration:.1f} сек | {info.samplerate} Гц | {info.channels} канал(а)"
            )
            self.run_button.configure(state="normal")
            self.export_button.configure(state="disabled")
            self.log(f"Выбран файл: {path}")
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Не удалось прочитать файл:\n{exc}")

    def current_settings(self) -> Settings:
        values = {key: float(var.get()) for key, var in self.gain_vars.items()}
        return Settings(
            **values,
            use_gentle_eq=self.eq_var.get(),
            use_vocal_ducking=self.vocal_duck_var.get(),
            use_kick_ducking=self.kick_duck_var.get(),
            use_stereo_widening=self.width_var.get(),
        )

    def start_processing(self):
        if self.processing or not self.input_file:
            return
        self.processing = True
        self.run_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.progress.start(10)
        self.output_file = self.input_file.with_name(self.input_file.stem + "_mix.wav")
        settings = self.current_settings()
        input_path, output_path = self.input_file, self.output_file
        threading.Thread(
            target=self._process_worker,
            args=(input_path, output_path, settings),
            daemon=True
        ).start()

    def _process_worker(self, input_path: Path, output_path: Path, settings: Settings):
        success = False
        try:
            process_track(input_path, output_path, settings, self.log)
            success = True
        except Exception as exc:
            self.log_exception_details(exc)
        finally:
            def finish():
                self.progress.stop()
                self.processing = False
                self.run_button.configure(state="normal" if self.input_file else "disabled")
                self.export_button.configure(state="normal" if success and output_path.exists() else "disabled")
                if success:
                    messagebox.showinfo(APP_NAME, f"Обработка завершена.\n\n{output_path}")
            self.root.after(0, finish)

    def log_exception_details(self, exc: BaseException):
        self.log(f"ОШИБКА: {type(exc).__name__}: {exc}")
        self.log("Подробности ошибки:")
        for line in traceback.format_exc().splitlines()[-12:]:
            self.log(line)

    def export_file(self):
        if not self.output_file or not self.output_file.is_file():
            return
        save_path = filedialog.asksaveasfilename(
            title="Экспортировать результат",
            defaultextension=".flac",
            initialfile=self.output_file.stem + ".flac",
            filetypes=[("FLAC PCM 24-bit", "*.flac"), ("WAV float 32-bit", "*.wav")]
        )
        if not save_path:
            return
        try:
            data, sr = sf.read(str(self.output_file), dtype="float32", always_2d=True)
            target = Path(save_path)
            if target.suffix.lower() == ".flac":
                sf.write(str(target), data, sr, format="FLAC", subtype="PCM_24")
            else:
                sf.write(str(target), data, sr, format="WAV", subtype="FLOAT")
            self.log(f"Экспортировано: {target}")
            messagebox.showinfo(APP_NAME, f"Файл сохранён:\n{target}")
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Ошибка экспорта:\n{exc}")


def main():
    root = TkinterDnD.Tk()
    MixerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
