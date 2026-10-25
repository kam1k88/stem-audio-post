"""
stem-audio-post.py — Единое приложение с автоустановкой зависимостей, Drag&Drop и мастерингом.
"""
import os
import sys
import subprocess
import importlib.util
import webbrowser

# ==========================================
# 0. АВТОУСТАНОВКА ЛЕГКИХ ЗАВИСИМОСТЕЙ
# ==========================================
def bootstrap():
    required = {
        "numpy": "numpy",
        "soundfile": "soundfile",
        "pedalboard": "pedalboard",
        "pyloudnorm": "pyloudnorm",
        "tkinterdnd2": "tkinterdnd2"
    }
    missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
    
    if missing:
        print(f"Устанавливаю базовые библиотеки: {', '.join(missing)}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        except subprocess.CalledProcessError:
            print("Ошибка установки. Попробуйте запустить от имени администратора.")
            sys.exit(1)

bootstrap()

import threading
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard, Compressor, Distortion, HighpassFilter, LowpassFilter,
    LowShelfFilter, PeakFilter, HighShelfFilter, Limiter, Reverb, PitchShift
)

SR = 44100
PARTS = ("drums", "bass", "other", "vocals")
TARGET_LUFS = -12.0
LIMIT_DBTP = -1.0

BUS = {
    "drums":  {"gain_db": +4.5, "widen": 1.2, "low_mono_hz": 120.0, "eq": "drum_punch"},
    "bass":   {"gain_db": +5.0, "eq": "bass_clean", "duck_from_kick": -2.0},
    "vocals": {"gain_db": +7.0, "eq": "vocal_surgical"},
    "other":  {"gain_db": +1.0, "duck_db": -3.0, "eq": "other_surgical"},
}

# ==========================================
# 1. УМНЫЙ ПОИСК И УСТАНОВКА DEMUCS
# ==========================================
def find_python_for_demucs():
    """Ищет Python 3.10-3.12 для Demucs."""
    try:
        import demucs
        return sys.executable
    except ImportError:
        pass

    candidates = []
    # Проверяем через launcher py
    for ver in ("3.12", "3.11", "3.10"):
        try:
            r = subprocess.run(["py", f"-{ver}", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
            if r.returncode == 0:
                candidates.append(r.stdout.strip())
        except Exception:
            pass

    # Проверяем стандартные пути
    candidates.extend([
        r"C:\Python312\python.exe",
        r"C:\Python311\python.exe",
        r"C:\Python310\python.exe"
    ])

    # Оставляем только существующие файлы
    return next((c for c in candidates if c and os.path.exists(c)), None)

def ensure_demucs(log_func):
    """Проверяет наличие Demucs и устанавливает его при необходимости."""
    py = find_python_for_demucs()
    
    if not py:
        log_func("КРИТИЧЕСКАЯ ОШИБКА: Не найден Python 3.10-3.12!")
        log_func("Сейчас откроется страница для скачивания Python 3.12.")
        log_func("При установке ОБЯЗАТЕЛЬНО поставьте галочку 'Add python.exe to PATH'.")
        webbrowser.open("https://www.python.org/downloads/release/python-31210/")
        return None
        
    # Проверяем, установлен ли demucs в найденном Python
    try:
        r = subprocess.run([py, "-c", "import demucs"], capture_output=True)
        if r.returncode == 0:
            return py
    except Exception:
        pass

    # Если demucs нет — устанавливаем
    log_func(f"Установка нейросети Demucs (это займет 5-10 минут, скачается ~2 ГБ)...")
    log_func("Пожалуйста, не закрывайте программу.")
    try:
        # Устанавливаем без вывода кучи текста в консоль, чтобы не вешать GUI
        subprocess.run([py, "-m", "pip", "install", "demucs"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_func("Demucs успешно установлен!")
        return py
    except subprocess.CalledProcessError:
        log_func("ОШИБКА: Не удалось установить Demucs. Проверьте интернет-соединение.")
        return None

# ==========================================
# 2. ФУНКЦИИ ОБРАБОТКИ ЗВУКА
# ==========================================
def separate_track(input_path: Path, log_func) -> Path | None:
    out_dir = input_path.parent / "stems"
    
    # Если стемы уже есть — пропускаем
    if (out_dir / "drums.wav").exists() and (out_dir / "bass.wav").exists():
        log_func(f"Стемы уже найдены в {out_dir}. Пропускаю разделение.")
        return out_dir

    py = ensure_demucs(log_func)
    if not py:
        return None
    
    out_dir.mkdir(exist_ok=True)
    log_func(f"Нейросеть разделяет {input_path.name} на стемы (2-5 минут)...")
    
    cmd = [py, "-m", "demucs", "-n", "htdemucs_ft", "-o", str(out_dir), str(input_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        log_func(f"ОШИБКА Demucs: {e}")
        return None
    
    demucs_out = out_dir / "htdemucs_ft" / input_path.stem
    if not demucs_out.exists():
        log_func(f"Demucs не создал папку с результатом: {demucs_out}")
        return None
    
    for part in PARTS:
        src = demucs_out / f"{part}.wav"
        dst = out_dir / f"{part}.wav"
        if src.exists():
            shutil.copyfile(src, dst)
    
    shutil.rmtree(out_dir / "htdemucs_ft", ignore_errors=True)
    log_func("Стемы готовы!")
    return out_dir

def load(path: Path, frames: int) -> np.ndarray | None:
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    if sr != SR: return None
    if x.shape[0] < frames:
        x = np.pad(x, ((0, frames - x.shape[0]), (0, 0)))
    else:
        x = x[:frames]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x.astype("float32")

def widen(x: np.ndarray, amount: float, low_hz: float) -> np.ndarray:
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    side_hp = Pedalboard([HighpassFilter(cutoff_frequency_hz=low_hz)])(side[:, None], SR)[:, 0]
    new_side = side_hp * amount
    return np.stack([mid + new_side, mid - new_side], axis=1).astype("float32")

def bass_clean(x: np.ndarray) -> np.ndarray:
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    sub = Pedalboard([LowpassFilter(100), LowShelfFilter(50, +3.0, 0.8), Compressor(-18, 10.0)])(mid[:, None], SR)[:, 0]
    mid_body = Pedalboard([HighpassFilter(100), LowpassFilter(500), PeakFilter(250, -4.0, 1.0), PeakFilter(150, +2.0, 1.0)])(mid[:, None], SR)[:, 0]
    side_body = Pedalboard([HighpassFilter(100), LowpassFilter(500), PeakFilter(250, -4.0, 1.0)])(side[:, None], SR)[:, 0]
    mid_top = Pedalboard([HighpassFilter(500), Distortion(+10.0), PeakFilter(800, +3.0, 1.5)])(mid[:, None], SR)[:, 0]
    side_top = Pedalboard([HighpassFilter(500), Distortion(+10.0), PeakFilter(800, +3.0, 1.5)])(side[:, None], SR)[:, 0]
    out_mid = sub + mid_body + mid_top
    out_side = (side_body + side_top) * 1.5
    return np.stack([out_mid + out_side, out_mid - out_side], axis=1).astype("float32")

def add_bass_harmonics(x: np.ndarray) -> np.ndarray:
    octave = PitchShift(12.0)(x.T, SR).T
    octave_processed = Pedalboard([HighpassFilter(300), Distortion(+8.0), LowpassFilter(6000), Compressor(-18, 4.0)])(octave.T, SR).T
    return x + octave_processed * 0.2

def bright_surgical(x: np.ndarray) -> np.ndarray:
    board = Pedalboard([HighpassFilter(150), PeakFilter(800, +2.0, 1.0), PeakFilter(1500, -2.0, 1.5), PeakFilter(2500, -2.0, 2.0), PeakFilter(3000, +3.0, 1.0)])
    return board(x.T, SR).T

def drum_punch(x: np.ndarray) -> np.ndarray:
    board_sat = Pedalboard([HighpassFilter(30), Distortion(+1.0)])
    x = board_sat(x.T, SR).T.astype("float32")
    board_eq = Pedalboard([PeakFilter(55, +3.0, 1.0), PeakFilter(200, -3.0, 2.0), PeakFilter(3500, +4.0, 1.0), HighShelfFilter(8000, +1.0)])
    return board_eq(x.T, SR).T.astype("float32")

def vocal_surgical(x: np.ndarray) -> np.ndarray:
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    leveler = Pedalboard([Compressor(-24.0, 3.0, 200.0, 1000.0)])
    mid_lev = leveler(mid[:, None], SR)[:, 0]
    side_lev = leveler(side[:, None], SR)[:, 0]
    mid_proc = Pedalboard([HighpassFilter(100), PeakFilter(500, -3.0, 1.5), PeakFilter(3000, +4.0, 2.0), PeakFilter(2500, +2.0, 1.5), HighShelfFilter(8000, +2.0)])(mid_lev[:, None], SR)[:, 0]
    side_proc = Pedalboard([HighpassFilter(250)])(side_lev[:, None], SR)[:, 0]
    return np.stack([mid_proc + side_proc*1.2, mid_proc - side_proc*1.2], axis=1).astype("float32")

def sidechain_duck(vox_mono, target, depth_db, attack_ms, release_ms, thresh_pct):
    block = int(SR * 0.005)
    n = (len(vox_mono) // block) * block
    if n < block * 8: return target
    env = np.sqrt(np.mean(vox_mono[:n].reshape(-1, block) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(env + 1e-12)
    thr = float(np.percentile(db, thresh_pct)) - 6.0
    drive = np.clip((db - thr) / 12.0, 0.0, 1.0)
    target_curve = depth_db * drive
    a_att = np.exp(-block / (SR * attack_ms / 1000.0))
    a_rel = np.exp(-block / (SR * release_ms / 1000.0))
    g = 0.0
    curve = np.empty_like(target_curve)
    for i in range(len(target_curve)):
        a = a_att if target_curve[i] < g else a_rel
        g = a * g + (1.0 - a) * target_curve[i]
        curve[i] = g
    full = np.interp(np.arange(len(target)), np.arange(len(curve)) * block, curve)
    return target * (10 ** (full / 20.0))[:, None]

def sidechain_kick_to_bass(kick_mono, bass, depth_db=-2.0):
    block = int(SR * 0.002)
    n = (len(kick_mono) // block) * block
    if n < block * 8: return bass
    env = np.sqrt(np.mean(kick_mono[:n].reshape(-1, block) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(env + 1e-12)
    thr = float(np.percentile(db, 90)) - 3.0
    drive = np.clip((db - thr) / 10.0, 0.0, 1.0)
    target_curve = depth_db * drive
    a_att = np.exp(-block / (SR * 0.003))
    a_rel = np.exp(-block / (SR * 0.080))
    g = 0.0
    curve = np.empty_like(target_curve)
    for i in range(len(target_curve)):
        a = a_att if target_curve[i] < g else a_rel
        g = a * g + (1.0 - a) * target_curve[i]
        curve[i] = g
    full = np.interp(np.arange(len(bass)), np.arange(len(curve)) * block, curve)
    return bass * (10 ** (full / 20.0))[:, None]

def master_stereo_imaging(mix: np.ndarray) -> np.ndarray:
    mid = (mix[:, 0] + mix[:, 1]) * 0.5
    side = (mix[:, 0] - mix[:, 1]) * 0.5
    side_hp = Pedalboard([HighpassFilter(120)])(side[:, None], SR)[:, 0]
    side_eq = Pedalboard([HighShelfFilter(10000, +3.0)])(side_hp[:, None], SR)[:, 0]
    mid_eq = Pedalboard([HighShelfFilter(10000, -1.5)])(mid[:, None], SR)[:, 0]
    return np.stack([mid_eq + side_eq * 1.3, mid_eq - side_eq * 1.3], axis=1).astype("float32")

def master_parallel_compression(mix: np.ndarray) -> np.ndarray:
    compressed = Pedalboard([Compressor(-12.0, 2.0, 20.0, 100.0)])(mix.T, SR).T
    return (mix * 0.95 + compressed * 0.05).astype("float32")

def master_glue(mix: np.ndarray) -> np.ndarray:
    board = Pedalboard([PeakFilter(150, +1.0, 0.8)])
    return board(mix.T, SR).T.astype("float32")

def lufs(x: np.ndarray) -> float:
    return float(pyln.Meter(SR).integrated_loudness(x))

def peak(x: np.ndarray) -> float:
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-12))

# ==========================================
# 3. ОРКЕСТРАТОР ПРОЦЕССА
# ==========================================
def process_track(input_path, output_path, log_func):
    stems_dir = separate_track(Path(input_path), log_func)
    if not stems_dir: return False

    stems = {}
    for part in PARTS:
        for ext in (".wav", ".flac", ".mp3"):
            f = stems_dir / f"{part}{ext}"
            if f.exists():
                stems[part] = f
                break
        if part not in stems:
            log_func(f"ОШИБКА: Не найден стем '{part}'")
            return False

    durations = [sf.info(str(p)).frames / sf.info(str(p)).samplerate for p in stems.values()]
    frames = int(sorted(durations)[1] * SR)
    
    log_func("Загружаю стемы...")
    buffers = {name: load(path, frames) for name, path in stems.items()}
    
    buffers["air"] = np.zeros_like(buffers["drums"])
    kick_mono = buffers["drums"].mean(axis=1)
    
    if BUS["drums"].get("widen"):
        buffers["drums"] = widen(buffers["drums"], BUS["drums"]["widen"], 120.0)

    log_func("Обработка баса (3 слоя + гармония + транзиент)...")
    buffers["bass"] = bass_clean(buffers["bass"])
    buffers["bass"] = add_bass_harmonics(buffers["bass"])
    
    kick_trans = Pedalboard([HighpassFilter(100), LowpassFilter(2000)])(kick_mono[:, None], SR)[:, 0]
    buffers["bass"] += np.stack([kick_trans, kick_trans], axis=1) * 0.3

    log_func("Эквалайзер инструментов и барабанов...")
    buffers["other"] = bright_surgical(buffers["other"])
    buffers["drums"] = drum_punch(buffers["drums"])
    
    log_func("Обработка вокала (Mid/Side)...")
    buffers["vocals"] = vocal_surgical(buffers["vocals"])

    log_func("Сайдчейн (вокал -> музыка, бочка -> бас)...")
    buffers["other"] = sidechain_duck(buffers["vocals"].mean(axis=1), buffers["other"], BUS["other"]["duck_db"], 3.0, 180.0, 55.0)
    buffers["bass"] = sidechain_kick_to_bass(kick_mono, buffers["bass"], BUS["bass"]["duck_from_kick"])

    for name in PARTS:
        buffers[name] = (buffers[name] * (10 ** (BUS[name].get("gain_db", 0.0) / 20.0))).astype("float32")

    log_func("Мастеринг: стереофазация и лимитер...")
    mix = np.sum(np.stack([buffers[p] for p in PARTS] + [buffers["air"]]), axis=0)
    mix = master_stereo_imaging(mix)
    mix = master_parallel_compression(mix)
    mix = master_glue(mix)

    mix = mix * (10 ** ((TARGET_LUFS - lufs(mix)) / 20.0))
    mix = Pedalboard([Limiter(LIMIT_DBTP)])(mix.T, SR).T

    log_func(f"Готово! LUFS: {lufs(mix):.1f}, Пик: {peak(mix):.1f} dBFS")
    sf.write(str(output_path), mix, SR, subtype="FLOAT")
    return True

# ==========================================
# 4. ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ==========================================
class MixerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("stem-audio-post")
        self.root.geometry("600x550")
        self.root.configure(bg="#2b2b2b")
        
        self.input_file = None
        self.output_file = None
        
        tk.Label(root, text="stem-audio-post", font=("Arial", 18, "bold"), fg="white", bg="#2b2b2b").pack(pady=10)
        
        self.drop_zone = tk.Label(root, text="Перетащите аудиофайл (MP3 / WAV / FLAC)\nсюда", 
                                  bg="#404040", fg="#a0a0a0", font=("Arial", 12), height=5, relief="groove", bd=2)
        self.drop_zone.pack(fill="x", padx=30, pady=10)
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self.on_drop)
        
        self.info_label = tk.Label(root, text="Файл не выбран", fg="#a0a0a0", bg="#2b2b2b")
        self.info_label.pack(pady=5)
        
        self.btn_run = tk.Button(root, text="Обработать трек", state="disabled", command=self.start_processing, 
                                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), relief="flat", padx=20, pady=8)
        self.btn_run.pack(pady=10)
        
        self.btn_save = tk.Button(root, text="Скачать WAV / FLAC", state="disabled", command=self.save_file, 
                                  bg="#2196F3", fg="white", font=("Arial", 12, "bold"), relief="flat", padx=20, pady=8)
        self.btn_save.pack(pady=5)
        
        self.log_box = tk.Text(root, height=10, state="disabled", bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True, padx=30, pady=15)
        
    def on_drop(self, event):
        file_path = event.data.strip('{}')
        if file_path.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            self.input_file = file_path
            try:
                info = sf.info(file_path)
                self.info_label.config(text=f"Выбран: {Path(file_path).name} | {info.duration:.1f} сек | {info.samplerate} Гц")
                self.btn_run.config(state="normal")
                self.btn_save.config(state="disabled")
                self.log("Файл загружен. Нажмите 'Обработать'.")
            except Exception as e:
                self.log(f"Ошибка чтения: {e}")
        else:
            self.log("Пожалуйста, перетащите аудиофайл.")
            
    def log(self, text):
        def update():
            self.log_box.config(state="normal")
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.root.after(0, update)
        
    def start_processing(self):
        if not self.input_file:
            return
        self.btn_run.config(state="disabled")
        self.log("Начинаю обработку...")
        
        self.output_file = str(Path(self.input_file).parent / f"{Path(self.input_file).stem}_remix.wav")
        threading.Thread(target=self._process_thread, daemon=True).start()
        
    def _process_thread(self):
        try:
            success = process_track(self.input_file, self.output_file, self.log)
            if success:
                self.log("Готово! Можно сохранить результат.")
                self.root.after(0, lambda: self.btn_save.config(state="normal"))
        except Exception as e:
            self.log(f"Критическая ошибка: {e}")
        finally:
            self.root.after(0, lambda: self.btn_run.config(state="normal"))

    def save_file(self):
        if not self.output_file or not os.path.exists(self.output_file):
            return
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".flac",
            filetypes=[("FLAC", "*.flac"), ("WAV", "*.wav")],
            title="Сохранить готовый трек"
        )
        if save_path:
            try:
                data, sr = sf.read(self.output_file)
                if save_path.endswith('.flac'):
                    sf.write(save_path, data, sr, format='FLAC', subtype='PCM_24')
                else:
                    sf.write(save_path, data, sr, format='WAV', subtype='FLOAT')
                self.log(f"Сохранено: {save_path}")
                messagebox.showinfo("Успех", f"Трек сохранен:\n{save_path}")
            except Exception as e:
                self.log(f"Ошибка сохранения: {e}")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = MixerApp(root)
    root.mainloop()
