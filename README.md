<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e7652310-6395-48e5-99d7-09d6a645fee9" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6d82f5e6-8cfa-46df-9622-7a603b10ffc5" />



## ℹ️ About

Automated track mixing and mastering using neural networks. Features a drag-and-drop interface, stem separation (Demucs), and professional audio processing (Pedalboard)

## 🚀 Quick Start
Just download 2 files: **`stem-audio-prod.py`** and **`start_stem-audio-prod.bat`** into a single folder.
Double-click the **`.bat`** file to launch the app — it will automatically download and install all required libraries on the first run.

---

## ⚙️ How it works (Processing Pipeline)

When you drag and drop a track into the app, it performs the following steps:

### 1. Preparation (Loading & Neural Network)
*   **Separation:** The Demucs (`htdemucs_ft`) neural network separates the track into 4 independent stems: `drums`, `bass`, `other` (instruments), and `vocals`.
*   **Alignment:** The program checks the length of all stems, converts mono files to stereo, and crops them to the same size so nothing goes out of sync.

### 2. Bass Processing (3 layers + harmonics + attack)
To make the bass tight and growling, it is split into frequency bands:
*   **SUB:** Cut everything above 100 Hz (`Lowpass`). Add `+3 dB` at 50 Hz and heavy compression (ratio 10:1). This is what hits you in the chest.
*   **BODY:** Keep 100–500 Hz, but **cut `-4 dB` at 250 Hz** (removes mud) and add `+2 dB` at 150 Hz (so the musical note is audible).
*   **TOP:** Everything above 500 Hz. Add `+10 dB` distortion and boost `+3 dB` at 800 Hz. This adds aggression and punch.
*   **HARMONICS:** Take the bass, pitch it up an octave (PitchShift +12), cut the low end (Highpass 300 Hz), add distortion, and mix it in quietly (20%). It sounds like a second synth playing.
*   **Transient:** Mix a piece of the kick drum (filtered 100–2000 Hz) into the bass at 30% volume so every note has a sharp attack.

### 3. Drum Processing (`drum_punch`)
*   Add light saturation for density (Distortion +1 dB).
*   Boost the kick (`+3 dB` at 55 Hz) and cut mud (`-3 dB` at 200 Hz).
*   Boost the snare click (`+4 dB` at 3.5 kHz) and add air (`+1 dB` at 8 kHz).
*   Widen the stereo field (but keep the low end below 120 Hz in mono).

### 4. Vocal Processing (`vocal_surgical`)
*   Split into Mid and Side channels.
*   Center: Apply a slow compressor (Leveler, ratio 3:1) so vocals don't drop in the chorus. Cut mud (`-3 dB` at 500 Hz), add clarity (`+4 dB` at 3000 Hz and `+2 dB` at 2500 Hz), and air (`+2 dB` at 8 kHz).
*   Sides (reverb): Widen (×1.2) for space.

### 5. Instrument Processing (`bright_surgical`)
*   This is "surgical" EQ. We cut frequencies around 1.5 kHz and 2.5 kHz, but **gently (`-2 dB`)** so we don't eat the distortion and keep the body of the instruments.
*   Add density at 800 Hz (`+2 dB`) and keep the brightness at 3 kHz (`+3 dB`).

### 6. Smart Effects (Sidechain)
*   **Vocal → Instruments:** When the singer sings, the music (`other`) ducks by **`-3.0 dB`** so it doesn't mask the voice.
*   **Kick → Bass:** When the kick hits, the bass ducks by **`-2.0 dB`**. This creates a "pumping" effect and prevents the low end from turning to mush.

### 7. Mastering (Master Bus)
*   **Stereo Imaging:** All low end below 120 Hz is forced to mono (so it doesn't fall apart in a club), while high frequencies (above 10 kHz) are widened (`+3 dB` on Side).
*   **Parallel Compression:** Very gentle (only 5% of the compressed signal is mixed in) just to add density.
*   **Subtle EQ:** Boost 150 Hz (`+1 dB`) for weight.
*   **Limiter:** Bring the final loudness to **-12.0 LUFS** with a ceiling of **-1.0 dBTP**. The limiter works gently, so there are no dips or "mush" in the track.

**Summary:** The script does all the routine work of a sound engineer (EQ, compression, panning, sidechain) while preserving the artist's original vision. You feed it the "raw" material, and it spits out a ready, clean, and powerful mix.

<br>
<hr>
<br>

## ℹ️ О чем это
Автоматическое сведение и мастеринг треков с использованием нейронных сетей. Имеет интерфейс перетаскивания файлов в окно, разделения на стемы (Demucs) и профессиональную обработку звука (Pedalboard).

## 🚀 Быстрый старт
Просто скачайте 2 файла: **`stem-audio-prod.py`** и **`start_stem-audio-prod.bat`** в одну папку.
Дважды кликните по **`.bat`** файлу для запуска — при первом старте программа сама скачает и установит все нужные библиотеки.

---

## ⚙️ Как это работает (Пайплайн обработки)

Когда вы перетаскиваете трек в окно программы, она выполняет следующие шаги:

### 1. Подготовка (Загрузка и нейросеть)
*   **Разделение:** Нейросеть Demucs (`htdemucs_ft`) разделяет трек на 4 независимых слоя: `drums` (барабаны), `bass` (бас), `other` (инструменты) и `vocals` (вокал).
*   **Выравнивание:** Программа проверяет длину всех стемов, конвертирует моно-файлы в стерео и обрезает их до одинакового размера, чтобы ничего не разъезжалось по времени.

### 2. Обработка баса (3 слоя + гармония + атака)
Чтобы сделать бас плотным и рычащим, он разделяется на полосы:
*   **SUB (Саб):** Отрезается всё выше 100 Гц (`Lowpass`). Добавляется `+3 дБ` на 50 Гц и жесткая компрессия (ratio 10:1). Это то, что качает живот.
*   **BODY (Тело):** Оставляем 100–500 Гц, но **вырезаем `-4 дБ` на 250 Гц** (убираем гудеж) и добавляем `+2 дБ` на 150 Гц (чтобы была слышна нота).
*   **TOP (Рык):** Всё выше 500 Гц. Добавляем дисторшн `+10 дБ` и поднимаем `+3 дБ` на 800 Гц. Это дает агрессию и пробивную способность.
*   **HARMONICS (Гармония):** Берем бас, сдвигаем его на октаву вверх (PitchShift +12), чистим от низа (Highpass 300 Гц), добавляем дисторшн и тихо подмешиваем (20%). Кажется, что играет второй синт.
*   **Транзиент:** Подмешиваем кусочек бочки (отфильтрованный 100–2000 Гц) к басу на 30% громкости, чтобы у каждой ноты был четкий удар.

### 3. Обработка барабанов (`drum_punch`)
*   Добавляем легкую сатурацию для плотности (Distortion +1 дБ).
*   Поднимаем бочку (`+3 дБ` на 55 Гц) и вырезаем мутность (`-3 дБ` на 200 Гц).
*   Поднимаем щелчок снейра (`+4 дБ` на 3.5 кГц) и добавляем воздух (`+1 дБ` на 8 кГц).
*   Расширяем стерео (но низ до 120 Гц оставляем в центре).

### 4. Обработка вокала (`vocal_surgical`)
*   Разделяем на центр (Mid) и бока (Side).
*   В центре ставим медленный компрессор (Leveler, ratio 3:1), чтобы вокал не проваливался в дропе.
*   Вырезаем бубнеж (`-3 дБ` на 500 Гц), добавляем четкость (`+4 дБ` на 3000 Гц и `+2 дБ` на 2500 Гц) и воздух (`+2 дБ` на 8 кГц).
*   Бока (реверб) расширяем (×1.2) для объема.

### 5. Обработка инструментов (`bright_surgical`)
*   Это «хирургия». Мы вырезаем частоты в районе 1.5 кГц и 2.5 кГц, но делаем это **мягко (`-2 дБ`)**, чтобы не съесть дисторшн и сохранить тело инструментов.
*   Добавляем плотность на 800 Гц (`+2 дБ`) и оставляем яркость на 3 кГц (`+3 дБ`).

### 6. Умные эффекты (Sidechain)
*   **Вокал → Инструменты:** Когда поет вокалист, музыка (`other`) проседает на **`-3.0 дБ`**, чтобы не мешать голосу.
*   **Бочка → Бас:** Когда бьет бочка, бас проседает на **`-2.0 дБ`**. Это создает «кач» и не дает низам слипнуться в кашу.

### 7. Мастеринг (Master Bus)
*   **Стереофазация:** Весь низ до 120 Гц принудительно сводится в моно (чтобы не разваливалось на клубе), а верхние частоты (выше 10 кГц) расширяются (`+3 дБ` на Side).
*   **Параллельная компрессия:** Очень деликатная (смешивается всего 5% от сжатого сигнала), просто чтобы добавить плотности.
*   **Легкий EQ:** Поднимаем 150 Гц (`+1 дБ`) для жира.
*   **Лимитер:** Приводим итоговую громкость к **-12.0 LUFS** и ставим потолок **-1.0 dBTP**. Лимитер работает мягко, поэтому в треке нет провалов и «каши».

**Итог:** Скрипт делает всю рутинную работу звукорежиссера (эквалайзер, компрессия, панорама, сайдчейн), при этом сохраняя оригинальную задумку автора. Ты даешь ему «сырье», а он выдает готовый, чистый и мощный микс.
