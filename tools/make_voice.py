"""Generate the spoken prompts for Word Train and embed them in index.html.

Every line the game passes to say() is generated with Kokoro, packed into one MP3
sprite, and written into index.html as base64 between the /*VOICE*/ markers, so the
game stays a single self-contained file. Keys are the exact spoken text; any line
missing here falls back to the browser's speechSynthesis.

Answer words are only read back after they've been answered (it's a reading game);
they're taken straight from the LEVELS lists in index.html.

Needs: kokoro-onnx, soundfile, numpy, Homebrew espeak-ng, lame, and the Kokoro model
files kokoro.onnx and voices.bin in the working directory.
Usage: python make_voice.py <path/to/index.html> [kokoro_voice]
"""
import base64, json, os, re, subprocess, sys, tempfile
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro, EspeakConfig

# Must match the strings in index.html (PROMPTS, PRAISE and the say() calls).
PROMPTS = ["Which word matches the picture?", "Find the word for this picture!", "Which word is it?"]
PRAISE = ["Great reading!", "Toot toot! You got it!", "Yes! Full steam ahead!", "Super reading!", "Faster and faster!"]
LINES = PROMPTS + PRAISE + [
    "All aboard! " + PROMPTS[0],
    "Choo choo! Off we go!",
    "Now tap this word!",
    "Moo! A cow is on the track! Blow the whistle!",
    "Moo! Cows on the track! Blow the whistle!",
    "Toot the whistle!",
    "You did it, Conductor! Hooray!",
    "Word Train! Pick a level. Green is easy. Yellow is medium. Red is hard.",
]


def trim(a, sr, thresh=0.01, pad=0.04):
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    return a[max(0, idx[0] - int(pad * sr)):min(len(a), idx[-1] + int(pad * sr))]


def main():
    html = sys.argv[1]
    voice = sys.argv[2] if len(sys.argv) > 2 else "af_heart"
    src = open(html, encoding="utf-8").read()
    words = re.findall(r'\["([a-z]+)","', src)
    lines = LINES + [w.capitalize() + "!" for w in words]
    k = Kokoro("kokoro.onnx", "voices.bin", espeak_config=EspeakConfig(
        lib_path="/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib",
        data_path="/opt/homebrew/opt/espeak-ng/share/espeak-ng-data"))
    gap, sr = 0.35, 24000
    chunks, index, pos = [], {}, 0.0
    for text in lines:
        a, sr = k.create(text, voice=voice, speed=0.9, lang="en-us")
        a = trim(np.asarray(a, dtype=np.float32), sr)
        a = a / max(1e-6, np.abs(a).max()) * 0.89  # even loudness across clips
        index[text] = [round(pos + gap, 3), round(len(a) / sr, 3)]
        chunks += [np.zeros(int(gap * sr), np.float32), a]
        pos += gap + len(a) / sr
        print(f"{pos:6.1f}s  {text}", flush=True)
    chunks.append(np.zeros(int(gap * sr), np.float32))
    with tempfile.TemporaryDirectory() as td:
        wav, mp3 = os.path.join(td, "v.wav"), os.path.join(td, "v.mp3")
        sf.write(wav, np.concatenate(chunks), sr)
        subprocess.run(["lame", "--quiet", "-m", "m", "-b", "40", wav, mp3], check=True)
        data = base64.b64encode(open(mp3, "rb").read()).decode()
    blob = json.dumps({"voice": voice, "clips": index, "mp3": data}, separators=(",", ":"))
    out, n = re.subn(r"/\*VOICE\*/.*?/\*/VOICE\*/", lambda _: f"/*VOICE*/{blob}/*/VOICE*/", src, flags=re.S)
    if n != 1:
        sys.exit("VOICE markers not found in " + html)
    open(html, "w", encoding="utf-8").write(out)
    print(f"done: {len(lines)} clips, {pos:.0f}s, {len(data) // 1024} KB base64")


if __name__ == "__main__":
    main()
