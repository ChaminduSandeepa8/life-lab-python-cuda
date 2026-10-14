# Life Lab — සිංහල quick start

මේ project එක Windows, Linux සහ macOS සඳහා CPU mode එකෙන් run කරන්න setup instructions තිබෙනවා. CUDA mode එකට Windows/Linux, compatible NVIDIA GPU එකක් සහ driver එකක් අවශ්‍යයි. macOS සහ GPU නැති computers වල **CPU only** භාවිත කරන්න.

## Clone කරන්න

Git සහ Python 3.10–3.13 install කරගන්න. අලුත් setup එකකට Python 3.12 භාවිත කරන්න.

```text
git clone https://github.com/ChaminduSandeepa8/life-lab-python-cuda.git
cd life-lab-python-cuda
```

## Windows

```powershell
py -3.12 tools/setup_environment.py
.venv\Scripts\python.exe app.py
```

ඊළඟ වතාවේ `run.bat` භාවිත කරන්න. PowerShell නම් `.\run.bat`. Environment එක activate කිරීම අනිවාර්ය නැහැ.

## Linux

Ubuntu/Debian වල මුලින් `sudo apt install python3-venv python3-tk git` run කරන්න.

```bash
python3 tools/setup_environment.py
bash run.sh
```

## macOS

Python.org එකෙන් Python 3.12 install කරලා:

```bash
python3.12 tools/setup_environment.py
bash run.sh
```

## CUDA තිබෙනවා නම්

NVIDIA driver එක install කරලා `nvidia-smi` command එකෙන් GPU එක පෙන්වනවාද බලන්න. Project environment එකේ Python භාවිත කරලා:

```text
python -m pip install -r requirements-cuda.txt
python app.py --check --require-cuda
```

Windows වල environment එක activate කරලා නැත්නම් `python` වෙනුවට `.venv\Scripts\python.exe` භාවිත කරන්න. Linux වල `.venv/bin/python` භාවිත කරන්න. CUDA check එක pass වුණාම **Both** තෝරලා CPU/GPU compare කරන්න.

## App එක භාවිත කිරීම

1. Settings window එකේ grid 64×64, iterations 100, Normal Python සහ Live view තෝරන්න.
2. Pattern/density වෙනස් කළාම preview එක වෙනස් වෙනවා.
3. Run click කළාම දෙවැනි window එක විවෘත වෙනවා. එහි Start simulation click කරන්න.
4. Pause/Resume, Stop සහ Reset controls භාවිත කරන්න.
5. Performance මැනීමට Benchmark mode භාවිත කරන්න. Graph සහ statistics පහළින් තිබෙනවා.
6. Export CSV සහ Save graphs + stat table මඟින් results save කරන්න.

Native app එකට browser/server එකක් අවශ්‍ය නැහැ. පරණ browser version එක අවශ්‍ය නම් `python app.py --web` run කරලා http://127.0.0.1:8000 විවෘත කරන්න. Terminal එක close කළාම server එකත් නවතිනවා.

සම්පූර්ණ instructions සහ error fixes: [INSTALL.md](INSTALL.md). Controls සහ timing විස්තර: [DESKTOP.md](DESKTOP.md).
