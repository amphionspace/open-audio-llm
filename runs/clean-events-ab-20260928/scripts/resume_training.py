import sys
from open_audio_llm.cli import main

raise SystemExit(main(['experiment', 'run', *sys.argv[1:]]))
