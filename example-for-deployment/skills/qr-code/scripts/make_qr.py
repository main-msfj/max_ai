"""Write a QR code PNG: make_qr.py TEXT OUTPUT.png"""

import sys
from pathlib import Path

try:
    import qrcode
except ModuleNotFoundError:
    sys.exit("missing package: qrcode (install it with: pip install 'qrcode[pil]')")

if len(sys.argv) != 3:
    sys.exit("usage: make_qr.py TEXT OUTPUT.png")

text, output = sys.argv[1], Path(sys.argv[2])
qrcode.make(text).save(output)
print(f"saved {output.resolve()} ({len(text)} characters encoded)")
