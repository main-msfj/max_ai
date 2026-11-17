---
name: qr-code
description: Makes a QR code image (PNG) from a text or a link. Use it when the user asks for a QR code.
---
# qr-code

Needs the Python package `qrcode` with Pillow (`qrcode[pil]`). It is not installed
by default: install it with pip when the script says it is missing.

1. Run `python3 scripts/make_qr.py "TEXT" OUTPUT.png` from this skill folder, with
   OUTPUT.png in your working directory.
2. Tell the user where the image is and what text it holds.
