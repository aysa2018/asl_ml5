### 12.02.2026 - 19.02.2026

* **Fixed import/export bug:** Implemented a robust JSON parser to handle plain text, objects, and base64 DataURLs. Dataset export → import now works reliably across browsers.
* **Improved hand handling:** System detects up to two hands but consistently tracks only one (left/right configurable), making single-hand use much more stable.
* **Model tuning:** Combined kNN weighting, confidence + margin gating, smoothing, and stronger NONE training to reduce false positives.
* **Visual upgrade:** Cropped canvas to video size, removed bottom debug panel, added minimal HUD and a clean top-right prediction badge overlay.


