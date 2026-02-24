### 22.02.2026 - 26.02.2026

* Cropped and prepared visual assets for all 26 ASL alphabet gestures
* Imported and integrated the visuals into the p5.js project, syncing them with the gesture prediction system
* Implemented progression mechanics and a progress bar that cycles through alphabets and advances only after the correct gesture is detected
* Developed a Word Mode allowing users to input their name or any word and learn the corresponding gesture sequence
* Enhanced overall UI and UX for improved clarity, feedback, and usability ( including a home screen and congratulations page )

### 12.02.2026 - 19.02.2026

* **Fixed import/export bug:** Implemented a robust JSON parser to handle plain text, objects, and base64 DataURLs. Dataset export → import now works reliably across browsers.
* **Improved hand handling:** System detects up to two hands but consistently tracks only one (left/right configurable), making single-hand use much more stable.
* **Model tuning:** Combined kNN weighting, confidence + margin gating, smoothing, and stronger NONE training to reduce false positives.
* **Visual upgrade:** Cropped canvas to video size, removed bottom debug panel, added minimal HUD and a clean top-right prediction badge overlay.

