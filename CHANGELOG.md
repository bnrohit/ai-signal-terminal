# Changelog

## 0.2.0 - 2026-08-21

### Added
- multi-channel CSV ingestion and aligned channel comparison
- magnitude-squared coherence and lag/correlation analysis
- low-pass, high-pass, band-pass and configurable notch filtering
- custom frequency-band power measurement
- normalized spectral entropy and zero-crossing features
- transparent signal-quality score with clipping/drift/high-frequency indicators
- anomaly event segmentation with start/end/duration/peak score
- downloadable structured engineering interpretation JSON
- upload size guardrail and irregular-timebase rejection
- richer tabbed Streamlit interface
- CI matrix across Python 3.11, 3.12 and 3.13
- Streamlit health smoke test in CI
- Docker image build verification in CI

### Safety and interpretation
The quality score and AI-assisted interpretation remain transparent engineering heuristics. They are not calibration results, medical diagnoses, or safety certifications.

## 0.1.0 - 2026-08-21

Initial open-source demonstrator with FFT, Welch PSD, spectrograms, filtering, feature extraction, Isolation Forest anomaly detection and interactive visualization.
