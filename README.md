# AI Signal Terminal

**AI-enabled signal processing terminal with interactive data interpretation display panel** — open-source software demonstrator associated with UK Registered Design application **6536215**.

> This repository is a functional software demonstrator and research project. It does not reproduce, define, interpret, or expand the legal scope of the UK registered design. A UK registered design concerns protected appearance; this software implementation should not be treated as a statement about patentability or design-right scope.

## What it does

AI Signal Terminal converts raw time-series data into an interactive engineering workspace for signal inspection, frequency analysis, anomaly detection, quality review, and explainable interpretation.

### v0.2 capabilities

- synthetic test-signal generation
- CSV ingestion with up to 8 aligned numeric signal channels
- automatic sample-rate inference from a regular time column
- rejection of strongly irregular timebases that would invalidate ordinary FFT/PSD assumptions
- time-domain waveform visualization
- low-pass, high-pass, band-pass, 50/60 Hz and custom notch filters
- FFT magnitude spectrum
- Welch power spectral density
- time-frequency spectrogram
- top dominant-frequency extraction
- custom frequency-band power calculation
- normalized spectral entropy and zero-crossing features
- Isolation Forest anomaly-window detection
- anomaly event segmentation with start/end/duration/peak score
- transparent signal-quality heuristic for clipping, baseline drift and high-frequency content
- multi-channel Pearson correlation, lag estimation and magnitude-squared coherence
- downloadable interpreted CSV
- downloadable structured engineering interpretation JSON
- local/offline operation by default

The project intentionally favors **transparent measurable features and bounded AI** over opaque conclusions.

## Architecture

```text
Sensor / CSV / generated signal
            |
            v
     Acquisition layer
            |
            v
 Validation + timebase checks
            |
            v
  Signal conditioning/filter
            |
      +-----+----------+----------------+
      |                |                |
      v                v                v
  FFT / PSD       Spectrogram    Channel coherence
      |                |                |
      +----------+-----+----------------+
                 |
                 v
            Feature engine
                 |
          +------+-------+
          |              |
          v              v
    Quality score   Anomaly model
          |              |
          +------+-------+
                 |
                 v
      Transparent interpretation
                 |
                 v
      Interactive display panel
                 |
                 v
        CSV / JSON engineering report
```

## Quick start

```bash
git clone https://github.com/bnrohit/ai-signal-terminal.git
cd ai-signal-terminal
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit, normally `http://localhost:8501`.

## Docker

```bash
docker compose up --build
```

Open `http://localhost:8501`.

The Docker image includes a Streamlit health check on `/_stcore/health`.

## CSV format

Single-channel example:

```csv
time,signal
0.000,0.12
0.002,0.31
0.004,0.52
```

Multi-channel example:

```csv
time,motor_x,motor_y,current
0.000,0.12,0.18,1.44
0.002,0.31,0.29,1.51
0.004,0.52,0.47,1.62
```

If there is no time column, choose the sample rate in the UI. When a time column is used, it must be strictly increasing and sufficiently regular for ordinary frequency-domain analysis.

## AI and engineering interpretation

The current anomaly detector is a bounded **Isolation Forest** applied to window-level statistical features. It does not call an external AI service or transmit signal data to a third-party model.

The interpretation layer explains measurable observations such as dominant frequency, RMS/crest factor, kurtosis, spectral entropy, anomaly windows, signal-quality indicators, and channel coherence.

This is **engineering decision-support software**. It is not medical diagnostic software, calibrated measurement equipment, a safety certification system, or an autonomous control system.

## Verification

The repository CI verifies Python 3.11/3.12/3.13, unit tests, Python compilation, Streamlit startup/health, and a production Docker image build.

Run locally:

```bash
pytest -q
python -m compileall signal_terminal app.py
```

## UK Registered Design reference

- Design application number: **6536215**
- Filing date shown on the supplied registration document: **01 July 2026**
- Design title: **AI enabled signal processing terminal with interactive data interpretation display panel**
- Supplied registration document: **uncertified copy**

The public registration document lists multiple owners. This repository intentionally omits private postal addresses. See [`docs/DESIGN_REGISTRATION.md`](docs/DESIGN_REGISTRATION.md) for public-safe attribution and status-check instructions.

Official UK design lookup: https://www.registered-design.service.gov.uk/find

Search for `6536215`.

Tutorial supplied by the project owner: https://www.youtube.com/watch?v=5aIIvAjUJuU

## Roadmap

Potential next stages include live serial/MQTT/UDP acquisition, synchronized streaming, wavelets, local ONNX inference, baseline drift history, edge-device profiles, signed reports, and optional local LLM interpretation with strict data-boundary controls.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Support open source

The project is free and open source. Optional community support helps fund maintenance, testing, documentation, and future development:

https://buy.stripe.com/00wdR99fp2R7aoHfku3Ru00

Support does not unlock hidden or paid functionality.

## License

MIT License. See [`LICENSE`](LICENSE).
