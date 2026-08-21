# AI Signal Terminal

**AI-enabled signal processing terminal with interactive data interpretation display panel** - open-source software demonstrator associated with UK Registered Design application **6536215**.

> This repository is a functional software demonstrator and research project. It does not reproduce, define, interpret, or expand the legal scope of the UK registered design. A UK registered design concerns protected appearance; this software implementation should not be treated as a statement about patentability or design-right scope.

## What this project does

AI Signal Terminal turns raw time-series data into an interactive engineering workspace:

- CSV signal import or synthetic demonstration signals
- time-domain waveform visualization
- optional Butterworth band-pass filtering
- FFT magnitude spectrum
- Welch power spectral density
- time-frequency spectrogram
- dominant-frequency and signal-statistics extraction
- bounded Isolation Forest anomaly detection
- transparent rule-based interpretation of signal behavior
- highlighted anomaly windows
- downloadable interpreted signal CSV
- local/offline operation by default

The goal is to make signal interpretation understandable instead of presenting only raw plots or opaque AI labels.

## Architecture

```text
Sensor / CSV / generated signal
            |
            v
     Acquisition layer
            |
            v
  Signal conditioning/filter
            |
      +-----+-----+
      |           |
      v           v
  FFT / PSD    Spectrogram
      |           |
      +-----+-----+
            |
            v
       Feature engine
            |
            v
     Anomaly detection
            |
            v
 Transparent interpretation
            |
            v
 Interactive display panel
```

## Quick start

```bash
git clone https://github.com/bnrohit/ai-signal-terminal.git
cd ai-signal-terminal
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown by Streamlit, normally `http://localhost:8501`.

## Docker

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## CSV format

Either provide a time column in seconds:

```csv
time,signal
0.000,0.12
0.002,0.31
0.004,0.52
```

or upload a signal-only CSV and enter the sample rate in the UI.

## AI and safety model

The initial release intentionally uses explainable, bounded analytics rather than sending signal data to a third-party model. Isolation Forest is used for anomaly-window scoring, while the interpretation layer explains measurable features such as dominant frequency, crest factor, kurtosis and anomaly occupancy.

This is decision-support software. It is not medical diagnostic software, a safety certification system, or calibrated measurement equipment.

## UK Registered Design reference

- Design application number: **6536215**
- Filing date shown on the registration document: **01 July 2026**
- Design title: **AI enabled signal processing terminal with interactive data interpretation display panel**
- Registration document supplied with this project: **uncertified copy**

The public registration document lists multiple owners. This repository intentionally omits private postal addresses. See [`docs/DESIGN_REGISTRATION.md`](docs/DESIGN_REGISTRATION.md) for public-safe attribution and status-check instructions.

## Design status lookup

Official UK service:

https://www.registered-design.service.gov.uk/find

Search for design application number `6536215`.

Tutorial supplied by the project owner:

https://www.youtube.com/watch?v=5aIIvAjUJuU

## Repository structure

```text
ai-signal-terminal/
├── app.py
├── signal_terminal/
│   ├── io.py
│   ├── processing.py
│   └── intelligence.py
├── sample_data/
│   └── example_signal.csv
├── tests/
│   └── test_processing.py
├── docs/
│   └── DESIGN_REGISTRATION.md
├── .github/workflows/ci.yml
├── .github/FUNDING.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── LICENSE
```

## Roadmap

- live serial/MQTT/UDP acquisition adapters
- multi-channel synchronized analysis
- wavelet transforms
- configurable feature pipelines
- explainable model comparison
- ONNX edge inference
- Raspberry Pi / industrial-PC deployment profile
- session reports and PDF export
- device health and data-quality scoring
- optional local LLM interpretation with strict data-boundary controls

## Contributing

Contributions are welcome, especially signal-processing methods, tests, input adapters, accessibility improvements, visualization enhancements and reproducible sample datasets.

## Support open source

The project is free and open source. Optional community support helps fund maintenance, testing, documentation and future development:

https://buy.stripe.com/00wdR99fp2R7aoHfku3Ru00

Support does not unlock hidden or paid functionality.

## License

MIT License. See [`LICENSE`](LICENSE).
