# AERIS

**Post-flight, vehicle-aware, SOP-referenced flight-test analysis and engineering decision-support platform.**

---

## Purpose

AERIS processes post-flight flight-test and telemetry data to support engineering analysis, deviation detection, and evidence-based decision-making.

### Core Pipeline

```
POST-FLIGHT EVIDENCE
  â†’ DATA UNDERSTANDING
    â†’ ENGINEERING ANALYSIS
      â†’ DEVIATION / ANOMALY DETECTION
        â†’ EVIDENCE
          â†’ INVESTIGATION SUPPORT
            â†’ ENGINEERING RECOMMENDATION
```

### Epistemic Integrity

AERIS maintains a strict distinction between levels of knowledge:

```
RAW DATA â†’ VALIDATED DATA â†’ CALCULATED RESULT â†’ OBSERVATION
  â†’ INFERENCE â†’ HYPOTHESIS â†’ RECOMMENDATION
```

An inference is never presented as an established fact.

## AERIS Is Not

- A real-time cockpit safety system
- An autopilot or flight-control system
- A certified safety-critical system
- A replacement for qualified engineering judgment

## Current Status

| Component | Status |
|-----------|--------|
| Project scaffold | âœ… Implemented |
| Core engine | ðŸ“‹ Planned |
| Backend API | ðŸ“‹ Planned |
| Frontend | ðŸ“‹ Planned |

## Development Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Architecture

Four-layer architecture:

1. **Frontend** (Plotly Dash) â€” Visualization and user interaction
2. **Backend** (FastAPI) â€” API, orchestration, file handling
3. **AERIS Core Engine** (Pure Python) â€” All engineering intelligence
4. **Data / Configuration** â€” Files, config, provenance

Engineering calculations reside exclusively in the core engine.

## License

Proprietary â€” All rights reserved.

