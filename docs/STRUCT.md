# Project Structure

- `src/aura_harness/`: Main package.
    - `llm/`: Ollama API client and exceptions.
    - `lab/`: Candidate generation logic and models.
    - `ui/`: PySide6 graphical user interface.
        - `main_window.py`: Primary application window and control logic.
        - `worker.py`: Background thread for non-blocking generation.
        - `theme.py`: UI styling and color constants.
- `tests/`: Project test suite.
    - `llm/`: Tests for the Ollama client.
    - `lab/`: Tests for the candidate generator.
    - `ui/`: Tests for the GUI components and workers.
