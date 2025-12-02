# How to Run NeuroNova

## Quick Start (Non-Interactive Demo)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo:**
   ```bash
   python examples/demo_noninteractive.py
   ```

3. **Check outputs:**
   - `assets/transcript.txt` - Conversation transcript
   - `assets/volt_example_*.png` - Membrane potential plots
   - `assets/conversation_graph.png` - Emotion heatmap graph

## Interactive Mode

1. **Run the CLI:**
   ```bash
   python -m neuronova.engine
   ```
   Or:
   ```bash
   python NNE.py
   ```

2. **Available commands:**
   - Type any text to analyze emotions
   - `graph` - Open live emotion visualization
   - `history` - Show conversation history
   - `bot` - Start chatbot session
   - `exit` or `quit` - Exit program

## Configuration

Set environment variables to customize behavior:

- `GEMINI_API_KEY` - Your Gemini API key (optional, for LLM responses)
- `NEURONOVA_DEBUG=1` - Enable debug logging
- `NEURONOVA_GUI=0` - Disable GUI (for headless servers)
- `NEURONOVA_DEMO=1` - Enable demo mode

## Requirements

- Python 3.7+
- numpy
- matplotlib

Optional (for enhanced features):
- sentence-transformers (for semantic embeddings)
- transformers + torch (for local chatbot)

