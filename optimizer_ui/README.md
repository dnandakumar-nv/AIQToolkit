# NeMo Agent Toolkit Optimizer UI

A professional, interactive web interface for the NeMo Agent Toolkit Optimizer. This UI makes it easy to configure, run, and analyze optimization experiments with real-time progress tracking and beautiful visualizations.

## Features

### 🎨 Modern, Professional Interface
- Clean, dark-themed UI optimized for long optimization sessions
- Responsive design that works on desktop and tablet
- Intuitive navigation with four main sections

### ⚙️ Configuration Management
- **Load configurations** from YAML/JSON files or file paths
- **Interactive YAML editor** with syntax highlighting powered by Monaco Editor
- **Visual parameter explorer** showing all optimizable parameters
- **Search space customization** for numeric, categorical, and prompt parameters
- **Config validation** with helpful error messages

### 🚀 Real-Time Optimization
- **Live progress tracking** via WebSocket connections
- **Trial-by-trial updates** showing current trial, total trials, and progress percentage
- **Start/Stop controls** with graceful cancellation
- **Live optimization log** with timestamped events
- **Visual progress bars** and statistics dashboard

### 📊 Interactive Visualizations
- **Pareto front plots** using Plotly for interactive exploration
- **Multi-objective visualization** with 2D scatter plots and parallel coordinates
- **Trials data table** showing detailed results for each trial
- **Hover interactions** to explore trial parameters and metrics
- **Export capabilities** for all visualizations

### 💡 AI-Powered Insights
- **Best trial identification** across multiple metrics
- **Parameter correlation analysis** to understand which parameters matter most
- **Convergence tracking** to see optimization progress over time
- **Automated statistics** summarizing optimization results
- **Actionable recommendations** based on results

## Architecture

### Backend (Python/FastAPI)
```
optimizer_ui/backend/
├── main.py                    # FastAPI application entry point
├── api/
│   ├── config_routes.py      # Configuration management endpoints
│   ├── optimization_routes.py # Optimization execution & WebSocket
│   └── results_routes.py     # Results retrieval & analytics
└── services/
    └── optimization_service.py # Core optimization logic & progress tracking
```

### Frontend (Vanilla JavaScript + Modern Libraries)
```
optimizer_ui/frontend/
├── index.html                # Single-page application
└── app.js                    # Application logic and API integration
```

**Key Technologies:**
- **Plotly.js** - Interactive visualizations
- **Monaco Editor** - Professional code editor
- **WebSocket** - Real-time progress updates
- **Material Icons** - Consistent iconography

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- NeMo Agent Toolkit installed
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Quick Start

1. **Navigate to the optimizer UI directory:**
   ```bash
   cd optimizer_ui
   ```

2. **Run the startup script:**

   **On macOS/Linux:**
   ```bash
   ./start.sh
   ```

   **On Windows:**
   ```batch
   start.bat
   ```

   The script will:
   - Create a virtual environment
   - Install dependencies
   - Start the backend server on http://localhost:8080

3. **Open your browser:**
   Navigate to http://localhost:8080

That's it! The UI will be running and ready to use.

## Usage Guide

### 1. Configure Your Optimization

**Step 1: Load a Configuration**
- Click the file upload area or "Load" button
- Select your YAML/JSON config file
- Alternatively, enter a file path and click "Load"

**Step 2: Review & Edit Configuration**
- The Monaco editor will display your config with syntax highlighting
- Make any needed changes directly in the editor
- Save your changes using the "Save Config" button

**Step 3: Explore Optimizable Parameters**
- Scroll down to see all parameters that will be optimized
- Review search spaces for numeric parameters (low, high, step)
- Check categorical options and prompt optimization settings
- Note which parameters are enabled for optimization

### 2. Run Optimization

**Step 1: Navigate to Optimize Tab**
- Click "Optimize" in the sidebar

**Step 2: Start Optimization**
- Click "Start Optimization" button
- The UI will immediately connect via WebSocket for real-time updates

**Step 3: Monitor Progress**
- Watch the progress bar advance
- See trial-by-trial updates in real-time
- View statistics: total trials, current trial, completion percentage
- Monitor the optimization log for detailed events

**Step 4: Control Execution**
- Click "Stop" button to gracefully cancel if needed
- Wait for completion notification

### 3. Analyze Results

**Step 1: Navigate to Results Tab**
- Click "Results" in the sidebar

**Step 2: Explore Visualizations**
- **Pareto Front Plot**: Interactive scatter plot showing trade-offs between objectives
  - Hover over points to see trial details
  - Zoom and pan to explore regions of interest
  - Identify optimal trials on the Pareto frontier

- **Trials Table**: Detailed data for each trial
  - Review parameters tested
  - Compare metric values
  - Sort by different columns

**Step 3: Download Results**
- Click "Download" to get optimized configuration
- Export visualizations as images
- Save trials data as CSV

### 4. Get Insights

**Step 1: Navigate to Insights Tab**
- Click "Insights" in the sidebar

**Step 2: Review AI-Generated Insights**
- **Best Trials**: See which trials performed best for each metric
- **Correlations**: Understand which parameters most influence outcomes
- **Convergence**: Track how optimization improved over time
- **Statistics**: Get overall summary of optimization run

**Step 3: Take Action**
- Use insights to refine your search space
- Identify promising parameter ranges
- Understand trade-offs between objectives
- Plan follow-up optimization runs

## API Documentation

The backend exposes a RESTful API and WebSocket endpoints:

### Configuration Endpoints

**POST /api/config/load**
- Upload and parse a configuration file
- Returns: config object and optimizable parameters

**POST /api/config/load-from-path**
- Load configuration from filesystem path
- Query param: `path` (string)

**POST /api/config/save**
- Save updated configuration to file
- Body: `{config: dict, config_path: string}`

**POST /api/config/validate**
- Validate configuration without saving
- Body: `{config: dict}`

### Optimization Endpoints

**POST /api/optimization/start**
- Start a new optimization run
- Body: `{config: dict, dataset_path?: string, ...}`
- Returns: `{run_id: string, status: string}`

**GET /api/optimization/status/{run_id}**
- Get current status of optimization run
- Returns: status, progress, current_trial, etc.

**POST /api/optimization/stop/{run_id}**
- Stop a running optimization

**GET /api/optimization/runs**
- List all optimization runs

**WebSocket /api/optimization/ws/{run_id}**
- Real-time progress updates
- Emits: status changes, trial updates, progress percentages

### Results Endpoints

**GET /api/results/{run_id}/summary**
- Get overall results summary

**GET /api/results/{run_id}/trials**
- Get detailed trials data with statistics

**GET /api/results/{run_id}/visualizations**
- Get all available visualizations as base64 images

**GET /api/results/{run_id}/download/{file_type}**
- Download specific result files
- Types: `config`, `trials`, `pareto_2d`, `pareto_parallel`, `pareto_pairwise`

**GET /api/results/{run_id}/insights**
- Get AI-generated insights and analytics

## Configuration Requirements

Your configuration file must include:

1. **Optimizer Section:**
   ```yaml
   optimizer:
     output_path: "optimizer_results"
     numeric:
       enabled: true
       n_trials: 50
     eval_metrics:
       latency:
         evaluator_name: "latency"
         direction: "minimize"
         weight: 0.3
       accuracy:
         evaluator_name: "accuracy"
         direction: "maximize"
         weight: 0.7
   ```

2. **Optimizable Parameters:**
   - Use `OptimizableField` in your data models
   - Enable parameters via `optimizable_params` list
   - Define search spaces with `low`, `high`, `step` or `values`

See the [main optimizer documentation](../docs/source/reference/optimizer.md) for details.

## Troubleshooting

### Port Already in Use
If port 8080 is already in use, edit `start.sh`/`start.bat` and change the port:
```bash
--port 8081
```
Then access the UI at http://localhost:8081

### Failed to Fetch Error on File Upload
If you see a "failed to fetch" error when uploading configuration files:

1. **Check Backend is Running**: The UI will show a notification on page load if it cannot connect to the backend
   - Ensure you ran `./start.sh` (macOS/Linux) or `start.bat` (Windows)
   - Check the terminal for any error messages
   - Try accessing http://localhost:8080/api/health directly in your browser

2. **Access UI Correctly**: Always access the UI through http://localhost:8080, not by opening the HTML file directly
   - ✅ Correct: http://localhost:8080
   - ❌ Wrong: file:///path/to/optimizer_ui/frontend/index.html

3. **Check Browser Console**: Press F12 and check the Console tab for detailed error messages
   - Network errors indicate the backend is not reachable
   - CORS errors suggest accessing from wrong origin

4. **Restart the Server**: Stop the server (Ctrl+C) and restart it with `./start.sh`

### WebSocket Connection Failed
- Ensure the backend is running
- Check browser console for errors
- Verify firewall/security settings allow WebSocket connections

### Configuration Not Loading
- Verify YAML/JSON syntax is correct
- Check that file path is absolute or relative to current directory
- Review backend logs for specific error messages

### Optimization Fails to Start
- Validate configuration using the /api/config/validate endpoint
- Ensure all required fields are present in optimizer config
- Check that optimizable parameters have valid search spaces

### Asyncio/Event Loop Errors
If you see errors like:
- `RuntimeError: asyncio.run() cannot be called from a running event loop`
- `ValueError: Can't patch loop of type <class 'uvloop.Loop'>`

These are caused by conflicts between the UI's async server and optimization libraries. The system automatically handles these by:
1. Running optimization in a separate thread
2. Using standard asyncio event loops for compatibility with libraries like ragas
3. Preserving uvloop performance for the main server

If issues persist, check the backend logs for detailed error messages.

## Development

### Running in Development Mode

**Backend:**
```bash
cd optimizer_ui/backend
python -m uvicorn main:app --reload --port 8080
```

**Frontend Development:**
The frontend is a static SPA, so you can:
- Open `frontend/index.html` directly in a browser
- Or use any static file server:
  ```bash
  cd optimizer_ui/frontend
  python -m http.server 8000
  ```

### Adding New Features

1. **Backend**: Add routes in `backend/api/` and logic in `backend/services/`
2. **Frontend**: Update `frontend/app.js` and styles in `frontend/index.html`
3. **Test**: Use `/api/health` endpoint to verify backend is running

## Security Considerations

This UI is designed for **local use only**:
- ✅ No authentication required
- ✅ CORS enabled for localhost
- ✅ File access limited to configured paths
- ⚠️ **Do not expose to the internet without adding:**
  - Authentication (e.g., OAuth, JWT)
  - HTTPS/TLS encryption
  - Input validation & sanitization
  - Rate limiting
  - CORS restrictions

For production deployment, implement proper security measures.

## Performance Tips

1. **Large Optimization Runs**: The UI handles hundreds of trials efficiently. For thousands of trials, consider:
   - Using pagination in trials table
   - Limiting visualization data points
   - Downloading raw data for offline analysis

2. **Multiple Concurrent Runs**: The backend supports multiple optimization runs, but be mindful of system resources.

3. **Browser Performance**: For best experience, use a modern browser with hardware acceleration enabled.

## License

SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

## Support

For issues or questions:
- Check the [main NAT documentation](../docs/)
- Review the [optimizer guide](../docs/source/reference/optimizer.md)
- Search for similar issues in the project repository

---

**Built with ❤️ for the NeMo Agent Toolkit community**
