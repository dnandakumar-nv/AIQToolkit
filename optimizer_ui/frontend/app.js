/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Main application logic for NeMo Agent Toolkit Optimizer UI
 */

const API_BASE = 'http://localhost:8080/api';

class OptimizerApp {
    constructor() {
        this.config = null;
        this.optimizableParams = null;
        this.editor = null;
        this.currentRunId = null;
        this.ws = null;

        this.init();
    }

    async init() {
        this.setupNavigation();
        this.setupMonacoEditor();
    }

    setupNavigation() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const page = item.dataset.page;
                this.navigateTo(page);
            });
        });
    }

    navigateTo(page) {
        // Update nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.page === page) {
                item.classList.add('active');
            }
        });

        // Update pages
        document.querySelectorAll('.page').forEach(p => {
            p.classList.add('hidden');
        });
        document.getElementById(`page-${page}`).classList.remove('hidden');
    }

    async setupMonacoEditor() {
        require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' } });

        require(['vs/editor/editor.main'], () => {
            this.editor = monaco.editor.create(document.getElementById('editor-container'), {
                value: '# Load a configuration file to begin',
                language: 'yaml',
                theme: 'vs-dark',
                automaticLayout: true,
                minimap: { enabled: false },
                fontSize: 13,
                scrollBeyondLastLine: false,
            });

            this.editor.onDidChangeModelContent(() => {
                if (this.config) {
                    try {
                        this.config = jsyaml.load(this.editor.getValue());
                    } catch (e) {
                        console.error('YAML parse error:', e);
                    }
                }
            });
        });
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.showNotification('Loading configuration...', 'info');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE}/config/load`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Failed to load config');

            const data = await response.json();
            this.config = data.config;
            this.optimizableParams = data.optimizable_params;

            // Update editor
            if (this.editor) {
                this.editor.setValue(jsyaml.dump(data.config, { lineWidth: -1 }));
            }

            // Show cards
            document.getElementById('config-editor-card').style.display = 'block';
            document.getElementById('params-card').style.display = 'block';

            // Render parameters
            this.renderParameters();

            this.showNotification('Configuration loaded successfully!', 'success');
        } catch (error) {
            console.error('Error loading config:', error);
            this.showNotification('Error loading configuration', 'error');
        }
    }

    async loadFromPath() {
        const path = document.getElementById('config-path').value;
        if (!path) {
            this.showNotification('Please enter a config path', 'error');
            return;
        }

        this.showNotification('Loading configuration...', 'info');

        try {
            const response = await fetch(`${API_BASE}/config/load-from-path?path=${encodeURIComponent(path)}`, {
                method: 'POST',
            });

            if (!response.ok) throw new Error('Failed to load config');

            const data = await response.json();
            this.config = data.config;
            this.optimizableParams = data.optimizable_params;

            // Update editor
            if (this.editor) {
                this.editor.setValue(jsyaml.dump(data.config, { lineWidth: -1 }));
            }

            // Show cards
            document.getElementById('config-editor-card').style.display = 'block';
            document.getElementById('params-card').style.display = 'block';

            // Render parameters
            this.renderParameters();

            this.showNotification('Configuration loaded successfully!', 'success');
        } catch (error) {
            console.error('Error loading config:', error);
            this.showNotification('Error loading configuration from path', 'error');
        }
    }

    renderParameters() {
        const container = document.getElementById('params-list');
        const paramCount = document.getElementById('param-count');

        const paramKeys = Object.keys(this.optimizableParams);
        paramCount.textContent = `${paramKeys.length} parameters`;

        container.innerHTML = '';

        paramKeys.forEach(key => {
            const param = this.optimizableParams[key];
            const paramDiv = document.createElement('div');
            paramDiv.className = 'param-item';

            let controlsHTML = '';

            if (param.is_prompt) {
                controlsHTML = `
                    <div class="form-group">
                        <label class="form-label">Prompt Purpose</label>
                        <input type="text" class="form-input" value="${param.prompt_purpose || ''}" readonly>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Base Prompt</label>
                        <textarea class="form-input" rows="3" readonly>${param.prompt || ''}</textarea>
                    </div>
                `;
            } else if (param.values) {
                controlsHTML = `
                    <div class="form-group">
                        <label class="form-label">Categorical Values</label>
                        <input type="text" class="form-input" value="${param.values.join(', ')}" readonly>
                    </div>
                `;
            } else {
                controlsHTML = `
                    <div class="form-group">
                        <label class="form-label">Low</label>
                        <input type="number" class="form-input" value="${param.low ?? ''}" readonly>
                    </div>
                    <div class="form-group">
                        <label class="form-label">High</label>
                        <input type="number" class="form-input" value="${param.high ?? ''}" readonly>
                    </div>
                    ${param.step ? `
                    <div class="form-group">
                        <label class="form-label">Step</label>
                        <input type="number" class="form-input" value="${param.step}" readonly>
                    </div>
                    ` : ''}
                    ${param.log ? '<div class="status-badge info">Log Scale</div>' : ''}
                `;
            }

            paramDiv.innerHTML = `
                <div class="param-header">
                    <div class="param-name">${key}</div>
                    <div class="status-badge" style="background: ${param.is_prompt ? 'var(--secondary-color)' : 'var(--surface-light)'};">
                        ${param.is_prompt ? 'Prompt' : param.values ? 'Categorical' : 'Numeric'}
                    </div>
                </div>
                <div class="param-controls">
                    ${controlsHTML}
                </div>
            `;

            container.appendChild(paramDiv);
        });
    }

    async saveConfig() {
        if (!this.config) {
            this.showNotification('No configuration to save', 'error');
            return;
        }

        const path = prompt('Enter save path:', 'optimized_config.yaml');
        if (!path) return;

        try {
            const response = await fetch(`${API_BASE}/config/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config: this.config,
                    config_path: path,
                }),
            });

            if (!response.ok) throw new Error('Failed to save config');

            this.showNotification('Configuration saved successfully!', 'success');
        } catch (error) {
            console.error('Error saving config:', error);
            this.showNotification('Error saving configuration', 'error');
        }
    }

    async startOptimization() {
        if (!this.config) {
            this.showNotification('Please load a configuration first', 'error');
            return;
        }

        // Get latest config from editor
        if (this.editor) {
            try {
                this.config = jsyaml.load(this.editor.getValue());
            } catch (e) {
                this.showNotification('Invalid YAML in editor', 'error');
                return;
            }
        }

        this.showNotification('Starting optimization...', 'info');

        try {
            const response = await fetch(`${API_BASE}/optimization/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config: this.config,
                }),
            });

            if (!response.ok) throw new Error('Failed to start optimization');

            const data = await response.json();
            this.currentRunId = data.run_id;

            // Update UI
            document.getElementById('start-btn').classList.add('hidden');
            document.getElementById('stop-btn').classList.remove('hidden');
            document.getElementById('log-card').style.display = 'block';
            document.getElementById('status-badge').textContent = 'Running';
            document.getElementById('status-badge').className = 'status-badge running';

            // Connect WebSocket
            this.connectWebSocket(this.currentRunId);

            this.showNotification('Optimization started!', 'success');
        } catch (error) {
            console.error('Error starting optimization:', error);
            this.showNotification('Error starting optimization', 'error');
        }
    }

    connectWebSocket(runId) {
        const wsUrl = `ws://localhost:8080/api/optimization/ws/${runId}`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.addLogEntry('Connected to optimization server');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleOptimizationUpdate(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.addLogEntry('WebSocket error occurred', 'error');
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
            this.addLogEntry('Disconnected from optimization server');
        };
    }

    handleOptimizationUpdate(data) {
        if (data.type === 'heartbeat') return;

        // Update stats
        if (data.total_trials) {
            document.getElementById('stat-trials').textContent = data.total_trials;
        }
        if (data.current_trial) {
            document.getElementById('stat-current').textContent = data.current_trial;
        }
        if (data.progress !== undefined) {
            const progress = Math.round(data.progress);
            document.getElementById('stat-progress').textContent = `${progress}%`;
            document.getElementById('progress-fill').style.width = `${progress}%`;
        }

        // Update status
        if (data.status) {
            const badge = document.getElementById('status-badge');
            badge.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
            badge.className = `status-badge ${data.status}`;
        }

        // Update message
        if (data.message) {
            document.getElementById('progress-text').textContent = data.message;
            this.addLogEntry(data.message);
        }

        // Handle completion
        if (data.status === 'completed') {
            this.onOptimizationComplete(data);
        } else if (data.status === 'failed') {
            this.onOptimizationFailed(data);
        }
    }

    onOptimizationComplete(data) {
        document.getElementById('start-btn').classList.remove('hidden');
        document.getElementById('stop-btn').classList.add('hidden');

        this.showNotification('Optimization completed successfully!', 'success');
        this.addLogEntry('Optimization completed!', 'success');

        // Load results
        if (this.currentRunId) {
            this.loadResults(this.currentRunId);
            this.loadInsights(this.currentRunId);
        }

        // Close WebSocket
        if (this.ws) {
            this.ws.close();
        }
    }

    onOptimizationFailed(data) {
        document.getElementById('start-btn').classList.remove('hidden');
        document.getElementById('stop-btn').classList.add('hidden');

        this.showNotification('Optimization failed: ' + (data.message || 'Unknown error'), 'error');
        this.addLogEntry('Optimization failed: ' + (data.message || 'Unknown error'), 'error');

        // Close WebSocket
        if (this.ws) {
            this.ws.close();
        }
    }

    async stopOptimization() {
        if (!this.currentRunId) return;

        try {
            const response = await fetch(`${API_BASE}/optimization/stop/${this.currentRunId}`, {
                method: 'POST',
            });

            if (!response.ok) throw new Error('Failed to stop optimization');

            this.showNotification('Optimization stopped', 'info');
            this.addLogEntry('Optimization stopped by user', 'info');

            document.getElementById('start-btn').classList.remove('hidden');
            document.getElementById('stop-btn').classList.add('hidden');
        } catch (error) {
            console.error('Error stopping optimization:', error);
            this.showNotification('Error stopping optimization', 'error');
        }
    }

    addLogEntry(message, type = 'info') {
        const logContainer = document.getElementById('log-container');
        const timestamp = new Date().toLocaleTimeString();

        const color = type === 'error' ? 'var(--error-color)' :
                     type === 'success' ? 'var(--success-color)' :
                     'var(--text-secondary)';

        const entry = document.createElement('div');
        entry.style.marginBottom = '8px';
        entry.innerHTML = `<span style="color: ${color};">[${timestamp}]</span> ${message}`;

        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    async loadResults(runId) {
        try {
            // Load trials data
            const trialsResponse = await fetch(`${API_BASE}/results/${runId}/trials`);
            if (!trialsResponse.ok) throw new Error('Failed to load trials');

            const trialsData = await trialsResponse.json();
            this.renderTrialsTable(trialsData);
            this.renderParetoVisualization(trialsData);

        } catch (error) {
            console.error('Error loading results:', error);
            this.showNotification('Error loading results', 'error');
        }
    }

    renderTrialsTable(data) {
        const tbody = document.querySelector('#trials-table tbody');
        tbody.innerHTML = '';

        data.trials.slice(0, 50).forEach(trial => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = trial.number || '-';
            row.insertCell(1).textContent = trial.state || '-';

            // Show value columns
            const valueColumns = data.metric_columns || [];
            const valuesCell = row.insertCell(2);
            valuesCell.innerHTML = valueColumns.map(col =>
                `<span style="margin-right: 12px;">${col}: ${trial[col]?.toFixed(4) || '-'}</span>`
            ).join('');
        });
    }

    renderParetoVisualization(data) {
        if (!data.trials || data.trials.length === 0) return;

        const valueColumns = data.metric_columns || [];
        if (valueColumns.length < 2) {
            document.getElementById('pareto-viz').innerHTML =
                '<p style="text-align: center; color: var(--text-secondary); padding: 40px;">Need at least 2 metrics for Pareto visualization</p>';
            return;
        }

        // Prepare data for 2D scatter plot
        const x = data.trials.map(t => t[valueColumns[0]]);
        const y = data.trials.map(t => t[valueColumns[1]]);

        const trace = {
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            name: 'Trials',
            marker: {
                size: 10,
                color: x.map((_, i) => i),
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'Trial #',
                },
            },
            text: data.trials.map(t => `Trial ${t.number}`),
        };

        const layout = {
            title: 'Pareto Front (2D)',
            xaxis: { title: valueColumns[0] },
            yaxis: { title: valueColumns[1] },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#fff' },
            hovermode: 'closest',
        };

        Plotly.newPlot('pareto-viz', [trace], layout, { responsive: true });
    }

    async loadInsights(runId) {
        try {
            const response = await fetch(`${API_BASE}/results/${runId}/insights`);
            if (!response.ok) throw new Error('Failed to load insights');

            const data = await response.json();
            this.renderInsights(data.insights);

        } catch (error) {
            console.error('Error loading insights:', error);
        }
    }

    renderInsights(insights) {
        const container = document.getElementById('insights-container');
        container.innerHTML = '';

        if (!insights || insights.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 20px;">No insights available</p>';
            return;
        }

        insights.forEach(insight => {
            const item = document.createElement('div');
            item.className = `insight-item ${insight.severity}`;

            const icon = insight.type === 'best_trial' ? 'star' :
                        insight.type === 'correlation' ? 'show_chart' :
                        insight.type === 'convergence' ? 'trending_up' :
                        'info';

            item.innerHTML = `
                <span class="material-icons insight-icon">${icon}</span>
                <div class="insight-content">
                    <h4>${insight.title}</h4>
                    <p>${insight.description}</p>
                </div>
            `;

            container.appendChild(item);
        });
    }

    async refreshInsights() {
        if (!this.currentRunId) {
            this.showNotification('No optimization run to refresh', 'error');
            return;
        }

        this.showNotification('Refreshing insights...', 'info');
        await this.loadInsights(this.currentRunId);
        this.showNotification('Insights refreshed', 'success');
    }

    async downloadResults() {
        if (!this.currentRunId) {
            this.showNotification('No results to download', 'error');
            return;
        }

        // Download optimized config
        window.open(`${API_BASE}/results/${this.currentRunId}/download/config`, '_blank');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = 'notification';

        const icon = type === 'success' ? 'check_circle' :
                    type === 'error' ? 'error' :
                    type === 'warning' ? 'warning' :
                    'info';

        const color = type === 'success' ? 'var(--success-color)' :
                     type === 'error' ? 'var(--error-color)' :
                     type === 'warning' ? 'var(--warning-color)' :
                     'var(--secondary-color)';

        notification.innerHTML = `
            <span class="material-icons" style="color: ${color};">${icon}</span>
            <span>${message}</span>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize app when DOM is ready
const app = new OptimizerApp();
