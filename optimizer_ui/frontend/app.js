/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Main application logic for NeMo Agent Toolkit Optimizer UI
 */

const API_BASE = 'http://localhost:8080/api';

class OptimizerApp {
    constructor() {
        this.configPath = null;  // Store config file path instead of config dict
        this.configFilename = null;
        this.optimizableParams = null;
        this.editor = null;  // For read-only viewing
        this.currentRunId = null;
        this.ws = null;
        this.diffMode = false;  // Toggle for diff view

        this.init();
    }

    async init() {
        this.setupNavigation();
        this.setupMonacoEditor();
        // Silently check backend connection
        this.checkBackendConnection();
    }

    async checkBackendConnection() {
        try {
            console.log('Checking backend connection...');
            const response = await fetch(`${API_BASE}/health`, {
                method: 'GET',
            }).catch(error => {
                console.error('Backend connection check failed:', error);
                throw error;
            });

            if (!response.ok) {
                throw new Error(`Backend returned status ${response.status}`);
            }

            const data = await response.json();
            console.log('Backend health check successful:', data);
        } catch (error) {
            // Only log to console, don't show notification on init
            // This avoids confusion when the page loads
            console.warn('Backend connection check failed (this is normal on first load):', error);
            console.warn(`Make sure the backend is running at ${API_BASE}`);
        }
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
                value: '# Load a configuration file to view its contents\n# Editing is disabled - config files are loaded directly',
                language: 'yaml',
                theme: 'vs-dark',
                automaticLayout: true,
                minimap: { enabled: false },
                fontSize: 13,
                scrollBeyondLastLine: false,
                readOnly: true,  // Make editor read-only
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
            }).catch(error => {
                console.error('Network error:', error);
                throw new Error(`Network error: ${error.message}. Please ensure the backend server is running on http://localhost:8080`);
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `HTTP ${response.status}: Failed to load config`);
            }

            const data = await response.json();
            console.log('Received config data:', data);

            // Store config path instead of config dict
            this.configPath = data.config_path;
            this.configFilename = data.config_filename;
            this.optimizableParams = data.optimizable_params || {};

            // Load raw config for read-only viewing
            await this.loadConfigContent(this.configFilename);

            // Show cards
            document.getElementById('config-editor-card').style.display = 'block';
            document.getElementById('params-card').style.display = 'block';

            // Render parameters
            this.renderParameters();

            this.showNotification(
                `Configuration loaded: ${data.config_filename} (${data.num_numeric_params} numeric, ${data.num_prompt_params} prompt parameters)`,
                'success'
            );
        } catch (error) {
            console.error('Error loading config:', error);
            this.showNotification(`Error loading configuration: ${error.message}`, 'error');
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
            }).catch(error => {
                console.error('Network error:', error);
                throw new Error(`Network error: ${error.message}. Please ensure the backend server is running on http://localhost:8080`);
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `HTTP ${response.status}: Failed to load config`);
            }

            const data = await response.json();
            console.log('Received config data:', data);

            // Store config path instead of config dict
            this.configPath = data.config_path;
            this.configFilename = data.config_filename;
            this.optimizableParams = data.optimizable_params || {};

            // Load raw config for read-only viewing
            await this.loadConfigContent(data.config_path);

            // Show cards
            document.getElementById('config-editor-card').style.display = 'block';
            document.getElementById('params-card').style.display = 'block';

            // Render parameters
            this.renderParameters();

            this.showNotification(
                `Configuration loaded: ${data.config_filename} (${data.num_numeric_params} numeric, ${data.num_prompt_params} prompt parameters)`,
                'success'
            );
        } catch (error) {
            console.error('Error loading config:', error);
            this.showNotification(`Error loading configuration from path: ${error.message}`, 'error');
        }
    }

    async loadConfigContent(configId) {
        /**
         * Load the raw config content for read-only display
         */
        try {
            const response = await fetch(`${API_BASE}/config/view/${encodeURIComponent(configId)}`);
            if (!response.ok) throw new Error('Failed to load config content');

            const data = await response.json();

            // Update editor with raw content (read-only)
            if (this.editor) {
                this.editor.setValue(data.content);
                this.editor.updateOptions({ readOnly: true });
            }
        } catch (error) {
            console.error('Error loading config content:', error);
            if (this.editor) {
                this.editor.setValue('# Could not load config file content');
            }
        }
    }

    renderParameters() {
        const container = document.getElementById('params-list');
        const paramCount = document.getElementById('param-count');

        if (!this.optimizableParams) {
            console.error('optimizableParams is null or undefined');
            paramCount.textContent = '0 parameters';
            container.innerHTML = '<p style="color: var(--text-secondary); padding: 20px;">No optimizable parameters found</p>';
            return;
        }

        const paramKeys = Object.keys(this.optimizableParams);
        paramCount.textContent = `${paramKeys.length} parameters`;

        container.innerHTML = '';

        if (paramKeys.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); padding: 20px;">No optimizable parameters found</p>';
            return;
        }

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
        // Config editing removed - configs are now read-only
        this.showNotification('Config editing disabled. Configs are loaded directly from files and remain unchanged.', 'info');
    }

    async startOptimization() {
        if (!this.configPath) {
            this.showNotification('Please load a configuration first', 'error');
            return;
        }

        this.showNotification('Starting optimization...', 'info');

        try {
            const response = await fetch(`${API_BASE}/optimization/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config_path: this.configPath,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || 'Failed to start optimization');
            }

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
            this.showNotification(`Error starting optimization: ${error.message}`, 'error');
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

        // Update numeric optimization stats
        if (data.total_trials) {
            document.getElementById('stat-trials').textContent = data.total_trials;
        }
        if (data.current_trial !== undefined && data.current_trial !== null) {
            document.getElementById('stat-current').textContent = data.current_trial;
        }

        // Update prompt optimization stats
        if (data.prompt_enabled) {
            document.getElementById('prompt-stats').style.display = 'grid';
            if (data.total_generations) {
                document.getElementById('stat-generations').textContent = data.total_generations;
            }
            if (data.current_generation !== undefined && data.current_generation !== null) {
                document.getElementById('stat-current-gen').textContent = data.current_generation;
            }
        }

        // Update overall progress
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
            console.log('Loaded trials data:', trialsData);
            console.log('First trial:', trialsData.trials?.[0]);

            // Render all visualizations
            this.renderTrialsTable(trialsData);
            this.renderColorCodedTable(trialsData);
            this.renderParetoVisualization(trialsData);
            this.renderHistograms(trialsData);
            this.renderConvergencePlot(trialsData);
            this.renderCorrelationHeatmap(trialsData);

            // Load prompts for comparison
            this.loadPromptComparisons(runId);

        } catch (error) {
            console.error('Error loading results:', error);
            this.showNotification('Error loading results', 'error');
        }
    }

    renderColorCodedTable(data) {
        try {
            console.log('Rendering color-coded table...');
            const container = document.getElementById('color-coded-table');
            if (!container) {
                console.error('color-coded-table container not found');
                return;
            }

            const thead = container.querySelector('thead tr');
            const tbody = container.querySelector('tbody');
            if (!thead || !tbody) {
                console.error('Table thead or tbody not found');
                return;
            }

            tbody.innerHTML = '';
            thead.innerHTML = '<th>Trial</th>'; // Reset and add trial column

            const valueColumns = data.metric_columns || [];
            const trials = data.trials || [];
            console.log('Color-coded table - trials count:', trials.length);

        // Add header columns for each metric
        valueColumns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col.replace('values_', '');
            thead.appendChild(th);
        });

        // Calculate min/max for each metric for color coding
        const metricRanges = {};
        valueColumns.forEach(col => {
            const values = trials.map(t => t[col]).filter(v => v != null);
            metricRanges[col] = {
                min: Math.min(...values),
                max: Math.max(...values),
            };
        });

        // Helper to get color based on value (green = better, red = worse)
        const getColorForValue = (value, col, direction = 'maximize') => {
            const range = metricRanges[col];
            if (!range || value == null) return 'transparent';

            const normalized = (value - range.min) / (range.max - range.min);

            // For minimize metrics, flip the colors
            const intensity = direction === 'minimize' ? (1 - normalized) : normalized;

            if (intensity > 0.8) return 'rgba(76, 175, 80, 0.3)'; // Green
            if (intensity > 0.6) return 'rgba(76, 175, 80, 0.2)';
            if (intensity < 0.2) return 'rgba(244, 67, 54, 0.3)'; // Red
            if (intensity < 0.4) return 'rgba(244, 67, 54, 0.2)';
            return 'transparent';
        };

        // Limit to top 50 trials for table display
        trials.slice(0, 50).forEach((trial, idx) => {
            const row = tbody.insertRow();

            // Trial number - handle 0 explicitly
            const trialCell = row.insertCell();
            const trialNumber = (trial.number !== null && trial.number !== undefined) ? trial.number : idx;
            trialCell.textContent = trialNumber;
            trialCell.style.fontWeight = '600';

            // Metric values with color coding
            valueColumns.forEach(col => {
                const cell = row.insertCell();
                const value = trial[col];

                if (value != null) {
                    cell.textContent = value.toFixed(4);
                    // Assume maximize by default (you can enhance this with direction info)
                    cell.style.backgroundColor = getColorForValue(value, col, 'maximize');
                } else {
                    cell.textContent = '-';
                }
            });
        });
        console.log('Color-coded table rendered successfully');
        } catch (error) {
            console.error('Error rendering color-coded table:', error);
        }
    }

    async loadPromptComparisons(runId) {
        try {
            console.log('Loading prompt comparisons...');

            // Get prompt comparisons from the new endpoint
            const promptsResponse = await fetch(`${API_BASE}/results/${runId}/prompts`);
            if (!promptsResponse.ok) {
                console.warn('Failed to load prompts, status:', promptsResponse.status);
                throw new Error('Failed to load prompts');
            }

            const promptsData = await promptsResponse.json();
            console.log('Prompts data received:', promptsData);

            if (!promptsData.comparisons || promptsData.comparisons.length === 0) {
                document.getElementById('prompts-container').innerHTML =
                    '<p class="no-prompts-message">No prompt optimization results found. ' +
                    'Prompts are only available if prompt optimization was enabled in the config.</p>';
                return;
            }

            this.renderPromptComparisons(promptsData.comparisons);
            console.log('Prompt comparisons loaded successfully:', promptsData.comparisons.length, 'prompts');

        } catch (error) {
            console.error('Error loading prompt comparisons:', error);
            document.getElementById('prompts-container').innerHTML =
                '<p class="no-prompts-message">Error loading prompt comparisons. See console for details.</p>';
        }
    }

    renderPromptComparisons(comparisons) {
        const container = document.getElementById('prompts-container');

        if (!comparisons || comparisons.length === 0) {
            container.innerHTML = '<p class="no-prompts-message">No prompt comparisons available</p>';
            return;
        }

        container.innerHTML = '';

        comparisons.forEach((comp, idx) => {
            const comparisonDiv = document.createElement('div');
            comparisonDiv.className = this.diffMode ? 'prompt-comparison diff-mode' : 'prompt-comparison';
            comparisonDiv.id = `comparison-${idx}`;

            if (this.diffMode) {
                // Diff mode - show unified diff
                const diffHtml = this.generateDiffHtml(comp.before, comp.after);
                comparisonDiv.innerHTML = `
                    <div class="prompt-box">
                        <h3 class="before">
                            <span class="material-icons">description</span>
                            ${comp.name}
                        </h3>
                        <div class="prompt-meta">
                            <strong>Purpose:</strong> ${comp.purpose}
                        </div>
                        <div class="prompt-content">${diffHtml}</div>
                    </div>
                `;
            } else {
                // Side-by-side mode
                comparisonDiv.innerHTML = `
                    <div class="prompt-box">
                        <h3 class="before">
                            <span class="material-icons">description</span>
                            Before Optimization
                        </h3>
                        <div class="prompt-meta">
                            <strong>Parameter:</strong> ${comp.name}<br>
                            <strong>Purpose:</strong> ${comp.purpose}
                        </div>
                        <div class="prompt-content">${this.escapeHtml(comp.before)}</div>
                    </div>
                    <div class="prompt-box">
                        <h3 class="after">
                            <span class="material-icons">auto_awesome</span>
                            After Optimization
                        </h3>
                        <div class="prompt-meta">
                            <strong>Parameter:</strong> ${comp.name}<br>
                            <strong>Purpose:</strong> ${comp.purpose}
                        </div>
                        <div class="prompt-content">${this.escapeHtml(comp.after)}</div>
                    </div>
                `;
            }

            container.appendChild(comparisonDiv);
        });
    }

    generateDiffHtml(before, after) {
        // Simple line-by-line diff
        const beforeLines = before.split('\n');
        const afterLines = after.split('\n');

        const maxLines = Math.max(beforeLines.length, afterLines.length);
        let diffHtml = '';

        for (let i = 0; i < maxLines; i++) {
            const beforeLine = beforeLines[i];
            const afterLine = afterLines[i];

            if (beforeLine === afterLine) {
                // Unchanged
                diffHtml += `<span class="diff-line diff-unchanged">${this.escapeHtml(beforeLine || '')}</span>\n`;
            } else {
                // Changed
                if (beforeLine !== undefined) {
                    diffHtml += `<span class="diff-line diff-removed">- ${this.escapeHtml(beforeLine)}</span>\n`;
                }
                if (afterLine !== undefined) {
                    diffHtml += `<span class="diff-line diff-added">+ ${this.escapeHtml(afterLine)}</span>\n`;
                }
            }
        }

        return diffHtml;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    toggleDiffMode() {
        this.diffMode = !this.diffMode;

        // Re-render prompt comparisons if they exist
        const container = document.getElementById('prompts-container');
        if (container && container.children.length > 0 && this.currentRunId) {
            this.loadPromptComparisons(this.currentRunId);
        }
    }

    renderHistograms(data) {
        try {
            console.log('Rendering histograms...');
        if (!data.trials || data.trials.length === 0) return;

        const valueColumns = data.metric_columns || [];
        const container = document.getElementById('histograms-viz');

        if (!container) return;

        container.innerHTML = '';

        // Create a histogram for each metric
        valueColumns.forEach((col, idx) => {
            const values = data.trials.map(t => t[col]).filter(v => v != null);

            if (values.length === 0) return;

            // Create div for this histogram
            const histDiv = document.createElement('div');
            histDiv.id = `histogram-${idx}`;
            histDiv.style.marginBottom = '20px';
            container.appendChild(histDiv);

            const trace = {
                x: values,
                type: 'histogram',
                name: col,
                marker: {
                    color: idx === 0 ? '#76b900' : idx === 1 ? '#00a3e0' : '#ff9800',
                    line: {
                        color: '#fff',
                        width: 1,
                    },
                },
                opacity: 0.7,
            };

            const layout = {
                title: `Distribution of ${col}`,
                xaxis: { title: col, color: '#fff' },
                yaxis: { title: 'Frequency', color: '#fff' },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#fff' },
                bargap: 0.1,
            };

            Plotly.newPlot(histDiv.id, [trace], layout, { responsive: true });
        });
        console.log('Histograms rendered successfully');
        } catch (error) {
            console.error('Error rendering histograms:', error);
        }
    }

    renderConvergencePlot(data) {
        try {
            console.log('Rendering convergence plot...');
        if (!data.trials || data.trials.length === 0) return;

        const valueColumns = data.metric_columns || [];
        const trials = data.trials;

        // Create traces for each metric showing how it changes over trials
        const traces = valueColumns.map((col, idx) => {
            const xValues = trials.map(t => (t.number !== null && t.number !== undefined) ? t.number : 0);
            const yValues = trials.map(t => t[col] ?? null);

            const colors = ['#76b900', '#00a3e0', '#ff9800', '#f44336', '#9c27b0'];

            return {
                x: xValues,
                y: yValues,
                mode: 'lines+markers',
                name: col,
                line: {
                    color: colors[idx % colors.length],
                    width: 2,
                },
                marker: {
                    size: 6,
                },
            };
        });

        const layout = {
            title: 'Optimization Convergence Over Time',
            xaxis: {
                title: 'Trial Number',
                color: '#fff',
            },
            yaxis: {
                title: 'Metric Value',
                color: '#fff',
            },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#fff' },
            hovermode: 'closest',
            showlegend: true,
            legend: {
                x: 1,
                y: 1,
                xanchor: 'right',
            },
        };

        Plotly.newPlot('convergence-viz', traces, layout, { responsive: true });
        console.log('Convergence plot rendered successfully');
        } catch (error) {
            console.error('Error rendering convergence plot:', error);
        }
    }

    renderCorrelationHeatmap(data) {
        try {
            console.log('Rendering correlation heatmap...');
        if (!data.trials || data.trials.length === 0) return;

        const paramColumns = data.param_columns || [];
        const valueColumns = data.metric_columns || [];

        if (paramColumns.length === 0 || valueColumns.length === 0) {
            document.getElementById('correlation-viz').innerHTML =
                '<p style="text-align: center; color: var(--text-secondary); padding: 40px;">No parameter data available for correlation analysis</p>';
            return;
        }

        // Calculate correlations between parameters and metrics
        const correlations = [];
        const labels = [];

        paramColumns.forEach(paramCol => {
            const paramValues = data.trials.map(t => t[paramCol]).filter(v => v != null && !isNaN(v));

            if (paramValues.length < 2) return; // Skip if not numeric or insufficient data

            const rowCorrelations = [];

            valueColumns.forEach(metricCol => {
                const metricValues = data.trials.map(t => t[metricCol]).filter(v => v != null);

                if (metricValues.length !== paramValues.length) {
                    rowCorrelations.push(0);
                    return;
                }

                // Calculate Pearson correlation
                const correlation = this.calculateCorrelation(paramValues, metricValues);
                rowCorrelations.push(correlation);
            });

            correlations.push(rowCorrelations);
            labels.push(paramCol.replace('params_', ''));
        });

        if (correlations.length === 0) {
            document.getElementById('correlation-viz').innerHTML =
                '<p style="text-align: center; color: var(--text-secondary); padding: 40px;">No numeric parameters for correlation analysis</p>';
            return;
        }

        const trace = {
            z: correlations,
            x: valueColumns.map(c => c.replace('values_', '')),
            y: labels,
            type: 'heatmap',
            colorscale: [
                [0, '#f44336'],
                [0.5, '#ffffff'],
                [1, '#76b900']
            ],
            zmid: 0,
            zmin: -1,
            zmax: 1,
            text: correlations.map(row => row.map(v => v.toFixed(3))),
            texttemplate: '%{text}',
            textfont: {
                color: '#000',
            },
            hovertemplate: 'Param: %{y}<br>Metric: %{x}<br>Correlation: %{z:.3f}<extra></extra>',
        };

        const layout = {
            title: 'Parameter-Metric Correlation Heatmap',
            xaxis: {
                title: 'Metrics',
                color: '#fff',
            },
            yaxis: {
                title: 'Parameters',
                color: '#fff',
            },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#fff' },
        };

        Plotly.newPlot('correlation-viz', [trace], layout, { responsive: true });
        console.log('Correlation heatmap rendered successfully');
        } catch (error) {
            console.error('Error rendering correlation heatmap:', error);
        }
    }

    calculateCorrelation(x, y) {
        const n = x.length;
        if (n === 0) return 0;

        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        if (denominator === 0) return 0;

        return numerator / denominator;
    }

    renderTrialsTable(data) {
        const tbody = document.querySelector('#trials-table tbody');
        tbody.innerHTML = '';

        data.trials.slice(0, 50).forEach(trial => {
            const row = tbody.insertRow();
            // Handle trial 0 explicitly
            const trialNumber = (trial.number !== null && trial.number !== undefined) ? trial.number : '-';
            row.insertCell(0).textContent = trialNumber;
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

    showNotification(message, type = 'info', duration = 5000) {
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
        }, duration);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new OptimizerApp();
    
    // Attach event handlers for file upload and path loading
    const configFile = document.getElementById('config-file');
    if (configFile) {
        configFile.addEventListener('change', (e) => app.handleFileUpload(e));
    }
    
    const loadPathBtn = document.getElementById('load-path-btn');
    if (loadPathBtn) {
        loadPathBtn.addEventListener('click', () => app.loadFromPath());
    }
    
    const startBtn = document.getElementById('start-btn');
    if (startBtn) {
        startBtn.addEventListener('click', () => app.startOptimization());
    }
    
    const stopBtn = document.getElementById('stop-btn');
    if (stopBtn) {
        stopBtn.addEventListener('click', () => app.stopOptimization());
    }
    
    const refreshInsightsBtn = document.getElementById('refresh-insights-btn');
    if (refreshInsightsBtn) {
        refreshInsightsBtn.addEventListener('click', () => app.refreshInsights());
    }
    
    const downloadResultsBtn = document.getElementById('download-results-btn');
    if (downloadResultsBtn) {
        downloadResultsBtn.addEventListener('click', () => app.downloadConfig());
    }

    const toggleDiffBtn = document.getElementById('toggle-diff-btn');
    if (toggleDiffBtn) {
        toggleDiffBtn.addEventListener('click', () => app.toggleDiffMode());
    }
});
