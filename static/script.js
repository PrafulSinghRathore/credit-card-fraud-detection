/**
 * Credit Card Fraud Detection - Frontend Logic
 * ==============================================
 * Handles prediction, UI updates, charts, and model info.
 */

// ============================================================
// Initialize
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    generateFeatureInputs();
    loadModelInfo();
    loadCharts();
});

// ============================================================
// Generate V1-V28 Feature Inputs
// ============================================================
function generateFeatureInputs() {
    const grid = document.getElementById('features-grid');
    let html = '';
    for (let i = 1; i <= 28; i++) {
        html += `
            <div class="v-input-group">
                <label class="v-input-label" for="input-v${i}">V${i}</label>
                <input type="number" id="input-v${i}" class="v-input-field" value="0" step="any">
            </div>
        `;
    }
    grid.innerHTML = html;
}

// ============================================================
// Toggle V1-V28 Section
// ============================================================
function toggleFeatures() {
    const grid = document.getElementById('features-grid');
    const chevron = document.getElementById('chevron');
    const isVisible = grid.style.display !== 'none';

    if (isVisible) {
        grid.style.display = 'none';
        chevron.classList.remove('open');
    } else {
        grid.style.display = 'grid';
        chevron.classList.add('open');
    }
}

// ============================================================
// Load Sample Transaction
// ============================================================
async function loadSample(type) {
    try {
        const res = await fetch(`/api/sample-transaction?type=${type}`);
        const data = await res.json();

        if (data.success) {
            const sample = data.sample;

            document.getElementById('input-time').value = sample.time;
            document.getElementById('input-amount').value = sample.amount;

            for (let i = 1; i <= 28; i++) {
                const input = document.getElementById(`input-v${i}`);
                if (input && sample[`v${i}`] !== undefined) {
                    input.value = sample[`v${i}`];
                }
            }

            // Show features if fraud sample (more interesting to see)
            if (type === 'fraud') {
                document.getElementById('features-grid').style.display = 'grid';
                document.getElementById('chevron').classList.add('open');
            }

            // Flash the button
            const btn = type === 'normal' ? document.getElementById('btn-sample-normal') : document.getElementById('btn-sample-fraud');
            btn.style.transform = 'scale(0.95)';
            setTimeout(() => btn.style.transform = '', 150);
        }
    } catch (err) {
        console.error('Failed to load sample:', err);
    }
}

// ============================================================
// Predict Fraud
// ============================================================
async function predictFraud() {
    const btn = document.getElementById('btn-predict');
    btn.classList.add('loading');
    btn.querySelector('span').textContent = 'Analyzing...';

    try {
        // Gather inputs
        const payload = {
            time: parseFloat(document.getElementById('input-time').value) || 0,
            amount: parseFloat(document.getElementById('input-amount').value) || 0,
        };

        for (let i = 1; i <= 28; i++) {
            payload[`v${i}`] = parseFloat(document.getElementById(`input-v${i}`).value) || 0;
        }

        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (data.success) {
            showResult(data);
        } else {
            alert('Prediction failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        alert('Failed to connect to server. Make sure the model is trained and the server is running.');
        console.error(err);
    } finally {
        btn.classList.remove('loading');
        btn.querySelector('span').textContent = 'Predict Fraud';
    }
}

// ============================================================
// Display Results
// ============================================================
function showResult(data) {
    // Hide default, show result
    document.getElementById('result-default').style.display = 'none';
    document.getElementById('result-content').style.display = 'block';

    // Animate gauge
    animateGauge(data.probability);

    // Verdict card
    const verdictCard = document.getElementById('verdict-card');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictText = document.getElementById('verdict-text');
    const verdictDetail = document.getElementById('verdict-detail');

    verdictCard.className = 'verdict-card';

    if (data.is_fraud) {
        verdictCard.classList.add('danger');
        verdictIcon.textContent = '🚨';
        verdictText.textContent = 'Fraudulent Transaction Detected!';
        verdictText.style.color = 'var(--accent-red)';
        verdictDetail.textContent = `High probability of fraud (${(data.probability * 100).toFixed(2)}%). Block this transaction.`;
    } else if (data.risk_level === 'medium') {
        verdictCard.classList.add('warning');
        verdictIcon.textContent = '⚠️';
        verdictText.textContent = 'Suspicious Activity';
        verdictText.style.color = 'var(--accent-amber)';
        verdictDetail.textContent = `Moderate fraud indicators detected. Manual review recommended.`;
    } else {
        verdictCard.classList.add('safe');
        verdictIcon.textContent = '✅';
        verdictText.textContent = 'Normal Transaction — Proceed Safely';
        verdictText.style.color = 'var(--accent-green)';
        verdictDetail.textContent = `Low fraud probability (${(data.probability * 100).toFixed(4)}%). Transaction appears legitimate.`;
    }

    // Metrics
    const riskColors = {
        'low': 'var(--accent-green)',
        'medium': 'var(--accent-amber)',
        'high': 'var(--accent-red)',
        'critical': 'var(--accent-red)'
    };

    const metricRisk = document.getElementById('metric-risk');
    metricRisk.textContent = data.risk_label;
    metricRisk.style.color = riskColors[data.risk_level] || 'var(--text-primary)';

    document.getElementById('metric-confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('metric-prediction').textContent = data.is_fraud ? 'FRAUD' : 'NORMAL';
    document.getElementById('metric-prediction').style.color = data.is_fraud ? 'var(--accent-red)' : 'var(--accent-green)';
    document.getElementById('metric-model').textContent = data.model_name;

    // Scroll to result on mobile
    if (window.innerWidth <= 1024) {
        document.getElementById('result-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ============================================================
// Animate Gauge
// ============================================================
function animateGauge(probability) {
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeValue = document.getElementById('gauge-value');
    const totalLength = 251.2; // Semi-circle arc length
    const targetOffset = totalLength * (1 - probability);

    // Animate the gauge fill
    let currentOffset = totalLength;
    const duration = 1200;
    const startTime = performance.now();

    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out cubic)
        const eased = 1 - Math.pow(1 - progress, 3);

        currentOffset = totalLength - (totalLength - targetOffset) * eased;
        gaugeFill.setAttribute('stroke-dashoffset', currentOffset);

        // Animate the value text
        const currentValue = probability * eased;
        gaugeValue.textContent = currentValue.toFixed(4);

        // Color the value based on probability
        if (currentValue < 0.1) {
            gaugeValue.style.color = 'var(--accent-green)';
        } else if (currentValue < 0.5) {
            gaugeValue.style.color = 'var(--accent-amber)';
        } else {
            gaugeValue.style.color = 'var(--accent-red)';
        }

        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }

    requestAnimationFrame(animate);
}

// ============================================================
// Load Model Info
// ============================================================
async function loadModelInfo() {
    try {
        const res = await fetch('/api/model-info');
        const data = await res.json();

        if (data.success) {
            // Update header
            document.getElementById('model-badge').textContent = data.model_name;
            document.getElementById('model-subtitle').textContent =
                `High-recall ${data.model_name} model — ROC-AUC: ${data.metrics.roc_auc.toFixed(4)}`;

            // Build metrics table
            const tbody = document.getElementById('metrics-table-body');
            let html = '';

            const modelColors = {
                'Logistic Regression': '#3b82f6',
                'Random Forest': '#10b981',
                'XGBoost': '#f59e0b'
            };

            for (const [name, metrics] of Object.entries(data.all_model_metrics)) {
                const isBest = name === data.model_name;
                const rowClass = isBest ? 'best-model-row' : '';
                const bestBadge = isBest ? '<span class="best-badge">★ BEST</span>' : '';
                const color = modelColors[name] || 'var(--text-primary)';

                html += `
                    <tr class="${rowClass}">
                        <td class="model-name-cell" style="color: ${color}">${name}${bestBadge}</td>
                        <td class="${getScoreClass(metrics.accuracy)}">${metrics.accuracy.toFixed(4)}</td>
                        <td class="${getScoreClass(metrics.precision)}">${metrics.precision.toFixed(4)}</td>
                        <td class="${getScoreClass(metrics.recall)}">${metrics.recall.toFixed(4)}</td>
                        <td class="${getScoreClass(metrics.f1_score)}">${metrics.f1_score.toFixed(4)}</td>
                        <td class="${getScoreClass(metrics.roc_auc)}">${metrics.roc_auc.toFixed(4)}</td>
                    </tr>
                `;
            }

            tbody.innerHTML = html;
        }
    } catch (err) {
        console.log('Model info not available yet. Train the model first.');
        document.getElementById('model-badge').textContent = 'Not trained';
        document.getElementById('metrics-table-body').innerHTML =
            '<tr><td colspan="6" class="loading-cell">⚠️ Model not trained yet. Run <code>python train_model.py</code> first.</td></tr>';
    }
}

function getScoreClass(score) {
    if (score >= 0.95) return 'score-excellent';
    if (score >= 0.85) return 'score-good';
    if (score >= 0.70) return 'score-fair';
    return 'score-poor';
}

// ============================================================
// Load Charts
// ============================================================
async function loadCharts() {
    try {
        const res = await fetch('/api/charts');
        const data = await res.json();

        if (data.success && data.charts.length > 0) {
            const grid = document.getElementById('charts-grid');
            const chartLabels = {
                'class_distribution.png': 'Class Distribution (Fraud vs Normal)',
                'amount_time_distribution.png': 'Amount & Time Distribution',
                'correlation_heatmap.png': 'Feature Correlation Heatmap',
                'feature_correlation_fraud.png': 'Feature Correlation with Fraud',
                'roc_curves.png': 'ROC Curve Comparison',
                'precision_recall_curves.png': 'Precision-Recall Curves',
                'confusion_matrices.png': 'Confusion Matrix Comparison',
                'metrics_comparison.png': 'Model Metrics Comparison',
                'feature_importance.png': 'Feature Importance (Best Model)',
            };

            let html = '';
            for (const chart of data.charts) {
                const label = chartLabels[chart] || chart.replace(/[_-]/g, ' ').replace('.png', '');
                html += `
                    <div class="chart-card" onclick="openLightbox('/static/charts/${chart}')">
                        <img src="/static/charts/${chart}" alt="${label}" loading="lazy">
                        <div class="chart-card-label">${label}</div>
                    </div>
                `;
            }

            grid.innerHTML = html;
        } else {
            document.getElementById('charts-grid').innerHTML =
                '<p class="loading-text">No charts available. Train the model to generate evaluation charts.</p>';
        }
    } catch (err) {
        document.getElementById('charts-grid').innerHTML =
            '<p class="loading-text">Charts will appear after training the model.</p>';
    }
}

// ============================================================
// Lightbox
// ============================================================
function openLightbox(src) {
    const lightbox = document.getElementById('lightbox');
    const img = document.getElementById('lightbox-img');
    img.src = src;
    lightbox.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
}

// Close lightbox on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeLightbox();
});

// ============================================================
// Batch Dataset CSV Upload & Processing
// ============================================================
let selectedBatchFile = null;

function setupDragAndDrop() {
    const dropArea = document.getElementById('upload-area');
    if (!dropArea) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

document.addEventListener('DOMContentLoaded', setupDragAndDrop);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a valid CSV file (.csv)');
        return;
    }

    selectedBatchFile = file;
    document.getElementById('selected-file-name').textContent = file.name;
    document.getElementById('selected-file-size').textContent = (file.size / 1024).toFixed(1) + ' KB';
    document.getElementById('upload-actions').style.display = 'flex';
    document.getElementById('batch-results').style.display = 'none';
}

function resetUpload() {
    selectedBatchFile = null;
    document.getElementById('csv-file-input').value = '';
    document.getElementById('upload-actions').style.display = 'none';
    document.getElementById('batch-results').style.display = 'none';
}

async function uploadAndAnalyzeCSV() {
    if (!selectedBatchFile) {
        alert('Please select a CSV file first');
        return;
    }

    const btn = document.getElementById('btn-analyze-csv');
    const originalText = btn.querySelector('span').textContent;
    btn.classList.add('loading');
    btn.querySelector('span').textContent = 'Analyzing Dataset...';

    const formData = new FormData();
    formData.append('file', selectedBatchFile);

    try {
        const res = await fetch('/api/batch-predict', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        if (data.success) {
            displayBatchResults(data);
        } else {
            alert('Analysis failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('Batch upload error:', err);
        alert('Failed to process CSV file. Ensure the server is running.');
    } finally {
        btn.classList.remove('loading');
        btn.querySelector('span').textContent = originalText;
    }
}

function displayBatchResults(data) {
    const resultsContainer = document.getElementById('batch-results');
    resultsContainer.style.display = 'block';

    document.getElementById('batch-total').textContent = data.total_analyzed.toLocaleString();
    document.getElementById('batch-fraud-count').textContent = data.fraud_count.toLocaleString();
    document.getElementById('batch-fraud-rate').textContent = data.fraud_percentage.toFixed(3) + '%';
    document.getElementById('batch-risk-amount').textContent = '$' + data.total_fraud_amount.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });

    const tbody = document.getElementById('flagged-table-body');
    if (!data.flagged_samples || data.flagged_samples.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="loading-cell" style="color: var(--accent-green) !important;">✅ No fraudulent transactions detected in this dataset!</td></tr>';
        return;
    }

    let html = '';
    data.flagged_samples.forEach(sample => {
        const badgeClass = sample.risk_level === 'Critical' ? 'badge-risk-critical' : 'badge-risk-high';
        html += `
            <tr>
                <td>#${sample.row_index}</td>
                <td>${sample.time.toFixed(1)}</td>
                <td>$${sample.amount.toFixed(2)}</td>
                <td style="color: var(--accent-red); font-weight: 700;">${(sample.probability * 100).toFixed(2)}%</td>
                <td><span class="badge-risk ${badgeClass}">${sample.risk_level}</span></td>
            </tr>
        `;
    });
    tbody.innerHTML = html;

    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

