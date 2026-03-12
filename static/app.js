// ── File Guessr - Frontend JavaScript ──

const API = '';

// ── State ──
let isSearching = false;
let pollingInterval = null;
let selectedFolderPath = null;
let attachedFile = null;

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();

    // Initial icon render
    if (window.lucide) lucide.createIcons();

    // Enter key to search
    document.getElementById('search-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doSearch();
    });

    // Drag & drop on search box
    const searchBox = document.querySelector('.search-box');
    searchBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        searchBox.classList.add('drag-over');
    });
    searchBox.addEventListener('dragleave', () => {
        searchBox.classList.remove('drag-over');
    });
    searchBox.addEventListener('drop', (e) => {
        e.preventDefault();
        searchBox.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            setAttachedFile(e.dataTransfer.files[0]);
        }
    });
});

// ── Health Check ──
async function checkHealth() {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    try {
        const res = await fetch(`${API}/api/health`);
        const data = await res.json();
        if (data.ollama_running && data.model_available) {
            dot.className = 'status-dot online';
            text.textContent = 'Service OK';
        } else {
            dot.className = 'status-dot offline';
            text.textContent = 'Ollama Error';
        }
    } catch {
        dot.className = 'status-dot offline';
        text.textContent = 'Server Offline';
    }
}

// ── File Attachment ──
function onFileAttached(input) {
    if (input.files.length > 0) {
        setAttachedFile(input.files[0]);
    }
}

function setAttachedFile(file) {
    attachedFile = file;
    const container = document.getElementById('attached-file');
    const nameEl = document.getElementById('attached-name');
    const iconEl = container.querySelector('.attached-icon');

    // Set icon based on type using data-lucide
    let iconName = 'file';
    if (file.type.startsWith('image/')) {
        iconName = 'image';
    } else if (file.name.endsWith('.pdf')) {
        iconName = 'file-text';
    } else if (file.name.endsWith('.docx') || file.name.endsWith('.doc')) {
        iconName = 'file-text';
    } else if (file.name.endsWith('.xlsx')) {
        iconName = 'table';
    }

    iconEl.setAttribute('data-lucide', iconName);
    nameEl.textContent = file.name;
    container.style.display = 'inline-flex';

    // Refresh icons
    if (window.lucide) lucide.createIcons();
}

function removeAttachment() {
    attachedFile = null;
    document.getElementById('attached-file').style.display = 'none';
    document.getElementById('search-file-input').value = '';
}

// ── Search ──
async function doSearch() {
    const input = document.getElementById('search-input');
    const query = input.value.trim();
    if (isSearching) return;

    isSearching = true;
    const btn = document.getElementById('btn-search');
    const originalBtnText = '搜尋';
    btn.disabled = true;
    btn.innerHTML = '<i data-lucide="loader-2" class="animate-spin icon-sm"></i>';
    if (window.lucide) lucide.createIcons();

    const meta = document.getElementById('search-meta');
    meta.innerHTML = '正在分析查詢...';

    const results = document.getElementById('results');
    results.innerHTML = '';

    try {
        let data;
        if (attachedFile) {
            const formData = new FormData();
            formData.append('file', attachedFile);
            formData.append('q', query);

            const res = await fetch(`${API}/api/search/multimodal`, {
                method: 'POST',
                body: formData,
            });
            data = await res.json();
        } else {
            const res = await fetch(`${API}/api/search?q=${encodeURIComponent(query)}`);
            data = await res.json();
        }

        if (attachedFile) {
            const fileTag = data.uploaded_file
                ? ` + <span class="attached-tag">分析物件: ${escapeHtml(data.uploaded_file)}</span>`
                : '';
            meta.innerHTML = `找到 ${data.total_results} 個相關對象${fileTag} — <span class="expanded-query">${escapeHtml(data.expanded_query)}</span>`;
        } else if (!query) {
            meta.innerHTML = `正在瀏覽所有檔案 — 顯示前 ${data.total_results} 個最新項目`;
        } else {
            meta.innerHTML = `找到 ${data.total_results} 個相關對象 — <span class="expanded-query">${escapeHtml(data.expanded_query)}</span>`;
        }

        if (data.results.length === 0) {
            results.innerHTML = `
                <div class="empty-state">
                    <div class="empty-illustration"><i data-lucide="search-x" size="48"></i></div>
                    <h2>未找到匹配項</h2>
                    <p>嘗試調整搜尋語句，或確認資料夾已完成索引</p>
                </div>`;
        } else {
            currentResults = []; // clear previous
            results.innerHTML = data.results.map((r, i) => renderResult(r, i, data.expanded_query, data.original_query)).join('');
        }

        // Final icon refresh after results render
        if (window.lucide) lucide.createIcons();

    } catch (err) {
        meta.innerHTML = `<span class="text-error">搜尋異常: ${err.message}</span>`;
    } finally {
        isSearching = false;
        btn.disabled = false;
        btn.innerHTML = originalBtnText;
        if (window.lucide) lucide.createIcons();
    }
}

// Store results globally so the modal can access them
let currentResults = [];

function renderResult(r, index, expandedQuery, originalQuery) {
    // Store in global array
    currentResults[index] = r;

    const iconName = getFileIcon(r.file_type);
    const isImage = isImageType(r.file_type);
    const size = formatSize(r.file_size);
    // Split by multiple possible separators (comma, semicolon, newline, bullet points)
    const keywords = (r.keywords || '').split(/[;,\n•]/).map(k => k.trim()).filter(k => k);

    // Combine original and expanded query for better highlighting coverage
    const combinedQuery = (originalQuery || '') + ' ' + (expandedQuery || '');
    const queryTerms = combinedQuery.toLowerCase().split(/\s+/).filter(t => t.length >= 2);

    const highlight = (text) => {
        if (!text) return '';
        // Sort by length descending to avoid partial matches of longer terms
        const sortedTerms = [...new Set(queryTerms)]
            .sort((a, b) => b.length - a.length)
            .map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

        if (!sortedTerms.length) return escapeHtml(text);

        const combinedRegex = new RegExp(`(${sortedTerms.join('|')})`, 'gi');
        const escapedText = escapeHtml(text);

        return escapedText.replace(combinedRegex, '<span class="highlight">$1</span>');
    };

    const highlightedName = highlight(r.file_name);
    const highlightedSummary = highlight(r.summary);

    // Tag rendering
    const displayTags = keywords.slice(0, 12).map(k => {
        const isMatch = queryTerms.some(t => k.toLowerCase().includes(t));
        return `<span class="tag ${isMatch ? 'match' : ''}">${escapeHtml(k)}</span>`;
    }).join('');

    let imagePreview = '';
    if (isImage) {
        imagePreview = `<img class="result-image-preview" 
            src="${API}/api/file/preview?path=${encodeURIComponent(r.file_path)}" 
            alt="${escapeHtml(r.file_name)}"
            loading="lazy">`;
    }

    return `
        <div class="result-card" style="animation-delay: ${index * 0.05}s">
            ${imagePreview}
            <div class="result-header" onclick="copyPath('${escapeAttr(r.file_path)}')">
                <div class="result-icon-container">
                    <i data-lucide="${iconName}" class="icon-md"></i>
                </div>
                <div class="result-title">
                    <h3>${highlightedName}</h3>
                    <div class="result-path">${escapeHtml(r.file_path)}</div>
                </div>
                <button class="btn-icon-ghost" onclick="event.stopPropagation(); showFileDetails(${index})" title="查看完整資訊" style="margin-left: auto;">
                    <i data-lucide="info" class="icon-sm"></i>
                </button>
            </div>
            <div class="result-summary">${highlightedSummary || '無詳細描述'}</div>
            <div class="result-tags">
                ${displayTags}
            </div>
            <div class="result-footer">
                <span>類型: ${r.file_type || '未知'}</span>
                <span>大小: ${size}</span>
            </div>
        </div>`;
}

// ── Folder Picker ──
async function browseFolderPath() {
    const btn = document.getElementById('btn-browse');
    const pathDisplay = document.getElementById('selected-folder-path');
    const indexBtn = document.getElementById('btn-index');

    btn.disabled = true;
    btn.textContent = '開啟中...';

    try {
        const res = await fetch(`${API}/api/browse`);
        const data = await res.json();
        if (data.path) {
            selectedFolderPath = data.path;
            pathDisplay.textContent = data.path;
            pathDisplay.style.color = 'var(--text-main)';
            indexBtn.disabled = false;
        }
    } catch (err) {
        alert('無法開啟選擇器: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '選擇資料夾';
    }
}

// ── Indexing ──
async function startIndex() {
    if (!selectedFolderPath) return;
    const btn = document.getElementById('btn-index');
    btn.disabled = true;
    try {
        const res = await fetch(`${API}/api/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: selectedFolderPath }),
        });
        document.getElementById('progress-card').style.display = 'block';
        startPolling();
    } catch (err) {
        btn.disabled = false;
    }
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(updateProgress, 1000);
}

async function updateProgress() {
    try {
        const res = await fetch(`${API}/api/index/status`);
        const data = await res.json();
        const pct = data.total_files > 0 ? Math.round((data.processed_files / data.total_files) * 100) : 0;
        document.getElementById('progress-bar').style.width = `${pct}%`;
        document.getElementById('progress-text').textContent = `${data.processed_files} / ${data.total_files} (${pct}%)`;
        document.getElementById('progress-file').textContent = data.current_file || '-';
        document.getElementById('progress-time').textContent = `耗時: ${data.elapsed_seconds}s`;

        const errEl = document.getElementById('progress-errors');
        if (data.errors && data.errors.length > 0) {
            errEl.textContent = `⚠ 跳過 ${data.errors.length} 個`;
            errEl.title = data.errors.map(e => typeof e === 'string' ? e : JSON.stringify(e)).join('\n');
        } else {
            errEl.textContent = '';
            errEl.title = '';
        }

        if (!data.is_indexing && data.total_files > 0) {
            clearInterval(pollingInterval);
            pollingInterval = null;
            document.getElementById('btn-index').disabled = false;
            document.getElementById('progress-file').textContent =
                data.errors && data.errors.length > 0
                    ? `完成！共跳過 ${data.errors.length} 個檔案（超時或錯誤）`
                    : '全部完成！';
            loadStats();
            loadWatchedFolders();
        }
    } catch { }
}

// ── Watched Folders ──
async function loadWatchedFolders() {
    const container = document.getElementById('watched-folders-list');
    try {
        const res = await fetch(`${API}/api/folders`);
        const data = await res.json();
        if (!data.folders || data.folders.length === 0) {
            container.innerHTML = '<p class="text-muted">尚未新增資料夾</p>';
            return;
        }
        container.innerHTML = data.folders.map(f => `
            <div class="watched-item">
                <span class="watched-name" title="${escapeAttr(f)}">${escapeHtml(f)}</span>
                <button class="btn-icon-danger" onclick="removeFolder('${escapeAttr(f)}')">
                    <i data-lucide="trash-2" class="icon-xs"></i>
                </button>
            </div>
        `).join('');
        if (window.lucide) lucide.createIcons();
    } catch {
        container.innerHTML = '<p class="text-error">載入失敗</p>';
    }
}

async function removeFolder(path) {
    if (!confirm(`確定要移除此資料夾的監控與索引嗎？\n${path}`)) return;
    try {
        await fetch(`${API}/api/folders/remove`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: path }),
        });
        loadWatchedFolders();
        loadStats();
    } catch { }
}

// ── Stats ──
async function loadStats() {
    const container = document.getElementById('stats-content');
    try {
        const res = await fetch(`${API}/api/stats`);
        const data = await res.json();
        const esStatus = data.search_engine === 'elasticsearch' ? 'Elasticsearch' : 'SQLite';
        container.innerHTML = `
            <div style="margin-bottom:12px; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase;">引擎: ${esStatus}</div>
            <div class="stats-grid">
                <div class="stat-box">
                    <span class="stat-val">${data.total_files}</span>
                    <span class="stat-lbl">檔案總額</span>
                </div>
                <div class="stat-box">
                    <span class="stat-val">${Object.keys(data.by_type).length}</span>
                    <span class="stat-lbl">格式類別</span>
                </div>
            </div>`;
    } catch {
        container.innerHTML = '<p class="text-error">無法讀取統計</p>';
    }
}




// ── UI UI UI ──
function showPanel(name) {
    document.getElementById('panel-overlay').classList.add('active');
    document.getElementById(`panel-${name}`).classList.add('active');
    if (name === 'settings') {
        loadStats();
        loadWatchedFolders();
        loadLLMSettings();
    }
}

function hidePanel() {
    document.getElementById('panel-overlay').classList.remove('active');
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        hidePanel();
        closeDetailsModal();
    }
});

// ── Details Modal ──
function showFileDetails(index) {
    const r = currentResults[index];
    if (!r) return;

    document.getElementById('details-title').textContent = r.file_name;
    document.getElementById('details-path').textContent = r.file_path;
    document.getElementById('details-summary').textContent = r.summary || '無總結資訊';

    const keywords = (r.keywords || '').split(',').map(k => k.trim()).filter(k => k);
    document.getElementById('details-keyword-count').textContent = keywords.length;

    const tagsHtml = keywords.map(k => `<span class="tag">${escapeHtml(k)}</span>`).join('');
    document.getElementById('details-keywords').innerHTML = tagsHtml || '<span class="text-muted">無關鍵字</span>';

    document.getElementById('details-overlay').classList.add('active');
    document.getElementById('panel-details').classList.add('active');
    if (window.lucide) lucide.createIcons();
}

function closeDetailsModal() {
    document.getElementById('details-overlay').classList.remove('active');
    document.getElementById('panel-details').classList.remove('active');
}

// ── LLM Settings ──
async function loadLLMSettings() {
    const select = document.getElementById('select-llm-model');
    try {
        const res = await fetch(`${API}/api/llm/models`);
        const data = await res.json();

        if (data.available_models && data.available_models.length > 0) {
            select.innerHTML = data.available_models.map(m =>
                `<option value="${escapeAttr(m)}" ${m === data.selected_model ? 'selected' : ''}>${escapeHtml(m)}</option>`
            ).join('');

            if (!data.model_available && data.selected_model) {
                // Warning if selected model is not in available list
                const opt = document.createElement('option');
                opt.value = data.selected_model;
                opt.selected = true;
                opt.textContent = `${data.selected_model} (未安裝!)`;
                opt.style.color = '#ef4444';
                select.prepend(opt);
            }
        } else {
            // No models found but Ollama might be running
            select.innerHTML = `<option value="">請先在 Ollama 中下載模型</option>`;
            if (data.selected_model) {
                const opt = document.createElement('option');
                opt.value = data.selected_model;
                opt.selected = true;
                opt.textContent = `${data.selected_model} (未安裝!)`;
                opt.style.color = '#ef4444';
                select.prepend(opt);
            }
        }
    } catch (err) {
        select.innerHTML = '<option value="">無法載入模型清單</option>';
    }
}

async function updateLLMModel() {
    const select = document.getElementById('select-llm-model');
    const model = select.value;
    if (!model) return;

    try {
        const res = await fetch(`${API}/api/llm/model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: model }),
        });
        const data = await res.json();
        console.log('[LLM] Model updated:', data.message);

        // Show a brief success highlight
        select.style.borderColor = '#10b981';
        setTimeout(() => select.style.borderColor = '', 1500);
    } catch (err) {
        alert('更新模型失敗: ' + err.message);
    }
}

// ── Helpers ──
function getFileIcon(ext) {
    const map = {
        '.pdf': 'file-text', '.docx': 'file-text', '.doc': 'file-text',
        '.xlsx': 'table', '.xls': 'table', '.pptx': 'presentation',
        '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.webp': 'image',
        '.py': 'code', '.js': 'code', '.html': 'code', '.css': 'code',
        '.txt': 'file-text', '.md': 'file-text', '.json': 'braces',
        '.zip': 'archive', '.mp3': 'music', '.mp4': 'video'
    };
    return map[ext] || 'file';
}

function isImageType(ext) {
    return ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'].includes(ext);
}

function formatSize(bytes) {
    if (!bytes) return '-';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0, size = bytes;
    while (size >= 1024 && i < units.length - 1) { size /= 1024; i++; }
    return `${size.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    return str.replace(/'/g, "\\'").replace(/\\/g, "\\\\");
}

function copyPath(path) {
    navigator.clipboard.writeText(path).then(() => {
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; bottom: 32px; left: 50%; transform: translateX(-50%);
            background: var(--zinc-800); color: var(--text-main);
            padding: 8px 16px; border-radius: 8px; font-size: 0.8rem;
            border: 1px solid var(--border); box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            z-index: 2000; animation: fadeIn 0.2s ease;
        `;
        toast.textContent = `路徑已複製`;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    });
}
async function showAiLogs() {
    try {
        const res = await fetch(`${API}/api/llm/logs`);
        const data = await res.json();
        const logContent = data.logs || '（目前沒有日誌記錄）';

        // Remove any existing overlay
        const existing = document.getElementById('log-modal-overlay');
        if (existing) existing.remove();

        // Build modal
        const overlay = document.createElement('div');
        overlay.id = 'log-modal-overlay';
        overlay.className = 'log-modal-overlay';
        overlay.innerHTML = `
            <div class="log-modal" role="dialog" aria-modal="true">
                <div class="log-modal-header">
                    <div class="log-modal-title">
                        <i data-lucide="terminal" class="icon-sm"></i>
                        <h3>AI 引擎日誌 <span style="font-weight:400; color:var(--text-dim); font-size:0.8rem;">（最後 100 行）</span></h3>
                    </div>
                    <button class="btn-close" id="log-close-btn" title="關閉">
                        <i data-lucide="x"></i>
                    </button>
                </div>
                <div class="log-modal-body">
                    <pre class="log-pre" id="log-pre-content"></pre>
                </div>
                <div class="log-modal-footer">
                    <button class="btn btn-ghost btn-xs" id="log-refresh-btn">
                        <i data-lucide="refresh-cw" class="icon-xs"></i>
                        重新整理
                    </button>
                    <button class="btn btn-ghost" id="log-close-btn-2">關閉</button>
                </div>
            </div>`;

        document.body.appendChild(overlay);
        if (window.lucide) lucide.createIcons();

        // Set content & scroll to bottom
        const pre = document.getElementById('log-pre-content');
        pre.textContent = logContent;
        pre.scrollTop = pre.scrollHeight;

        // Event handlers
        const closeLog = () => overlay.remove();
        document.getElementById('log-close-btn').addEventListener('click', closeLog);
        document.getElementById('log-close-btn-2').addEventListener('click', closeLog);
        document.getElementById('log-refresh-btn').addEventListener('click', async () => {
            const r2 = await fetch(`${API}/api/llm/logs`);
            const d2 = await r2.json();
            pre.textContent = d2.logs || '（沒有日誌記錄）';
            pre.scrollTop = pre.scrollHeight;
        });
        // Click outside to close
        overlay.addEventListener('click', (e) => { if (e.target === overlay) closeLog(); });
        // Escape key to close
        const escHandler = (e) => { if (e.key === 'Escape') { closeLog(); document.removeEventListener('keydown', escHandler); } };
        document.addEventListener('keydown', escHandler);

    } catch (e) {
        alert(`讀取日誌失敗: ${e.message}`);
    }
}

async function clearIndex() {
    if (!confirm('確定要清除所有索引資料嗎？\n此操作無法復原，需要重新索引所有資料夾。')) return;
    try {
        const res = await fetch(`${API}/api/clear`, { method: 'POST' });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || '清除失敗 (HTTP ' + res.status + ')');
        }

        loadStats();
        loadWatchedFolders();
        // Show success toast
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; bottom: 32px; left: 50%; transform: translateX(-50%);
            background: var(--zinc-800); color: var(--text-main);
            padding: 10px 18px; border-radius: 8px; font-size: 0.8rem;
            border: 1px solid var(--border); box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            z-index: 2000; animation: fadeIn 0.2s ease; display:flex; gap:8px; align-items:center;
        `;
        toast.innerHTML = `<i data-lucide="check-circle" style="width:14px;height:14px;color:#10b981"></i> 索引已清除`;
        document.body.appendChild(toast);
        if (window.lucide) lucide.createIcons();
        setTimeout(() => toast.remove(), 2500);
    } catch (err) {
        alert('清除失敗: ' + err.message);
    }
}

