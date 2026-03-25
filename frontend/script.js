document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const promptInput = document.getElementById('prompt-input'),
        processPromptBtn = document.getElementById('process-prompt-btn'),
        selectedTiffNameEl = document.getElementById('selected-tiff-name'),
        classListContainer = document.getElementById('class-list'),
        newClassInput = document.getElementById('new-class-input'),
        addClassBtn = document.getElementById('add-class-btn'),
        runPipelineBtn = document.getElementById('run-pipeline-btn'),
        generateCostmapBtn = document.getElementById('generate-costmap-btn'),
        // [NEW]
        downloadPlanBtn = document.getElementById('download-plan-btn'),
        costmapDeviceSelect = document.getElementById('costmap-device'),
        saveMasksBtn = document.getElementById('save-masks-btn'),
        saveCostmapImgBtn = document.getElementById('save-costmap-img-btn'),
        browseCostmapsBtn = document.getElementById('browse-costmaps-btn'),
        savedGoalsList = document.getElementById('saved-goals-list'),
        saveCurrentGoalBtn = document.getElementById('save-current-goal-btn'),
        loadServerMasksBtn = document.getElementById('load-server-masks-btn'),
        updateParamsBtn = document.getElementById('update-params-btn'),
        confirmClassesBtn = document.getElementById('confirm-classes-btn'),
        tiffPreviewContainer = document.getElementById('tiff-preview-container'),
        resultsArea = document.getElementById('results-area'),
        resultsMasksContainer = document.getElementById('results-masks-container'),
        costmapDisplayArea = document.getElementById('costmap-display-area'),
        costmapOverlayBaseImg = document.getElementById('costmap-overlay-base'),
        costmapOverlayImg = document.getElementById('costmap-overlay-img'),
        costmapOverlaySlider = document.getElementById('costmap-overlay-slider'),
        resultsPlaceholder = document.getElementById('results-placeholder'),
        loader = document.getElementById('loader'),
        cancelPipelineBtn = document.getElementById('cancel-pipeline-btn');

    const paramInputs = {
        areal_threshold: document.getElementById('areal-threshold'),
        linear_threshold: document.getElementById('linear-threshold'),
        refiner_areal_threshold: document.getElementById('refiner-areal-threshold'),
        refiner_linear_threshold: document.getElementById('refiner-linear-threshold'),
        model_name: document.getElementById('model-name'),
        mask_refiner_name: document.getElementById('mask-refiner-name'),
        sam_model: document.getElementById('sam-model'),
        cmap_device: document.getElementById('cmap-device'),
        sam_device: document.getElementById('sam-device'),
        semseg_device: document.getElementById('semseg-device'),
        semseg_tile_size: document.getElementById('semseg-tile-size'),
        semseg_stride: document.getElementById('semseg-stride'),
        refiner_tile_size: document.getElementById('refiner-tile-size'),
        refiner_stride: document.getElementById('refiner-stride'),
        semseg_combine_method: document.getElementById('semseg-combine-method'),
        refiner_combine_method: document.getElementById('refiner-combine-method'),
    };

    const gpuModal = document.getElementById('gpu-modal'),
        showGpuModalBtn = document.getElementById('show-gpu-modal-btn'),
        closeGpuModalBtn = document.getElementById('close-gpu-modal-btn'),
        gpuStatusContainer = document.getElementById('gpu-status-container');

    const consoleModal = document.getElementById('console-modal'),
        showConsoleBtn = document.getElementById('show-console-btn'),
        closeConsoleBtn = document.getElementById('close-console-btn'),
        consoleOutput = document.getElementById('console-output');

    const costmapModal = document.getElementById('costmap-modal'),
        showCostmapBtn = document.getElementById('show-costmap-btn'),
        closeCostmapBtn = document.getElementById('close-costmap-btn'),
        editCostmapBtn = document.getElementById('edit-costmap-btn'),
        saveCostmapBtn = document.getElementById('save-costmap-btn'),
        restoreCostmapBtn = document.getElementById('restore-costmap-btn'),
        defaultCostmapCode = document.getElementById('default-costmap-code'),
        generatedCostmapCode = document.getElementById('generated-costmap-code');

    const serverMasksModal = document.getElementById('server-masks-modal'),
        closeServerMasksBtn = document.getElementById('close-server-masks-btn'),
        serverMasksList = document.getElementById('server-masks-list'),
        loadSelectedMasksBtn = document.getElementById('load-selected-masks-btn'),
        browserUpBtn = document.getElementById('browser-up-btn'),
        browserBreadcrumb = document.getElementById('browser-breadcrumb'),
        browserModalTitle = document.getElementById('browser-modal-title'),
        browserSearchArea = document.getElementById('browser-search-area'),
        browserSearchInput = document.getElementById('browser-search-input'),
        browserSearchResults = document.getElementById('browser-search-results'),
        browseTiffsBtn = document.getElementById('browse-tiffs-btn');

    const worldMapModal = document.getElementById('world-map-modal'),
        showWorldMapBtn = document.getElementById('show-world-map-btn'),
        closeWorldMapBtn = document.getElementById('close-world-map-btn'),
        saveMapAreaBtn = document.getElementById('save-map-area-btn'),
        snipFilenameInput = document.getElementById('snip-filename'),
        snipTileSizeInput = document.getElementById('snip-tile-size'),
        snipResolutionInput = document.getElementById('snip-resolution'),
        rasterizeProgressArea = document.getElementById('rasterize-progress-area'),
        rasterizeProgressBar = document.getElementById('rasterize-progress-bar'),
        rasterizeStatusText = document.getElementById('rasterize-status-text');

    const imageViewerModal = document.getElementById('image-viewer-modal'),
        imageViewerContent = document.getElementById('image-viewer-content'),
        closeImageViewerBtn = document.getElementById('close-image-viewer-btn');

    const classDetailModal = document.getElementById('class-detail-modal'),
        closeClassDetailBtn = document.getElementById('close-class-detail-btn'),
        editMaskBtn = document.getElementById('edit-mask-btn'),
        classDetailTitle = document.getElementById('class-detail-title'),
        detailRgbImg = document.getElementById('detail-rgb'),
        detailSemsegOverlayBase = document.getElementById('detail-semseg-overlay-base'),
        detailSemsegOverlayMask = document.getElementById('detail-semseg-overlay-mask'),
        semsegSlider = document.getElementById('semseg-slider'),
        detailRefinedOverlayBase = document.getElementById('detail-refined-overlay-base'),
        detailRefinedOverlayMask = document.getElementById('detail-refined-overlay-mask'),
        refinedSlider = document.getElementById('refined-slider');

    const pixelEditorModal = document.getElementById('pixel-editor-modal'),
        pixelEditorTitle = document.getElementById('pixel-editor-title'),
        pixelEditorSubtitle = document.getElementById('pixel-editor-subtitle'),
        closePixelEditorBtn = document.getElementById('close-pixel-editor-btn'),
        pixelEditorCanvas = document.getElementById('pixel-editor-canvas'),
        pixelSelectionCount = document.getElementById('pixel-selection-count'),
        pixelToolButtons = document.getElementById('pixel-tool-buttons'),
        pixelBrushSizeInput = document.getElementById('pixel-brush-size-input'),
        pixelBrushSizeValue = document.getElementById('pixel-brush-size-value'),
        pixelToolHint = document.getElementById('pixel-tool-hint'),
        pixelClosePolygonBtn = document.getElementById('pixel-close-polygon-btn'),
        pixelZoomLabel = document.getElementById('pixel-zoom-label'),
        pixelZoomOutBtn = document.getElementById('pixel-zoom-out-btn'),
        pixelZoomResetBtn = document.getElementById('pixel-zoom-reset-btn'),
        pixelZoomInBtn = document.getElementById('pixel-zoom-in-btn'),
        pixelClearSelectionBtn = document.getElementById('pixel-clear-selection-btn'),
        pixelUndoSelectionBtn = document.getElementById('pixel-undo-selection-btn'),
        pixelValueInput = document.getElementById('pixel-value-input'),
        pixelApplyValueBtn = document.getElementById('pixel-apply-value-btn'),
        pixelSaveBtn = document.getElementById('pixel-save-btn'),
        pixelEditorStatus = document.getElementById('pixel-editor-status'),
        editCostmapPixelsBtn = document.getElementById('edit-costmap-pixels-btn');

    const planOverMapBtn = document.getElementById('plan-over-map-btn'),
        downloadCostmapBtn = document.getElementById('download-costmap-btn'),
        plannerModal = document.getElementById('planner-modal'),
        closePlannerBtn = document.getElementById('close-planner-btn'),
        plannerImgBg = document.getElementById('planner-img-bg'),
        plannerCanvas = document.getElementById('planner-canvas'),
        plannerImgOverlay = document.getElementById('planner-img-overlay'),
        plannerOverlaySlider = document.getElementById('planner-overlay-slider'),
        selectStartBtn = document.getElementById('select-start-btn'),
        selectEndBtn = document.getElementById('select-end-btn'),
        planPathBtn = document.getElementById('plan-path-btn'),
        clearPlanBtn = document.getElementById('clear-plan-btn'),
        plannerStatus = document.getElementById('planner-status'),
        plannerCoords = document.getElementById('planner-coords');

    const downloadZipNameInput = document.getElementById('download-zip-name');
    const clearTempDownloadsBtn = document.getElementById('clear-temp-downloads-btn');

    const API_BASE_URL = '';
    let currentClasses = [], worldMap, drawnItems, capturedBounds, progressInterval, gpuPollInterval, consolePollInterval;
    let panzoomState = { scale: 1, translateX: 0, translateY: 0, isPanning: false, startX: 0, startY: 0 };
    let currentResultData = null;
    let pipelinePollInterval = null;
    let currentTaskId = null;
    let currentCostmapUrl = null;
    let currentCostmapOverlayUrl = null;
    let currentCostmapOverlayFallbackUrl = null;
    let plannerState = {};
    let plannerResizeObserver = null;
    let selectedRunId = null;
    let lastKnownCoords = null;
    let currentTiffFolder = null;
    let browserCurrentPath = '';  // For file browser navigation
    let browserSelectedPath = null;  // Selected mask folder path
    let browserMode = 'masks';  // 'tiff' or 'masks'
    let searchDebounceTimer = null;
    let selectedTiffName = null;  // Currently selected TIFF file/folder name
    const assetVersions = new Map();
    const pixelEditorState = {
        imageType: null,
        imageUrl: null,
        colorUrl: null,
        className: null,
        imageData: null,
        width: 0,
        height: 0,
        zoom: 1,
        selectedPixels: new Set(),
        permanentPixels: new Set(),
        selectionHistory: [],
        currentStroke: null,
        isDrawing: false,
        lastPoint: null,
        isDirty: false,
        tool: 'single',
        brushSize: 1,
        polygonPoints: [],
        hoverPoint: null
    };

    // --- Helper Functions ---
    function rgbToHex(rgbString) {
        if (!rgbString) return '#ffffff';
        const match = rgbString.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (!match) return '#ffffff';
        const r = parseInt(match[1]).toString(16).padStart(2, '0');
        const g = parseInt(match[2]).toString(16).padStart(2, '0');
        const b = parseInt(match[3]).toString(16).padStart(2, '0');
        return `#${r}${g}${b}`;
    }

    function hexToRgb(hexString) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hexString);
        return result ? `rgb(${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)})` : 'rgb(0, 0, 0)';
    }

    function getRandomRgbColor() {
        const r = Math.floor(Math.random() * 256);
        const g = Math.floor(Math.random() * 256);
        const b = Math.floor(Math.random() * 256);
        return `rgb(${r}, ${g}, ${b})`;
    }

    function getCurrentTiffName() {
        return currentTiffFolder || selectedTiffName || '';
    }

    function getPreviewUrlForTiff(tiffName) {
        if (!tiffName) return '';
        return `${API_BASE_URL}/results/${tiffName}/preview.png`;
    }

    function deriveOverlayAndBwUrls(url) {
        // url is a /results/... path (usually .png)
        if (!url) return { overlayUrl: null, overlayFallbackUrl: null, bwUrl: null };
        const overlayFallbackUrl = url;
        let bwUrl = url;
        if (url.endsWith('costmap.png')) {
            bwUrl = url.replace(/costmap\.png$/, 'costmap_bw.png');
        } else if (url.endsWith('costmap_bw.png')) {
            bwUrl = url;
        } else if (url.endsWith('_bw.png')) {
            bwUrl = url;
        } else if (url.endsWith('.png')) {
            // Assume it's an overlay; try sibling *_bw.png for planning
            bwUrl = url.replace(/\.png$/, '_bw.png');
        }
        // Prefer BW image for overlays by default; fall back to whatever was clicked.
        return { overlayUrl: bwUrl, overlayFallbackUrl, bwUrl };
    }

    function withAssetVersion(url) {
        if (!url || url.startsWith('data:')) return url;
        const version = assetVersions.get(url);
        const suffix = version ? `?v=${version}` : '';
        return `${API_BASE_URL}${url}${suffix}`;
    }

    function bumpAssetVersion(url) {
        if (!url || url.startsWith('data:')) return;
        assetVersions.set(url, Date.now());
    }

    function getPixelKey(x, y) {
        return `${x},${y}`;
    }

    function parsePixelKey(key) {
        const [x, y] = key.split(',').map(Number);
        return { x, y };
    }

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
            img.src = src;
        });
    }

    function setImgOpacityFromSlider(imgEl, sliderEl) {
        if (!imgEl || !sliderEl) return;
        imgEl.style.opacity = (sliderEl.value / 100).toString();
    }

    function updateCostmapOverlayUI() {
        if (!costmapOverlayBaseImg || !costmapOverlayImg) return;

        const tiffName = getCurrentTiffName();
        const previewUrl = getPreviewUrlForTiff(tiffName);
        const rgbPreviewImg = tiffPreviewContainer.querySelector('img');
        costmapOverlayBaseImg.src = (rgbPreviewImg ? rgbPreviewImg.src : '') || previewUrl;

        if (currentCostmapOverlayUrl) {
            const primary = withAssetVersion(currentCostmapOverlayUrl);
            const fallback = currentCostmapOverlayFallbackUrl ? withAssetVersion(currentCostmapOverlayFallbackUrl) : '';
            costmapOverlayImg.onerror = () => {
                if (fallback && costmapOverlayImg.src !== fallback) {
                    costmapOverlayImg.onerror = null;
                    costmapOverlayImg.src = fallback;
                }
            };
            costmapOverlayImg.src = primary;
            costmapOverlayImg.classList.remove('hidden');
        } else {
            costmapOverlayImg.src = '';
            costmapOverlayImg.classList.add('hidden');
        }

        if (costmapOverlaySlider) {
            setImgOpacityFromSlider(costmapOverlayImg, costmapOverlaySlider);
        }

        // If planner is open, also update its overlay image.
        if (plannerModal && !plannerModal.classList.contains('hidden') && plannerImgOverlay) {
            if (currentCostmapOverlayUrl) {
                const primary = withAssetVersion(currentCostmapOverlayUrl);
                const fallback = currentCostmapOverlayFallbackUrl ? withAssetVersion(currentCostmapOverlayFallbackUrl) : '';
                plannerImgOverlay.onerror = () => {
                    if (fallback && plannerImgOverlay.src !== fallback) {
                        plannerImgOverlay.onerror = null;
                        plannerImgOverlay.src = fallback;
                    }
                };
                plannerImgOverlay.src = primary;
                plannerImgOverlay.classList.remove('hidden');
            } else {
                plannerImgOverlay.src = '';
                plannerImgOverlay.classList.add('hidden');
            }
            if (plannerOverlaySlider) {
                setImgOpacityFromSlider(plannerImgOverlay, plannerOverlaySlider);
            }
        }
    }

    // --- Core Functions ---
    async function fetchDefaultConfig() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/get-default-config`);
            const data = await response.json();
            if (!response.ok) throw new Error("Failed to fetch default config.");
            currentClasses = data.classes;
            renderClasses();
        } catch (error) {
            console.error("Error fetching default config:", error);
            classListContainer.innerHTML = `<p class="text-red-500">Could not load default classes.</p>`;
        }
    }

    async function fetchTiffFiles() {
        // Just ensure results folder exists - TIFFs are now selected via Browse
        try {
            await fetch(`${API_BASE_URL}/api/get-tiff-files`);
        } catch (error) { console.error("Failed to fetch TIFF files:", error); }
    }

    function updateDeviceDropdowns(gpuCount) {
        const dropdowns = [paramInputs.cmap_device, paramInputs.sam_device, paramInputs.semseg_device];
        dropdowns.forEach(dd => {
            const currentVal = dd.value;
            dd.innerHTML = '';
            for (let i = 0; i < gpuCount; i++) dd.add(new Option(`cuda:${i}`, `cuda:${i}`));
            if (dd.querySelector(`[value="${currentVal}"]`)) {
                dd.value = currentVal;
            }
        });
    }

    function renderClasses() {
        classListContainer.innerHTML = '';
        if (currentClasses.length === 0) {
            classListContainer.innerHTML = `<p class="text-gray-500 italic">No classes defined.</p>`;
        } else {
            currentClasses.forEach((classObj, index) => {
                const classItem = document.createElement('div');
                classItem.className = 'class-item p-3 bg-gray-100 rounded-lg border w-full';
                classItem.dataset.index = index;

                let thresholdValue, isReadOnly = true;
                if (classObj.type === 'Areal') thresholdValue = paramInputs.areal_threshold.value;
                else if (classObj.type === 'Linear') thresholdValue = paramInputs.linear_threshold.value;
                else { thresholdValue = classObj.threshold; isReadOnly = false; }

                const hexColor = rgbToHex(classObj.color);

                classItem.innerHTML = `
                    <div class="grid grid-cols-[auto,1fr,auto] items-center gap-3">
                        <input type="color" value="${hexColor}" class="class-color-input">
                        <span class="font-semibold text-gray-800">${classObj.name}</span>
                        <button class="remove-class-btn text-red-500 hover:text-red-700 font-bold text-xl">&times;</button>
                    </div>
                    <div class="grid grid-cols-2 gap-3 mt-2">
                        <select class="class-type-select w-full p-1.5 border border-gray-300 rounded-md text-sm">
                            <option value="Areal" ${classObj.type === 'Areal' ? 'selected' : ''}>Areal</option>
                            <option value="Linear" ${classObj.type === 'Linear' ? 'selected' : ''}>Linear</option>
                            <option value="Custom" ${classObj.type === 'Custom' ? 'selected' : ''}>Custom</option>
                        </select>
                        <input type="number" step="0.1" value="${thresholdValue}" ${isReadOnly ? 'readonly' : ''} class="class-threshold-input w-full p-1.5 border border-gray-300 rounded-md text-sm ${isReadOnly ? 'bg-gray-200' : 'bg-white'}">
                    </div>
                `;
                classListContainer.appendChild(classItem);
            });
        }
        updateRunButtonState();
    }

    function updateRunButtonState() {
        const hasTiff = !!selectedTiffName;
        const hasClasses = currentClasses.length > 0;
        const hasResults = !!currentResultData;

        runPipelineBtn.disabled = !(hasTiff && hasClasses);
        generateCostmapBtn.disabled = !hasResults;
    }

    function displayResults(data) {
        currentResultData = data;
        currentTiffFolder = data.tiff_folder || null;
        resultsArea.classList.remove('hidden');
        resultsPlaceholder.classList.add('hidden');
        costmapDisplayArea.classList.add('hidden');
        if (costmapOverlayBaseImg) costmapOverlayBaseImg.src = '';
        if (costmapOverlayImg) costmapOverlayImg.src = '';
        resultsMasksContainer.innerHTML = '';

        // Enable Save Masks button if we have a tiff_folder
        if (currentTiffFolder) {
            saveMasksBtn.disabled = false;
        }

        const masksToDisplay = data.refined_masks || data.local_masks;

        if (!masksToDisplay || Object.keys(masksToDisplay).length === 0) {
            resultsMasksContainer.innerHTML = '<p class="text-gray-400 col-span-full text-center">No masks generated.</p>';
            generateCostmapBtn.disabled = true;
            return;
        }

        for (const [className, maskUrl] of Object.entries(masksToDisplay)) {
            const maskCard = document.createElement('div');
            maskCard.className = 'bg-gray-50 rounded-lg shadow-md overflow-hidden border border-gray-200 cursor-pointer hover:shadow-xl hover:border-blue-500 transition-all';
            maskCard.dataset.className = className;
            const finalUrl = maskUrl.startsWith('data:') ? maskUrl : withAssetVersion(maskUrl);
            maskCard.innerHTML = `
                <img src="${finalUrl}" alt="Mask for ${className}" class="w-full h-48 object-cover pointer-events-none">
                <div class="p-4"><h3 class="font-bold text-lg text-center pointer-events-none">${className}</h3></div>`;
            resultsMasksContainer.appendChild(maskCard);
        }
        generateCostmapBtn.disabled = false;
    }

    async function handleProcessPrompt() {
        const prompt = promptInput.value.trim();
        if (!prompt) return alert("Please enter a prompt first.");

        processPromptBtn.textContent = 'Processing...';
        processPromptBtn.disabled = true;
        try {
            const response = await fetch(`${API_BASE_URL}/api/process-prompt`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to process prompt.');
            currentClasses = data.classes;
            renderClasses();

            // [ADD THIS FIX]
            // After rendering the new classes, check if the code editor is open.
            // If it is, re-apply its highlighting to prevent it from disappearing.
            if (!costmapModal.classList.contains('hidden')) {
                hljs.highlightElement(defaultCostmapCode);
                hljs.highlightElement(generatedCostmapCode);
                hljs.lineNumbersBlock(defaultCostmapCode);
                hljs.lineNumbersBlock(generatedCostmapCode);
            }
            // [END OF FIX]

        } catch (error) {
            console.error("Prompt processing error:", error);
            alert(`Error: ${error.message}`);
        } finally {
            processPromptBtn.textContent = 'Run LLM';
            processPromptBtn.disabled = false;
        }
    }

    function handleAddClass() {
        const newClassName = newClassInput.value.trim();
        if (newClassName && !currentClasses.some(c => c.name === newClassName)) {
            currentClasses.push({ name: newClassName, type: 'Areal', threshold: paramInputs.areal_threshold.value, color: getRandomRgbColor() });
            renderClasses();
            newClassInput.value = '';
        } else if (currentClasses.some(c => c.name === newClassName)) {
            alert(`Class "${newClassName}" already exists.`);
        }
    }

    function handleClassListChange(e) {
        const classItem = e.target.closest('.class-item');
        if (!classItem) return;
        const index = parseInt(classItem.dataset.index, 10);

        if (e.target.classList.contains('remove-class-btn')) {
            currentClasses.splice(index, 1);
            renderClasses();
            return;
        }

        const classObj = currentClasses[index];
        const colorInput = classItem.querySelector('.class-color-input');
        const typeSelect = classItem.querySelector('.class-type-select');
        const thresholdInput = classItem.querySelector('.class-threshold-input');

        if (colorInput.matches(':focus')) return;

        classObj.color = hexToRgb(colorInput.value);
        classObj.type = typeSelect.value;

        let isReadOnly = true;
        let newThreshold;

        if (classObj.type === 'Areal') newThreshold = paramInputs.areal_threshold.value;
        else if (classObj.type === 'Linear') newThreshold = paramInputs.linear_threshold.value;
        else { isReadOnly = false; newThreshold = thresholdInput.value; }

        classObj.threshold = newThreshold;
        thresholdInput.value = newThreshold;
        thresholdInput.readOnly = isReadOnly;
        thresholdInput.classList.toggle('bg-gray-200', isReadOnly);
        thresholdInput.classList.toggle('bg-white', !isReadOnly);
    }

    function setSelectedTiff(name) {
        selectedTiffName = name;
        if (name) {
            selectedTiffNameEl.textContent = name;
            selectedTiffNameEl.classList.remove('text-gray-500', 'italic');
            selectedTiffNameEl.classList.add('text-gray-800');
        } else {
            selectedTiffNameEl.textContent = 'No TIFF selected';
            selectedTiffNameEl.classList.add('text-gray-500', 'italic');
            selectedTiffNameEl.classList.remove('text-gray-800');
        }
        updateRunButtonState();
        // Generate preview if selected
        if (name) {
            tiffPreviewContainer.innerHTML = `<div class="flex items-center"><div class="loader !w-6 !h-6"></div><p class="ml-3 text-gray-600">Generating preview...</p></div>`;
            fetch(`${API_BASE_URL}/api/generate-preview`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: name }) })
                .then(r => r.json())
                .then(data => {
                    if (data.preview_url) {
                        tiffPreviewContainer.innerHTML = `<img src="${API_BASE_URL}${data.preview_url}" alt="Preview of ${name}" class="max-w-full max-h-[400px] rounded-lg shadow-md">`;
                    } else {
                        tiffPreviewContainer.innerHTML = `<p class="text-red-500">Error: ${data.error || 'Failed to generate preview.'}</p>`;
                    }
                })
                .catch(err => {
                    tiffPreviewContainer.innerHTML = `<p class="text-red-500">Error: ${err.message}</p>`;
                });
        } else {
            tiffPreviewContainer.innerHTML = '<p class="text-gray-500">Select a TIFF file to see a preview.</p>';
        }
    }

    function collectFinalConfig() {
        const classesToSend = currentClasses.map(classObj => {
            let finalThreshold = classObj.threshold;
            if (classObj.type === 'Areal') finalThreshold = paramInputs.areal_threshold.value;
            if (classObj.type === 'Linear') finalThreshold = paramInputs.linear_threshold.value;
            return { name: classObj.name, type: classObj.type, threshold: finalThreshold, color: classObj.color };
        });

        const config = {
            tiff_file: selectedTiffName || '',
            classes: classesToSend,
            params: {}
        };
        for (const key in paramInputs) {
            const input = paramInputs[key];
            if (input.classList.contains('toggle-btn-group')) {
                config.params[key] = input.querySelector('button.active').dataset.value;
            } else {
                config.params[key] = input.value;
            }
        }
        return config;
    }

    async function handleRunPipeline() {
        const config = collectFinalConfig();
        if (!config.tiff_file || config.classes.length === 0) return alert("Select a TIFF file and ensure classes are present.");

        loader.classList.remove('hidden');
        resultsPlaceholder.classList.add('hidden');
        resultsArea.classList.add('hidden');
        costmapDisplayArea.classList.add('hidden');
        runPipelineBtn.disabled = true;
        generateCostmapBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/api/run-pipeline`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to start pipeline.');

            currentTaskId = data.task_id;
            pipelinePollInterval = setInterval(() => checkPipelineStatus(currentTaskId), 2500);

        } catch (error) {
            console.error("Pipeline start error:", error);
            alert(`Error: ${error.message}`);
            stopPolling();
        }
    }

    function stopPolling() {
        if (pipelinePollInterval) clearInterval(pipelinePollInterval);
        pipelinePollInterval = null;
        currentTaskId = null;
        loader.classList.add('hidden');
        updateRunButtonState();
    }

    async function checkPipelineStatus(taskId) {
        if (!taskId) { stopPolling(); return; }
        try {
            const response = await fetch(`${API_BASE_URL}/api/pipeline-status/${taskId}`);
            const data = await response.json();

            if (data.status === 'completed') {
                stopPolling();
                displayResults(data.results);
            } else if (data.status === 'error') {
                stopPolling();
                alert(`Pipeline Error: ${data.message}`);
                resultsPlaceholder.textContent = `An error occurred: ${data.message}`;
                resultsPlaceholder.classList.remove('hidden');
            } else if (data.status === 'canceled') {
                stopPolling();
                resultsPlaceholder.textContent = 'Pipeline execution was canceled by the user.';
                resultsPlaceholder.classList.remove('hidden');
            }
        } catch (error) {
            console.error("Status check failed:", error);
            stopPolling();
            alert("Error checking pipeline status. Check the console.");
        }
    }

    async function handleCancelPipeline() {
        if (!currentTaskId) return;
        console.log(`Requesting cancellation for task: ${currentTaskId}`);
        try {
            await fetch(`${API_BASE_URL}/api/cancel-pipeline/${currentTaskId}`, { method: 'POST' });
        } catch (error) {
            console.error("Failed to send cancel request:", error);
            alert("Could not send cancel request to the server.");
        }
    }

    async function handleGenerateCostmap() {
        if (!currentResultData || (!currentResultData.refined_masks && !currentResultData.local_masks)) {
            alert("Please run the vision pipeline or load local masks first.");
            return;
        }

        const masks_to_use = currentResultData.refined_masks || currentResultData.local_masks;

        generateCostmapBtn.disabled = true;
        generateCostmapBtn.textContent = 'Generating...';

        try {
            const response = await fetch(`${API_BASE_URL}/api/generate-costmap`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    refined_masks: masks_to_use,
                    tiff_folder: currentTiffFolder || selectedTiffName || '',
                    device: document.getElementById('costmap-device').value,
                    t_dict: {
                        t_a: parseFloat(paramInputs.refiner_areal_threshold.value),
                        t_l: parseFloat(paramInputs.refiner_linear_threshold.value)
                    }
                })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to generate costmap.");

            currentCostmapUrl = data.costmap_url;
            currentCostmapOverlayUrl = data.costmap_url; // overlay uses BW by default
            currentCostmapOverlayFallbackUrl = data.colored_url || null;
            costmapDisplayArea.classList.remove('hidden');
            updateCostmapOverlayUI();
            saveCostmapImgBtn.disabled = false;

            // Load saved goals for this TIFF
            loadSavedGoals(currentTiffFolder || selectedTiffName);

        } catch (error) {
            alert(`Costmap Generation Error: ${error.message}`);
        } finally {
            generateCostmapBtn.disabled = false;
            generateCostmapBtn.textContent = 'Generate Costmap';
        }
    }

    async function updateGpuStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/get-gpu-status`);
            if (!response.ok) throw new Error("Could not fetch GPU status.");
            const data = await response.json();

            gpuStatusContainer.innerHTML = '';
            gpuStatusContainer.className = 'grid grid-cols-1 md:grid-cols-2 gap-4';

            if (data.source.startsWith("mock")) {
                const mockMessage = document.createElement('p');
                mockMessage.className = "text-center text-amber-600 bg-amber-100 p-3 rounded-md col-span-full";
                mockMessage.textContent = "Note: Displaying mock data. Install 'pynvml' on the server for live stats.";
                gpuStatusContainer.appendChild(mockMessage);
            }

            updateDeviceDropdowns(data.gpus.length);

            data.gpus.forEach(gpu => {
                const memoryUsage = (gpu.memory_total > 0) ? (gpu.memory_used / gpu.memory_total * 100).toFixed(1) : 0;
                const gpuCard = document.createElement('div');
                gpuCard.className = 'p-4 border rounded-lg bg-gray-50';
                gpuCard.innerHTML = `
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="font-bold text-lg">GPU ${gpu.id}</h3>
                        <span class="font-mono text-sm bg-indigo-100 text-indigo-800 px-2 py-1 rounded">Usage: ${gpu.usage}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2.5">
                        <div class="bg-blue-600 h-2.5 rounded-full" style="width: ${memoryUsage}%"></div>
                    </div>
                    <p class="text-right text-sm text-gray-600 mt-1">Memory: ${gpu.memory_used}MB / ${gpu.memory_total}MB (${memoryUsage}%)</p>
                `;
                gpuStatusContainer.appendChild(gpuCard);
            });

        } catch (error) {
            gpuStatusContainer.innerHTML = `<p class="text-red-500 col-span-full">${error.message}</p>`;
        }
    }

    function showGpuStatus() {
        gpuModal.classList.remove('hidden');
        gpuModal.classList.add('flex');
        updateGpuStatus();
        if (!gpuPollInterval) {
            gpuPollInterval = setInterval(updateGpuStatus, 2000);
        }
    }

    function hideGpuStatus() {
        gpuModal.classList.add('hidden');
        gpuModal.classList.remove('flex');
        if (gpuPollInterval) {
            clearInterval(gpuPollInterval);
            gpuPollInterval = null;
        }
    }

    async function updateConsole() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/get-console-output`);
            const data = await response.json();
            if (response.ok && consoleOutput.textContent !== data.logs) {
                consoleOutput.textContent = data.logs;
                consoleOutput.scrollTop = consoleOutput.scrollHeight;
            }
        } catch (error) {
            console.error("Failed to fetch console logs:", error);
        }
    }

    function showConsole() {
        consoleModal.classList.remove('hidden');
        consoleModal.classList.add('flex');
        updateConsole();
        if (!consolePollInterval) consolePollInterval = setInterval(updateConsole, 1500);
    }

    function hideConsole() {
        consoleModal.classList.add('hidden');
        consoleModal.classList.remove('flex');
        clearInterval(consolePollInterval);
        consolePollInterval = null;
    }

    async function showCostmapModal() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/get-costmap-functions`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to load costmap functions.");

            defaultCostmapCode.textContent = data.default;
            generatedCostmapCode.textContent = data.generated;
            defaultCostmapCode.dataset.raw = data.default;
            generatedCostmapCode.dataset.raw = data.generated;

            // Apply syntax highlighting first
            hljs.highlightElement(defaultCostmapCode);
            hljs.highlightElement(generatedCostmapCode);

            // [ADD THIS] Then, apply line numbers
            hljs.lineNumbersBlock(defaultCostmapCode);
            hljs.lineNumbersBlock(generatedCostmapCode);

            generatedCostmapCode.setAttribute('contenteditable', 'false');
            generatedCostmapCode.classList.remove('ring-2', 'ring-blue-500');
            saveCostmapBtn.disabled = true;
            editCostmapBtn.disabled = false;

            costmapModal.classList.remove('hidden');
            costmapModal.classList.add('flex');
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    function hideCostmapModal() {
        costmapModal.classList.add('hidden');
        costmapModal.classList.remove('flex');
    }

    function handleEditCostmap() {
        const rawCode = generatedCostmapCode.dataset.raw || generatedCostmapCode.textContent;
        generatedCostmapCode.textContent = rawCode;
        generatedCostmapCode.setAttribute('contenteditable', 'true');
        generatedCostmapCode.classList.add('ring-2', 'ring-blue-500');
        generatedCostmapCode.focus();
        saveCostmapBtn.disabled = false;
        editCostmapBtn.disabled = true;
    }

    async function handleSaveCostmapCode() {
        // Use our new, more reliable function to get the code
        const newCode = getTextFromEditable(generatedCostmapCode);

        saveCostmapBtn.textContent = 'Saving...';
        try {
            const response = await fetch(`${API_BASE_URL}/api/save-costmap-function`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: newCode })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to save.");

            alert(data.message);
            generatedCostmapCode.setAttribute('contenteditable', 'false');
            generatedCostmapCode.classList.remove('ring-2', 'ring-blue-500');
            saveCostmapBtn.disabled = true;
            editCostmapBtn.disabled = false;

            // Re-apply highlighting and line numbers after saving
            generatedCostmapCode.textContent = newCode;
            generatedCostmapCode.dataset.raw = newCode;
            hljs.highlightElement(generatedCostmapCode);
            hljs.lineNumbersBlock(generatedCostmapCode);

        } catch (error) {
            alert(`Save failed: ${error.message}`);
        } finally {
            saveCostmapBtn.textContent = 'Save';
        }
    }

    function getTextFromEditable(element) {
        // This function is better at preserving whitespace than .innerText
        let html = element.innerHTML;
        // Convert common contenteditable newlines (<br> and <div>) to the \n character
        html = html.replace(/<br\s*\/?>/gi, "\n");
        html = html.replace(/<div[^>]*>/gi, "\n").replace(/<\/div>/gi, "");
        html = html.replace(/<p[^>]*>/gi, "\n").replace(/<\/p>/gi, "");

        // Use a temporary element to strip any remaining HTML tags (like <span> from highlighting)
        // and decode HTML entities (like &nbsp;) into actual spaces.
        const tempEl = document.createElement('div');
        tempEl.innerHTML = html;
        let text = tempEl.textContent || tempEl.innerText || "";

        // Finally, remove any trailing newlines that might have been added
        return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trimEnd();
    }

    async function handleRestoreCostmap() {
        if (!confirm("Are you sure you want to overwrite the generated costmap with the default version? This cannot be undone.")) {
            return;
        }

        restoreCostmapBtn.textContent = 'Restoring...';
        restoreCostmapBtn.disabled = true;

        try {
            // 1. Tell the server to restore the file
            const restoreResponse = await fetch(`${API_BASE_URL}/api/restore-costmap-function`, { method: 'POST' });
            const restoreData = await restoreResponse.json();
            if (!restoreResponse.ok) throw new Error(restoreData.error || "Failed to restore.");

            // 2. Fetch the new content of the generated file
            const contentResponse = await fetch(`${API_BASE_URL}/api/get-costmap-functions`);
            const contentData = await contentResponse.json();
            if (!contentResponse.ok) throw new Error(contentData.error || "Failed to reload content.");

            // 3. Update the UI with the new content
            generatedCostmapCode.textContent = contentData.generated;
            generatedCostmapCode.dataset.raw = contentData.generated;
            hljs.highlightElement(generatedCostmapCode);

            // 4. Reset the editor state
            generatedCostmapCode.setAttribute('contenteditable', 'false');
            generatedCostmapCode.classList.remove('ring-2', 'ring-blue-500');
            saveCostmapBtn.disabled = true;
            editCostmapBtn.disabled = false;

            alert(restoreData.message);

        } catch (error) {
            alert(`Restore failed: ${error.message}`);
        } finally {
            restoreCostmapBtn.textContent = 'Restore Default';
            restoreCostmapBtn.disabled = false;
        }
    }

    function handleToggleButtons(e) {
        const button = e.target.closest('button');
        if (!button) return;
        const group = button.parentElement;
        group.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    }

    function initializeWorldMap() {
        if (worldMap) return;

        // worldMap = L.map('world-map').setView([30.2672, -97.7431], 13);
        worldMap = L.map('world-map').setView([50, -90], 3);
        L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', { maxZoom: 20, subdomains: ['mt0', 'mt1', 'mt2', 'mt3'], attribution: '&copy; Google' }).addTo(worldMap);
        drawnItems = new L.FeatureGroup();
        worldMap.addLayer(drawnItems);
        const drawControl = new L.Control.Draw({ draw: { polygon: false, polyline: false, circle: false, marker: false, circlemarker: false, rectangle: { shapeOptions: { color: '#f06eaa' } } }, edit: { featureGroup: drawnItems } });
        worldMap.addControl(drawControl);
        worldMap.on(L.Draw.Event.CREATED, (event) => { drawnItems.clearLayers(); drawnItems.addLayer(event.layer); capturedBounds = event.layer.getBounds(); saveMapAreaBtn.disabled = false; });
        worldMap.on('draw:edited', (e) => e.layers.eachLayer(layer => capturedBounds = layer.getBounds()));
        worldMap.on('draw:deleted', () => { saveMapAreaBtn.disabled = true; capturedBounds = null; });
    }

    async function handleSaveMapArea() {
        const filename = snipFilenameInput.value.trim();
        if (!filename || !capturedBounds) return alert('Please provide a filename and draw a rectangle on the map.');

        saveMapAreaBtn.disabled = true;
        rasterizeProgressArea.classList.remove('hidden');
        try {
            const response = await fetch(`${API_BASE_URL}/api/rasterize-area`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, bounds: capturedBounds.toBBoxString(), tileSize: snipTileSizeInput.value, resolution: snipResolutionInput.value })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to start rasterization.');
            progressInterval = setInterval(() => checkRasterizeProgress(data.task_id), 1000);
        } catch (error) {
            alert(`Error: ${error.message}`);
            saveMapAreaBtn.disabled = false;
            rasterizeProgressArea.classList.add('hidden');
        }
    }

    async function checkRasterizeProgress(taskId) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/rasterize-status/${taskId}`);
            const data = await response.json();
            rasterizeStatusText.textContent = data.status || 'Waiting...';
            rasterizeProgressBar.style.width = `${data.progress || 0}%`;
            rasterizeProgressBar.classList.remove('bg-red-500');

            if (data.error) {
                clearInterval(progressInterval);
                rasterizeStatusText.textContent = `Error: ${data.status}`;
                rasterizeProgressBar.classList.add('bg-red-500');
                saveMapAreaBtn.disabled = false;
                return;
            }

            if (data.progress >= 100) {
                clearInterval(progressInterval);
                alert(`Successfully saved: ${data.filename}`);
                hideWorldMap();
                fetchTiffFiles();
                rasterizeProgressArea.classList.add('hidden');
                snipFilenameInput.value = '';
                saveMapAreaBtn.disabled = true;
                drawnItems.clearLayers();
            }
        } catch (error) {
            console.error('Progress check failed:', error);
            clearInterval(progressInterval);
            rasterizeStatusText.textContent = 'Error checking progress.';
            saveMapAreaBtn.disabled = false;
        }
    }

    function showWorldMap() {
        worldMapModal.classList.remove('hidden');
        worldMapModal.classList.add('flex');
        setTimeout(() => { initializeWorldMap(); worldMap.invalidateSize(); }, 10);
    }

    function hideWorldMap() {
        worldMapModal.classList.add('hidden');
        worldMapModal.classList.remove('flex');
    }

    function openImageViewer(src) {
        imageViewerContent.setAttribute('src', src);
        imageViewerModal.classList.remove('hidden');
        imageViewerModal.classList.add('flex');
        document.body.style.overflow = 'hidden';
        resetPanzoom();
    }

    function closeImageViewer() {
        imageViewerModal.classList.add('hidden');
        imageViewerModal.classList.remove('flex');
        document.body.style.overflow = '';
    }

    function updateTransform() {
        imageViewerContent.style.transform = `translate(${panzoomState.translateX}px, ${panzoomState.translateY}px) scale(${panzoomState.scale})`;
    }

    function resetPanzoom() {
        panzoomState = { scale: 1, translateX: 0, translateY: 0, isPanning: false, startX: 0, startY: 0 };
        updateTransform();
    }

    function onWheel(e) {
        if (!imageViewerModal.classList.contains('hidden')) {
            e.preventDefault();
            const scaleAmount = 0.1;
            panzoomState.scale += e.deltaY * -scaleAmount;
            panzoomState.scale = Math.min(Math.max(0.5, panzoomState.scale), 10);
            updateTransform();
        }
    }

    function onMouseDown(e) { if (!imageViewerModal.classList.contains('hidden')) { e.preventDefault(); panzoomState.isPanning = true; panzoomState.startX = e.clientX - panzoomState.translateX; panzoomState.startY = e.clientY - panzoomState.translateY; } }
    function onMouseUp() { panzoomState.isPanning = false; }
    function onMouseMove(e) { if (panzoomState.isPanning) { e.preventDefault(); panzoomState.translateX = e.clientX - panzoomState.startX; panzoomState.translateY = e.clientY - panzoomState.startY; updateTransform(); } }

    function showClassDetailModal(className) {
        if (!currentResultData) return;

        const rgbPreviewImg = tiffPreviewContainer.querySelector('img');
        if (!rgbPreviewImg) { alert("Original image preview not available."); return; }
        const rgbUrl = rgbPreviewImg.src;

        const semsegUrl = currentResultData.semantic_masks ? currentResultData.semantic_masks[className] : null;
        const finalUrl = (currentResultData.refined_masks || currentResultData.local_masks)[className];

        if (!finalUrl) { alert(`Mask for class "${className}" is not available.`); return; }


        classDetailTitle.textContent = `Details for: ${className}`;
        editMaskBtn.dataset.className = className;
        detailRgbImg.src = rgbUrl;

        detailSemsegOverlayBase.src = rgbUrl;
        detailSemsegOverlayMask.src = semsegUrl ? withAssetVersion(semsegUrl) : "https://placehold.co/600x400/2d3748/9ca3af?text=N/A";

        detailRefinedOverlayBase.src = rgbUrl;
        detailRefinedOverlayMask.src = finalUrl.startsWith('data:') ? finalUrl : withAssetVersion(finalUrl);

        semsegSlider.value = 60;
        refinedSlider.value = 60;
        detailSemsegOverlayMask.style.opacity = 0.6;
        detailRefinedOverlayMask.style.opacity = 0.6;

        classDetailModal.classList.remove('hidden');
        classDetailModal.classList.add('flex');
        document.body.style.overflow = 'hidden';
    }

    function hideClassDetailModal() {
        classDetailModal.classList.add('hidden');
        classDetailModal.classList.remove('flex');
        document.body.style.overflow = '';
    }

    function updatePixelEditorSelectionUI() {
        pixelSelectionCount.textContent = pixelEditorState.selectedPixels.size.toString();
        pixelUndoSelectionBtn.disabled = pixelEditorState.selectionHistory.length === 0 && pixelEditorState.polygonPoints.length === 0;
        pixelApplyValueBtn.disabled = pixelEditorState.selectedPixels.size === 0;
        pixelSaveBtn.disabled = !pixelEditorState.isDirty;
        pixelEditorStatus.textContent = pixelEditorState.isDirty ? 'Local changes ready to save.' : 'No local changes yet.';
        pixelBrushSizeInput.disabled = !['square', 'circle'].includes(pixelEditorState.tool);
        pixelClosePolygonBtn.disabled = pixelEditorState.tool !== 'polygon' || pixelEditorState.polygonPoints.length < 3;

        Array.from(pixelToolButtons.querySelectorAll('[data-tool]')).forEach((button) => {
            const isActive = button.dataset.tool === pixelEditorState.tool;
            button.classList.toggle('bg-blue-600', isActive);
            button.classList.toggle('text-white', isActive);
            button.classList.toggle('hover:bg-blue-700', isActive);
            button.classList.toggle('bg-gray-200', !isActive);
            button.classList.toggle('text-gray-800', !isActive);
            button.classList.toggle('hover:bg-gray-300', !isActive);
        });

        pixelBrushSizeValue.textContent = `${pixelEditorState.brushSize} px`;
        if (pixelEditorState.tool === 'single') {
            pixelToolHint.textContent = 'Single pixel mode selects exactly one pixel per hit.';
        } else if (pixelEditorState.tool === 'square') {
            pixelToolHint.textContent = 'Square mode paints a square brush as you click or drag.';
        } else if (pixelEditorState.tool === 'circle') {
            pixelToolHint.textContent = 'Circle mode paints a circular brush as you click or drag.';
        } else {
            pixelToolHint.textContent = 'Polygon mode places vertices on click. Close the polygon to add the enclosed area to the selection.';
        }
    }

    function updatePixelEditorZoomUI() {
        const zoomPercent = Math.round(pixelEditorState.zoom * 100);
        pixelZoomLabel.textContent = `${zoomPercent}%`;
        pixelEditorCanvas.style.width = `${pixelEditorState.width * pixelEditorState.zoom}px`;
        pixelEditorCanvas.style.height = `${pixelEditorState.height * pixelEditorState.zoom}px`;
    }

    function renderPixelEditorCanvas() {
        if (!pixelEditorState.imageData) return;
        const ctx = pixelEditorCanvas.getContext('2d');
        ctx.putImageData(pixelEditorState.imageData, 0, 0);
        ctx.save();
        ctx.fillStyle = 'rgba(59, 130, 246, 0.75)';
        for (const key of pixelEditorState.selectedPixels) {
            const { x, y } = parsePixelKey(key);
            ctx.fillRect(x, y, 1, 1);
        }
        if (pixelEditorState.tool === 'polygon' && pixelEditorState.polygonPoints.length > 0) {
            ctx.save();
            ctx.strokeStyle = 'rgba(79, 70, 229, 0.95)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pixelEditorState.polygonPoints[0].x + 0.5, pixelEditorState.polygonPoints[0].y + 0.5);
            for (let i = 1; i < pixelEditorState.polygonPoints.length; i += 1) {
                ctx.lineTo(pixelEditorState.polygonPoints[i].x + 0.5, pixelEditorState.polygonPoints[i].y + 0.5);
            }
            if (pixelEditorState.hoverPoint) {
                ctx.lineTo(pixelEditorState.hoverPoint.x + 0.5, pixelEditorState.hoverPoint.y + 0.5);
            }
            ctx.stroke();
            pixelEditorState.polygonPoints.forEach((point) => {
                ctx.fillStyle = 'rgba(79, 70, 229, 1)';
                ctx.fillRect(point.x - 1, point.y - 1, 3, 3);
            });
            ctx.restore();
        }
        ctx.restore();
        updatePixelEditorSelectionUI();
        updatePixelEditorZoomUI();
    }

    function rebuildPixelSelection() {
        pixelEditorState.selectedPixels = new Set(pixelEditorState.permanentPixels);
        pixelEditorState.selectionHistory.forEach((stroke) => {
            stroke.forEach((key) => pixelEditorState.selectedPixels.add(key));
        });
    }

    function clearPixelSelection() {
        pixelEditorState.selectedPixels.clear();
        pixelEditorState.permanentPixels.clear();
        pixelEditorState.selectionHistory = [];
        pixelEditorState.currentStroke = null;
        pixelEditorState.lastPoint = null;
        pixelEditorState.polygonPoints = [];
        pixelEditorState.hoverPoint = null;
        renderPixelEditorCanvas();
    }

    function commitPixelStroke(stroke) {
        if (!stroke || stroke.size === 0) return;
        pixelEditorState.selectionHistory.push(Array.from(stroke));
        if (pixelEditorState.selectionHistory.length > 5) {
            const oldestStroke = pixelEditorState.selectionHistory.shift();
            oldestStroke.forEach((key) => pixelEditorState.permanentPixels.add(key));
        }
        rebuildPixelSelection();
        renderPixelEditorCanvas();
    }

    function undoLastPixelSelection() {
        if (pixelEditorState.polygonPoints.length > 0) {
            pixelEditorState.polygonPoints.pop();
            renderPixelEditorCanvas();
            return;
        }
        if (pixelEditorState.selectionHistory.length === 0) return;
        pixelEditorState.selectionHistory.pop();
        rebuildPixelSelection();
        renderPixelEditorCanvas();
    }

    function addPixelToStroke(x, y) {
        if (x < 0 || y < 0 || x >= pixelEditorState.width || y >= pixelEditorState.height) return;
        const key = getPixelKey(x, y);
        pixelEditorState.currentStroke.add(key);
        pixelEditorState.selectedPixels.add(key);
    }

    function addSquareToStroke(centerX, centerY) {
        const size = pixelEditorState.brushSize;
        const startX = centerX - Math.floor(size / 2);
        const startY = centerY - Math.floor(size / 2);
        for (let y = startY; y < startY + size; y += 1) {
            for (let x = startX; x < startX + size; x += 1) {
                addPixelToStroke(x, y);
            }
        }
    }

    function addCircleToStroke(centerX, centerY) {
        const radius = (pixelEditorState.brushSize - 1) / 2;
        const minX = Math.floor(centerX - radius);
        const maxX = Math.ceil(centerX + radius);
        const minY = Math.floor(centerY - radius);
        const maxY = Math.ceil(centerY + radius);
        for (let y = minY; y <= maxY; y += 1) {
            for (let x = minX; x <= maxX; x += 1) {
                const dx = x - centerX;
                const dy = y - centerY;
                if ((dx * dx) + (dy * dy) <= radius * radius + 0.25) {
                    addPixelToStroke(x, y);
                }
            }
        }
    }

    function addToolStampToStroke(x, y) {
        if (pixelEditorState.tool === 'square') {
            addSquareToStroke(x, y);
            return;
        }
        if (pixelEditorState.tool === 'circle') {
            addCircleToStroke(x, y);
            return;
        }
        addPixelToStroke(x, y);
    }

    function addLineToStroke(x0, y0, x1, y1) {
        let currentX = x0;
        let currentY = y0;
        const dx = Math.abs(x1 - x0);
        const sx = x0 < x1 ? 1 : -1;
        const dy = -Math.abs(y1 - y0);
        const sy = y0 < y1 ? 1 : -1;
        let err = dx + dy;

        while (true) {
            addToolStampToStroke(currentX, currentY);
            if (currentX === x1 && currentY === y1) break;
            const e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                currentX += sx;
            }
            if (e2 <= dx) {
                err += dx;
                currentY += sy;
            }
        }
    }

    function getCanvasPixelFromEvent(event) {
        const rect = pixelEditorCanvas.getBoundingClientRect();
        const x = Math.floor(((event.clientX - rect.left) / rect.width) * pixelEditorState.width);
        const y = Math.floor(((event.clientY - rect.top) / rect.height) * pixelEditorState.height);
        return {
            x: Math.max(0, Math.min(pixelEditorState.width - 1, x)),
            y: Math.max(0, Math.min(pixelEditorState.height - 1, y))
        };
    }

    function startPixelSelection(event) {
        if (event.button !== 0 || !pixelEditorState.imageData) return;
        event.preventDefault();
        const { x, y } = getCanvasPixelFromEvent(event);
        if (pixelEditorState.tool === 'polygon') {
            if (pixelEditorState.polygonPoints.length >= 3) {
                const firstPoint = pixelEditorState.polygonPoints[0];
                const dx = x - firstPoint.x;
                const dy = y - firstPoint.y;
                if ((dx * dx) + (dy * dy) <= 9) {
                    finalizePolygonSelection();
                    return;
                }
            }
            pixelEditorState.polygonPoints.push({ x, y });
            pixelEditorState.hoverPoint = { x, y };
            renderPixelEditorCanvas();
            return;
        }
        pixelEditorState.isDrawing = true;
        pixelEditorState.currentStroke = new Set();
        pixelEditorState.lastPoint = { x, y };
        addToolStampToStroke(x, y);
        renderPixelEditorCanvas();
    }

    function movePixelSelection(event) {
        const { x, y } = getCanvasPixelFromEvent(event);
        if (pixelEditorState.tool === 'polygon') {
            pixelEditorState.hoverPoint = { x, y };
            renderPixelEditorCanvas();
            return;
        }
        if (!pixelEditorState.isDrawing || !pixelEditorState.currentStroke) return;
        const lastPoint = pixelEditorState.lastPoint || { x, y };
        addLineToStroke(lastPoint.x, lastPoint.y, x, y);
        pixelEditorState.lastPoint = { x, y };
        renderPixelEditorCanvas();
    }

    function endPixelSelection() {
        if (!pixelEditorState.isDrawing) return;
        pixelEditorState.isDrawing = false;
        commitPixelStroke(pixelEditorState.currentStroke);
        pixelEditorState.currentStroke = null;
        pixelEditorState.lastPoint = null;
    }

    function finalizePolygonSelection() {
        if (pixelEditorState.polygonPoints.length < 3) return;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = pixelEditorState.width;
        tempCanvas.height = pixelEditorState.height;
        const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
        tempCtx.fillStyle = '#ffffff';
        tempCtx.beginPath();
        tempCtx.moveTo(pixelEditorState.polygonPoints[0].x + 0.5, pixelEditorState.polygonPoints[0].y + 0.5);
        for (let i = 1; i < pixelEditorState.polygonPoints.length; i += 1) {
            tempCtx.lineTo(pixelEditorState.polygonPoints[i].x + 0.5, pixelEditorState.polygonPoints[i].y + 0.5);
        }
        tempCtx.closePath();
        tempCtx.fill();

        const tempData = tempCtx.getImageData(0, 0, pixelEditorState.width, pixelEditorState.height).data;
        const stroke = new Set();
        for (let y = 0; y < pixelEditorState.height; y += 1) {
            for (let x = 0; x < pixelEditorState.width; x += 1) {
                const alphaIndex = ((y * pixelEditorState.width) + x) * 4 + 3;
                if (tempData[alphaIndex] > 0) {
                    stroke.add(getPixelKey(x, y));
                }
            }
        }

        pixelEditorState.polygonPoints = [];
        pixelEditorState.hoverPoint = null;
        commitPixelStroke(stroke);
    }

    function setPixelEditorTool(tool) {
        pixelEditorState.tool = tool;
        pixelEditorState.isDrawing = false;
        pixelEditorState.currentStroke = null;
        pixelEditorState.lastPoint = null;
        pixelEditorState.polygonPoints = [];
        pixelEditorState.hoverPoint = null;
        renderPixelEditorCanvas();
    }

    function zoomPixelEditor(multiplier) {
        pixelEditorState.zoom = Math.max(1, Math.min(40, pixelEditorState.zoom * multiplier));
        updatePixelEditorZoomUI();
    }

    function resetPixelEditorZoom() {
        pixelEditorState.zoom = 1;
        updatePixelEditorZoomUI();
    }

    async function openPixelEditor({ imageType, imageUrl, colorUrl = null, className = null, title }) {
        try {
            const image = await loadImage(withAssetVersion(imageUrl));
            pixelEditorState.imageType = imageType;
            pixelEditorState.imageUrl = imageUrl;
            pixelEditorState.colorUrl = colorUrl;
            pixelEditorState.className = className;
            pixelEditorState.width = image.naturalWidth || image.width;
            pixelEditorState.height = image.naturalHeight || image.height;
            pixelEditorState.zoom = 1;
            pixelEditorState.isDirty = false;
            pixelEditorState.selectedPixels = new Set();
            pixelEditorState.permanentPixels = new Set();
            pixelEditorState.selectionHistory = [];
            pixelEditorState.currentStroke = null;
            pixelEditorState.lastPoint = null;
            pixelEditorState.tool = 'single';
            pixelEditorState.brushSize = 1;
            pixelEditorState.polygonPoints = [];
            pixelEditorState.hoverPoint = null;
            pixelBrushSizeInput.value = '1';

            pixelEditorCanvas.width = pixelEditorState.width;
            pixelEditorCanvas.height = pixelEditorState.height;

            const ctx = pixelEditorCanvas.getContext('2d', { willReadFrequently: true });
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(image, 0, 0);
            pixelEditorState.imageData = ctx.getImageData(0, 0, pixelEditorState.width, pixelEditorState.height);

            pixelEditorTitle.textContent = title;
            pixelEditorSubtitle.textContent = imageType === 'mask'
                ? 'Edit the grayscale mask directly. Value 0 is black and 255 is white.'
                : 'Edit the grayscale costmap directly. Value 0 is low cost and 255 is high cost.';

            renderPixelEditorCanvas();
            pixelEditorModal.classList.remove('hidden');
            pixelEditorModal.classList.add('flex');
            document.body.style.overflow = 'hidden';
        } catch (error) {
            alert(`Failed to open editor: ${error.message}`);
        }
    }

    function closePixelEditor() {
        pixelEditorModal.classList.add('hidden');
        pixelEditorModal.classList.remove('flex');
        if (classDetailModal.classList.contains('hidden') && imageViewerModal.classList.contains('hidden')) {
            document.body.style.overflow = '';
        }
    }

    function applyPixelValueToSelection() {
        if (!pixelEditorState.imageData || pixelEditorState.selectedPixels.size === 0) return;

        const value = Number(pixelValueInput.value);
        if (!Number.isFinite(value) || value < 0 || value > 255) {
            alert('Enter a value between 0 and 255.');
            return;
        }

        const clamped = Math.round(value);
        for (const key of pixelEditorState.selectedPixels) {
            const { x, y } = parsePixelKey(key);
            const index = (y * pixelEditorState.width + x) * 4;
            pixelEditorState.imageData.data[index] = clamped;
            pixelEditorState.imageData.data[index + 1] = clamped;
            pixelEditorState.imageData.data[index + 2] = clamped;
            pixelEditorState.imageData.data[index + 3] = 255;
        }

        pixelEditorState.isDirty = true;
        clearPixelSelection();
        renderPixelEditorCanvas();
    }

    function refreshMaskPreview(className) {
        if (!currentResultData) return;
        const maskUrl = (currentResultData.refined_masks || currentResultData.local_masks || {})[className];
        if (!maskUrl) return;

        Array.from(resultsMasksContainer.querySelectorAll('[data-class-name]')).forEach((card) => {
            if (card.dataset.className !== className) return;
            const img = card.querySelector('img');
            if (img) img.src = withAssetVersion(maskUrl);
        });

        if (!classDetailModal.classList.contains('hidden') && classDetailTitle.textContent === `Details for: ${className}`) {
            detailRefinedOverlayMask.src = withAssetVersion(maskUrl);
        }
    }

    async function savePixelEditorChanges() {
        if (!pixelEditorState.isDirty || !pixelEditorState.imageData) return;

        pixelSaveBtn.disabled = true;
        pixelEditorStatus.textContent = 'Saving changes...';

        try {
            const imageDataUrl = pixelEditorCanvas.toDataURL('image/png');
            const response = await fetch(`${API_BASE_URL}/api/update-edited-image`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_type: pixelEditorState.imageType,
                    image_url: pixelEditorState.imageUrl,
                    image_data: imageDataUrl
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to save image changes.');

            bumpAssetVersion(data.image_url || pixelEditorState.imageUrl);
            if (data.colored_url) bumpAssetVersion(data.colored_url);

            pixelEditorState.isDirty = false;
            updatePixelEditorSelectionUI();

            if (pixelEditorState.imageType === 'mask' && pixelEditorState.className) {
                refreshMaskPreview(pixelEditorState.className);
            } else {
                if (data.image_url) currentCostmapUrl = data.image_url;
                if (data.image_url) currentCostmapOverlayUrl = data.image_url;
                if (data.colored_url) currentCostmapOverlayFallbackUrl = data.colored_url;
                updateCostmapOverlayUI();
            }

            pixelEditorStatus.textContent = 'Changes saved.';
        } catch (error) {
            pixelEditorStatus.textContent = 'Save failed.';
            alert(`Save error: ${error.message}`);
        } finally {
            pixelSaveBtn.disabled = !pixelEditorState.isDirty;
        }
    }
    async function handleDownloadCostmap() {
        if (!currentCostmapUrl) {
            alert("No costmap available for download.");
            return;
        }

        // --- Get user choice from dropdown ---
        const formatChoice = document.getElementById("download-format").value;
        // Optional: if your app tracks the original GeoTIFF name
        const tiffFile = selectedTiffName || null;

        try {
            // Show feedback
            const downloadBtn = document.getElementById("download-costmap-btn");
            const originalText = downloadBtn.textContent;
            downloadBtn.textContent = "Preparing...";
            downloadBtn.disabled = true;

            // --- Ask backend to prepare and send the file ---
            const response = await fetch(`${API_BASE_URL}/api/download-costmap`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    format: formatChoice,
                    costmap_url: currentCostmapUrl,
                    tiff_filename: tiffFile
                })
            });

            if (!response.ok) {
                const err = await response.json();
                alert("Download failed: " + (err.error || "Unknown error"));
                downloadBtn.textContent = originalText;
                downloadBtn.disabled = false;
                return;
            }

            // --- Convert to blob and trigger download ---
            const blob = await response.blob();
            const ext = formatChoice === "tiff" ? "tif" : "png";
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = `costmap_${Date.now()}.${ext}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

        } catch (error) {
            console.error("Download error:", error);
            alert("An error occurred while downloading the costmap.");
        } finally {
            // Restore button state
            const downloadBtn = document.getElementById("download-costmap-btn");
            downloadBtn.textContent = "Download";
            downloadBtn.disabled = false;
        }
    }



    // --- [MODIFIED] Planner Functions ---

    function redrawAllPlannerElements() {
        const ctx = plannerCanvas.getContext('2d');
        ctx.clearRect(0, 0, plannerCanvas.width, plannerCanvas.height);
        drawPlannerPoints();
        if (plannerState.path) {
            drawPath(plannerState.path);
        }
    }

    const setupPlannerCanvas = () => {
        if (plannerModal.classList.contains('hidden') || plannerImgBg.clientWidth === 0 || plannerImgBg.clientHeight === 0) {
            return;
        }

        const imgRect = plannerImgBg.getBoundingClientRect();
        const containerRect = plannerImgBg.parentElement.getBoundingClientRect();

        plannerCanvas.width = imgRect.width;
        plannerCanvas.height = imgRect.height;

        // Position the canvas directly over the visible image
        plannerCanvas.style.left = `${imgRect.left - containerRect.left}px`;
        plannerCanvas.style.top = `${imgRect.top - containerRect.top}px`;

        redrawAllPlannerElements();
    };

    function resetPlannerState() {
        plannerState = {
            selectingStart: true,
            selectingEnd: false,
            startPoint: null,
            endPoint: null,
            displayDimensions: null,
            path: null,
            originalDimensions: null // Also clear original dimensions
        };
        if (downloadPlanBtn) downloadPlanBtn.disabled = true;
        const optsBtn = document.getElementById('download-plan-options-btn');
        if (optsBtn) optsBtn.disabled = true;
        setDownloadDropdownOpen(false);
        lastKnownCoords = null;
        plannerCoords.textContent = 'Mouse: (-, -)';
        redrawAllPlannerElements();
        updatePlannerUI();
    }

    function updatePlannerUI() {
        const buttons = [selectStartBtn, selectEndBtn];
        buttons.forEach(btn => btn.classList.remove('ring-2', 'ring-green-500', 'ring-red-500'));

        if (plannerState.selectingStart) {
            selectStartBtn.classList.add('ring-2', 'ring-green-500');
            plannerStatus.textContent = 'Select a start point';
        } else if (plannerState.selectingEnd) {
            selectEndBtn.classList.add('ring-2', 'ring-red-500');
            plannerStatus.textContent = 'Select an end point';
        } else if (plannerState.startPoint && plannerState.endPoint) {
            plannerStatus.textContent = 'Ready to plan';
        } else {
            plannerStatus.textContent = 'Planner Ready';
        }
        planPathBtn.disabled = !(plannerState.startPoint && plannerState.endPoint);
    }

    function showPlannerModal() {
        const rgbPreviewImg = tiffPreviewContainer.querySelector('img');
        if (!rgbPreviewImg || !rgbPreviewImg.src) {
            alert("Cannot open planner: Please select a TIFF and generate a preview first.");
            return;
        }

        plannerModal.classList.remove('hidden');
        plannerModal.classList.add('flex');

        const tiffName = currentTiffFolder || selectedTiffName;
        plannerImgBg.onerror = () => {
            plannerImgBg.onerror = null;
            plannerImgBg.src = rgbPreviewImg.src;
        };
        plannerImgBg.src = tiffName ? `${API_BASE_URL}/results/${tiffName}/preview.png` : rgbPreviewImg.src;

        if (plannerImgOverlay) {
            if (currentCostmapOverlayUrl) {
                const primary = `${API_BASE_URL}${currentCostmapOverlayUrl}`;
                const fallback = currentCostmapOverlayFallbackUrl ? `${API_BASE_URL}${currentCostmapOverlayFallbackUrl}` : '';
                plannerImgOverlay.onerror = () => {
                    if (fallback && plannerImgOverlay.src !== fallback) {
                        plannerImgOverlay.onerror = null;
                        plannerImgOverlay.src = fallback;
                    }
                };
                plannerImgOverlay.src = primary;
                plannerImgOverlay.classList.remove('hidden');
            } else {
                plannerImgOverlay.src = '';
                plannerImgOverlay.classList.add('hidden');
            }
        }
        if (plannerOverlaySlider) {
            setImgOpacityFromSlider(plannerImgOverlay, plannerOverlaySlider);
        }
        if (downloadZipNameInput && tiffName && !downloadZipNameInput.value.trim()) {
            downloadZipNameInput.value = `${tiffName}_plan.zip`;
            localStorage.setItem('downloadZipName', downloadZipNameInput.value);
        }

        if (plannerImgBg.complete) {
            resetPlannerState();
            setupPlannerCanvas();
        } else {
            plannerImgBg.onload = () => {
                resetPlannerState();
                setupPlannerCanvas();
            };
        }

        if (plannerResizeObserver) plannerResizeObserver.disconnect();
        plannerResizeObserver = new ResizeObserver(setupPlannerCanvas);
        plannerResizeObserver.observe(plannerImgBg.parentElement);

        // Load Saved Goals for current TIFF
        if (tiffName) {
            loadSavedGoals(tiffName);
        }
    }

    function hidePlannerModal() {
        plannerModal.classList.add('hidden');
        plannerModal.classList.remove('flex');
        setDownloadDropdownOpen(false);
        if (plannerResizeObserver) {
            plannerResizeObserver.disconnect();
            plannerResizeObserver = null;
        }
    }

    function handlePlannerClick(e) {
        if (!plannerState.selectingStart && !plannerState.selectingEnd) return;

        const rect = plannerCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const point = { x, y };

        if (plannerState.selectingStart) {
            plannerState.startPoint = point;
            plannerState.selectingStart = false;
            plannerState.selectingEnd = true;
        } else if (plannerState.selectingEnd) {
            plannerState.endPoint = point;
            plannerState.selectingEnd = false;
        }

        plannerState.displayDimensions = { width: plannerCanvas.width, height: plannerCanvas.height };

        redrawAllPlannerElements();
        updatePlannerUI();
    }

    function drawPlannerPoints() {
        const ctx = plannerCanvas.getContext('2d');

        if (plannerState.startPoint) {
            ctx.beginPath();
            ctx.arc(plannerState.startPoint.x, plannerState.startPoint.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = '#22c55e';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        if (plannerState.endPoint) {
            ctx.beginPath();
            ctx.arc(plannerState.endPoint.x, plannerState.endPoint.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = '#ef4444';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    async function handlePlanPath() {
        if (!plannerState.startPoint || !plannerState.endPoint || !currentCostmapUrl) {
            alert("Please select a start and end point, and generate a costmap first.");
            return;
        }

        // Goals loaded from disk may not set displayDimensions (it’s usually set on click).
        if (!plannerState.displayDimensions && plannerCanvas && plannerCanvas.width && plannerCanvas.height) {
            plannerState.displayDimensions = { width: plannerCanvas.width, height: plannerCanvas.height };
        }

        planPathBtn.disabled = true;
        plannerStatus.textContent = 'Planning...';

        try {
            const response = await fetch(`${API_BASE_URL}/api/plan-path`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    costmap_url: currentCostmapUrl,
                    tiff_filename: selectedTiffName || '',
                    start_point: plannerState.startPoint,
                    goal_point: plannerState.endPoint,
                    display_dimensions: plannerState.displayDimensions
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Planning failed.");

            plannerState.path = data.path;
            plannerState.originalDimensions = data.original_dimensions; // [MODIFIED] Store original dimensions

            // Enable Download button
            if (downloadPlanBtn) downloadPlanBtn.disabled = false;
            const optsBtn = document.getElementById('download-plan-options-btn');
            if (optsBtn) optsBtn.disabled = false;
            redrawAllPlannerElements();
            plannerStatus.textContent = `Path found with length: ${data.path.length}`;

        } catch (error) {
            alert(`Planning Error: ${error.message}`);
        } finally {
            planPathBtn.disabled = false;
            plannerStatus.textContent = plannerState.path ? `Path found: ${plannerState.path.length} steps` : 'Planning failed.';
        }
    }

    function getDownloadOptionsFromUI() {
        return {
            rgb_plan: document.getElementById('dl-rgb')?.checked ?? true,
            white_plan: document.getElementById('dl-white')?.checked ?? true,
            metadata: document.getElementById('dl-meta')?.checked ?? true,
            costmap_files: document.getElementById('dl-costmap-files')?.checked ?? true,
            costmap_tiff: document.getElementById('dl-costmap-tiff')?.checked ?? true,
            original_tiff: document.getElementById('dl-orig-tiff')?.checked ?? true,
            masks: document.getElementById('dl-masks')?.checked ?? true
        };
    }

    function getDownloadOutputFromUI() {
        return {
            zip: document.getElementById('dl-out-zip')?.checked ?? true,
            individual: document.getElementById('dl-out-files')?.checked ?? false
        };
    }

    function loadDownloadOptionsIntoUI() {
        const savedOptions = JSON.parse(localStorage.getItem('downloadOptions') || '{}');
        const hasSaved = savedOptions && Object.keys(savedOptions).length > 0;
        const merged = {
            rgb_plan: true,
            white_plan: true,
            metadata: true,
            costmap_files: true,
            costmap_tiff: true,
            original_tiff: true,
            masks: true,
            out_zip: true,
            out_files: false,
            ...(hasSaved ? savedOptions : {})
        };

        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.checked = val;
        };
        set('dl-rgb', merged.rgb_plan !== false);
        set('dl-white', merged.white_plan !== false);
        set('dl-meta', merged.metadata !== false);
        set('dl-costmap-files', merged.costmap_files !== false);
        set('dl-costmap-tiff', merged.costmap_tiff !== false);
        set('dl-orig-tiff', merged.original_tiff !== false);
        set('dl-masks', merged.masks !== false);
        set('dl-out-zip', merged.out_zip !== false);
        set('dl-out-files', merged.out_files === true);
    }

    function saveDownloadOptionsFromUI() {
        const options = getDownloadOptionsFromUI();
        const output = getDownloadOutputFromUI();
        localStorage.setItem('downloadOptions', JSON.stringify({
            ...options,
            out_zip: output.zip,
            out_files: output.individual
        }));
    }

    function setDownloadDropdownOpen(open) {
        const dropdown = document.getElementById('download-plan-dropdown');
        const btn = document.getElementById('download-plan-options-btn');
        if (!dropdown || !btn) return;
        dropdown.classList.toggle('hidden', !open);
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    async function handleClearTempDownloads() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/clear-temp-downloads`, { method: 'POST' });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to clear temp downloads.');
            alert(`Cleared temp downloads (${data.deleted || 0} items).`);
        } catch (e) {
            alert(`Clear temp downloads error: ${e.message}`);
        } finally {
            setDownloadDropdownOpen(false);
        }
    }

    async function handleDownloadPlan() {
        if (!plannerState.path || plannerState.path.length === 0) {
            alert("No plan available to download. Please generate a path first.");
            return;
        }
        await executeDownloadPlan();
    }

    async function executeDownloadPlan() {
        const options = getDownloadOptionsFromUI();
        const output = getDownloadOutputFromUI();
        localStorage.setItem('downloadOptions', JSON.stringify({
            ...options,
            out_zip: output.zip,
            out_files: output.individual
        }));
        const zipName = (downloadZipNameInput && downloadZipNameInput.value) ? downloadZipNameInput.value.trim() : '';

        if (!output.zip && !output.individual) {
            alert('Select at least one output type: ZIP and/or Files.');
            return;
        }

        const originalText = downloadPlanBtn ? downloadPlanBtn.textContent : '';
        if (downloadPlanBtn) {
            downloadPlanBtn.disabled = true;
            downloadPlanBtn.textContent = 'Generating...';
        }
        const optsBtn = document.getElementById('download-plan-options-btn');
        if (optsBtn) optsBtn.disabled = true;
        try {
            const response = await fetch(`${API_BASE_URL}/api/download-plan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tiff_folder: currentTiffFolder || selectedTiffName || '',
                    costmap_url: currentCostmapUrl,
                    path: plannerState.path,
                    start: plannerState.startPoint,
                    end: plannerState.endPoint,
                    options: options,
                    zip_name: zipName,
                    output: output
                })
            });

            if (response.ok) {
                const data = await response.json();

                if (output.zip) {
                    if (data.download_url) {
                        const link = document.createElement('a');
                        link.href = `${API_BASE_URL}${data.download_url}`;
                        link.download = data.zip_filename || (zipName || '');
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    } else {
                        alert("ZIP download URL missing in response.");
                    }
                }

                if (output.individual) {
                    const files = Array.isArray(data.individual_files) ? data.individual_files : [];
                    if (files.length === 0) {
                        alert('No individual files were generated for the selected options.');
                    }
                    for (const f of files) {
                        if (!f || !f.url) continue;
                        const link = document.createElement('a');
                        link.href = `${API_BASE_URL}${f.url}`;
                        link.download = f.filename || '';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        // Light throttling to avoid the browser dropping some downloads.
                        // eslint-disable-next-line no-await-in-loop
                        await new Promise(r => setTimeout(r, 150));
                    }
                }
            } else {
                const err = await response.json();
                alert(`Download Error: ${err.error}`);
            }
        } catch (error) {
            alert(`Download Error: ${error.message}`);
        } finally {
            if (downloadPlanBtn) {
                downloadPlanBtn.disabled = false;
                downloadPlanBtn.textContent = originalText || 'Download Plan';
            }
            if (optsBtn) optsBtn.disabled = false;
        }
    }

    function drawPath(path) {
        if (!path || path.length < 2 || !plannerState.originalDimensions) return;

        const ctx = plannerCanvas.getContext('2d');

        const scaleX = plannerCanvas.width / plannerState.originalDimensions.width;
        const scaleY = plannerCanvas.height / plannerState.originalDimensions.height;

        ctx.beginPath();
        ctx.strokeStyle = '#FF0000 ';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const firstPoint = path[0];
        ctx.moveTo(firstPoint[0] * scaleX, firstPoint[1] * scaleY);

        for (let i = 1; i < path.length; i++) {
            const point = path[i];
            ctx.lineTo(point[0] * scaleX, point[1] * scaleY);
        }
        ctx.stroke();
    }

    function handlePlannerMouseMove(e) {
        const rect = plannerCanvas.getBoundingClientRect();
        const x = Math.round(e.clientX - rect.left);
        const y = Math.round(e.clientY - rect.top);

        lastKnownCoords = { x, y };
        plannerCoords.textContent = `Mouse: (${x}, ${y})`;
    }

    function handlePlannerMouseLeave() {
        if (lastKnownCoords) {
            plannerCoords.textContent = `Last: (${lastKnownCoords.x}, ${lastKnownCoords.y})`;
        }
    }

    // --- File Browser Functions ---

    function openBrowser(mode, startPath = '') {
        browserMode = mode;
        browserCurrentPath = startPath;
        browserSelectedPath = null;
        loadSelectedMasksBtn.disabled = true;

        // Configure modal for mode
        if (mode === 'tiff') {
            browserModalTitle.textContent = 'Browse TIFFs';
            browserSearchArea.classList.add('hidden');
            loadSelectedMasksBtn.classList.add('hidden');
        } else if (mode === 'all') {
            browserModalTitle.textContent = 'Browse Costmaps';
            browserSearchArea.classList.add('hidden');
            loadSelectedMasksBtn.classList.add('hidden');
        } else {
            browserModalTitle.textContent = 'Browse Masks';
            browserSearchArea.classList.remove('hidden');
            loadSelectedMasksBtn.classList.remove('hidden');
            browserSearchInput.value = '';
            browserSearchResults.classList.add('hidden');
        }

        serverMasksModal.classList.remove('hidden');
        serverMasksModal.classList.add('flex');
        browseResults(startPath);
    }

    async function showServerMasksModal() {
        openBrowser('masks');
    }

    function hideServerMasksModal() {
        serverMasksModal.classList.add('hidden');
        serverMasksModal.classList.remove('flex');
    }

    async function browseResults(path) {
        browserCurrentPath = path;
        browserSelectedPath = null;
        loadSelectedMasksBtn.disabled = true;
        browserUpBtn.disabled = !path;  // Disable Up when at root

        // Update breadcrumb
        renderBreadcrumb(path);

        serverMasksList.innerHTML = '<p class="text-gray-400 text-center py-4">Loading...</p>';

        try {
            const response = await fetch(`${API_BASE_URL}/api/browse-results`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path, mode: browserMode })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to browse.');

            renderFileBrowser(data.items, path);
        } catch (error) {
            serverMasksList.innerHTML = `<p class="text-red-500 text-center py-4">Error: ${error.message}</p>`;
        }
    }

    // Search handler for mask folders
    function handleSearchInput() {
        const query = browserSearchInput.value.trim();
        clearTimeout(searchDebounceTimer);

        if (!query) {
            browserSearchResults.classList.add('hidden');
            return;
        }

        searchDebounceTimer = setTimeout(async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/search-masks`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await response.json();

                if (data.results && data.results.length > 0) {
                    browserSearchResults.classList.remove('hidden');
                    browserSearchResults.innerHTML = '';
                    data.results.forEach(result => {
                        const row = document.createElement('div');
                        row.className = 'flex items-center gap-2 p-2 hover:bg-blue-50 cursor-pointer border-b last:border-b-0 transition';
                        row.innerHTML = `
                            <span class="text-lg">📁</span>
                            <div class="flex-1 min-w-0">
                                <div class="font-medium text-sm text-gray-800 truncate">${result.name}</div>
                                <div class="text-xs text-gray-500 truncate">${result.path}</div>
                            </div>
                            <span class="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium">masks</span>
                        `;
                        row.addEventListener('click', () => {
                            browserSearchResults.classList.add('hidden');
                            browserSearchInput.value = '';
                            // Navigate to the parent of the found mask folder
                            browseResults(result.path);
                        });

                        // Add Load button
                        const loadBtn = document.createElement('button');
                        loadBtn.className = 'text-xs bg-blue-500 text-white px-3 py-1 rounded-full hover:bg-blue-600 transition ml-1 whitespace-nowrap';
                        loadBtn.textContent = 'Load';
                        loadBtn.addEventListener('click', (e) => {
                            e.stopPropagation();
                            browserSelectedPath = result.path;
                            loadSelectedMasks();
                            browserSearchResults.classList.add('hidden');
                        });
                        row.appendChild(loadBtn);

                        browserSearchResults.appendChild(row);
                    });
                } else {
                    browserSearchResults.classList.remove('hidden');
                    browserSearchResults.innerHTML = '<p class="text-gray-500 text-sm text-center p-3">No mask folders found.</p>';
                }
            } catch (error) {
                browserSearchResults.classList.remove('hidden');
                browserSearchResults.innerHTML = `<p class="text-red-500 text-sm text-center p-3">Search error: ${error.message}</p>`;
            }
        }, 300);  // 300ms debounce
    }

    function renderBreadcrumb(path) {
        browserBreadcrumb.innerHTML = '';

        // Root crumb
        const rootCrumb = document.createElement('span');
        rootCrumb.className = 'cursor-pointer text-blue-600 hover:text-blue-800 font-semibold';
        rootCrumb.textContent = '📁 results';
        rootCrumb.addEventListener('click', () => browseResults(''));
        browserBreadcrumb.appendChild(rootCrumb);

        if (path) {
            const parts = path.split('/');
            let accumulated = '';
            parts.forEach((part, idx) => {
                // Separator
                const sep = document.createElement('span');
                sep.className = 'text-gray-400 mx-1';
                sep.textContent = '/';
                browserBreadcrumb.appendChild(sep);

                accumulated += (accumulated ? '/' : '') + part;
                const crumb = document.createElement('span');
                if (idx === parts.length - 1) {
                    crumb.className = 'font-semibold text-gray-800';
                    crumb.textContent = part;
                } else {
                    const crumbPath = accumulated;
                    crumb.className = 'cursor-pointer text-blue-600 hover:text-blue-800 font-semibold';
                    crumb.textContent = part;
                    crumb.addEventListener('click', () => browseResults(crumbPath));
                }
                browserBreadcrumb.appendChild(crumb);
            });
        }
    }

    function renderFileBrowser(items, currentPath) {
        serverMasksList.innerHTML = '';

        if (items.length === 0) {
            const emptyMsg = browserMode === 'tiff' ? 'No TIFF files found.' : 'This folder is empty.';
            serverMasksList.innerHTML = `<p class="text-gray-500 text-center py-8 italic">${emptyMsg}</p>`;
            return;
        }

        // Separate dirs and files
        const dirs = items.filter(i => i.type === 'dir');
        const files = items.filter(i => i.type === 'file');

        // Render directories
        dirs.forEach(item => {
            const itemPath = currentPath ? `${currentPath}/${item.name}` : item.name;
            const row = document.createElement('div');
            row.className = 'flex items-center gap-3 p-3 rounded-lg cursor-pointer hover:bg-blue-50 border border-transparent hover:border-blue-200 transition-all group';

            // Decide icon
            let icon = '📁';  // Default folder
            let badge = '';
            if (item.has_masks) {
                icon = '📁';
                badge = '<span class="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium">masks</span>';
            } else if (item.name === 'temp_latest') {
                icon = '⏳';
                badge = '<span class="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full font-medium">temp</span>';
            } else if (item.name.match(/^\d{4}-\d{2}-\d{2}$/)) {
                icon = '📅';
            } else if (item.name === 'mask') {
                icon = '🎭';
            } else if (item.name === 'costmap') {
                icon = '🗺️';
            } else if (item.name === 'semantic' || item.name === 'refined') {
                icon = '🖼️';
            }

            row.innerHTML = `
                <span class="text-2xl">${icon}</span>
                <span class="flex-1 font-medium text-gray-800 group-hover:text-blue-700">${item.name}</span>
                ${badge}
                <span class="text-gray-400 group-hover:text-blue-500">›</span>
            `;

            // All folders are navigable on click
            row.addEventListener('click', () => browseResults(itemPath));

            // If it has masks and we're in masks mode, add inline Load button
            if (item.has_masks && browserMode === 'masks') {
                const loadBtn = document.createElement('button');
                loadBtn.className = 'text-xs bg-blue-500 text-white px-3 py-1 rounded-full hover:bg-blue-600 transition ml-2 whitespace-nowrap';
                loadBtn.textContent = 'Load';
                loadBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    browserSelectedPath = itemPath;
                    loadSelectedMasks();
                });
                row.insertBefore(loadBtn, row.lastElementChild);
            }

            // Add rename button for root-level folders (TIFF folders)
            if (!currentPath) {
                const renameBtn = document.createElement('button');
                renameBtn.className = 'text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded-full hover:bg-gray-300 transition whitespace-nowrap';
                renameBtn.textContent = '✏️ Rename';
                renameBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    handleRenameTiffFolder(item.name);
                });
                row.insertBefore(renameBtn, row.lastElementChild);
            }

            serverMasksList.appendChild(row);
        });

        // Render files
        files.forEach(item => {
            const itemPath = currentPath ? `${currentPath}/${item.name}` : item.name;
            const row = document.createElement('div');
            row.className = 'flex items-center gap-3 p-3 rounded-lg hover:bg-gray-100 transition border border-transparent';

            let icon = '📄';
            if (item.name.endsWith('.tif') || item.name.endsWith('.tiff')) {
                icon = '🌍';
            } else if (item.name.endsWith('.png')) {
                icon = '🖼️';
            } else if (item.name.endsWith('.npy')) {
                icon = '📊';
            }

            row.innerHTML = `
                <span class="text-2xl">${icon}</span>
                <span class="flex-1 font-medium text-gray-800">${item.name}</span>
            `;

            // In tiff mode, clicking a file selects it as the active TIFF
            if (browserMode === 'tiff' && (item.name.endsWith('.tif') || item.name.endsWith('.tiff'))) {
                row.classList.add('cursor-pointer');
                // Add a "Select" button
                const selectBtn = document.createElement('button');
                selectBtn.className = 'text-xs bg-green-500 text-white px-3 py-1 rounded-full hover:bg-green-600 transition whitespace-nowrap';
                selectBtn.textContent = 'Select';
                selectBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    // Extract tiff folder name from the path (first part)
                    const parts = currentPath.split('/');
                    const tiffFolderName = parts[0] || item.name.replace(/\.(tiff?|tif)$/i, '');
                    // Set the dropdown to this TIFF
                    selectTiffFromBrowser(tiffFolderName);
                    hideServerMasksModal();
                });
                row.appendChild(selectBtn);
            }

            // Download button for all files
            const dlBtn = document.createElement('button');
            dlBtn.className = 'text-xs bg-gray-200 text-gray-700 px-3 py-1 rounded-full hover:bg-gray-300 transition whitespace-nowrap';
            dlBtn.textContent = '⬇ Download';
            dlBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                downloadResultFile(itemPath);
            });
            row.appendChild(dlBtn);

            // In 'all' mode (costmap browsing), allow clicking to load costmap image
            if (browserMode === 'all' && (item.name.endsWith('.png') || item.name.endsWith('.jpg'))) {
                row.classList.add('cursor-pointer');
                row.addEventListener('click', (e) => {
                    // Only if we clicked the row, not the download button
                    if (e.target === dlBtn) return;

                    const costmapUrl = `/api/download-result-file?path=${itemPath}`; // We can use the static file serving or download endpoint?
                    // Actually, results are served statically
                    // path is e.g. results/subdir/file.png
                    // We need to convert path to URL
                    // Usually results are mapped? 
                    // app.py: @app.route('/results/<path:filename>') -> send_from_directory(RESULTS_FOLDER, filename)
                    // itemPath is relative to RESULTS_FOLDER?
                    // browseResults gets items from RESULTS_FOLDER.
                    // itemPath is relative to RESULTS_FOLDER.

                    const clickedCostmapUrl = `/results/${itemPath}`;
                    costmapDisplayArea.classList.remove('hidden');
                    saveCostmapImgBtn.disabled = true; // Loaded costmap, maybe not re-saveable directly?
                    const urls = deriveOverlayAndBwUrls(clickedCostmapUrl);
                    currentCostmapUrl = urls.bwUrl;
                    currentCostmapOverlayUrl = urls.overlayUrl;
                    currentCostmapOverlayFallbackUrl = urls.overlayFallbackUrl;
                    updateCostmapOverlayUI();

                    // Also load goals for the TIFF
                    // itemPath: tiff_folder/costmap/.../file.png
                    const parts = itemPath.split('/');
                    const tiffName = parts[0];
                    if (tiffName && tiffName !== selectedTiffName) {
                        setSelectedTiff(tiffName);
                    }
                    loadSavedGoals(tiffName);

                    hideServerMasksModal();
                });
            }

            serverMasksList.appendChild(row);
        });
    }

    function selectTiffFromBrowser(tiffFolderName) {
        setSelectedTiff(tiffFolderName);
        hideServerMasksModal();
    }

    async function handleRenameTiffFolder(oldName) {
        const newName = prompt(`Rename "${oldName}" to:`, oldName);
        if (!newName || newName === oldName) return;

        try {
            const response = await fetch(`${API_BASE_URL}/api/rename-tiff-folder`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ old_name: oldName, new_name: newName })
            });
            const data = await response.json();
            if (!response.ok) {
                alert('Rename failed: ' + (data.error || 'Unknown error'));
                return;
            }
            alert(`Renamed to "${newName}"`);
            // If this was the selected TIFF, update the selection
            if (selectedTiffName === oldName) {
                setSelectedTiff(newName);
            }
            // Refresh the current browser view
            browseResults(browserCurrentPath);
        } catch (error) {
            alert('Rename error: ' + error.message);
        }
    }

    function handleBrowseCostmaps() {
        // Browse costmaps for the selected TIFF (or all results if none selected)
        const tiffName = selectedTiffName;
        openBrowser('all', tiffName ? `${tiffName}/costmap` : '');
    }

    async function handleSaveCostmap() {
        if (!currentCostmapUrl) {
            alert("No costmap generated yet.");
            return;
        }

        const suffix = prompt('Enter a suffix for this costmap run (optional):');
        if (suffix === null) return;

        saveCostmapImgBtn.disabled = true;
        saveCostmapImgBtn.textContent = 'Saving...';

        try {
            const response = await fetch(`${API_BASE_URL}/api/save-costmap`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tiff_folder: currentTiffFolder || selectedTiffName,
                    costmap_url: currentCostmapUrl,
                    suffix: suffix.trim()
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to save costmap.");

            alert(`✅ ${data.message}`);
        } catch (e) {
            alert(`Error: ${e.message}`);
        } finally {
            saveCostmapImgBtn.disabled = false;
            saveCostmapImgBtn.textContent = '💾 Save';
        }
    }

    async function downloadResultFile(path) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/download-result-file`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path })
            });
            if (!response.ok) {
                const err = await response.json();
                alert('Download failed: ' + (err.error || 'Unknown error'));
                return;
            }
            const blob = await response.blob();
            const filename = path.split('/').pop();
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);
        } catch (error) {
            alert('Download error: ' + error.message);
        }
    }



    function handleBrowserUp() {
        if (!browserCurrentPath) return;
        const parts = browserCurrentPath.split('/');
        parts.pop();
        browseResults(parts.join('/'));
    }

    async function loadSelectedMasks() {
        if (!browserSelectedPath) return;

        loadSelectedMasksBtn.textContent = 'Loading...';
        loadSelectedMasksBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/api/load-masks-from-path`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: browserSelectedPath })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to load mask set.');

            displayResults(data);
            hideServerMasksModal();

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            loadSelectedMasksBtn.textContent = 'Load Selected Masks';
        }
    }

    async function handleSaveMasks() {
        if (!currentTiffFolder) {
            alert('No pipeline results to save. Run the pipeline first.');
            return;
        }

        const suffix = prompt('Enter a suffix for this save (optional):');
        if (suffix === null) return;  // User cancelled

        saveMasksBtn.disabled = true;
        saveMasksBtn.textContent = 'Saving...';

        try {
            const response = await fetch(`${API_BASE_URL}/api/save-masks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tiff_folder: currentTiffFolder,
                    suffix: suffix.trim()
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to save masks.');

            alert(`✅ ${data.message}`);
        } catch (error) {
            alert(`Save Error: ${error.message}`);
        } finally {
            saveMasksBtn.disabled = false;
            saveMasksBtn.textContent = '💾 Save';
        }
    }


    // --- Initial Setup & Event Listeners ---
    fetchTiffFiles();
    fetchDefaultConfig();
    updateGpuStatus();

    runPipelineBtn.addEventListener('click', handleRunPipeline);
    cancelPipelineBtn.addEventListener('click', handleCancelPipeline);
    generateCostmapBtn.addEventListener('click', handleGenerateCostmap);
    loadServerMasksBtn.addEventListener('click', showServerMasksModal);

    processPromptBtn.addEventListener('click', handleProcessPrompt);
    addClassBtn.addEventListener('click', handleAddClass);
    newClassInput.addEventListener('keypress', (e) => e.key === 'Enter' && handleAddClass());
    classListContainer.addEventListener('change', handleClassListChange);
    classListContainer.addEventListener('click', (e) => e.target.classList.contains('remove-class-btn') && handleClassListChange(e));


    confirmClassesBtn.addEventListener('click', () => {
        const config = collectFinalConfig();
        console.log("--- CONFIRMED CONFIGURATION ---", config);
        confirmClassesBtn.textContent = "Config Logged!";
        setTimeout(() => { confirmClassesBtn.textContent = "Confirm Classes & Log Config"; }, 2000);
    });

    document.getElementById("tiff-upload").addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`${API_BASE_URL}/api/upload-tiff`, {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            if (!response.ok) {
                alert(`Upload failed: ${result.error}`);
                return;
            }

            alert("✅ TIFF uploaded successfully!");

            // Auto-select the uploaded TIFF
            const folderName = result.filename.replace(/\.(tiff?|tif)$/i, '');
            setSelectedTiff(folderName);

        } catch (error) {
            console.error("Error uploading TIFF:", error);
            alert("Upload failed due to network error.");
        } finally {
            // reset input for next upload
            event.target.value = "";
        }
    });


    // --- Parameter Persistence ---
    async function loadParamsFromServer() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/params`);
            const data = await response.json();
            if (data.params) {
                applyParams(data.params);
                console.log(`Parameters loaded (${data.source})`);
            }
        } catch (err) {
            console.warn('Could not load saved parameters:', err.message);
        }
    }

    function applyParams(params) {
        for (const key in params) {
            const input = paramInputs[key];
            if (!input) continue;
            if (input.classList && input.classList.contains('toggle-btn-group')) {
                // Toggle button group
                input.querySelectorAll('button').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.value === params[key]);
                });
            } else if (input.tagName === 'SELECT') {
                // Only set if option exists
                const optionExists = Array.from(input.options).some(o => o.value === params[key]);
                if (optionExists) input.value = params[key];
            } else {
                input.value = params[key];
            }
        }
    }

    function collectParams() {
        const params = {};
        for (const key in paramInputs) {
            const input = paramInputs[key];
            if (input.classList && input.classList.contains('toggle-btn-group')) {
                const active = input.querySelector('button.active');
                params[key] = active ? active.dataset.value : 'max';
            } else {
                params[key] = input.value;
            }
        }
        return params;
    }

    async function saveParamsToServer() {
        try {
            await fetch(`${API_BASE_URL}/api/params`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ params: collectParams() })
            });
        } catch (err) {
            console.warn('Could not save parameters:', err.message);
        }
    }

    updateParamsBtn.addEventListener('click', async () => {
        console.log("Parameters Updated:", collectFinalConfig().params);
        await saveParamsToServer();
        updateParamsBtn.textContent = "Saved!";
        setTimeout(() => { updateParamsBtn.textContent = "Update Parameters"; }, 2000);
    });

    const resetParamsBtn = document.getElementById('reset-params-btn');
    resetParamsBtn.addEventListener('click', async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/params`, { method: 'DELETE' });
            const data = await response.json();
            if (data.params) {
                applyParams(data.params);
                renderClasses();
            }
            resetParamsBtn.textContent = "Reset!";
            setTimeout(() => { resetParamsBtn.textContent = "\u21ba Reset to Default"; }, 2000);
        } catch (err) {
            alert('Reset failed: ' + err.message);
        }
    });

    // Load saved parameters on startup
    loadParamsFromServer();

    showGpuModalBtn.addEventListener('click', showGpuStatus);
    closeGpuModalBtn.addEventListener('click', hideGpuStatus);
    showConsoleBtn.addEventListener('click', showConsole);
    closeConsoleBtn.addEventListener('click', hideConsole);
    showCostmapBtn.addEventListener('click', showCostmapModal);
    closeCostmapBtn.addEventListener('click', hideCostmapModal);
    editCostmapBtn.addEventListener('click', handleEditCostmap);
    saveCostmapBtn.addEventListener('click', handleSaveCostmapCode);
    restoreCostmapBtn.addEventListener('click', handleRestoreCostmap);
    showWorldMapBtn.addEventListener('click', showWorldMap);
    closeWorldMapBtn.addEventListener('click', hideWorldMap);
    saveMapAreaBtn.addEventListener('click', handleSaveMapArea);
    paramInputs.semseg_combine_method.addEventListener('click', handleToggleButtons);
    paramInputs.refiner_combine_method.addEventListener('click', handleToggleButtons);
    paramInputs.areal_threshold.addEventListener('change', renderClasses);
    paramInputs.linear_threshold.addEventListener('change', renderClasses);
    tiffPreviewContainer.addEventListener('click', (e) => {
        if (e.target.tagName === 'IMG') { openImageViewer(e.target.src); }
    });
    closeImageViewerBtn.addEventListener('click', closeImageViewer);

    imageViewerModal.addEventListener('wheel', onWheel, { passive: false });
    imageViewerModal.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('mousemove', onMouseMove);

    resultsMasksContainer.addEventListener('click', (e) => {
        const card = e.target.closest('[data-class-name]');
        if (card) showClassDetailModal(card.dataset.className);
    });
    closeClassDetailBtn.addEventListener('click', hideClassDetailModal);
    editMaskBtn.addEventListener('click', async () => {
        const className = editMaskBtn.dataset.className;
        if (!className || !currentResultData) return;
        const maskUrl = (currentResultData.refined_masks || currentResultData.local_masks || {})[className];
        if (!maskUrl) {
            alert(`Mask for "${className}" is not available.`);
            return;
        }
        hideClassDetailModal();
        await openPixelEditor({
            imageType: 'mask',
            imageUrl: maskUrl,
            className,
            title: `Edit Mask: ${className}`
        });
    });

    editCostmapPixelsBtn.addEventListener('click', async () => {
        if (!currentCostmapUrl) {
            alert('Generate or load a costmap first.');
            return;
        }
        await openPixelEditor({
            imageType: 'costmap',
            imageUrl: currentCostmapUrl,
            colorUrl: currentCostmapOverlayFallbackUrl,
            title: 'Edit Costmap Pixels'
        });
    });

    closePixelEditorBtn.addEventListener('click', closePixelEditor);
    pixelToolButtons.addEventListener('click', (event) => {
        const button = event.target.closest('[data-tool]');
        if (!button) return;
        setPixelEditorTool(button.dataset.tool);
    });
    pixelBrushSizeInput.addEventListener('input', () => {
        pixelEditorState.brushSize = Math.max(1, Math.min(15, Number(pixelBrushSizeInput.value) || 1));
        updatePixelEditorSelectionUI();
    });
    pixelClosePolygonBtn.addEventListener('click', finalizePolygonSelection);
    pixelZoomInBtn.addEventListener('click', () => zoomPixelEditor(1.5));
    pixelZoomOutBtn.addEventListener('click', () => zoomPixelEditor(1 / 1.5));
    pixelZoomResetBtn.addEventListener('click', resetPixelEditorZoom);
    pixelClearSelectionBtn.addEventListener('click', clearPixelSelection);
    pixelUndoSelectionBtn.addEventListener('click', undoLastPixelSelection);
    pixelApplyValueBtn.addEventListener('click', applyPixelValueToSelection);
    pixelSaveBtn.addEventListener('click', savePixelEditorChanges);

    pixelEditorCanvas.addEventListener('mousedown', startPixelSelection);
    pixelEditorCanvas.addEventListener('mousemove', movePixelSelection);
    pixelEditorCanvas.addEventListener('mouseleave', () => {
        if (pixelEditorState.tool !== 'polygon') return;
        pixelEditorState.hoverPoint = null;
        renderPixelEditorCanvas();
    });
    pixelEditorCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
    pixelEditorCanvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        zoomPixelEditor(e.deltaY < 0 ? 1.2 : 1 / 1.2);
    }, { passive: false });
    window.addEventListener('mouseup', endPixelSelection);
    window.addEventListener('mouseleave', endPixelSelection);

    semsegSlider.addEventListener('input', (e) => {
        detailSemsegOverlayMask.style.opacity = e.target.value / 100;
    });

    refinedSlider.addEventListener('input', (e) => {
        detailRefinedOverlayMask.style.opacity = e.target.value / 100;
    });

    planOverMapBtn.addEventListener('click', showPlannerModal);
    downloadCostmapBtn.addEventListener('click', handleDownloadCostmap);
    closePlannerBtn.addEventListener('click', hidePlannerModal);
    plannerCanvas.addEventListener('click', handlePlannerClick);
    plannerCanvas.addEventListener('mousemove', handlePlannerMouseMove);
    plannerCanvas.addEventListener('mouseleave', handlePlannerMouseLeave);
    selectStartBtn.addEventListener('click', () => {
        plannerState.selectingStart = true;
        plannerState.selectingEnd = false;
        updatePlannerUI();
    });
    selectEndBtn.addEventListener('click', () => {
        plannerState.selectingStart = false;
        plannerState.selectingEnd = true;
        updatePlannerUI();
    });
    planPathBtn.addEventListener('click', handlePlanPath);
    if (downloadPlanBtn) {
        downloadPlanBtn.addEventListener('click', handleDownloadPlan);

        // Download dropdown (options)
        loadDownloadOptionsIntoUI();
        ['dl-rgb', 'dl-white', 'dl-meta', 'dl-costmap-files', 'dl-costmap-tiff', 'dl-orig-tiff', 'dl-masks', 'dl-out-zip', 'dl-out-files'].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', saveDownloadOptionsFromUI);
        });

        const optsBtn = document.getElementById('download-plan-options-btn');
        if (optsBtn) {
            optsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const dropdown = document.getElementById('download-plan-dropdown');
                const isOpen = dropdown && !dropdown.classList.contains('hidden');
                setDownloadDropdownOpen(!isOpen);
            });
        }

        document.addEventListener('click', (e) => {
            const wrapper = document.getElementById('download-plan-wrapper');
            if (!wrapper) return;
            if (!wrapper.contains(e.target)) setDownloadDropdownOpen(false);
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') setDownloadDropdownOpen(false);
        });
    }
    clearPlanBtn.addEventListener('click', resetPlannerState);

    if (downloadZipNameInput) {
        const savedZipName = localStorage.getItem('downloadZipName') || '';
        if (savedZipName) downloadZipNameInput.value = savedZipName;
        downloadZipNameInput.addEventListener('input', () => {
            localStorage.setItem('downloadZipName', downloadZipNameInput.value);
        });
    }

    if (clearTempDownloadsBtn) {
        clearTempDownloadsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            handleClearTempDownloads();
        });
    }

    if (costmapOverlaySlider) {
        const saved = localStorage.getItem('costmapOverlayOpacity');
        if (saved) costmapOverlaySlider.value = saved;
        costmapOverlaySlider.addEventListener('input', () => {
            localStorage.setItem('costmapOverlayOpacity', costmapOverlaySlider.value);
            setImgOpacityFromSlider(costmapOverlayImg, costmapOverlaySlider);
        });
        setImgOpacityFromSlider(costmapOverlayImg, costmapOverlaySlider);
    }

    if (plannerOverlaySlider) {
        const saved = localStorage.getItem('plannerOverlayOpacity');
        if (saved) plannerOverlaySlider.value = saved;
        plannerOverlaySlider.addEventListener('input', () => {
            localStorage.setItem('plannerOverlayOpacity', plannerOverlaySlider.value);
            setImgOpacityFromSlider(plannerImgOverlay, plannerOverlaySlider);
        });
        setImgOpacityFromSlider(plannerImgOverlay, plannerOverlaySlider);
    }

    closeServerMasksBtn.addEventListener('click', hideServerMasksModal);
    browserUpBtn.addEventListener('click', handleBrowserUp);
    loadSelectedMasksBtn.addEventListener('click', loadSelectedMasks);
    saveMasksBtn.addEventListener('click', handleSaveMasks);
    browseTiffsBtn.addEventListener('click', () => openBrowser('tiff'));
    browserSearchInput.addEventListener('input', handleSearchInput);
    browseCostmapsBtn.addEventListener('click', handleBrowseCostmaps);
    saveCostmapImgBtn.addEventListener('click', handleSaveCostmap);
    // --- Saved Goals Logic ---
    async function loadSavedGoals(tiffName) {
        if (!tiffName) return;
        try {
            const response = await fetch(`${API_BASE_URL}/api/goals?tiff_name=${encodeURIComponent(tiffName)}`);
            const data = await response.json();

            savedGoalsList.innerHTML = '';
            if (data.goals && data.goals.length > 0) {
                data.goals.forEach(goal => {
                    const div = document.createElement('div');
                    div.className = "flex justify-between items-center bg-white p-2 rounded shadow-sm border border-gray-100 group hover:bg-gray-50 transition";

                    // Name/Date Container
                    const infoDiv = document.createElement('div');
                    infoDiv.className = "flex-1 cursor-text select-none";
                    infoDiv.title = "Double-click to rename";

                    const nameEl = document.createElement('div');
                    nameEl.className = "font-medium text-gray-700";
                    nameEl.textContent = goal.name;

                    const dateEl = document.createElement('div');
                    dateEl.className = "text-xs text-gray-400";
                    dateEl.textContent = new Date(goal.timestamp * 1000).toLocaleString();

                    infoDiv.appendChild(nameEl);
                    infoDiv.appendChild(dateEl);

                    // Rename Logic
                    infoDiv.addEventListener('dblclick', () => {
                        const input = document.createElement('input');
                        input.type = 'text';
                        input.value = goal.name;
                        input.className = "border rounded px-1 py-0.5 text-sm w-full outline-blue-500 shadow-sm";

                        const savedName = goal.name;
                        let isSaving = false;

                        async function saveRename() {
                            if (isSaving) return;
                            isSaving = true;
                            const newName = input.value.trim();
                            if (newName && newName !== savedName) {
                                try {
                                    const res = await fetch(`${API_BASE_URL}/api/goals/${goal.id}`, {
                                        method: 'PUT',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ tiff_name: tiffName, name: newName })
                                    });
                                    if (res.ok) {
                                        goal.name = newName;
                                        nameEl.textContent = newName;
                                        infoDiv.replaceChild(nameEl, input);
                                    } else {
                                        alert("Failed to rename.");
                                        infoDiv.replaceChild(nameEl, input);
                                    }
                                } catch (e) { console.error(e); infoDiv.replaceChild(nameEl, input); }
                            } else {
                                if (infoDiv.contains(input)) infoDiv.replaceChild(nameEl, input);
                            }
                        }

                        input.addEventListener('blur', saveRename);
                        input.addEventListener('keydown', (e) => {
                            if (e.key === 'Enter') { input.blur(); }
                        });

                        infoDiv.replaceChild(input, nameEl);
                        input.focus();
                    });

                    div.appendChild(infoDiv);

                    // Actions Container
                    const actionsDiv = document.createElement('div');
                    actionsDiv.className = "flex gap-2 ml-2";

                    // Load Button
                    const loadBtn = document.createElement('button');
                    loadBtn.className = "text-xs bg-green-100 text-green-700 font-bold px-2 py-1 rounded hover:bg-green-200 border border-green-200 transition shadow-sm";
                    loadBtn.textContent = 'Load';
                    loadBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        if (goal.start && goal.end) {
                            plannerState.startPoint = goal.start;
                            plannerState.endPoint = goal.end;
                            plannerState.displayDimensions = { width: plannerCanvas.width, height: plannerCanvas.height };
                            plannerState.path = null;
                            plannerState.originalDimensions = null;
                            if (downloadPlanBtn) downloadPlanBtn.disabled = true;
                            const optsBtn = document.getElementById('download-plan-options-btn');
                            if (optsBtn) optsBtn.disabled = true;
                            setDownloadDropdownOpen(false);
                            // Reset selection modes and update UI
                            plannerState.selectingStart = false;
                            plannerState.selectingEnd = false;

                            redrawAllPlannerElements();
                            updatePlannerUI();

                            // Enable planning since points are set
                            planPathBtn.disabled = false;

                            // Visual feedback
                            const originalText = loadBtn.textContent;
                            loadBtn.textContent = "Loaded!";
                            loadBtn.classList.remove('bg-green-100', 'text-green-700');
                            loadBtn.classList.add('bg-gray-800', 'text-white');
                            setTimeout(() => {
                                loadBtn.textContent = originalText;
                                loadBtn.classList.add('bg-green-100', 'text-green-700');
                                loadBtn.classList.remove('bg-gray-800', 'text-white');
                            }, 1000);
                        }
                    });

                    // Delete Button
                    const deleteBtn = document.createElement('button');
                    deleteBtn.className = "text-xs bg-red-50 text-red-600 font-bold px-2 py-1 rounded hover:bg-red-100 border border-red-200 transition opacity-100 md:opacity-0 group-hover:opacity-100 shadow-sm";
                    deleteBtn.textContent = '✕';
                    deleteBtn.title = "Delete Goal";
                    deleteBtn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        if (confirm(`Are you sure you want to delete "${goal.name}"?`)) {
                            try {
                                const res = await fetch(`${API_BASE_URL}/api/goals/${goal.id}?tiff_name=${encodeURIComponent(tiffName)}`, {
                                    method: 'DELETE'
                                });
                                if (res.ok) {
                                    loadSavedGoals(tiffName);
                                } else {
                                    alert("Failed to delete goal.");
                                }
                            } catch (e) { console.error(e); }
                        }
                    });

                    actionsDiv.appendChild(loadBtn);
                    actionsDiv.appendChild(deleteBtn);
                    div.appendChild(actionsDiv);

                    savedGoalsList.appendChild(div);
                });
            } else {
                savedGoalsList.innerHTML = '<span class="text-sm text-gray-400 italic">No goals saved.</span>';
            }
        } catch (err) {
            console.error("Error loading goals:", err);
            savedGoalsList.innerHTML = '<span class="text-sm text-red-400 italic">Error loading goals.</span>';
        }
    }

    saveCurrentGoalBtn.addEventListener('click', async () => {
        if (!plannerState.startPoint || !plannerState.endPoint) {
            alert("Please set both Start and End points on the costmap first.");
            return;
        }

        const tiffName = currentTiffFolder || selectedTiffName;
        if (!tiffName) {
            alert("No TIFF selected.");
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/api/goals`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tiff_name: tiffName,
                    start: plannerState.startPoint,
                    end: plannerState.endPoint
                })
            });
            const data = await response.json();
            if (data.goal) {
                loadSavedGoals(tiffName); // Refresh list
                const originalText = saveCurrentGoalBtn.textContent;
                saveCurrentGoalBtn.textContent = "Saved!";
                setTimeout(() => { saveCurrentGoalBtn.textContent = originalText; }, 2000);
            } else {
                alert("Failed to save goal: " + (data.error || "Unknown error"));
            }
        } catch (err) {
            console.error("Error saving goal:", err);
            alert("Error saving goal.");
        }
    });

});
