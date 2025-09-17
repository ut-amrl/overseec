document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const promptInput = document.getElementById('prompt-input'),
        processPromptBtn = document.getElementById('process-prompt-btn'),
        tiffSelect = document.getElementById('tiff-select'),
        classListContainer = document.getElementById('class-list'),
        newClassInput = document.getElementById('new-class-input'),
        addClassBtn = document.getElementById('add-class-btn'),
        runPipelineBtn = document.getElementById('run-pipeline-btn'),
        generateCostmapBtn = document.getElementById('generate-costmap-btn'),
        loadServerMasksBtn = document.getElementById('load-server-masks-btn'),
        updateParamsBtn = document.getElementById('update-params-btn'),
        confirmClassesBtn = document.getElementById('confirm-classes-btn'),
        tiffPreviewContainer = document.getElementById('tiff-preview-container'),
        resultsArea = document.getElementById('results-area'),
        resultsMasksContainer = document.getElementById('results-masks-container'),
        costmapDisplayArea = document.getElementById('costmap-display-area'),
        costmapImageContainer = document.getElementById('costmap-image-container'),
        resultsPlaceholder = document.getElementById('results-placeholder'),
        loader = document.getElementById('loader'),
        cancelPipelineBtn = document.getElementById('cancel-pipeline-btn');

    const paramInputs = {
        areal_threshold: document.getElementById('areal-threshold'),
        linear_threshold: document.getElementById('linear-threshold'),
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
        loadSelectedMasksBtn = document.getElementById('load-selected-masks-btn');

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
        classDetailTitle = document.getElementById('class-detail-title'),
        detailRgbImg = document.getElementById('detail-rgb'),
        detailSemsegOverlayBase = document.getElementById('detail-semseg-overlay-base'),
        detailSemsegOverlayMask = document.getElementById('detail-semseg-overlay-mask'),
        semsegSlider = document.getElementById('semseg-slider'),
        detailRefinedOverlayBase = document.getElementById('detail-refined-overlay-base'),
        detailRefinedOverlayMask = document.getElementById('detail-refined-overlay-mask'),
        refinedSlider = document.getElementById('refined-slider');

    const planOverMapBtn = document.getElementById('plan-over-map-btn'),
        plannerModal = document.getElementById('planner-modal'),
        closePlannerBtn = document.getElementById('close-planner-btn'),
        plannerImgBg = document.getElementById('planner-img-bg'),
        plannerCanvas = document.getElementById('planner-canvas'),
        selectStartBtn = document.getElementById('select-start-btn'),
        selectEndBtn = document.getElementById('select-end-btn'),
        planPathBtn = document.getElementById('plan-path-btn'),
        clearPlanBtn = document.getElementById('clear-plan-btn'),
        plannerStatus = document.getElementById('planner-status'),
        plannerCoords = document.getElementById('planner-coords');

    const API_BASE_URL = 'http://127.0.0.1:5002';
    let currentClasses = [], worldMap, drawnItems, capturedBounds, progressInterval, gpuPollInterval, consolePollInterval;
    let panzoomState = { scale: 1, translateX: 0, translateY: 0, isPanning: false, startX: 0, startY: 0 };
    let currentResultData = null; 
    let pipelinePollInterval = null; 
    let currentTaskId = null;
    let currentCostmapUrl = null;
    let plannerState = {};
    let plannerResizeObserver = null;
    let selectedRunId = null;
    let lastKnownCoords = null;

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
        try {
            const response = await fetch(`${API_BASE_URL}/api/get-tiff-files`);
            const data = await response.json();
            const currentlySelected = tiffSelect.value;
            tiffSelect.innerHTML = '<option value="">-- Select a file --</option>';
            data.files.forEach(file => tiffSelect.add(new Option(file, file)));
            if (data.files.includes(currentlySelected)) tiffSelect.value = currentlySelected;
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
        const hasTiff = !!tiffSelect.value;
        const hasClasses = currentClasses.length > 0;
        const hasResults = !!currentResultData;

        loadServerMasksBtn.disabled = !hasTiff;
        runPipelineBtn.disabled = !(hasTiff && hasClasses);
        generateCostmapBtn.disabled = !hasResults;
    }

    function displayResults(data) {
        currentResultData = data; 
        resultsArea.classList.remove('hidden');
        resultsPlaceholder.classList.add('hidden');
        costmapDisplayArea.classList.add('hidden');
        costmapImageContainer.innerHTML = '';
        resultsMasksContainer.innerHTML = '';

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
            const finalUrl = maskUrl.startsWith('data:') ? maskUrl : `${API_BASE_URL}${maskUrl}`;
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
        } catch (error) {
            console.error("Prompt processing error:", error);
            alert(`Error: ${error.message}`);
        } finally {
            processPromptBtn.textContent = 'Get Classes from Prompt';
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

    async function handleTiffSelect(e) {
        const filename = e.target.value;
        updateRunButtonState();
        if (!filename) {
            tiffPreviewContainer.innerHTML = '<p class="text-gray-500">Select a TIFF file to see a preview.</p>';
            loadServerMasksBtn.disabled = true;
            return;
        }
        loadServerMasksBtn.disabled = false;

        tiffPreviewContainer.innerHTML = `<div class="flex items-center"><div class="loader !w-6 !h-6"></div><p class="ml-3 text-gray-600">Generating preview...</p></div>`;
        try {
            const response = await fetch(`${API_BASE_URL}/api/generate-preview`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename }) });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to generate preview.');
            tiffPreviewContainer.innerHTML = `<img src="${API_BASE_URL}${data.preview_url}" alt="Preview of ${filename}" class="max-w-full max-h-[400px] rounded-lg shadow-md">`;
        } catch (error) {
            console.error("Preview generation error:", error);
            tiffPreviewContainer.innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
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
            tiff_file: tiffSelect.value,
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
        } catch(error) {
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
                body: JSON.stringify({ refined_masks: masks_to_use })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to generate costmap.");
            
            currentCostmapUrl = data.costmap_url;
            costmapImageContainer.innerHTML = `<img src="${API_BASE_URL}${data.costmap_url}" alt="Final Costmap" class="max-w-full max-h-[400px] rounded-lg shadow-md">`;
            costmapDisplayArea.classList.remove('hidden');

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

            if(data.source.startsWith("mock")) {
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

            hljs.highlightElement(defaultCostmapCode);
            hljs.highlightElement(generatedCostmapCode);

            generatedCostmapCode.setAttribute('contenteditable', 'false');
            generatedCostmapCode.classList.remove('ring-2', 'ring-blue-500');
            saveCostmapBtn.disabled = true;
            editCostmapBtn.disabled = false;
            
            costmapModal.classList.remove('hidden');
            costmapModal.classList.add('flex');
        } catch(error) {
            alert(`Error: ${error.message}`);
        }
    }

    function hideCostmapModal() {
        costmapModal.classList.add('hidden');
        costmapModal.classList.remove('flex');
    }

    function handleEditCostmap() {
        generatedCostmapCode.setAttribute('contenteditable', 'true');
        generatedCostmapCode.classList.add('ring-2', 'ring-blue-500');
        generatedCostmapCode.focus();
        saveCostmapBtn.disabled = false;
        editCostmapBtn.disabled = true;
    }

    async function handleSaveCostmap() {
        const newCode = generatedCostmapCode.textContent;
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
        } catch(error) {
            alert(`Save failed: ${error.message}`);
        } finally {
            saveCostmapBtn.textContent = 'Save';
        }
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

        worldMap = L.map('world-map').setView([30.2672, -97.7431], 13);
        L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',{ maxZoom: 20, subdomains:['mt0','mt1','mt2','mt3'], attribution: '&copy; Google' }).addTo(worldMap);
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

    function onMouseDown(e) { if (!imageViewerModal.classList.contains('hidden')) { e.preventDefault(); panzoomState.isPanning = true; panzoomState.startX = e.clientX - panzoomState.translateX; panzoomState.startY = e.clientY - panzoomState.translateY; }}
    function onMouseUp() { panzoomState.isPanning = false; }
    function onMouseMove(e) { if (panzoomState.isPanning) { e.preventDefault(); panzoomState.translateX = e.clientX - panzoomState.startX; panzoomState.translateY = e.clientY - panzoomState.startY; updateTransform(); }}

    function showClassDetailModal(className) {
        if (!currentResultData) return;

        const rgbPreviewImg = tiffPreviewContainer.querySelector('img');
        if (!rgbPreviewImg) { alert("Original image preview not available."); return; }
        const rgbUrl = rgbPreviewImg.src;

        const semsegUrl = currentResultData.semantic_masks ? currentResultData.semantic_masks[className] : null;
        const finalUrl = (currentResultData.refined_masks || currentResultData.local_masks)[className];
        
        if (!finalUrl) { alert(`Mask for class "${className}" is not available.`); return; }


        classDetailTitle.textContent = `Details for: ${className}`;
        detailRgbImg.src = rgbUrl;
        
        detailSemsegOverlayBase.src = rgbUrl;
        detailSemsegOverlayMask.src = semsegUrl ? `${API_BASE_URL}${semsegUrl}` : "https://placehold.co/600x400/2d3748/9ca3af?text=N/A";
        
        detailRefinedOverlayBase.src = rgbUrl;
        detailRefinedOverlayMask.src = finalUrl.startsWith('data:') ? finalUrl : `${API_BASE_URL}${finalUrl}`;

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
        
        plannerImgBg.src = rgbPreviewImg.src;
        
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
    }

    function hidePlannerModal() {
        plannerModal.classList.add('hidden');
        plannerModal.classList.remove('flex');
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

        planPathBtn.disabled = true;
        plannerStatus.textContent = 'Planning...';

        try {
            const response = await fetch(`${API_BASE_URL}/api/plan-path`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    costmap_url: currentCostmapUrl,
                    tiff_filename: tiffSelect.value,
                    start_point: plannerState.startPoint,
                    goal_point: plannerState.endPoint,
                    display_dimensions: plannerState.displayDimensions
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Planning failed.");
            
            plannerState.path = data.path; 
            plannerState.originalDimensions = data.original_dimensions; // [MODIFIED] Store original dimensions
            redrawAllPlannerElements();
            plannerStatus.textContent = `Path found with length: ${data.path.length}`;

        } catch(error) {
            alert(`Planning Error: ${error.message}`);
        } finally {
            planPathBtn.disabled = false;
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
        
        lastKnownCoords = {x, y};
        plannerCoords.textContent = `Mouse: (${x}, ${y})`;
    }

    function handlePlannerMouseLeave() {
        if(lastKnownCoords) {
            plannerCoords.textContent = `Last: (${lastKnownCoords.x}, ${lastKnownCoords.y})`;
        }
    }

    async function showServerMasksModal() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/get-result-folders`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to get result folders.");

            serverMasksList.innerHTML = '';
            if (data.folders.length === 0) {
                serverMasksList.innerHTML = '<p class="text-gray-500">No previous results found on the server.</p>';
            } else {
                data.folders.forEach(folder => {
                    const item = document.createElement('div');
                    item.className = 'p-3 border rounded-lg cursor-pointer hover:bg-gray-100';
                    item.textContent = folder;
                    item.dataset.runId = folder;
                    serverMasksList.appendChild(item);
                });
            }
            
            selectedRunId = null;
            loadSelectedMasksBtn.disabled = true;
            serverMasksModal.classList.remove('hidden');
            serverMasksModal.classList.add('flex');
        } catch(error) {
            alert(`Error: ${error.message}`);
        }
    }

    function hideServerMasksModal() {
        serverMasksModal.classList.add('hidden');
        serverMasksModal.classList.remove('flex');
    }

    function handleServerMaskSelection(e) {
        const target = e.target.closest('[data-run-id]');
        if (!target) return;
        
        serverMasksList.querySelectorAll('.bg-blue-200').forEach(el => el.classList.remove('bg-blue-200', 'border-blue-500'));
        
        target.classList.add('bg-blue-200', 'border-blue-500');
        
        selectedRunId = target.dataset.runId;
        loadSelectedMasksBtn.disabled = false;
    }

    async function loadSelectedMasks() {
        if (!selectedRunId) return;
        
        loadSelectedMasksBtn.textContent = 'Loading...';
        loadSelectedMasksBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/api/load-masks-from-folder/${selectedRunId}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to load mask set.');
            
            displayResults(data);
            hideServerMasksModal();

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            loadSelectedMasksBtn.textContent = 'Load Selected Set';
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

    tiffSelect.addEventListener('change', handleTiffSelect);
    
    confirmClassesBtn.addEventListener('click', () => {
        const config = collectFinalConfig();
        console.log("--- CONFIRMED CONFIGURATION ---", config);
        confirmClassesBtn.textContent = "Config Logged!";
        setTimeout(() => { confirmClassesBtn.textContent = "Confirm Classes & Log Config"; }, 2000);
    });
    
    updateParamsBtn.addEventListener('click', () => {
        console.log("Parameters Updated:", collectFinalConfig().params);
        updateParamsBtn.textContent = "Updated!";
        setTimeout(() => { updateParamsBtn.textContent = "Update Parameters"; }, 2000);
    });
    
    showGpuModalBtn.addEventListener('click', showGpuStatus);
    closeGpuModalBtn.addEventListener('click', hideGpuStatus);
    showConsoleBtn.addEventListener('click', showConsole);
    closeConsoleBtn.addEventListener('click', hideConsole);
    showCostmapBtn.addEventListener('click', showCostmapModal);
    closeCostmapBtn.addEventListener('click', hideCostmapModal);
    editCostmapBtn.addEventListener('click', handleEditCostmap);
    saveCostmapBtn.addEventListener('click', handleSaveCostmap);
    restoreCostmapBtn.addEventListener('click', handleRestoreCostmap);
    showWorldMapBtn.addEventListener('click', showWorldMap);
    closeWorldMapBtn.addEventListener('click', hideWorldMap);
    saveMapAreaBtn.addEventListener('click', handleSaveMapArea);
    paramInputs.semseg_combine_method.addEventListener('click', handleToggleButtons);
    paramInputs.refiner_combine_method.addEventListener('click', handleToggleButtons);
    paramInputs.areal_threshold.addEventListener('change', renderClasses);
    paramInputs.linear_threshold.addEventListener('change', renderClasses);
    tiffPreviewContainer.addEventListener('click', (e) => {
        if(e.target.tagName === 'IMG') { openImageViewer(e.target.src); }
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

    semsegSlider.addEventListener('input', (e) => {
        detailSemsegOverlayMask.style.opacity = e.target.value / 100;
    });

    refinedSlider.addEventListener('input', (e) => {
        detailRefinedOverlayMask.style.opacity = e.target.value / 100;
    });

    planOverMapBtn.addEventListener('click', showPlannerModal);
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
    clearPlanBtn.addEventListener('click', resetPlannerState);

    closeServerMasksBtn.addEventListener('click', hideServerMasksModal);
    serverMasksList.addEventListener('click', handleServerMaskSelection);
    loadSelectedMasksBtn.addEventListener('click', loadSelectedMasks);
});
