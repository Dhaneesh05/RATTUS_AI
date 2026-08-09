document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const btnPowerToggle = document.getElementById("btnPowerToggle");
    const powerBtnText = document.getElementById("powerBtnText");

    const rodentCountVal = document.getElementById("rodentCountVal");
    const rodentPeakVal = document.getElementById("rodentPeakVal");
    const fpsCounter = document.getElementById("fpsCounter");
    const videoOverlayBadge = document.getElementById("videoOverlayBadge");
    const videoOverlayText = document.getElementById("videoOverlayText");
    const activeModelBadge = document.getElementById("activeModelBadge");

    const riskScoreVal = document.getElementById("riskScoreVal");
    const riskLevelBadge = document.getElementById("riskLevelBadge");
    const gaugeFill = document.getElementById("gaugeFill");

    const rainfallVal = document.getElementById("rainfallVal");
    const btnRefreshWeather = document.getElementById("btnRefreshWeather");
    const weatherTempBadge = document.getElementById("weatherTempBadge");
    const weatherLocationSelect = document.getElementById("weatherLocationSelect");
    const weatherSyncStatus = document.getElementById("weatherSyncStatus");

    const waterLevelSlider = document.getElementById("waterLevelSlider");
    const waterLevelValText = document.getElementById("waterLevelValText");

    const btnSourceWebcam = document.getElementById("btnSourceWebcam");
    const btnSourceIpCam = document.getElementById("btnSourceIpCam");
    const btnSourceVideoFile = document.getElementById("btnSourceVideoFile");

    const webcamRow = document.getElementById("webcamRow");
    const ipCamRow = document.getElementById("ipCamRow");
    const videoFileRow = document.getElementById("videoFileRow");

    const webcamIndexSelect = document.getElementById("webcamIndex");
    const ipCamUrlInput = document.getElementById("ipCamUrl");
    const btnApplyIpCam = document.getElementById("btnApplyIpCam");

    const videoFileInput = document.getElementById("videoFileInput");
    const btnUploadVideo = document.getElementById("btnUploadVideo");
    const videoUploadStatus = document.getElementById("videoUploadStatus");

    const modelWeightsSelect = document.getElementById("modelWeightsSelect");
    const confSlider = document.getElementById("confSlider");
    const confValText = document.getElementById("confValText");
    const chkHumanFP = document.getElementById("chkHumanFP");

    const chkManualOverride = document.getElementById("chkManualOverride");
    const manualCountRow = document.getElementById("manualCountRow");
    const manualCountInput = document.getElementById("manualCountInput");

    const synergyBanner = document.getElementById("synergyBanner");
    const expRodent = document.getElementById("expRodent");
    const expRain = document.getElementById("expRain");
    const expWaterLevel = document.getElementById("expWaterLevel");
    const expHist = document.getElementById("expHist");
    const municipalActionText = document.getElementById("municipalActionText");
    const citizenActionText = document.getElementById("citizenActionText");

    // Local State
    let isPowerOn = true;
    let currentStats = {
        current_count: 0,
        max_session_count: 0,
        fps: 0,
    };
    let rainfallMm = 18.0;
    let waterLevelScore = 90.0; // Suggested based on default rainfallMm (18 * 5 = 90)
    let historicalRiskScore = 45.0;
    let isManualOverride = false;
    let manualCount = 3;

    // Fetch initial config
    async function loadConfig() {
        try {
            const resp = await fetch("/api/config");
            if (!resp.ok) return;
            const data = await resp.json();
            
            confSlider.value = data.conf_threshold || 0.50;
            confValText.textContent = `${Math.round(data.conf_threshold * 100)}%`;
            chkHumanFP.checked = data.suppress_human_fp !== undefined ? data.suppress_human_fp : true;
            
            if (data.weights_path) {
                modelWeightsSelect.value = data.weights_path;
                activeModelBadge.innerHTML = `Model: <strong>${data.weights_path.split(/[\\/]/).pop()}</strong>`;
            }

            if (data.source_type === "off") {
                setPowerState(false);
            } else {
                setPowerState(true);
                updateSourceTab(data.source_type);
            }
        } catch (e) {
            console.error("Config fetch failed:", e);
        }
    }

    // Power Toggle State
    function setPowerState(on) {
        isPowerOn = on;
        if (on) {
            btnPowerToggle.className = "btn-power active";
            powerBtnText.textContent = "Camera ON";
        } else {
            btnPowerToggle.className = "btn-power off";
            powerBtnText.textContent = "Camera OFF";
        }
    }

    btnPowerToggle.addEventListener("click", () => {
        const nextState = !isPowerOn;
        setPowerState(nextState);
        if (nextState) {
            const activeTab = document.querySelector(".btn-toggle-group .btn-toggle.active");
            if (activeTab === btnSourceIpCam) {
                updateBackendConfig({ source_type: "ip_cam", camera_url: ipCamUrlInput.value });
            } else if (activeTab === btnSourceVideoFile) {
                updateBackendConfig({ source_type: "video_file" });
            } else {
                updateBackendConfig({ source_type: "webcam", camera_index: parseInt(webcamIndexSelect.value) });
            }
        } else {
            updateBackendConfig({ source_type: "off" });
        }
    });

    // Switch Tab UI
    function updateSourceTab(sourceType) {
        btnSourceWebcam.classList.remove("active");
        btnSourceIpCam.classList.remove("active");
        btnSourceVideoFile.classList.remove("active");

        webcamRow.classList.add("hidden");
        ipCamRow.classList.add("hidden");
        videoFileRow.classList.add("hidden");

        if (sourceType === "ip_cam") {
            btnSourceIpCam.classList.add("active");
            ipCamRow.classList.remove("hidden");
        } else if (sourceType === "video_file") {
            btnSourceVideoFile.classList.add("active");
            videoFileRow.classList.remove("hidden");
        } else {
            btnSourceWebcam.classList.add("active");
            webcamRow.classList.remove("hidden");
        }
    }

    // Update backend config
    async function updateBackendConfig(payload) {
        try {
            const resp = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (resp.ok) {
                const data = await resp.json();
                if (payload.weights_path) {
                    activeModelBadge.innerHTML = `Model: <strong>${payload.weights_path.split(/[\\/]/).pop()}</strong>`;
                }
            }
        } catch (e) {
            console.error("Failed to update config:", e);
        }
    }

    // Video File Upload
    btnUploadVideo.addEventListener("click", async () => {
        const file = videoFileInput.files[0];
        if (!file) {
            videoUploadStatus.textContent = "❌ Please select a video file first.";
            videoUploadStatus.style.color = "#ef4444";
            return;
        }

        videoUploadStatus.textContent = "⏳ Uploading video file...";
        videoUploadStatus.style.color = "#38bdf8";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/api/upload_video", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const res = await resp.json();

            videoUploadStatus.textContent = `✅ Playing: ${res.filename}`;
            videoUploadStatus.style.color = "#10b981";
            setPowerState(true);
            updateSourceTab("video_file");
        } catch (e) {
            console.error("Upload error:", e);
            videoUploadStatus.textContent = `❌ Upload failed: ${e.message}`;
            videoUploadStatus.style.color = "#ef4444";
        }
    });

    // Poll Stats & Update Vision HUD
    async function fetchStats() {
        try {
            const resp = await fetch("/api/stats");
            if (!resp.ok) return;
            const stats = await resp.json();
            currentStats = stats;

            const activeCount = isManualOverride ? manualCount : stats.current_count;
            rodentCountVal.textContent = activeCount;
            rodentPeakVal.textContent = `Peak: ${stats.max_session_count}`;
            fpsCounter.textContent = `FPS: ${stats.fps || "--"}`;

            if (!isPowerOn) {
                videoOverlayBadge.style.borderColor = "rgba(100, 116, 139, 0.4)";
                videoOverlayText.textContent = "CAMERA POWERED OFF";
            } else if (activeCount > 0) {
                videoOverlayBadge.style.borderColor = "rgba(239, 68, 68, 0.8)";
                videoOverlayText.textContent = `RODENTS DETECTED: ${activeCount}`;
            } else {
                videoOverlayBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
                videoOverlayText.textContent = `LIVE STREAMING (0 DETECTED)`;
            }

            recalculateRisk();
        } catch (e) {
            console.error("Error fetching stats:", e);
        }
    }

    // Recalculate Risk Score
    async function recalculateRisk() {
        const payload = {
            rodent_count: currentStats.current_count,
            rainfall_mm: rainfallMm,
            water_level: waterLevelScore,
            historical_risk: historicalRiskScore,
            manual_override: isManualOverride,
            manual_count: manualCount
        };

        try {
            const resp = await fetch("/api/risk", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (!resp.ok) return;
            const res = await resp.json();

            riskScoreVal.textContent = res.score;
            gaugeFill.style.width = `${res.score}%`;

            gaugeFill.className = "gauge-fill";
            riskLevelBadge.className = "risk-badge";
            const levelUpper = res.level.toUpperCase();
            riskLevelBadge.textContent = levelUpper;

            if (res.level === "Low") {
                gaugeFill.classList.add("fill-low");
                riskLevelBadge.classList.add("badge-low");
            } else if (res.level === "Moderate") {
                gaugeFill.classList.add("fill-moderate");
                riskLevelBadge.classList.add("badge-moderate");
            } else if (res.level === "High") {
                gaugeFill.classList.add("fill-high");
                riskLevelBadge.classList.add("badge-high");
            } else {
                gaugeFill.classList.add("fill-critical");
                riskLevelBadge.classList.add("badge-critical");
            }

            expRodent.textContent = `${res.rodent_index}/100`;
            expRain.textContent = `${res.rainfall_index}/100`;
            expWaterLevel.textContent = `${res.water_level_index}/100`;
            expHist.textContent = `${res.historical_index}/100`;

            if (res.synergy_boost) {
                synergyBanner.classList.remove("hidden");
            } else {
                synergyBanner.classList.add("hidden");
            }

            municipalActionText.textContent = res.municipal_action;
            citizenActionText.textContent = res.citizen_action;

        } catch (e) {
            console.error("Error recalculating risk:", e);
        }
    }

    // Weather Sync
    async function syncWeather() {
        btnRefreshWeather.textContent = "⏳ Syncing...";
        const locKey = weatherLocationSelect ? weatherLocationSelect.value : "kuala_lumpur";
        try {
            const resp = await fetch(`/api/weather?location=${encodeURIComponent(locKey)}`);
            const data = await resp.json();
            if (data.rainfall_mm !== undefined) {
                rainfallMm = data.rainfall_mm;
                rainfallVal.textContent = rainfallMm.toFixed(1);

                if (weatherTempBadge && data.temperature_c !== undefined) {
                    weatherTempBadge.textContent = `🌤️ ${data.temperature_c.toFixed(1)}°C`;
                    weatherTempBadge.title = `Humidity: ${data.humidity_pct}% | Rain: ${data.current_rain_mm} mm`;
                }

                // Auto-suggest and update the water level (1mm rain = 5% water level)
                waterLevelScore = Math.min(100, Math.round(rainfallMm * 5.0));
                waterLevelSlider.value = waterLevelScore;
                waterLevelValText.textContent = `${waterLevelScore}/100`;

                recalculateRisk();
            }
        } catch (e) {
            console.error("Weather sync failed:", e);
        } finally {
            btnRefreshWeather.textContent = "🔄 Sync";
        }
    }

    // Event Listeners
    waterLevelSlider.addEventListener("input", (e) => {
        waterLevelScore = parseFloat(e.target.value);
        waterLevelValText.textContent = `${waterLevelScore.toFixed(0)}/100`;
        recalculateRisk();
    });

    confSlider.addEventListener("change", (e) => {
        const val = parseFloat(e.target.value);
        confValText.textContent = `${Math.round(val * 100)}%`;
        updateBackendConfig({ conf_threshold: val });
    });

    chkHumanFP.addEventListener("change", (e) => {
        updateBackendConfig({ suppress_human_fp: e.target.checked });
    });

    modelWeightsSelect.addEventListener("change", (e) => {
        updateBackendConfig({ weights_path: e.target.value });
    });

    // Camera source toggles
    btnSourceWebcam.addEventListener("click", () => {
        setPowerState(true);
        updateSourceTab("webcam");
        updateBackendConfig({ source_type: "webcam", camera_index: parseInt(webcamIndexSelect.value) });
    });

    btnSourceIpCam.addEventListener("click", () => {
        setPowerState(true);
        updateSourceTab("ip_cam");
    });

    btnSourceVideoFile.addEventListener("click", () => {
        setPowerState(true);
        updateSourceTab("video_file");
    });

    webcamIndexSelect.addEventListener("change", (e) => {
        updateBackendConfig({ source_type: "webcam", camera_index: parseInt(e.target.value) });
    });

    btnApplyIpCam.addEventListener("click", () => {
        const url = ipCamUrlInput.value.trim();
        if (url) {
            updateBackendConfig({ source_type: "ip_cam", camera_url: url });
        }
    });

    // Manual Override
    chkManualOverride.addEventListener("change", (e) => {
        isManualOverride = e.target.checked;
        if (isManualOverride) {
            manualCountRow.classList.remove("hidden");
        } else {
            manualCountRow.classList.add("hidden");
        }
        recalculateRisk();
    });

    manualCountInput.addEventListener("input", (e) => {
        manualCount = parseInt(e.target.value) || 0;
        recalculateRisk();
    });

    btnRefreshWeather.addEventListener("click", syncWeather);
    if (weatherLocationSelect) {
        weatherLocationSelect.addEventListener("change", syncWeather);
    }

    // Auto-refresh weather every 5 minutes (300,000 ms)
    setInterval(syncWeather, 300000);
    // Initial weather sync on load
    syncWeather();
    // PDF Report — Handled by standalone /static/pdf_report.js

    // Tab Navigation & Geospatial Map Logic
    const tabBtnVision = document.getElementById("tabBtnVision");
    const tabBtnMap = document.getElementById("tabBtnMap");
    const tabViewVision = document.getElementById("tabViewVision");
    const tabViewMap = document.getElementById("tabViewMap");

    let leafletMap = null;
    let mapMarkers = [];

    tabBtnVision.addEventListener("click", () => {
        tabBtnVision.classList.add("active");
        tabBtnMap.classList.remove("active");
        tabViewVision.classList.remove("hidden");
        tabViewMap.classList.add("hidden");
    });

    tabBtnMap.addEventListener("click", () => {
        tabBtnMap.classList.add("active");
        tabBtnVision.classList.remove("active");
        tabViewMap.classList.remove("hidden");
        tabViewVision.classList.add("hidden");

        if (!leafletMap) {
            initLeafletMap();
        } else {
            setTimeout(() => leafletMap.invalidateSize(), 200);
        }
        loadNodesData();
    });

    function getBadgeClass(level) {
        if (level === "Critical") return "badge-critical";
        if (level === "High") return "badge-high";
        if (level === "Moderate") return "badge-moderate";
        return "badge-low";
    }

    function getMarkerColor(level) {
        if (level === "Critical") return "#ef4444";
        if (level === "High") return "#f97316";
        if (level === "Moderate") return "#f59e0b";
        return "#10b981";
    }

    function initLeafletMap() {
        if (typeof L === "undefined") return;
        
        // Center on Kuala Lumpur
        leafletMap = L.map("nodeMap").setView([3.1390, 101.6869], 13);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 18,
        }).addTo(leafletMap);
    }

    async function loadNodesData() {
        try {
            const resp = await fetch("/api/nodes");
            if (!resp.ok) return;
            const data = await resp.json();
            if (!data.nodes) return;

            const nodes = data.nodes;
            
            // Update Summary KPI Bar
            document.getElementById("totalNodesCount").textContent = nodes.length;
            const criticalCount = nodes.filter(n => n.risk_level === "Critical").length;
            const highCount = nodes.filter(n => n.risk_level === "High").length;
            const avgScore = Math.round(nodes.reduce((acc, n) => acc + n.lers_score, 0) / nodes.length);

            document.getElementById("criticalNodesCount").textContent = criticalCount;
            document.getElementById("highNodesCount").textContent = highCount;
            document.getElementById("avgLersScore").textContent = avgScore;

            // Clear old markers
            mapMarkers.forEach(m => leafletMap.removeLayer(m));
            mapMarkers = [];

            // Populate Table & Map Markers
            const tbody = document.getElementById("nodeTableBody");
            tbody.innerHTML = "";

            nodes.forEach(node => {
                const color = getMarkerColor(node.risk_level);
                
                // Circle Marker
                const circle = L.circleMarker([node.latitude, node.longitude], {
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.8,
                    radius: 10
                }).addTo(leafletMap);

                const popupContent = `
                    <div class="popup-node-card">
                        <h4>📍 ${node.name} (${node.id})</h4>
                        <div class="popup-metrics">
                            <div><span>LERS Score:</span> <strong>${node.lers_score}/100 (${node.risk_level})</strong></div>
                            <div><span>Rodents Detected:</span> <strong>${node.rodent_count}</strong></div>
                            <div><span>Water Level:</span> <strong>${node.water_level_pct}%</strong></div>
                            <div><span>Rainfall:</span> <strong>${node.rainfall_mm} mm</strong></div>
                        </div>
                        <div style="font-size:0.75rem; color:#94a3b8; border-top:1px solid #334155; padding-top:0.3rem;">
                            <strong>Municipal Action:</strong> ${node.municipal_action}
                        </div>
                    </div>
                `;

                circle.bindPopup(popupContent);
                mapMarkers.push(circle);

                // Table Row
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><code>${node.id}</code></td>
                    <td><strong>${node.name}</strong></td>
                    <td>${node.rodent_count}</td>
                    <td>${node.rainfall_mm} mm</td>
                    <td>${node.water_level_pct}%</td>
                    <td><strong>${node.lers_score}/100</strong></td>
                    <td><span class="risk-badge ${getBadgeClass(node.risk_level)}">${node.risk_level.toUpperCase()}</span></td>
                    <td><button class="btn-focus">Center Map</button></td>
                `;

                tr.querySelector(".btn-focus").addEventListener("click", (e) => {
                    e.stopPropagation();
                    leafletMap.flyTo([node.latitude, node.longitude], 15, { duration: 1.2 });
                    circle.openPopup();
                });

                tr.addEventListener("click", () => {
                    leafletMap.flyTo([node.latitude, node.longitude], 15, { duration: 1.2 });
                    circle.openPopup();
                });

                tbody.appendChild(tr);
            });

        } catch (e) {
            console.error("Failed to load node data:", e);
        }
    }

    // Init
    loadConfig();
    recalculateRisk();
    setInterval(fetchStats, 500);
});

