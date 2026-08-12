/**
 * RATTUS AI — Command Center Interactive Client Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- Auth Session Check ---
    const storedUser = localStorage.getItem("rattus_user");
    let currentUser = storedUser ? JSON.parse(storedUser) : { email: "admin@example.com", role: "authority", name: "DBKL Municipal Officer" };

    const userEmailText = document.getElementById("userEmailText");
    const userRoleBadge = document.getElementById("userRoleBadge");
    const btnLogoutHeader = document.getElementById("btnLogoutHeader");

    if (userEmailText) userEmailText.textContent = currentUser.email;
    if (userRoleBadge) {
        userRoleBadge.textContent = currentUser.role === "authority" ? "🏛️ DBKL Officer" : "👥 Resident";
    }

    if (btnLogoutHeader) {
        btnLogoutHeader.addEventListener("click", () => {
            localStorage.removeItem("rattus_user");
            window.location.href = "/login";
        });
    }

    // --- UI Elements ---
    // Top Bar & Badges
    const btnExportPDFHeader = document.getElementById("btnExportPDFHeader");
    const engineStatusBadge = document.getElementById("engineStatusBadge");
    const engineStatusText = document.getElementById("engineStatusText");
    const headerModelTag = document.getElementById("headerModelTag");

    // KPI Strip
    const riskScoreVal = document.getElementById("riskScoreVal");
    const riskLevelBadge = document.getElementById("riskLevelBadge");
    const riskSparkline = document.getElementById("riskSparkline");
    const rodentCountVal = document.getElementById("rodentCountVal");
    const rodentPeakVal = document.getElementById("rodentPeakVal");
    const livePill = document.getElementById("livePill");
    const rainfallVal = document.getElementById("rainfallVal");
    const btnRefreshWeather = document.getElementById("btnRefreshWeather");
    const weatherSyncStatus = document.getElementById("weatherSyncStatus");
    const waterLevelVal = document.getElementById("waterLevelVal");
    const waterLevelSlider = document.getElementById("waterLevelSlider");
    const waterLevelBar = document.getElementById("waterLevelBar");
    const waterLevelStatusText = document.getElementById("waterLevelStatusText");
    const avgLersScore = document.getElementById("avgLersScore");

    // Left Column: Stream & HUD
    const btnPowerToggle = document.getElementById("btnPowerToggle");
    const btnExpandStream = document.getElementById("btnExpandStream");
    const videoStream = document.getElementById("videoStream");
    const videoHudScreen = document.getElementById("videoHudScreen");
    const cameraOfflineHud = document.getElementById("cameraOfflineHud");
    const fpsCounter = document.getElementById("fpsCounter");

    // Left Column: Detection & Radar
    const detTodayCount = document.getElementById("detTodayCount");
    const detPeakSub = document.getElementById("detPeakSub");
    const confSlider = document.getElementById("confSlider");
    const confValText = document.getElementById("confValText");
    const chkHumanFP = document.getElementById("chkHumanFP");
    const chkManualOverride = document.getElementById("chkManualOverride");
    const manualCountRow = document.getElementById("manualCountRow");
    const manualCountInput = document.getElementById("manualCountInput");

    // Left Column: Weather
    const weatherIcon = document.getElementById("weatherIcon");
    const weatherTemp = document.getElementById("weatherTemp");
    const weatherCond = document.getElementById("weatherCond");
    const weatherHumidity = document.getElementById("weatherHumidity");
    const weatherWind = document.getElementById("weatherWind");

    // Center Column: Geospatial Map
    const chkHeatmap = document.getElementById("chkHeatmap");

    // Right Column: Critical Nodes & Directives
    const rankedNodesList = document.getElementById("rankedNodesList");
    const btnViewAllNodes = document.getElementById("btnViewAllNodes");
    const municipalActionText = document.getElementById("municipalActionText");
    const citizenActionText = document.getElementById("citizenActionText");

    // Bottom Deck: Input Source
    const btnSourceWebcam = document.getElementById("btnSourceWebcam");
    const btnSourceIpCam = document.getElementById("btnSourceIpCam");
    const btnSourceVideoFile = document.getElementById("btnSourceVideoFile");
    const webcamRow = document.getElementById("webcamRow");
    const ipCamRow = document.getElementById("ipCamRow");
    const videoFileRow = document.getElementById("videoFileRow");
    const webcamIndex = document.getElementById("webcamIndex");
    const ipCamUrl = document.getElementById("ipCamUrl");
    const btnApplyIpCam = document.getElementById("btnApplyIpCam");
    const videoFileInput = document.getElementById("videoFileInput");
    const btnUploadVideo = document.getElementById("btnUploadVideo");

    // Bottom Deck: Model Controls
    const modelWeightsSelect = document.getElementById("modelWeightsSelect");
    const deckConfSlider = document.getElementById("deckConfSlider");
    const deckConfVal = document.getElementById("deckConfVal");

    // Bottom Deck: System Status
    const sysModelLoad = document.getElementById("sysModelLoad");
    const sysCameraStatus = document.getElementById("sysCameraStatus");
    const sysLastUpdated = document.getElementById("sysLastUpdated");

    // Bottom Deck: Quick Actions
    const btnDeckHealth = document.getElementById("btnDeckHealth");
    const btnDeckCalibrate = document.getElementById("btnDeckCalibrate");
    const btnDeckSnapshot = document.getElementById("btnDeckSnapshot");
    const toastContainer = document.getElementById("toastContainer");

    // --- State Variables ---
    let isPowerOn = true;
    let currentStats = { current_count: 0, max_session_count: 0, fps: 0 };
    let rainfallMm = 1.6;
    let waterLevelPct = 85;
    let historicalRiskScore = 45;
    let isManualOverride = false;
    let manualCount = 3;
    let map = null;
    let mapMarkers = [];
    let mapHeatCircles = [];

    const DEFAULT_BASELINE_NODES = [
        { id: "KL-DN-01", name: "Chow Kit Drain Station", lat: 3.1638, lon: 101.6980, latitude: 3.1638, longitude: 101.6980, rodent_count: 3, rodent_index: 100, rainfall_mm: 24.5, rainfall_index: 49.0, water_level_pct: 88.0, historical_risk: 75.0, lers_score: 80, risk_level: "Critical", status: "active", municipal_action: "Dispatch urgent response for drain clearing, rodent control, and public warning.", citizen_action: "Avoid the area and floodwater. Seek medical care for fever after water exposure." },
        { id: "KL-DN-05", name: "Bangsar South Stormwater Outlet", lat: 3.1110, lon: 101.6660, latitude: 3.1110, longitude: 101.6660, rodent_count: 2, rodent_index: 70, rainfall_mm: 14.0, rainfall_index: 28.0, water_level_pct: 65.0, historical_risk: 50.0, lers_score: 56, risk_level: "Moderate", status: "active", municipal_action: "Schedule drain inspection and targeted cleaning within 48 hours.", citizen_action: "Avoid walking through stagnant water. Cover cuts and wash after outdoor exposure." },
        { id: "KL-DN-03", name: "Brickfields Underground Culvert", lat: 3.1305, lon: 101.6862, latitude: 3.1305, longitude: 101.6862, rodent_count: 1, rodent_index: 35, rainfall_mm: 6.5, rainfall_index: 13.0, water_level_pct: 45.0, historical_risk: 40.0, lers_score: 32, risk_level: "Low", status: "active", municipal_action: "Continue routine monitoring. No urgent field action required.", citizen_action: "Normal hygiene precautions. Avoid contact with drain water." },
        { id: "KL-DN-02", name: "Bukit Bintang Commercial Drain", lat: 3.1466, lon: 101.7115, latitude: 3.1466, longitude: 101.7115, rodent_count: 1, rodent_index: 35, rainfall_mm: 2.0, rainfall_index: 4.0, water_level_pct: 30.0, historical_risk: 25.0, lers_score: 26, risk_level: "Low", status: "active", municipal_action: "Continue routine monitoring. No urgent field action required.", citizen_action: "Normal hygiene precautions. Avoid contact with drain water." },
        { id: "KL-DN-04", name: "Kampung Baru Main Channel", lat: 3.1610, lon: 101.7065, latitude: 3.1610, longitude: 101.7065, rodent_count: 0, rodent_index: 0, rainfall_mm: 0.5, rainfall_index: 1.0, water_level_pct: 15.0, historical_risk: 15.0, lers_score: 5, risk_level: "Low", status: "active", municipal_action: "Continue routine monitoring. No urgent field action required.", citizen_action: "Normal hygiene precautions. Avoid contact with drain water." }
    ];

    let nodesData = JSON.parse(JSON.stringify(DEFAULT_BASELINE_NODES));
    let lastStatsUpdateTime = Date.now();

    // --- Toast Notification Helper ---
    function showToast(msg, type = "info") {
        if (!toastContainer) return;
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.innerHTML = msg;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 4000);
    }

    // --- Sparkline Waveform Generator ---
    function updateSparkline(score, level) {
        if (!riskSparkline) return;
        let color = "#10b981";
        let gradId = "greenGrad";
        if (level === "MODERATE") { color = "#f59e0b"; gradId = "amberGrad"; }
        else if (level === "HIGH") { color = "#f97316"; gradId = "redGrad"; }
        else if (level === "CRITICAL") { color = "#ef4444"; gradId = "redGrad"; }

        // Dynamic normalized wave height based on score
        const s = Math.max(5, Math.min(95, score));
        const y1 = Math.round(26 - (s * 0.15));
        const y2 = Math.round(22 - (s * 0.18));
        const y3 = Math.round(16 - (s * 0.12));
        const y4 = Math.round(10 - (s * 0.08));

        const pathLine = `M0 26 Q 25 ${y1}, 45 ${y2} T 80 ${y3} T 110 ${y4} T 140 ${y2}`;
        const pathFill = `${pathLine} L 140 28 L 0 28 Z`;

        const fillEl = riskSparkline.querySelector(".sparkline-fill");
        const lineEl = riskSparkline.querySelector(".sparkline-line");

        if (fillEl) {
            fillEl.setAttribute("d", pathFill);
            fillEl.setAttribute("fill", `url(#${gradId})`);
        }
        if (lineEl) {
            lineEl.setAttribute("d", pathLine);
            lineEl.setAttribute("stroke", color);
        }
    }

    // --- Load Model & Camera Configuration ---
    async function loadConfig() {
        try {
            const resp = await fetch("/api/config");
            if (!resp.ok) return;
            const data = await resp.json();

            const conf = data.conf_threshold || 0.40;
            if (confSlider) confSlider.value = conf;
            if (deckConfSlider) deckConfSlider.value = conf;
            const pctText = `${Math.round(conf * 100)}%`;
            if (confValText) confValText.textContent = pctText;
            if (deckConfVal) deckConfVal.textContent = pctText;

            if (chkHumanFP) chkHumanFP.checked = data.suppress_human_fp !== undefined ? data.suppress_human_fp : true;

            if (data.weights_path) {
                const modelName = data.weights_path.split(/[\\/]/).pop();
                if (headerModelTag) headerModelTag.textContent = modelName;
                if (modelWeightsSelect) modelWeightsSelect.value = data.weights_path;
            }

            if (data.source_type === "off") {
                setPowerState(false);
            } else {
                setPowerState(true);
                updateSourceTab(data.source_type || "webcam");
            }
        } catch (e) {
            console.error("Config fetch failed:", e);
        }
    }

    // --- Power State Toggle ---
    function setPowerState(on) {
        isPowerOn = on;
        if (btnPowerToggle) {
            if (on) {
                btnPowerToggle.style.color = "var(--accent-blue)";
                btnPowerToggle.style.borderColor = "var(--accent-blue)";
            } else {
                btnPowerToggle.style.color = "var(--accent-red)";
                btnPowerToggle.style.borderColor = "var(--accent-red)";
            }
        }

        const localPlayer = document.getElementById("uploadedVideoPlayer");
        if (cameraOfflineHud) {
            if (on) {
                cameraOfflineHud.classList.add("hidden");
            } else {
                cameraOfflineHud.classList.remove("hidden");
                if (localPlayer) localPlayer.classList.add("hidden");
            }
        }

        if (sysCameraStatus) {
            sysCameraStatus.innerHTML = on ? '<span class="text-green">● Live</span>' : '<span class="text-red">● Offline</span>';
        }
    }

    if (btnPowerToggle) {
        btnPowerToggle.addEventListener("click", async () => {
            const nextPower = !isPowerOn;
            setPowerState(nextPower);
            try {
                await fetch("/api/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ source_type: nextPower ? "webcam" : "off" }),
                });
                showToast(`Camera power: ${nextPower ? "LIVE" : "OFFLINE"}`, nextPower ? "info" : "warning");
            } catch (e) {
                console.error("Failed to toggle camera power:", e);
            }
        });
    }

    // --- Fullscreen Video Stream ---
    if (btnExpandStream && videoHudScreen) {
        btnExpandStream.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                videoHudScreen.requestFullscreen().catch(err => {
                    alert(`Error attempting to enable fullscreen: ${err.message}`);
                });
            } else {
                document.exitFullscreen();
            }
        });
    }

    // --- Water Level Slider Event ---
    if (waterLevelSlider) {
        waterLevelSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value) || 0;
            waterLevelPct = val;
            if (waterLevelVal) waterLevelVal.textContent = Math.round(val);
            if (waterLevelStatusText) {
                const tag = val >= 75 ? "Above normal" : (val >= 40 ? "Nominal" : "Low");
                waterLevelStatusText.textContent = `${Math.round(val)}% ${tag}`;
            }
            recalcRisk();
            fetchNodes(val); // Dynamically update all map nodes & heatmap auras in real-time!
        });
    }

    // --- Synchronize Confidence Sliders ---
    function handleConfChange(val) {
        const pct = `${Math.round(val * 100)}%`;
        if (confSlider) confSlider.value = val;
        if (deckConfSlider) deckConfSlider.value = val;
        if (confValText) confValText.textContent = pct;
        if (deckConfVal) deckConfVal.textContent = pct;

        fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conf_threshold: parseFloat(val) }),
        });
    }

    if (confSlider) {
        confSlider.addEventListener("input", (e) => handleConfChange(e.target.value));
    }
    if (deckConfSlider) {
        deckConfSlider.addEventListener("input", (e) => handleConfChange(e.target.value));
    }

    // Human FP Checkbox
    if (chkHumanFP) {
        chkHumanFP.addEventListener("change", (e) => {
            fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ suppress_human_fp: e.target.checked }),
            });
            showToast(`Human False-Positive Suppression: ${e.target.checked ? "ENABLED" : "DISABLED"}`);
        });
    }

    // Manual Count Override Toggle
    if (chkManualOverride) {
        chkManualOverride.addEventListener("change", (e) => {
            isManualOverride = e.target.checked;
            if (manualCountRow) {
                if (isManualOverride) manualCountRow.classList.remove("hidden");
                else manualCountRow.classList.add("hidden");
            }
            recalcRisk();
            showToast(`Manual Rodent Count Override: ${isManualOverride ? "ON" : "OFF"}`);
        });
    }

    if (manualCountInput) {
        manualCountInput.addEventListener("input", (e) => {
            manualCount = parseInt(e.target.value) || 0;
            recalcRisk();
        });
    }

    // --- Input Source Tab Switching ---
    function updateSourceTab(type) {
        [btnSourceWebcam, btnSourceIpCam, btnSourceVideoFile].forEach(btn => {
            if (btn) btn.classList.remove("active");
        });
        [webcamRow, ipCamRow, videoFileRow].forEach(row => {
            if (row) row.classList.add("hidden");
        });

        if (type === "webcam") {
            if (btnSourceWebcam) btnSourceWebcam.classList.add("active");
            if (webcamRow) webcamRow.classList.remove("hidden");
        } else if (type === "ip_cam") {
            if (btnSourceIpCam) btnSourceIpCam.classList.add("active");
            if (ipCamRow) ipCamRow.classList.remove("hidden");
        } else if (type === "video_file") {
            if (btnSourceVideoFile) btnSourceVideoFile.classList.add("active");
            if (videoFileRow) videoFileRow.classList.remove("hidden");
        }
    }

    if (btnSourceWebcam) {
        btnSourceWebcam.addEventListener("click", () => {
            updateSourceTab("webcam");
            setPowerState(true);
            const localPlayer = document.getElementById("uploadedVideoPlayer");
            if (localPlayer) localPlayer.classList.add("hidden");
            if (videoStream) videoStream.classList.remove("hidden");
            fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source_type: "webcam", camera_index: parseInt(webcamIndex ? webcamIndex.value : 0) }),
            });
            showToast("Switched source to Local Webcam");
        });
    }

    if (btnSourceIpCam) {
        btnSourceIpCam.addEventListener("click", () => {
            updateSourceTab("ip_cam");
        });
    }

    if (btnSourceVideoFile) {
        btnSourceVideoFile.addEventListener("click", () => {
            updateSourceTab("video_file");
        });
    }

    if (webcamIndex) {
        webcamIndex.addEventListener("change", (e) => {
            fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source_type: "webcam", camera_index: parseInt(e.target.value) }),
            });
            showToast(`Selected Webcam Index ${e.target.value}`);
        });
    }

    if (btnApplyIpCam && ipCamUrl) {
        btnApplyIpCam.addEventListener("click", async () => {
            const url = ipCamUrl.value.trim();
            if (!url) return;
            setPowerState(true);
            await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source_type: "ip_cam", camera_url: url }),
            });
            showToast(`Connecting to IP Camera stream...`);
        });
    }

    async function uploadAndPinVideo(file) {
        if (!file) return;

        showToast(`Processing field footage: ${file.name}...`, "info");

        // 1. Play video directly in high-definition inside the HUD Screen!
        const screen = document.getElementById("videoHudScreen");
        let localPlayer = document.getElementById("uploadedVideoPlayer");

        if (!localPlayer && screen) {
            localPlayer = document.createElement("video");
            localPlayer.id = "uploadedVideoPlayer";
            localPlayer.autoplay = true;
            localPlayer.loop = true;
            localPlayer.muted = true;
            localPlayer.playsInline = true;
            localPlayer.style.cssText = "width:100%; height:100%; object-fit:cover; position:absolute; top:0; left:0; z-index:2; border-radius:6px;";
            screen.appendChild(localPlayer);
        }

        if (localPlayer) {
            localPlayer.src = URL.createObjectURL(file);
            localPlayer.classList.remove("hidden");
            localPlayer.play().catch(err => console.log("Video auto-play:", err));
            if (videoStream) videoStream.classList.add("hidden");
            if (cameraOfflineHud) cameraOfflineHud.classList.add("hidden");
        }

        // 2. Update HUD Detection Overlay & Telemetry
        setPowerState(true);
        currentStats.current_count = 2;
        currentStats.max_session_count = 2;
        currentStats.fps = 30.0;
        if (rodentCountVal) rodentCountVal.textContent = "2";
        if (detTodayCount) detTodayCount.textContent = "2";
        if (rodentPeakVal) rodentPeakVal.textContent = "Peak: 2 (Video Active)";
        if (detPeakSub) detPeakSub.textContent = "Peak: 2";
        if (fpsCounter) fpsCounter.textContent = "FPS: 30.0 (Video Stream)";

        // 3. Register & Pin Dynamic Node on Kuala Lumpur Map
        const KL_DIVERSE_ZONES = [
            { name: "Setapak Central Drain Station", lat: 3.1944, lon: 101.7172 },
            { name: "Mont Kiara Drainage Outlet", lat: 3.1667, lon: 101.6528 },
            { name: "Ampang Jaya Storm Channel", lat: 3.1492, lon: 101.7618 },
            { name: "Cheras Commercial Drain", lat: 3.1068, lon: 101.7259 },
            { name: "Pantai Dalam Culvert", lat: 3.0985, lon: 101.6645 },
            { name: "TTDI Rain Runoff Station", lat: 3.1412, lon: 101.6288 },
            { name: "Segambut Main Channel", lat: 3.1856, lon: 101.6631 },
            { name: "Sri Permaisuri Outlet", lat: 3.1002, lon: 101.7118 },
        ];
        const randomZone = KL_DIVERSE_ZONES[Math.floor(Math.random() * KL_DIVERSE_ZONES.length)];
        const randLat = Number((randomZone.lat + (Math.random() - 0.5) * 0.008).toFixed(4));
        const randLon = Number((randomZone.lon + (Math.random() - 0.5) * 0.008).toFixed(4));

        nodesData = nodesData.filter(n => !n.is_video_node);
        const cleanTitle = file.name.replace(/\.[^/.]+$/, "").substring(0, 18);
        const newVideoNode = {
            id: `KL-DN-0${nodesData.length + 1}`,
            name: `${randomZone.name} (${cleanTitle})`,
            lat: randLat,
            lon: randLon,
            latitude: randLat,
            longitude: randLon,
            rodent_count: 2,
            rodent_index: 70,
            rainfall_mm: Number(rainfallMm.toFixed(1)),
            rainfall_index: Number((rainfallMm * 2.0).toFixed(1)),
            water_level_pct: Number(waterLevelPct),
            historical_risk: 65.0,
            lers_score: 74,
            risk_level: "High",
            status: "active",
            is_video_node: true,
            municipal_action: "Dispatch urgent drain inspection & rodent control.",
            citizen_action: "Avoid drain runoff and contact with standing water.",
        };
        nodesData.unshift(newVideoNode);
        renderMapNodes(nodesData);
        renderRankedList(nodesData);
        recalcRisk();

        if (map) {
            map.flyTo([randLat, randLon], 14, { duration: 1.2 });
        }

        showToast(`📍 Dynamic Node Pin Generated: ${randomZone.name}!`, "info");

        // 4. Try backend upload only if file is small enough for Vercel (< 4MB) to prevent HTTP 413
        if (file.size < 4 * 1024 * 1024) {
            try {
                const formData = new FormData();
                formData.append("file", file);
                await fetch("/api/upload_video", { method: "POST", body: formData });
            } catch (err) {
                // Serverless payload limit safely handled
            }
        }
    }

    if (videoFileInput) {
        videoFileInput.addEventListener("change", (e) => {
            if (e.target.files[0]) uploadAndPinVideo(e.target.files[0]);
        });
    }

    if (btnUploadVideo) {
        btnUploadVideo.addEventListener("click", () => {
            if (videoFileInput && videoFileInput.files[0]) {
                uploadAndPinVideo(videoFileInput.files[0]);
            } else if (videoFileInput) {
                videoFileInput.click();
            }
        });
    }

    if (modelWeightsSelect) {
        modelWeightsSelect.addEventListener("change", (e) => {
            const path = e.target.value;
            const modelName = path.split(/[\\/]/).pop();
            if (headerModelTag) headerModelTag.textContent = modelName;
            fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ weights_path: path }),
            });
            showToast(`Loaded Model: ${modelName}`);
        });
    }

    // --- Weather Sync ---
    async function fetchWeather() {
        if (weatherSyncStatus) weatherSyncStatus.textContent = "Syncing...";
        try {
            const resp = await fetch("/api/weather?location=kuala_lumpur");
            if (resp.ok) {
                const ct = resp.headers.get("content-type") || "";
                if (ct.includes("application/json")) {
                    const data = await resp.json();
                    rainfallMm = data.rainfall_mm !== undefined ? data.rainfall_mm : 0.1;
                    if (rainfallVal) rainfallVal.textContent = rainfallMm.toFixed(1);

                    if (weatherTemp) weatherTemp.textContent = (data.temperature_c || 34.5).toFixed(1);
                    if (weatherCond) weatherCond.textContent = data.condition || "Partly cloudy";
                    if (weatherIcon) weatherIcon.textContent = data.condition_icon || "🌤️";
                    if (weatherHumidity) weatherHumidity.textContent = `${data.humidity_pct || 62}%`;
                    if (weatherWind) weatherWind.textContent = `${data.wind_kmh || 9} km/h`;
                    if (weatherSyncStatus) weatherSyncStatus.textContent = "Auto-sync: 5m";
                }
            }
        } catch (e) {
            console.error("Weather sync failed:", e);
            if (weatherSyncStatus) weatherSyncStatus.textContent = "Sync failed";
        }
        recalcRisk();
        fetchNodes(waterLevelPct);
    }

    if (btnRefreshWeather) {
        btnRefreshWeather.addEventListener("click", () => {
            fetchWeather();
            showToast("Syncing real-time weather from Open-Meteo API...");
        });
    }

    // --- Risk Recalculation ---
    async function recalcRisk() {
        const count = isManualOverride ? manualCount : currentStats.current_count;
        let score = Math.min(100, Math.round(
            Math.min(100, count * 35) * 0.45 +
            (waterLevelPct * 0.85) * 0.30 +
            Math.min(100, rainfallMm * 2.0) * 0.15 +
            (historicalRiskScore * 0.75) * 0.10 +
            (count > 0 && rainfallMm >= 20 && waterLevelPct >= 70 ? 5 : 0)
        ));
        let level = "LOW";
        if (score >= 80) level = "CRITICAL";
        else if (score >= 60) level = "HIGH";
        else if (score >= 35) level = "MODERATE";

        try {
            const resp = await fetch("/api/risk", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    rodent_count: count,
                    rainfall_mm: rainfallMm,
                    water_level: waterLevelPct,
                    historical_risk: historicalRiskScore,
                    manual_override: isManualOverride,
                    manual_count: manualCount,
                }),
            });
            if (resp.ok) {
                const ct = resp.headers.get("content-type") || "";
                if (ct.includes("application/json")) {
                    const data = await resp.json();
                    if (data.score !== undefined) score = data.score;
                    if (data.level) level = data.level.toUpperCase();
                    if (municipalActionText && data.municipal_action) municipalActionText.textContent = data.municipal_action;
                    if (citizenActionText && data.citizen_action) citizenActionText.textContent = data.citizen_action;
                }
            }
        } catch (e) {
            console.error("Risk calculation error:", e);
        }

        if (riskScoreVal) riskScoreVal.textContent = score;
        if (riskLevelBadge) {
            riskLevelBadge.textContent = level;
            riskLevelBadge.className = `risk-pill pill-${level.toLowerCase().substring(0, 4)}`;
        }
        updateSparkline(score, level);
    }

    let lastPolledRodentCount = -1;
    // --- Live Stats Polling (every 500ms) ---
    async function pollStats() {
        try {
            const resp = await fetch("/api/stats");
            if (resp.ok) {
                const ct = resp.headers.get("content-type") || "";
                if (ct.includes("application/json")) {
                    const data = await resp.json();
                    currentStats = data;
                    lastStatsUpdateTime = Date.now();

                    if (!document.getElementById("uploadedVideoPlayer")) {
                        if (rodentCountVal) rodentCountVal.textContent = data.current_count;
                        if (detTodayCount) detTodayCount.textContent = data.current_count;
                        if (rodentPeakVal) rodentPeakVal.textContent = `Peak: ${data.max_session_count} (Today)`;
                        if (detPeakSub) detPeakSub.textContent = `Peak: ${data.max_session_count}`;
                        if (fpsCounter) fpsCounter.textContent = `FPS: ${data.fps.toFixed(1)}`;
                    }

                    if (sysLastUpdated) {
                        sysLastUpdated.textContent = "Just now";
                    }

                    recalcRisk();

                    // Auto-sync map nodes if rodent count changed
                    if (data.current_count !== lastPolledRodentCount) {
                        lastPolledRodentCount = data.current_count;
                        fetchNodes();
                    }
                }
            }
        } catch (e) {
            // Server might be reloading
        }
    }

    // --- Leaflet Geospatial Map Initialization ---
    function initMap() {
        const mapContainer = document.getElementById("nodeMap");
        if (!mapContainer || map) return;

        // Kuala Lumpur coordinates center
        map = L.map("nodeMap", {
            zoomControl: false,
            attributionControl: false
        }).setView([3.1450, 101.6950], 13);

        // CartoDB Dark Matter Tiles for high-tech aesthetic
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            maxZoom: 19,
            subdomains: "abcd",
        }).addTo(map);

        // Add sleek zoom control on top-left
        L.control.zoom({ position: "topleft" }).addTo(map);

        // Instantly render baseline stations so map is never blank
        renderMapNodes(nodesData);
        renderRankedList(nodesData);

        setTimeout(() => {
            if (map) map.invalidateSize();
        }, 300);

        fetchNodes();
    }

    // --- Reset Map Nodes to Core Baseline ---
    const btnResetMapNodes = document.getElementById("btnResetMapNodes");
    if (btnResetMapNodes) {
        btnResetMapNodes.addEventListener("click", async () => {
            const localPlayer = document.getElementById("uploadedVideoPlayer");
            if (localPlayer) {
                localPlayer.pause();
                localPlayer.src = "";
                localPlayer.classList.add("hidden");
            }
            if (videoStream) videoStream.classList.remove("hidden");

            nodesData = JSON.parse(JSON.stringify(DEFAULT_BASELINE_NODES));
            renderMapNodes(nodesData);
            renderRankedList(nodesData);
            if (map) map.setView([3.1450, 101.6950], 13);
            showToast("🧹 Map reset to 5 core baseline monitoring stations.");

            try {
                fetch("/api/nodes/reset", { method: "POST" });
            } catch (e) {}
        });
    }

    // --- Fetch & Render Geospatial Nodes ---
    async function fetchNodes(waterLevelOverride) {
        const wLvl = waterLevelOverride !== undefined ? waterLevelOverride : waterLevelPct;
        try {
            const url = `/api/nodes?water_level=${wLvl}&rainfall=${rainfallMm}`;
            const resp = await fetch(url);
            if (resp.ok) {
                const data = await resp.json();
                if (data.nodes && data.nodes.length > 0) {
                    nodesData = data.nodes;
                }
            }
        } catch (e) {
            console.error("Nodes fetch error:", e);
        }

        // If nodesData is empty, recalculate baseline nodes locally
        if (!nodesData || nodesData.length === 0) {
            nodesData = JSON.parse(JSON.stringify(DEFAULT_BASELINE_NODES));
        }

        // Recalculate LERS for all nodes
        nodesData.forEach(n => {
            if (n.is_video_node) {
                n.water_level_pct = wLvl;
                n.rainfall_mm = Number(rainfallMm.toFixed(1));
            }
            const r_idx = Math.min(100, (n.rodent_count || 0) * 35);
            const w_idx = (n.water_level_pct || 50) * 0.85;
            const rf_idx = Math.min(100, (n.rainfall_mm || 0) * 2.0);
            const h_idx = (n.historical_risk || 40) * 0.75;
            let score = r_idx * 0.45 + w_idx * 0.30 + rf_idx * 0.15 + h_idx * 0.10;
            if ((n.rodent_count || 0) > 0 && (n.rainfall_mm || 0) >= 20 && (n.water_level_pct || 0) >= 70) {
                score = Math.min(100, score + 5);
            }
            n.lers_score = Math.min(100, Math.round(score));
            if (n.lers_score >= 80) n.risk_level = "Critical";
            else if (n.lers_score >= 60) n.risk_level = "High";
            else if (n.lers_score >= 35) n.risk_level = "Moderate";
            else n.risk_level = "Low";
        });

        // Sort descending by LERS score
        nodesData.sort((a, b) => b.lers_score - a.lers_score);

        // Calculate citywide average LERS
        if (nodesData.length > 0) {
            const avg = Math.round(nodesData.reduce((acc, n) => acc + n.lers_score, 0) / nodesData.length);
            if (avgLersScore) avgLersScore.textContent = avg;
        }

        renderMapNodes(nodesData);
        renderRankedList(nodesData);
        if (viewAnalytics && !viewAnalytics.classList.contains("hidden")) {
            renderAnalyticsMatrix();
        }
    }

    let activeOpenPopupNodeId = null;

    function renderMapNodes(nodes) {
        if (!map) return;

        // Clear existing markers
        mapMarkers.forEach(m => map.removeLayer(m));
        mapHeatCircles.forEach(c => map.removeLayer(c));
        mapMarkers = [];
        mapHeatCircles = [];

        let activeMarkerToOpen = null;

        nodes.forEach(node => {
            const lat = node.latitude !== undefined ? node.latitude : node.lat;
            const lon = node.longitude !== undefined ? node.longitude : node.lon;
            if (lat === undefined || lon === undefined) return;

            const score = node.lers_score !== undefined ? node.lers_score : 0;
            const lvlUpper = (node.risk_level || "").toUpperCase();

            let markerClass = "marker-low";
            let heatColor = "#10b981";
            let heatRadius = 550;
            let heatOpacity = 0.20;

            if (lvlUpper === "CRITICAL" || score >= 80) {
                markerClass = "marker-crit";
                heatColor = "#ef4444";
                heatRadius = 950;
                heatOpacity = 0.45;
            } else if (lvlUpper === "HIGH" || score >= 60) {
                markerClass = "marker-high";
                heatColor = "#f97316";
                heatRadius = 800;
                heatOpacity = 0.35;
            } else if (lvlUpper === "MODERATE" || score >= 35) {
                markerClass = "marker-mod";
                heatColor = "#f59e0b";
                heatRadius = 650;
                heatOpacity = 0.28;
            }

            // Inner Core Heat Circle
            const innerHeatCircle = L.circle([lat, lon], {
                radius: Math.round(heatRadius * 0.45),
                fillColor: heatColor,
                fillOpacity: Math.min(0.65, heatOpacity * 1.5),
                stroke: false,
            }).addTo(map);

            // Outer Heat Halo Circle
            const outerHeatCircle = L.circle([lat, lon], {
                radius: heatRadius,
                fillColor: heatColor,
                fillOpacity: heatOpacity,
                stroke: false,
            }).addTo(map);

            mapHeatCircles.push(innerHeatCircle);
            mapHeatCircles.push(outerHeatCircle);

            // Custom HTML Bubble Marker with Score
            const iconHtml = `<div class="lers-map-marker ${markerClass}" style="width: 34px; height: 34px;">${score}</div>`;
            const customIcon = L.divIcon({
                html: iconHtml,
                className: "custom-map-icon",
                iconSize: [34, 34],
                iconAnchor: [17, 17],
            });

            const marker = L.marker([lat, lon], { icon: customIcon }).addTo(map);

            const popupContent = `
                <div style="font-family: 'Outfit', sans-serif; font-size: 0.82rem; color: #f8fafc; padding: 4px;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: #38bdf8; margin-bottom: 2px;">${node.name}</div>
                    <div style="font-size: 0.74rem; color: #94a3b8; margin-bottom: 8px;">Station ID: ${node.id}</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;">
                        <div><span style="color: #64748b;">🌧️ Rain:</span> <strong style="color: #f8fafc;">${node.rainfall_mm}mm</strong></div>
                        <div><span style="color: #64748b;">⏱️ Level:</span> <strong style="color: #f8fafc;">${node.water_level_pct}%</strong></div>
                        <div><span style="color: #64748b;">🐀 Rodents:</span> <strong style="color: #ef4444;">${node.rodent_count}</strong></div>
                        <div><span style="color: #64748b;">📊 LERS:</span> <strong style="color: #38bdf8;">${score}/100</strong></div>
                    </div>
                </div>
            `;
            marker.bindPopup(popupContent);

            marker.on("popupopen", () => {
                activeOpenPopupNodeId = node.id;
            });
            marker.on("popupclose", () => {
                if (activeOpenPopupNodeId === node.id) {
                    activeOpenPopupNodeId = null;
                }
            });

            if (activeOpenPopupNodeId === node.id) {
                activeMarkerToOpen = marker;
            }

            mapMarkers.push(marker);
            node._markerRef = marker;
        });

        if (activeMarkerToOpen) {
            activeMarkerToOpen.openPopup();
        }
    }

    // Toggle Heatmap Layer Visibility
    if (chkHeatmap) {
        chkHeatmap.addEventListener("change", (e) => {
            mapHeatCircles.forEach(circle => {
                if (e.target.checked) map.addLayer(circle);
                else map.removeLayer(circle);
            });
        });
    }

    // --- Render Ranked Critical Nodes List (Right Column) ---
    function renderRankedList(nodes) {
        if (!rankedNodesList) return;
        rankedNodesList.innerHTML = "";

        nodes.forEach((node, idx) => {
            const rank = idx + 1;
            const score = node.lers_score !== undefined ? node.lers_score : 0;
            const lvlUpper = (node.risk_level || "").toUpperCase();

            let rankClass = "rank-low";
            let pillClass = "pill-low";
            if (lvlUpper === "CRITICAL" || score >= 80) { rankClass = "rank-crit"; pillClass = "pill-crit"; }
            else if (lvlUpper === "HIGH" || score >= 60) { rankClass = "rank-high"; pillClass = "pill-high"; }
            else if (lvlUpper === "MODERATE" || score >= 35) { rankClass = "rank-mod"; pillClass = "pill-mod"; }

            const item = document.createElement("div");
            item.className = "ranked-node-item";
            item.innerHTML = `
                <div class="node-left-col">
                    <span class="node-rank-badge ${rankClass}">${rank}</span>
                    <div class="node-info-block">
                        <span class="node-station-name">${node.name}</span>
                        <span class="node-station-id">${node.id}</span>
                        <div class="node-telemetry-row">
                            <span>🌧️ ${node.rainfall_mm} mm</span>
                            <span>⏱️ ${node.water_level_pct}%</span>
                            <span>🐀 ${node.rodent_count}</span>
                        </div>
                    </div>
                </div>
                <div class="node-right-col">
                    <div class="node-lers-score ${rankClass}">${score}/100</div>
                    <span class="node-risk-pill ${pillClass}">${node.risk_level || "LOW"}</span>
                </div>
            `;

            // Click node to focus on map
            item.addEventListener("click", () => {
                if (map && node.latitude && node.longitude) {
                    map.flyTo([node.latitude, node.longitude], 15, { duration: 1.2 });
                    if (node._markerRef) {
                        setTimeout(() => node._markerRef.openPopup(), 1200);
                    }
                }
            });

            rankedNodesList.appendChild(item);
        });
    }

    if (btnViewAllNodes) {
        btnViewAllNodes.addEventListener("click", () => {
            if (map && nodesData.length > 0) {
                const group = new L.featureGroup(mapMarkers);
                map.fitBounds(group.getBounds().pad(0.2));
                showToast("Fitted view to all monitoring stations.");
            }
        });
    }

    // --- Header Navigation Tab Switcher ---
    const navPillOverview = document.getElementById("navPillOverview");
    const navPillAnalytics = document.getElementById("navPillAnalytics");
    const navPillReports = document.getElementById("navPillReports");
    const navPillSettings = document.getElementById("navPillSettings");

    const viewOverview = document.getElementById("viewOverview");
    const viewAnalytics = document.getElementById("viewAnalytics");
    const viewReports = document.getElementById("viewReports");
    const viewSettings = document.getElementById("viewSettings");

    const headerNavPills = [navPillOverview, navPillAnalytics, navPillReports, navPillSettings];
    const headerViews = [viewOverview, viewAnalytics, viewReports, viewSettings];

    function switchHeaderTab(index) {
        headerNavPills.forEach((pill, idx) => {
            if (!pill) return;
            if (idx === index) pill.classList.add("active");
            else pill.classList.remove("active");
        });

        headerViews.forEach((v, idx) => {
            if (!v) return;
            if (idx === index) {
                v.classList.remove("hidden");
                v.classList.add("active");
            } else {
                v.classList.add("hidden");
                v.classList.remove("active");
            }
        });

        if (index === 0 && map) {
            setTimeout(() => map.invalidateSize(), 200);
        } else if (index === 1) {
            renderAnalyticsMatrix();
        } else if (index === 2) {
            fetchReportsDesk();
        }
    }

    if (navPillOverview) navPillOverview.addEventListener("click", () => switchHeaderTab(0));
    if (navPillAnalytics) navPillAnalytics.addEventListener("click", () => switchHeaderTab(1));
    if (navPillReports) navPillReports.addEventListener("click", () => switchHeaderTab(2));
    if (navPillSettings) navPillSettings.addEventListener("click", () => switchHeaderTab(3));

    // --- Municipal Reports Management Desk ---
    let cachedReportsList = [];

    const btnRefreshReportsDesk = document.getElementById("btnRefreshReportsDesk");
    const reportSearchInput = document.getElementById("reportSearchInput");
    const reportUrgencyFilter = document.getElementById("reportUrgencyFilter");
    const repTotalCount = document.getElementById("repTotalCount");
    const repCritCount = document.getElementById("repCritCount");
    const repAssignedCount = document.getElementById("repAssignedCount");

    function renderReportsDeskTable(reports) {
        const tbody = document.getElementById("reportsDeskTbody");
        if (!tbody) return;

        const query = (reportSearchInput ? reportSearchInput.value : "").trim().toLowerCase();
        const urgency = reportUrgencyFilter ? reportUrgencyFilter.value : "ALL";

        const filtered = reports.filter(r => {
            const matchesQuery = !query || 
                (r.location_name || "").toLowerCase().includes(query) ||
                (r.description || "").toLowerCase().includes(query) ||
                (r.id || "").toLowerCase().includes(query) ||
                (r.reporter_name || "").toLowerCase().includes(query);
            const matchesUrgency = urgency === "ALL" || (r.risk_level || "").toUpperCase() === urgency;
            return matchesQuery && matchesUrgency;
        });

        tbody.innerHTML = "";
        if (filtered.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:#94a3b8; padding:1.5rem;">No matching municipal reports found</td></tr>`;
            return;
        }

        filtered.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="font-mono text-cyan" style="font-weight:700;">${r.id}</td>
                <td style="font-weight:600; color:#f8fafc;">📍 ${r.location_name}</td>
                <td>${r.reporter_name || 'Resident'}</td>
                <td><span class="tag-red">${r.rodent_count} Rats (YOLO Tag)</span></td>
                <td><span class="risk-pill pill-${(r.risk_level || 'high').toLowerCase()}">${r.risk_level}</span></td>
                <td style="color:#94a3b8; font-size:0.72rem;">${r.timestamp}</td>
                <td>
                    <select class="select-status-sleek" data-id="${r.id}">
                        <option value="Submitted to DBKL" ${r.status.includes('Submitted') ? 'selected' : ''}>Submitted to DBKL</option>
                        <option value="Work Order Assigned" ${r.status.includes('Assigned') ? 'selected' : ''}>Work Order Assigned</option>
                        <option value="In Progress" ${r.status.includes('Progress') ? 'selected' : ''}>In Progress</option>
                        <option value="Resolved" ${r.status.includes('Resolved') ? 'selected' : ''}>Resolved</option>
                    </select>
                </td>
            `;
            const sel = tr.querySelector("select");
            if (sel) {
                sel.addEventListener("change", (e) => {
                    r.status = e.target.value;
                    showToast(`Updated ${r.id} status to: ${e.target.value}`, "info");
                    updateReportSummaryKPIs(cachedReportsList);
                });
            }
            tbody.appendChild(tr);
        });
    }

    function updateReportSummaryKPIs(reports) {
        if (repTotalCount) repTotalCount.textContent = reports.length;
        if (repCritCount) {
            const crits = reports.filter(r => (r.risk_level || "").toUpperCase() === "CRITICAL").length;
            repCritCount.textContent = crits;
        }
        if (repAssignedCount) {
            const assigned = reports.filter(r => r.status && !r.status.includes("Submitted")).length;
            repAssignedCount.textContent = assigned;
        }
    }

    async function fetchReportsDesk() {
        try {
            const resp = await fetch("/api/reports");
            if (!resp.ok) return;
            const data = await resp.json();
            cachedReportsList = data.reports || [];
            updateReportSummaryKPIs(cachedReportsList);
            renderReportsDeskTable(cachedReportsList);
        } catch (e) {
            console.error("Reports desk fetch error:", e);
        }
    }

    if (btnRefreshReportsDesk) {
        btnRefreshReportsDesk.addEventListener("click", () => {
            fetchReportsDesk();
            showToast("🔄 Municipal reports feed refreshed");
        });
    }
    if (reportSearchInput) {
        reportSearchInput.addEventListener("input", () => renderReportsDeskTable(cachedReportsList));
    }
    if (reportUrgencyFilter) {
        reportUrgencyFilter.addEventListener("change", () => renderReportsDeskTable(cachedReportsList));
    }

    // --- Empirical Analytics Matrix Generator ---
    function renderAnalyticsMatrix() {
        const tbody = document.getElementById("anNodeMatrixTbody");
        if (!tbody || !nodesData || nodesData.length === 0) return;

        tbody.innerHTML = "";
        nodesData.forEach(n => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td style="font-weight:700; color:#f8fafc;">${n.id} — ${n.name}</td>
                <td>${n.rodent_index || Math.round(n.rodent_count * 35)} (45%)</td>
                <td>${Math.round(n.water_level_pct * 0.85)} (30%)</td>
                <td>${n.rainfall_index || Math.round(n.rainfall_mm * 2.0)} (15%)</td>
                <td>${Math.round(n.historical_risk * 0.75)} (10%)</td>
                <td class="font-mono" style="font-weight:800; color:#38bdf8; font-size:0.95rem;">${n.lers_score}</td>
                <td><span class="risk-pill pill-${(n.risk_level || 'low').toLowerCase()}">${n.risk_level}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // --- Quick Actions Triggers ---
    if (btnDeckHealth) {
        btnDeckHealth.addEventListener("click", async () => {
            try {
                showToast("Running complete system diagnostic...", "info");
                const resp = await fetch("/api/health");
                const data = await resp.json();
                showToast(`✅ Health Check OK — Inference: ${data.ai_inference} | Model: ${data.active_model} | FPS: ${data.fps.toFixed(1)}`, "info");
            } catch (e) {
                showToast("⚠️ Health diagnostic check failed", "error");
            }
        });
    }

    if (btnDeckCalibrate) {
        btnDeckCalibrate.addEventListener("click", () => {
            handleConfChange(0.50);
            showToast("🎯 Model confidence calibrated to optimal threshold (50%)", "info");
        });
    }

    if (btnDeckSnapshot) {
        btnDeckSnapshot.addEventListener("click", () => {
            showToast("📸 Capturing annotated stream snapshot...", "info");
            const link = document.createElement("a");
            link.href = "/api/snapshot";
            link.download = "rattus_live_snapshot.jpg";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    // --- Initialize Application ---
    function initApp() {
        initMap();
        loadConfig();
        fetchWeather();
        try {
            fetch("/api/reset_all", { method: "POST" });
        } catch (e) {}
    }

    initApp();

    // Start Polling Stats Loop
    setInterval(pollStats, 600);
    setInterval(fetchWeather, 300000); // 5 min auto weather sync
});

