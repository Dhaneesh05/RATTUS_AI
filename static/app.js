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

    const drainRiskSlider = document.getElementById("drainRiskSlider");
    const drainSliderValText = document.getElementById("drainSliderValText");

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
    const expDrain = document.getElementById("expDrain");
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
    let drainRiskScore = 55.0;
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
            drain_risk: drainRiskScore,
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
            expDrain.textContent = `${res.drain_index}/100`;
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
        try {
            const resp = await fetch("/api/weather");
            const data = await resp.json();
            if (data.rainfall_mm !== undefined) {
                rainfallMm = data.rainfall_mm;
                rainfallVal.textContent = rainfallMm.toFixed(1);
                recalculateRisk();
            }
        } catch (e) {
            console.error("Weather sync failed:", e);
        } finally {
            btnRefreshWeather.textContent = "🔄 Sync";
        }
    }

    // Event Listeners
    drainRiskSlider.addEventListener("input", (e) => {
        drainRiskScore = parseFloat(e.target.value);
        drainSliderValText.textContent = `${drainRiskScore.toFixed(0)}/100`;
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

    // Init
    loadConfig();
    recalculateRisk();
    setInterval(fetchStats, 500);
});
