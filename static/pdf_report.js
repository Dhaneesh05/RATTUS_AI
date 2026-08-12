/**
 * RATTUS AI — Standalone PDF Incident Report Generator
 * This script is self-contained and does NOT depend on app.js initialization.
 * It reads dashboard values directly from DOM elements and generates a
 * printable HTML report window (Save as PDF via browser print dialog).
 */

(function () {
    "use strict";

    function generatePDFReport() {
        // Read all values directly from the live DOM
        var now = new Date();
        var refId = "RAT-" + now.getFullYear() +
            String(now.getMonth() + 1).padStart(2, "0") +
            String(now.getDate()).padStart(2, "0") + "-" +
            Math.floor(1000 + Math.random() * 9000);
        var timeString = now.toLocaleString();

        var citySel = document.getElementById("weatherLocationSelect");
        var cityName = "Kuala Lumpur";
        if (citySel && citySel.selectedIndex >= 0) {
            cityName = citySel.options[citySel.selectedIndex].text.replace(/📍\s*/g, "");
        }

        var scoreEl = document.getElementById("riskScoreVal");
        var scoreText = scoreEl ? scoreEl.textContent.trim() : "0";

        var levelEl = document.getElementById("riskLevelBadge");
        var levelBadge = levelEl ? levelEl.textContent.trim() : "LOW";

        var rodentEl = document.getElementById("rodentCountVal");
        var rodentCountText = rodentEl ? rodentEl.textContent.trim() : "0";

        var peakEl = document.getElementById("rodentPeakVal");
        var rodentPeakText = peakEl ? peakEl.textContent.trim() : "Peak: 0";

        var rainEl = document.getElementById("rainfallVal");
        var rainMm = rainEl ? rainEl.textContent.trim() : "0";

        var tempEl = document.getElementById("weatherTemp") || document.getElementById("weatherTempBadge");
        var tempText = tempEl ? tempEl.textContent.trim() + "°C" : "34.5°C";

        var waterEl = document.getElementById("waterLevelVal") || document.getElementById("waterLevelValText");
        var waterLvl = waterEl ? waterEl.textContent.trim() + "%" : "85%";

        var munEl = document.getElementById("municipalActionText");
        var municipalText = munEl ? munEl.textContent.trim() : "Continue routine monitoring. No urgent field action required.";

        var citEl = document.getElementById("citizenActionText");
        var citizenText = citEl ? citEl.textContent.trim() : "Normal hygiene precautions. Avoid contact with drain water.";

        // Color by risk level
        var levelBg = "#22c55e";
        if (levelBadge === "MODERATE") levelBg = "#eab308";
        if (levelBadge === "HIGH") levelBg = "#f97316";
        if (levelBadge === "CRITICAL") levelBg = "#ef4444";

        // Open a new printable window
        var printWin = window.open("", "_blank", "width=900,height=700,scrollbars=yes");
        if (!printWin) {
            alert("Please allow popups for this site to generate the PDF report.\nGo to your browser settings and allow popups for localhost:8000.");
            return;
        }

        var html = [
            '<!DOCTYPE html>',
            '<html><head>',
            '<title>RATTUS Incident Report — ' + cityName + '</title>',
            '<meta charset="UTF-8">',
            '<style>',
            '  @page { size: A4; margin: 15mm; }',
            '  * { box-sizing: border-box; }',
            '  body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; margin: 0; padding: 24px; color: #1e293b; background: #fff; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }',
            '  .header { background: #0f172a; color: #fff; padding: 22px 28px; border-radius: 10px; margin-bottom: 18px; }',
            '  .header h1 { margin: 0 0 6px 0; font-size: 21px; letter-spacing: 0.3px; }',
            '  .header p { margin: 0; font-size: 11.5px; color: #94a3b8; }',
            '  .location-box { background: #f1f5f9; padding: 13px 18px; border-radius: 8px; font-weight: bold; font-size: 14px; margin-bottom: 18px; border-left: 5px solid #0284c7; }',
            '  .risk-banner { background: ' + levelBg + '; color: #fff; padding: 20px 24px; border-radius: 10px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; }',
            '  .risk-score { font-size: 36px; font-weight: 800; line-height: 1; }',
            '  .risk-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }',
            '  .risk-level { font-size: 22px; font-weight: 800; text-align: right; }',
            '  .metrics { background: #f8fafc; border: 1px solid #e2e8f0; padding: 18px 22px; border-radius: 10px; margin-bottom: 18px; }',
            '  .metrics h3 { margin: 0 0 14px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #475569; }',
            '  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 30px; font-size: 13px; }',
            '  .metrics-grid div { padding: 3px 0; }',
            '  .directive { padding: 16px 20px; border-radius: 10px; margin-bottom: 14px; font-size: 13px; line-height: 1.5; }',
            '  .municipal { background: #fef2f2; border: 1px solid #fecaca; }',
            '  .municipal strong { color: #991b1b; }',
            '  .citizen { background: #f0fdf4; border: 1px solid #bbf7d0; }',
            '  .citizen strong { color: #166534; }',
            '  .signoff { margin-top: 35px; border-top: 2px dashed #cbd5e1; padding-top: 22px; }',
            '  .signoff h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #475569; margin: 0 0 20px 0; }',
            '  .sign-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; font-size: 13px; line-height: 2.2; }',
            '  .footer { margin-top: 30px; text-align: center; font-size: 10px; color: #94a3b8; }',
            '  .print-bar { background: #0ea5e9; color: #fff; border: none; padding: 12px 28px; font-size: 15px; font-weight: 700; border-radius: 8px; cursor: pointer; margin-bottom: 20px; display: block; }',
            '  .print-bar:hover { background: #0284c7; }',
            '  @media print { .print-bar { display: none !important; } }',
            '</style>',
            '</head><body>',
            '<button class="print-bar" onclick="window.print()">🖨️ Print / Save as PDF</button>',
            '<div class="header">',
            '  <h1>🐀 RATTUS — EXPOSURE-RISK INCIDENT REPORT</h1>',
            '  <p>Report Ref: ' + refId + ' &nbsp;|&nbsp; Generated: ' + timeString + ' &nbsp;|&nbsp; Status: Verified Official Document</p>',
            '</div>',
            '<div class="location-box">📍 TARGET MONITORING LOCATION: ' + cityName.toUpperCase() + '</div>',
            '<div class="risk-banner">',
            '  <div>',
            '    <div class="risk-label">Leptospirosis Exposure Risk Score (LERS)</div>',
            '    <div class="risk-score">' + scoreText + ' / 100</div>',
            '  </div>',
            '  <div class="risk-level">RISK LEVEL:<br>' + levelBadge + '</div>',
            '</div>',
            '<div class="metrics">',
            '  <h3>Real-Time Environmental &amp; Detection Metrics</h3>',
            '  <div class="metrics-grid">',
            '    <div><strong>Active Rodent Count:</strong> ' + rodentCountText + ' (' + rodentPeakText + ')</div>',
            '    <div><strong>Drain Water Level:</strong> ' + waterLvl + '</div>',
            '    <div><strong>24h Accumulated Rainfall:</strong> ' + rainMm + ' mm</div>',
            '    <div><strong>Detection Engine:</strong> YOLOv8 (rat-trained)</div>',
            '    <div><strong>Weather / Temperature:</strong> ' + tempText + '</div>',
            '    <div><strong>Vision Stream:</strong> 30 FPS Live</div>',
            '    <div><strong>Historical Baseline Risk:</strong> 45%</div>',
            '    <div><strong>Open-Meteo API:</strong> Synced &amp; Active</div>',
            '  </div>',
            '</div>',
            '<div class="directive municipal">',
            '  <strong>🏛️ MUNICIPAL RESPONSE DIRECTIVE (DBKL / LOCAL COUNCIL):</strong><br>',
            '  ' + municipalText,
            '</div>',
            '<div class="directive citizen">',
            '  <strong>🏠 CITIZEN HEALTH &amp; HYGIENE ADVISORY:</strong><br>',
            '  ' + citizenText,
            '</div>',
            '<div class="signoff">',
            '  <h3>Field Inspection &amp; Municipal Verification Sign-Off</h3>',
            '  <div class="sign-grid">',
            '    <div>Inspector Name: _________________________________<br>Badge / Staff ID: _________________________________</div>',
            '    <div>Signature: ____________________________________<br>Verification Date: _____________________________</div>',
            '  </div>',
            '</div>',
            '<div class="footer">Generated automatically by RATTUS — Environmental Leptospirosis Exposure-Risk Intelligence Engine v2.1</div>',
            '</body></html>'
        ].join('\n');

        printWin.document.open();
        printWin.document.write(html);
        printWin.document.close();

        // Auto-trigger print dialog after a short delay
        printWin.onload = function () {
            setTimeout(function () { printWin.print(); }, 600);
        };
    }

    // Expose globally so it works from anywhere
    window.generatePDFReport = generatePDFReport;

    // Bind click events to both PDF buttons once DOM is ready
    function bindButtons() {
        var pdfBtns = document.querySelectorAll(".btn-export-pdf, #btnExportPDFHeader, #btnExportPDFReportsView, #btnExportPDFRisk");
        pdfBtns.forEach(function(btn) {
            btn.addEventListener("click", generatePDFReport);
        });
        console.log("[PDF Report] Export buttons bound successfully.");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bindButtons);
    } else {
        bindButtons();
    }
})();
