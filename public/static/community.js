/**
 * RATTUS Community Mobile App Client Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- Auth Session Check ---
    const storedUser = localStorage.getItem("rattus_user");
    let currentUser = storedUser ? JSON.parse(storedUser) : { email: "community@example.com", role: "community", name: "Citizen Resident" };

    const commEmailText = document.getElementById("commEmailText");
    const btnCommLogout = document.getElementById("btnCommLogout");

    if (commEmailText) commEmailText.textContent = currentUser.email;

    if (btnCommLogout) {
        btnCommLogout.addEventListener("click", () => {
            localStorage.removeItem("rattus_user");
            window.location.href = "/login";
        });
    }

    // --- Mobile Bottom Navigation ---
    const tabReport = document.getElementById("tabReport");
    const tabMap = document.getElementById("tabMap");
    const tabForum = document.getElementById("tabForum");
    const tabAlerts = document.getElementById("tabAlerts");

    const viewReport = document.getElementById("viewReport");
    const viewMap = document.getElementById("viewMap");
    const viewForum = document.getElementById("viewForum");
    const viewAlerts = document.getElementById("viewAlerts");

    const navItems = [tabReport, tabMap, tabForum, tabAlerts];
    const views = [viewReport, viewMap, viewForum, viewAlerts];

    function switchTab(index) {
        navItems.forEach((item, idx) => {
            if (idx === index) item.classList.add("active");
            else item.classList.remove("active");
        });

        views.forEach((v, idx) => {
            if (idx === index) v.classList.add("active");
            else v.classList.remove("active");
        });

        if (index === 1 && communityMap) {
            setTimeout(() => communityMap.invalidateSize(), 200);
        }
    }

    if (tabReport) tabReport.addEventListener("click", () => switchTab(0));
    if (tabMap) tabMap.addEventListener("click", () => switchTab(1));
    if (tabForum) tabForum.addEventListener("click", () => switchTab(2));
    if (tabAlerts) tabAlerts.addEventListener("click", () => switchTab(3));

    // --- Toast Notifications ---
    const toastContainer = document.getElementById("communityToastContainer");
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

    // --- Video Capture / Upload Converter ---
    const videoCaptureDropzone = document.getElementById("videoCaptureDropzone");
    const citizenVideoInput = document.getElementById("citizenVideoInput");
    const automatedReportCard = document.getElementById("automatedReportCard");
    const citizenReportForm = document.getElementById("citizenReportForm");
    const citizenReportsList = document.getElementById("citizenReportsList");

    if (videoCaptureDropzone && citizenVideoInput) {
        videoCaptureDropzone.addEventListener("click", () => citizenVideoInput.click());

        citizenVideoInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) {
                if (automatedReportCard) automatedReportCard.classList.remove("hidden");
                showToast(`YOLO Vision Engine analyzing footage: ${file.name}...`, "info");
            }
        });
    }

    if (citizenReportForm) {
        citizenReportForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const location = document.getElementById("reportLocation").value;
            const desc = document.getElementById("reportDescription").value;
            const reporter = document.getElementById("reporterName").value;

            try {
                const resp = await fetch("/api/reports", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        location_name: location,
                        latitude: 3.1638,
                        longitude: 101.6980,
                        rodent_count: 5,
                        description: desc,
                        reporter_name: reporter,
                    }),
                });
                const res = await resp.json();
                if (resp.ok) {
                    showToast("🚀 Report submitted & dynamic pin generated on Kuala Lumpur Heatmap!");
                    fetchCitizenReports();
                    fetchCommunityNodes();
                } else {
                    showToast("Report submission failed", "error");
                }
            } catch (err) {
                console.error("Report submit error:", err);
                showToast("Submission error", "error");
            }
        });
    }

    async function fetchCitizenReports() {
        if (!citizenReportsList) return;
        try {
            const resp = await fetch("/api/reports");
            if (!resp.ok) return;
            const data = await resp.json();
            const reports = data.reports || [];

            citizenReportsList.innerHTML = "";
            reports.forEach(r => {
                const item = document.createElement("div");
                item.className = "report-item-card";
                item.innerHTML = `
                    <div class="ric-top">
                        <span class="ric-id">${r.id}</span>
                        <span class="ric-status">${r.status}</span>
                    </div>
                    <div class="ric-loc">📍 ${r.location_name}</div>
                    <div class="ric-desc">${r.description}</div>
                `;
                citizenReportsList.appendChild(item);
            });
        } catch (e) {
            console.error("Fetch reports error:", e);
        }
    }

    // --- Scoreless Community Risk Map ---
    let communityMap = null;
    function initCommunityMap() {
        const container = document.getElementById("communityMap");
        if (!container) return;

        communityMap = L.map("communityMap", {
            zoomControl: false,
            attributionControl: false
        }).setView([3.1450, 101.6950], 13);

        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            maxZoom: 19,
            subdomains: "abcd",
        }).addTo(communityMap);

        L.control.zoom({ position: "topleft" }).addTo(communityMap);

        fetchCommunityNodes();
    }

    async function fetchCommunityNodes() {
        try {
            const resp = await fetch("/api/nodes");
            if (!resp.ok) return;
            const data = await resp.json();
            const nodes = data.nodes || [];

            nodes.forEach(node => {
                let heatColor = "#10b981";
                let markerBg = "#10b981";
                const lvl = (node.risk_level || "").toUpperCase();

                if (lvl === "CRITICAL" || node.lers_score >= 80) { heatColor = "#ef4444"; markerBg = "#ef4444"; }
                else if (lvl === "HIGH" || node.lers_score >= 60) { heatColor = "#f97316"; markerBg = "#f97316"; }
                else if (lvl === "MODERATE" || node.lers_score >= 35) { heatColor = "#f59e0b"; markerBg = "#f59e0b"; }

                // Scoreless Aura Circle
                L.circle([node.latitude, node.longitude], {
                    radius: 750,
                    fillColor: heatColor,
                    fillOpacity: 0.35,
                    stroke: false,
                }).addTo(communityMap);

                // Scoreless Marker Bubble (Icon instead of raw LERS number)
                const iconHtml = `<div class="scoreless-marker" style="width:30px; height:30px; background:${markerBg}; border:2px solid #fff;">🐀</div>`;
                const customIcon = L.divIcon({
                    html: iconHtml,
                    className: "scoreless-map-icon",
                    iconSize: [30, 30],
                    iconAnchor: [15, 15],
                });

                const marker = L.marker([node.latitude, node.longitude], { icon: customIcon }).addTo(communityMap);

                const popupContent = `
                    <div style="font-family:'Outfit',sans-serif; font-size:0.8rem; color:#f8fafc; padding:4px;">
                        <div style="font-weight:700; color:#38bdf8; font-size:0.9rem; margin-bottom:2px;">${node.name}</div>
                        <div style="display:inline-block; font-size:0.65rem; font-weight:700; color:#fff; background:${markerBg}; padding:2px 6px; border-radius:4px; margin-bottom:6px;">${node.risk_level} RISK AREA</div>
                        <div style="font-size:0.72rem; color:#cbd5e1; line-height:1.35;">
                            ⚠️ Public Health Advisory: Rodent activity and rain runoff reported near this zone. Avoid contact with drain floodwater.
                        </div>
                    </div>
                `;
                marker.bindPopup(popupContent);
            });
        } catch (e) {
            console.error("Community nodes fetch error:", e);
        }
    }

    // --- Community Forum (Blogspot Style) ---
    const forumPostsContainer = document.getElementById("forumPostsContainer");
    const categoryPills = document.querySelectorAll(".cat-pill");
    let currentCategory = "all";

    categoryPills.forEach(pill => {
        pill.addEventListener("click", () => {
            categoryPills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            currentCategory = pill.getAttribute("data-cat");
            fetchForumPosts();
        });
    });

    async function fetchForumPosts() {
        if (!forumPostsContainer) return;
        try {
            const resp = await fetch("/api/forum");
            if (!resp.ok) return;
            const data = await resp.json();
            let posts = data.posts || [];

            if (currentCategory !== "all") {
                posts = posts.filter(p => p.tag === currentCategory);
            }

            renderForumPosts(posts);
        } catch (e) {
            console.error("Fetch forum error:", e);
        }
    }

    function renderForumPosts(posts) {
        forumPostsContainer.innerHTML = "";
        posts.forEach(post => {
            const card = document.createElement("div");
            card.className = "forum-post-card";
            card.innerHTML = `
                <div class="post-header">
                    <div class="post-author-row">
                        <div class="post-avatar">👤</div>
                        <div>
                            <div class="post-author-name">${post.author}</div>
                            <div class="post-meta">📍 ${post.location} • ${post.timestamp}</div>
                        </div>
                    </div>
                    <span class="post-tag-badge">${post.tag}</span>
                </div>
                <div class="post-content">${post.content}</div>
                <div class="post-footer-actions">
                    <button class="btn-upvote" data-id="${post.id}">
                        ▲ Upvote <strong class="upvote-count">${post.upvotes}</strong>
                    </button>
                    <span class="comments-count-toggle">💬 ${post.comments ? post.comments.length : 0} Comments</span>
                </div>
                <div class="post-comments-drawer">
                    ${(post.comments || []).map(c => `
                        <div class="comment-item">
                            <span class="comment-author">${c.author}:</span>
                            <span>${c.text}</span>
                        </div>
                    `).join('')}
                    <div class="add-comment-row">
                        <input type="text" placeholder="Add a comment..." class="input-comment-sleek" id="inputComment_${post.id}">
                        <button class="btn-send-comment" data-id="${post.id}">Send</button>
                    </div>
                </div>
            `;

            // Upvote Button Handler
            const upvoteBtn = card.querySelector(".btn-upvote");
            if (upvoteBtn) {
                upvoteBtn.addEventListener("click", async () => {
                    try {
                        const r = await fetch("/api/forum/upvote", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ post_id: post.id }),
                        });
                        const res = await r.json();
                        if (r.ok) {
                            upvoteBtn.querySelector(".upvote-count").textContent = res.upvotes;
                            upvoteBtn.classList.add("upvoted");
                        }
                    } catch (err) {
                        console.error("Upvote error:", err);
                    }
                });
            }

            // Comment Send Handler
            const sendCommentBtn = card.querySelector(".btn-send-comment");
            if (sendCommentBtn) {
                sendCommentBtn.addEventListener("click", async () => {
                    const input = card.querySelector(`#inputComment_${post.id}`);
                    if (!input || !input.value.trim()) return;

                    try {
                        const r = await fetch("/api/forum/comment", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ post_id: post.id, author: "Resident", comment: input.value.trim() }),
                        });
                        if (r.ok) {
                            fetchForumPosts();
                        }
                    } catch (err) {
                        console.error("Comment error:", err);
                    }
                });
            }

            forumPostsContainer.appendChild(card);
        });
    }

    // --- New Post Modal ---
    const btnOpenNewPostModal = document.getElementById("btnOpenNewPostModal");
    const btnCloseNewPostModal = document.getElementById("btnCloseNewPostModal");
    const newPostModal = document.getElementById("newPostModal");
    const newPostForm = document.getElementById("newPostForm");

    if (btnOpenNewPostModal && newPostModal) {
        btnOpenNewPostModal.addEventListener("click", () => newPostModal.classList.remove("hidden"));
    }
    if (btnCloseNewPostModal && newPostModal) {
        btnCloseNewPostModal.addEventListener("click", () => newPostModal.classList.add("hidden"));
    }

    if (newPostForm) {
        newPostForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const author = document.getElementById("postAuthor").value;
            const location = document.getElementById("postLocation").value;
            const tag = document.getElementById("postTag").value;
            const content = document.getElementById("postContent").value;

            try {
                const resp = await fetch("/api/forum", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ author, location, tag, content }),
                });
                if (resp.ok) {
                    newPostModal.classList.add("hidden");
                    newPostForm.reset();
                    showToast("Post published to community feed!");
                    fetchForumPosts();
                }
            } catch (err) {
                console.error("New post error:", err);
            }
        });
    }

    // Initialize Community App
    fetchCitizenReports();
    initCommunityMap();
    fetchForumPosts();
});
