let traineeVideos = [];
let selectedTutorialId = null;
let tutorialActions = [];

/* ============================================================
TRAINEE LIBRARY
============================================================ */

async function loadTraineeVideos() {
const grid = document.getElementById("videoGrid");

if (!grid) {
    return;
}

try {
    /*
     * Expected backend response:
     *
     * {
     *   "videos": [
     *      {
     *          "id": "...",
     *          "title": "...",
     *          "description": "...",
     *          "video_url": "...",
     *          "published": true
     *      }
     *   ]
     * }
     */

    const response = await fetch("/api/trainee/videos");

    if (!response.ok) {
        throw new Error("Unable to load tutorials");
    }

    const data = await response.json();

    traineeVideos = data.videos || [];

    renderTraineeVideos(traineeVideos);

} catch (error) {
    console.error(error);

    /*
     * Keeps the frontend usable while the backend route
     * is being connected.
     */
    grid.innerHTML = `
        <div class="loading-card">
            No published tutorials are available yet.
        </div>
    `;
}

}

function renderTraineeVideos(videos) {

const grid = document.getElementById("videoGrid");
const count = document.getElementById("videoCount");

if (!grid) {
    return;
}

if (count) {
    count.textContent = videos.length;
}

if (!videos.length) {
    grid.innerHTML = `
        <div class="loading-card">
            No published tutorials are available.
        </div>
    `;
    return;
}

grid.innerHTML = videos.map(video => {

    const title = escapeHtml(
        video.title || "Untitled Tutorial"
    );

    const description = escapeHtml(
        video.description ||
        "Interactive Teamcenter training tutorial."
    );

    const id = encodeURIComponent(video.id);

    return `
        <article class="trainee-video-card">

            <div class="trainee-video-thumbnail" aria-hidden="true">
                <span class="trainee-video-icon">▶</span>
            </div>

            <div class="trainee-video-content">
                <div class="trainee-video-kicker">${title}</div>
                <h3>Complete Tutorial</h3>
                <p>${description}</p>

                <div class="trainee-video-footer">
                    <span class="trainee-video-status">Published</span>
                    <button
                        class="watch-button"
                        onclick="openTutorial('${id}')">
                        Watch Tutorial
                    </button>
                </div>
            </div>

        </article>
    `;

}).join("");

}

/* ============================================================
SEARCH
============================================================ */

function filterVideos() {

const input = document.getElementById("searchInput");

if (!input) {
    return;
}

const query = input.value
    .trim()
    .toLowerCase();

const filtered = traineeVideos.filter(video => {

    const title =
        (video.title || "").toLowerCase();

    const description =
        (video.description || "").toLowerCase();

    return (
        title.includes(query) ||
        description.includes(query)
    );
});

renderTraineeVideos(filtered);

}

/* ============================================================
OPEN TUTORIAL
============================================================ */

function openTutorial(id) {

window.location.href =
    "/trainee/videos/" + id;

}

/* ============================================================
WATCH PAGE
============================================================ */

async function loadTutorial() {

const videoElement =
    document.getElementById("tutorialVideo");

if (!videoElement) {
    return;
}

const pathParts =
    window.location.pathname.split("/");

const rawId =
    pathParts[pathParts.length - 1];

const id =
    decodeURIComponent(rawId);

try {

    const response =
        await fetch(
            "/api/trainee/videos/" +
            encodeURIComponent(id)
        );

    if (!response.ok) {
        throw new Error(
            "Tutorial not found"
        );
    }

    const tutorial =
        await response.json();

    renderTutorial(tutorial);

} catch (error) {

    console.error(error);

    document.getElementById(
        "tutorialTitle"
    ).textContent =
        "Tutorial unavailable";
}

}

function renderTutorial(tutorial) {

    const title =
        tutorial.title ||
        "Teamcenter Tutorial";

    const description =
        tutorial.description ||
        "";

    const generatedDescription =
        "Tutorial generated from UI screenshots.";

    const visibleDescription =
        description === generatedDescription
            ? ""
            : description;

    const videoUrl =
        tutorial.video_url ||
        tutorial.videoUrl ||
        "";

    document.title =
        title + " - Teamcenter AI Studio";

    document.getElementById(
        "tutorialTitle"
    ).textContent = title;

    document.getElementById(
        "tutorialDescription"
    ).textContent = visibleDescription;

    const topicIntro =
        tutorial.topic_intro ||
        tutorial.intro ||
        "";

    document.getElementById(
        "topicIntro"
    ).textContent =
        topicIntro === generatedDescription
            ? ""
            : topicIntro;

    const video =
        document.getElementById(
            "tutorialVideo"
        );

    video.src = videoUrl;

    renderActions(
        tutorial.actions ||
        tutorial.steps ||
        []
    );

    renderPrerequisites(
        tutorial.prerequisites ||
        []
    );

    /*
     * Change the dialogue according to
     * the current position of the video.
     */
    video.addEventListener(
        "timeupdate",
        function () {

            if (
                !video.duration ||
                !tutorialActions.length
            ) {
                return;
            }

            const progress =
                video.currentTime /
                video.duration;

            const index = Math.min(
                tutorialActions.length - 1,
                Math.floor(
                    progress *
                    tutorialActions.length
                )
            );

            setCurrentAction(index);
        }
    );

    /*
     * Make sure the first action is shown
     * immediately when metadata is available.
     */
    video.addEventListener(
        "loadedmetadata",
        function () {

            if (tutorialActions.length) {
                setCurrentAction(0);
            }

        }
    );
}


function renderActions(actions) {

    tutorialActions = actions
        .map(action => {

            /*
             * If the backend already provides
             * a string, keep it.
             */
            if (typeof action === "string") {

                return {
                    title: action,
                    description: ""
                };
            }

            /*
             * Support the different action formats
             * already used by the tutorial pipeline.
             */
            const title =
                action.title ||
                action.action ||
                action.caption ||
                "";

            const description =
                action.description ||
                action.content ||
                action.narration ||
                "";

            return {
                title: title,
                description: description
            };

        })
        .filter(action =>
            action.title ||
            action.description
        )
        .slice(0, 8);

    const titleElement =
        document.getElementById(
            "currentActionTitle"
        );

    const descriptionElement =
        document.getElementById(
            "currentActionDescription"
        );

    const progressElement =
        document.getElementById(
            "actionProgress"
        );

    if (!tutorialActions.length) {

        if (titleElement) {
            titleElement.textContent =
                "Follow the tutorial";
        }

        if (descriptionElement) {
            descriptionElement.textContent =
                "Follow the actions demonstrated in the video.";
        }

        if (progressElement) {
            progressElement.textContent =
                "1 / 1";
        }

        return;
    }

    setCurrentAction(0);
}


function setCurrentAction(index) {

    if (!tutorialActions.length) {
        return;
    }

    const safeIndex =
        Math.max(
            0,
            Math.min(
                index,
                tutorialActions.length - 1
            )
        );

    const action =
        tutorialActions[safeIndex];

    const numberElement =
        document.querySelector(
            ".action-number"
        );

    const titleElement =
        document.getElementById(
            "currentActionTitle"
        );

    const descriptionElement =
        document.getElementById(
            "currentActionDescription"
        );

    const progressElement =
        document.getElementById(
            "actionProgress"
        );

    if (numberElement) {

        numberElement.textContent =
            safeIndex + 1;
    }

    if (titleElement) {

        titleElement.textContent =
            action.title ||
            "Current action";
    }

    if (descriptionElement) {

        descriptionElement.textContent =
            action.description ||
            "Follow the action demonstrated in the video.";
    }

    if (progressElement) {

        progressElement.textContent =
            `${safeIndex + 1} / ${tutorialActions.length}`;
    }

}

function renderPrerequisites(prerequisites) {

const container =
    document.getElementById(
        "prerequisiteList"
    );

if (!container) {
    return;
}

if (!prerequisites.length) {

    container.innerHTML = `
        <div class="empty-state">
            There is no prerequisite video for this. You're good to go.
        </div>
    `;

    return;
}

const validPrerequisites = prerequisites.filter(item => {
    if (typeof item === "string") {
        return !!item.trim();
    }

    const url = item && (item.url || item.video_url);
    const title = item && (item.title || url || "");
    return !!(url || title);
});

if (!validPrerequisites.length) {
    container.innerHTML = `
        <div class="empty-state">
            There is no prerequisite video for this. You're good to go.
        </div>
    `;
    return;
}

container.innerHTML =
    validPrerequisites
        .map(item => {

            const url =
                typeof item === "string"
                    ? item
                    : item.url || item.video_url;

            const title =
                typeof item === "string"
                    ? item
                    : item.title || url;

            if (!url) {
                return "";
            }

            return `
                <div class="prerequisite-item">

                    <span>▶</span>

                    <a
                        href="${escapeAttribute(url)}"
                        target="_blank"
                        rel="noopener noreferrer">
                        ${escapeHtml(title)}
                    </a>

                </div>
            `;
        })
        .join("");

}

/* ============================================================
TRAINER
============================================================ */

let trainerVideos = [];

async function loadTrainerVideos() {

const grid =
    document.getElementById(
        "trainerGrid"
    );

if (!grid) {
    return;
}

try {

    const response =
        await fetch(
            "/api/trainer/videos"
        );

    if (!response.ok) {
        throw new Error(
            "Unable to load generated videos"
        );
    }

    const data =
        await response.json();

    trainerVideos =
        data.videos || [];

    renderTrainerVideos(
        trainerVideos
    );

} catch (error) {

    console.error(error);

    grid.innerHTML = `
        <div class="loading-card">
            No generated tutorials found.
        </div>
    `;
}

}

function renderTrainerVideos(videos) {

const grid =
    document.getElementById(
        "trainerGrid"
    );

if (!grid) {
    return;
}

if (!videos.length) {

    grid.innerHTML = `
        <div class="loading-card">
            No generated tutorials found.
        </div>
    `;

    return;
}

grid.innerHTML =
    videos.map(video => {

        const id =
            encodeURIComponent(
                video.id
            );

        const title =
            escapeHtml(
                video.title ||
                "Untitled Tutorial"
            );

        const published =
            Boolean(video.published);

        return `
            <article class="trainer-card">

                <div class="trainer-info">

                    <h3>${title}</h3>

                    <p>
                        Generated tutorial
                    </p>

                    <span class="status-badge ${
                        published
                            ? "status-published"
                            : "status-draft"
                    }">
                        ${
                            published
                                ? "Published"
                                : "Draft"
                        }
                    </span>

                </div>

                <div class="trainer-actions">

                    <button
                        class="secondary-button"
                        onclick="openPrerequisiteModal('${id}')">
                        Prerequisites
                    </button>

                    <button
                        class="primary-button"
                        onclick="togglePublish('${id}', ${!published})">
                        ${
                            published
                                ? "Unpublish"
                                : "Publish"
                        }
                    </button>

                </div>

            </article>
        `;

    }).join("");

}

/* ============================================================
PUBLISH
============================================================ */

async function togglePublish(id, publish) {

try {

    const response =
        await fetch(
            "/api/trainer/publish/" +
            encodeURIComponent(id),
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    published: publish
                })
            }
        );

    if (!response.ok) {
        throw new Error(
            "Publish operation failed"
        );
    }

    await loadTrainerVideos();

} catch (error) {

    console.error(error);

    alert(
        "Unable to update publication status."
    );
}

}

/* ============================================================
PREREQUISITES
============================================================ */

function openPrerequisiteModal(id) {

selectedTutorialId =
    decodeURIComponent(id);

const tutorial =
    trainerVideos.find(
        video =>
            String(video.id) ===
            String(selectedTutorialId)
    );

const modal =
    document.getElementById(
        "prerequisiteModal"
    );

const title =
    document.getElementById(
        "modalTitle"
    );

if (title && tutorial) {
    title.textContent =
        "Prerequisites — " +
        (tutorial.title ||
         "Tutorial");
}

renderExistingPrerequisites(
    tutorial
        ? tutorial.prerequisites || []
        : []
);

modal.classList.remove("hidden");

}

function closePrerequisiteModal() {

const modal =
    document.getElementById(
        "prerequisiteModal"
    );

if (modal) {
    modal.classList.add(
        "hidden"
    );
}

selectedTutorialId = null;

}

function renderExistingPrerequisites(
prerequisites
) {

const container =
    document.getElementById(
        "existingPrerequisites"
    );

if (!container) {
    return;
}

if (!prerequisites.length) {

    container.innerHTML = `
        <div class="empty-state">
            No prerequisites added yet.
        </div>
    `;

    return;
}

container.innerHTML =
    prerequisites.map(item => {

        const url =
            typeof item === "string"
                ? item
                : item.url ||
                  item.video_url;

        return `
            <div class="prerequisite-item">
                <span>▶</span>
                <a
                    href="${escapeAttribute(url)}"
                    target="_blank"
                    rel="noopener noreferrer">
                    ${escapeHtml(url)}
                </a>
            </div>
        `;
    }).join("");

}

async function addPrerequisite() {

if (!selectedTutorialId) {
    return;
}

const input =
    document.getElementById(
        "prerequisiteUrl"
    );

const url =
    input.value.trim();

if (!url) {
    return;
}

try {

    const response =
        await fetch(
            "/api/trainer/prerequisites/" +
            encodeURIComponent(
                selectedTutorialId
            ),
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    url: url
                })
            }
        );

    if (!response.ok) {
        throw new Error(
            "Unable to add prerequisite"
        );
    }

    input.value = "";

    await loadTrainerVideos();

    const tutorial =
        trainerVideos.find(
            video =>
                String(video.id) ===
                String(selectedTutorialId)
        );

    renderExistingPrerequisites(
        tutorial
            ? tutorial.prerequisites || []
            : []
    );

} catch (error) {

    console.error(error);

    alert(
        "Unable to add prerequisite."
    );
}

}

/* ============================================================
SECURITY / DISPLAY HELPERS
============================================================ */

function escapeHtml(value) {

return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

}

function escapeAttribute(value) {

return escapeHtml(value)
    .replaceAll("`", "&#096;");

}