const pdf = document.getElementById("pdf");
const nameEl = document.getElementById("name");
const go = document.getElementById("go");
const language = document.getElementById("language");
const voice = document.getElementById("voice");

const err = document.getElementById("err");

const prog = document.getElementById("prog");
const result = document.getElementById("result");

const st = document.getElementById("st");
const msg = document.getElementById("msg");
const pct = document.getElementById("pct");
const fill = document.getElementById("fill");

const title = document.getElementById("title");
const meta = document.getElementById("meta");

const vid = document.getElementById("vid");
const open = document.getElementById("open");


let selectedFile = null;
let polling = null;


// ============================================================
// FILE SELECTION
// ============================================================

pdf.addEventListener("change", () => {

    selectedFile = pdf.files[0];

    if (!selectedFile) {
        nameEl.textContent = "Choose a PDF";
        return;
    }

    nameEl.textContent = selectedFile.name;

});


// ============================================================
// ERROR
// ============================================================

function showError(message) {

    err.textContent = message;
    err.classList.remove("hidden");

}

function hideError() {

    err.textContent = "";
    err.classList.add("hidden");

}


// ============================================================
// PROGRESS
// ============================================================

function updateProgress(data) {

    const progress = Number(
        data.progress || 0
    );

    pct.textContent = `${progress}%`;

    fill.style.width = `${progress}%`;

    st.textContent =
        data.status === "complete"
            ? "Complete"
            : "Processing";

    msg.textContent =
        data.message || "Processing...";

}


const KOKORO_VOICES = {
    "hinglish": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "hi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "en-us": ["af_alloy", "af_heart", "af_jessica", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_echo", "am_michael", "am_onyx", "am_puck"],
    "en-gb": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    "es": ["ef_dora", "em_alex", "em_santa"],
    "fr": ["ff_siwis"],
    "it": ["if_sara", "im_nicola"],
    "ja": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "pt-br": ["pf_dora", "pm_alex", "pm_santa"],
    "zh": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]
};

function refreshVoices() {
    const voices = KOKORO_VOICES[language.value] || KOKORO_VOICES.hinglish;
    voice.innerHTML = voices.map(v => `<option value="${v}">${v}</option>`).join("");
}

language.addEventListener("change", refreshVoices);
refreshVoices();

// ============================================================
// GENERATE
// ============================================================

go.addEventListener("click", async () => {

    hideError();

    if (!selectedFile) {

        showError(
            "Please select a PDF first."
        );

        return;
    }

    if (
        !selectedFile.name
            .toLowerCase()
            .endsWith(".pdf")
    ) {

        showError(
            "Please upload a PDF file."
        );

        return;
    }


    go.disabled = true;

    prog.classList.remove("hidden");
    result.classList.add("hidden");

    st.textContent = "Uploading";
    msg.textContent = "Uploading PDF...";
    pct.textContent = "0%";
    fill.style.width = "0%";


    try {

        const formData = new FormData();

        formData.append(
            "file",
            selectedFile
        );
        formData.append("narration_language", language.value);
        formData.append("tts_voice", voice.value);


        const response = await fetch(
            "/generate",
            {
                method: "POST",
                body: formData
            }
        );


        if (!response.ok) {

            const text =
                await response.text();

            throw new Error(
                text || "Generation failed."
            );
        }


        const data =
            await response.json();


        if (!data.job_id) {

            throw new Error(
                "Backend did not return a job ID."
            );
        }


        pollStatus(
            data.job_id
        );


    } catch (error) {

        console.error(error);

        showError(
            error.message ||
            "Could not start generation."
        );

        go.disabled = false;
    }

});


// ============================================================
// POLL STATUS
// ============================================================

function pollStatus(jobId) {

    if (polling) {
        clearInterval(polling);
    }


    async function check() {

        try {

            const response =
                await fetch(
                    `/status/${jobId}`
                );


            if (!response.ok) {

                throw new Error(
                    "Could not read job status."
                );
            }


            const data =
                await response.json();


            console.log(
                "JOB STATUS:",
                data
            );


            updateProgress(data);


            // ---------------------------------------------
            // ERROR
            // ---------------------------------------------

            if (
                data.status === "error" ||
                data.error
            ) {

                clearInterval(polling);

                showError(
                    data.error ||
                    data.message ||
                    "Tutorial generation failed."
                );

                go.disabled = false;

                return;
            }


            // ---------------------------------------------
            // COMPLETE
            // ---------------------------------------------

            if (
                data.status === "complete" ||
                data.status === "completed" ||
                data.video_url
            ) {

                clearInterval(polling);

                displayVideo(
                    data
                );

                go.disabled = false;

                return;
            }


        } catch (error) {

            console.error(
                "Status error:",
                error
            );

            clearInterval(polling);

            showError(
                error.message ||
                "Unable to check generation status."
            );

            go.disabled = false;
        }

    }


    check();

    polling = setInterval(
        check,
        1000
    );
}


// ============================================================
// DISPLAY VIDEO
// ============================================================

function displayVideo(data) {

    console.log(
        "FINAL JOB DATA:",
        data
    );


    let videoUrl =
        data.video_url ||
        data.video ||
        data.url;


    if (!videoUrl) {

        showError(
            "Tutorial was generated, but the backend did not provide a video URL."
        );

        return;
    }


    // If backend accidentally returns a Windows path,
    // convert it to the media URL.
    if (
        videoUrl.includes("\\") ||
        /^[A-Za-z]:/.test(videoUrl)
    ) {

        console.warn(
            "Backend returned filesystem path:",
            videoUrl
        );

        const match =
            videoUrl.match(
                /jobs[\\/]+([^\\/]+)[\\/]+([^\\/]+\.mp4)$/i
            );

        if (match) {

            videoUrl =
                `/media/${match[1]}/${match[2]}`;

        }

    }


    // Prevent browser caching an old generated video.
    const cacheBuster =
        videoUrl.includes("?")
            ? "&"
            : "?";

    const playableUrl =
        `${videoUrl}${cacheBuster}t=${Date.now()}`;


    console.log(
        "VIDEO URL:",
        playableUrl
    );


    // ---------------------------------------------
    // Set video
    // ---------------------------------------------

    vid.pause();

    vid.removeAttribute("src");

    vid.load();

    vid.src = playableUrl;

    vid.controls = true;
    vid.playsInline = true;

    vid.preload = "metadata";


    // ---------------------------------------------
    // Open-video link
    // ---------------------------------------------

    open.href =
        playableUrl;


    // ---------------------------------------------
    // Metadata
    // ---------------------------------------------

    title.textContent =
        data.title ||
        "Teamcenter Tutorial";


    meta.textContent =
        data.message ||
        "Tutorial generated successfully";


    // ---------------------------------------------
    // Show result card
    // ---------------------------------------------

    result.classList.remove(
        "hidden"
    );


    // ---------------------------------------------
    // Load video
    // ---------------------------------------------

    vid.load();


    vid.onloadedmetadata = () => {

        console.log(
            "VIDEO LOADED:",
            vid.videoWidth,
            "x",
            vid.videoHeight,
            "duration:",
            vid.duration
        );

    };


    vid.onerror = () => {

        console.error(
            "VIDEO LOAD ERROR:",
            vid.error
        );

        showError(
            "The tutorial was generated, but the browser could not load the MP4. Check the video URL and MP4 file."
        );

    };

}