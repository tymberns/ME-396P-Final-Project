import hashlib
import logging
import json
import math
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd
import streamlit as st
from supabase import create_client


# ----------------------------------
# 1. Configuration and Supabase
# ----------------------------------

logger = logging.getLogger(__name__)

MOT_FOLDER = "OpenSimData/Kinematics"
GAIT_SCRIPT = Path(__file__).resolve().parent / "gait_score.py"
MAX_ZIP_SIZE = 200 * 1024 * 1024  # 200 MB compressed upload
MAX_MOT_SIZE = 20 * 1024 * 1024  # 20 MB uncompressed MOT


@st.cache_resource
def get_database():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


try:
    supabase = get_database()
except Exception:
    logger.exception("Failed to initialize Supabase")
    st.error("The leaderboard service is unavailable. Please try again later.")
    st.stop()


# ----------------------------------
# 2. Extract all MOT files and calculate gait score
# ----------------------------------

def read_mot_files_from_zip(zip_bytes):
    """Return [(original ZIP path, MOT bytes), ...] in deterministic order."""
    if len(zip_bytes) > MAX_ZIP_SIZE:
        raise ValueError("The ZIP exceeds the 200 MB upload limit.")

    with ZipFile(BytesIO(zip_bytes)) as archive:
        matches = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            # Allow an optional outer ZIP folder; require MOT files directly
            # inside OpenSimData/Kinematics, regardless of capitalization.
            parts = info.filename.replace("\\", "/").split("/")
            if (len(parts) >= 3
                    and parts[-3].casefold() == "opensimdata"
                    and parts[-2].casefold() == "kinematics"
                    and parts[-1].casefold().endswith(".mot")
                    and parts[-1] not in ("", ".", "..")
                    and all(part not in ("", ".", "..") for part in parts)):
                matches.append(info)

        if not matches:
            raise FileNotFoundError(
                f"No .mot files found directly inside '{MOT_FOLDER}'."
            )
        if len(matches) > 100:
            raise ValueError("The ZIP contains more than 100 MOT files.")

        matches.sort(key=lambda info: (info.filename.casefold(), info.filename))
        files = []
        total_extracted = 0
        for info in matches:
            if info.file_size > MAX_MOT_SIZE:
                raise ValueError(f"MOT file exceeds 20 MB: {info.filename}")
            with archive.open(info) as entry:
                contents = entry.read(MAX_MOT_SIZE + 1)
            if len(contents) > MAX_MOT_SIZE:
                raise ValueError(f"MOT file exceeds 20 MB: {info.filename}")
            total_extracted += len(contents)
            if total_extracted > MAX_ZIP_SIZE:
                raise ValueError("Combined extracted MOT files exceed 200 MB.")
            files.append((info.filename, contents))
    return files


def score_uploaded_mots(mot_files):
    """Run the unchanged gait_score.py CLI, returning score, report, chart, hash."""
    if not GAIT_SCRIPT.is_file():
        raise RuntimeError("gait_score.py is missing from the website directory.")

    # Hash only MOT contents, not filenames or ZIP paths. Sorting the
    # individual hashes makes file order irrelevant while preserving duplicates.
    digest = hashlib.sha256()
    content_hashes = sorted(
        hashlib.sha256(contents).digest() for _, contents in mot_files
    )
    for content_hash in content_hashes:
        digest.update(content_hash)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        file_paths = []
        used_names = set()
        for number, (name, contents) in enumerate(mot_files, start=1):
            # Only use the basename, never a path from the ZIP archive.
            basename = name.replace("\\", "/").split("/")[-1]
            if basename.casefold() in used_names:
                basename = f"trial_{number:03d}.mot"
            used_names.add(basename.casefold())
            file_path = temp_path / basename
            file_path.write_bytes(contents)
            file_paths.append(str(file_path))

        chart_path = temp_path / "chart.png"
        json_path = temp_path / "result.json"
        command = [
            sys.executable, str(GAIT_SCRIPT), *file_paths,
            "--plot", str(chart_path), "--json", str(json_path),
        ]
        try:
            process = subprocess.run(
                command,
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=None,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Gait analysis timed out after 120 seconds.") from exc

        if process.returncode != 0:
            details = (process.stderr or process.stdout).strip()
            raise ValueError("Gait analysis failed: " + details[:2000])
        if not json_path.is_file() or not chart_path.is_file():
            raise ValueError("Gait analysis did not generate its score and chart.")

        result = json.loads(json_path.read_text(encoding="utf-8"))
        score = round(float(result["score"]), 3)
        if not math.isfinite(score):
            raise ValueError("Gait analysis returned an invalid score.")
        return score, process.stdout, chart_path.read_bytes(), digest.hexdigest()


@st.dialog("Gait Analysis Results", width="large")
def show_gait_results(report, chart_bytes):
    st.subheader("Command-line report")
    st.code(report, language="text")
    st.subheader("Score vs. number of steps")
    st.image(chart_bytes, caption="Chart generated by gait_score.py", use_container_width=True)


# ----------------------------------
# 4. Username session state
# ----------------------------------

if "username" not in st.session_state:
    st.session_state.username = ""


def clear_username():
    st.session_state.username = ""


# ----------------------------------
# 5. Streamlit website and submission
# ----------------------------------

st.title("Gait Efficiency Leaderboard", anchor=False)
st.write(
    "Enter your username and upload a ZIP containing one or more `.mot` files "
    f"directly inside `{MOT_FOLDER}` to compete on the leaderboard!"
)

username = st.text_input(
    "Enter your username",
    max_chars=30,
    key="username",
)
st.button("Clear Username", on_click=clear_username)

uploaded_file = st.file_uploader(
    "Choose a ZIP file",
    type=["zip"],
)

if uploaded_file is not None:
    try:
        zip_bytes = uploaded_file.getvalue()
        upload_id = hashlib.sha256(zip_bytes).hexdigest()

        # Cache the most recently analyzed upload across Streamlit reruns
        # (e.g., typing a username or clicking a button).
        if st.session_state.get("analyzed_upload_id") != upload_id:
            with st.spinner("Analyzing gait trials and generating chart..."):
                mot_files = read_mot_files_from_zip(zip_bytes)
                score, report, chart_bytes, file_hash = score_uploaded_mots(mot_files)
            st.session_state.analyzed_upload_id = upload_id
            st.session_state.gait_analysis = (
                score, report, chart_bytes, file_hash, len(mot_files)
            )

        total, report, chart_bytes, file_hash, mot_count = st.session_state.gait_analysis
        st.caption(f"Analyzed {mot_count} MOT file(s) from {MOT_FOLDER}.")
        st.subheader("Your Score", anchor=False)
        st.metric("Gait Efficiency Score (out of 100)", f"{total:.3f}")

        # Open automatically once for each newly uploaded ZIP.
        if st.session_state.get("gait_dialog_shown_for") != upload_id:
            st.session_state.gait_dialog_shown_for = upload_id
            show_gait_results(report, chart_bytes)

        if st.button("View Full Analysis and Chart"):
            show_gait_results(report, chart_bytes)

        if st.button("Submit to Leaderboard"):
            if not username.strip():
                st.error("Please enter a username.")
            else:
                try:
                    existing = (
                        supabase.table("leaderboard")
                        .select("id")
                        .eq("file_hash", file_hash)
                        .limit(1)
                        .execute()
                    )

                    if existing.data:
                        st.info("These MOT files were already submitted.")
                    else:
                        supabase.table("leaderboard").insert({
                            "username": username.strip(),
                            "filename": uploaded_file.name,
                            "file_hash": file_hash,
                            "total": total,
                        }).execute()
                        st.success("Score added to leaderboard!")
                except Exception as exc:
                    if str(getattr(exc, "code", "")) == "23505":
                        st.info("These MOT files were already submitted.")
                    else:
                        logger.exception("Supabase submission failed")
                        st.error(
                            "Unable to submit your score right now. "
                            "Please try again later."
                        )

    except FileNotFoundError as exc:
        st.error(str(exc))
    except BadZipFile:
        st.error("The uploaded file is not a valid ZIP archive.")
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        st.error(str(exc))
    except (RuntimeError, EOFError, OSError, KeyError, TypeError):
        logger.exception("Unable to analyze uploaded MOT files")
        st.error("The uploaded ZIP could not be analyzed. Please check its contents.")


# ----------------------------------
# 6. Display leaderboard
# ----------------------------------

st.header("Leaderboard", anchor=False)
if st.button("Refresh Leaderboard"):
    st.rerun()

try:
    response = (
        supabase.table("leaderboard")
        .select("username,filename,total")
        .order("total", desc=True)
        .limit(1000)
        .execute()
    )
    leaderboard = pd.DataFrame(response.data)
    if not leaderboard.empty:
        leaderboard.columns = ["Username", "File Name", "Gait Efficiency Score"]
        # Rank using the same three-decimal precision as result.json and new submissions.
        leaderboard["Gait Efficiency Score"] = (
            pd.to_numeric(leaderboard["Gait Efficiency Score"]).round(3)
        )
        leaderboard["Rank"] = leaderboard["Gait Efficiency Score"].rank(method="min", ascending=False).astype(int)
        leaderboard = leaderboard.sort_values(
            by=["Gait Efficiency Score", "Username"],
            ascending=[False, True],
            key=lambda col: col.str.casefold() if col.name == "Username" else col,
            kind="stable",
        )
        leaderboard = leaderboard.set_index("Rank")
        st.dataframe(
            leaderboard,
            width="stretch",
            column_config={
                "Rank": st.column_config.TextColumn(
                    width=50
                ),
                "Username": st.column_config.TextColumn(
                    width=150
                ),
                "File Name": st.column_config.TextColumn(
                    width=300
                ),
                "Gait Efficiency Score": st.column_config.NumberColumn(
                    width=100,
                    format="%.3f",
                ),
            }
        )
    else:
        st.write("No MOT files submitted yet.")
except Exception:
    logger.exception("Supabase leaderboard retrieval failed")
    st.error(
        "Unable to load the leaderboard right now. "
        "Please refresh the page and try again."
    )
