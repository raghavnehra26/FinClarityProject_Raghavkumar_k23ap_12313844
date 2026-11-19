from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from ai_agent import answer_chat, deterministic_explanation, ollama_status
from app_logic import validate_applicant_input
from config import DATA_PATH, DATASET_REPORT_PATH, MODEL_PATH
from config import MODEL_VERSIONS_DIR, PROJECT_ROOT
from dataset_manager import (
    FEATURE_ALIASES,
    FEATURE_LABELS,
    POST_DECISION_FEATURES,
    compatible_features,
    detect_column_mapping,
    load_registry,
    register_uploaded_csv,
    register_existing_csv,
    remove_dataset,
)
from dataset_discovery import CATALOG, DISCOVERED_ROOT, download_curated_dataset
from explainability import explain_prediction, global_importance, make_synthetic_examples
from inspect_dataset import inspect_dataset
from multi_dataset_training import activate_model_version, model_history, train_from_registry


st.set_page_config(
    page_title="FinClarity | Model Explainer",
    page_icon="F",
    layout="wide",
)

def apply_theme(dark: bool) -> None:
    colors = ({"background": "#111827", "surface": "#1f2937", "text": "#f3f4f6", "muted": "#cbd5e1", "border": "#374151", "sidebar": "#0b1220", "accent": "#38bdf8"}
              if dark else {"background": "#f5f7fa", "surface": "#ffffff", "text": "#102a43", "muted": "#486581", "border": "#d9e2ec", "sidebar": "#102a43", "accent": "#087e8b"})
    pio.templates.default = "plotly_dark" if dark else "plotly_white"
    st.markdown(f"""<style>
:root {{ color-scheme: {'dark' if dark else 'light'}; }}
.stApp {{ background: {colors['background']}; color: {colors['text']}; }}
[data-testid="stSidebar"] {{ background: {colors['sidebar']}; }}
[data-testid="stSidebar"] * {{ color: #f0f5fa !important; }}
[data-testid="stSidebar"] [aria-checked="true"] {{ background: #174e70; }}
[data-testid="stHeader"] {{ background: {colors['background']}; }}
h1, h2, h3 {{ color: {colors['text']}; letter-spacing: -.02em; }}
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {{ color: {colors['muted']}; }}
[data-testid="stMetric"], [data-testid="stChatMessage"] {{ background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 12px; padding: 16px 18px; }}
[data-testid="stMetricLabel"] {{ color: {colors['muted']} !important; }}
[data-testid="stMetricValue"] {{ color: {colors['text']} !important; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 9px; min-height: 2.5rem; font-weight: 600; }}
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div, [data-testid="stChatInput"] textarea {{ background: {colors['surface']}; color: {colors['text']}; }}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ color: {colors['text']}; line-height: 1.65; }}
[data-testid="stDataFrame"] {{ border: 1px solid {colors['border']}; border-radius: 10px; }}
[role="listbox"], [role="option"], [data-baseweb="menu"] {{ background: {colors['surface']}; color: {colors['text']}; }}
button:disabled {{ opacity: .72 !important; color: {colors['muted']} !important; }}
[data-testid="stCode"] {{ border: 1px solid {colors['border']}; border-radius: 10px; }}
[data-testid="stAlert"] {{ border-radius: 10px; }}
div[data-testid="stPlotlyChart"] {{ background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 12px; padding: 8px; }}
@media (max-width: 700px) {{ [data-testid="stMetric"] {{ padding: 12px; }} h1 {{ font-size: 1.8rem; }} }}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_artifact(model_path: str, modified_time: float) -> dict[str, Any]:
    del modified_time
    return joblib.load(model_path)


def get_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        return None
    try:
        return load_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except (OSError, EOFError, ValueError, TypeError, KeyError) as error:
        st.error(f"Could not load the saved model artifact: {error}")
        return None


@st.cache_data
def load_csv_preview(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path, nrows=100)


def show_prediction(result: dict[str, Any]) -> None:
    st.subheader("Model result")
    title = (
        f"Higher {result['positive_score_name']}"
        if result["prediction"] == 1
        else f"Lower {result['positive_score_name']}"
    )
    st.markdown(f"### {title}")
    st.caption(f"Predicted target class: **{result['prediction_label']}**")
    left, right = st.columns(2)
    left.metric(
        "Uncalibrated model score",
        f"{result['model_score']:.3f}",
    )
    right.metric("Model threshold", "0.500")
    st.warning(result["score_warning"])
    st.caption(f"Target: {result['target']}")

    st.subheader("What influenced this result?")
    st.caption(
        "Ranked from strongest to weakest for this profile. A score shift is "
        "relative to the model's reference profile; it is not a real-world "
        "loan effect."
    )
    explanation_frame = pd.DataFrame(result["explanations"])
    if not explanation_frame.empty:
        shown = explanation_frame.copy()
        strongest = shown.iloc[0]
        if strongest["contribution"] > 0:
            strongest_direction = result["positive_class_label"]
        elif strongest["contribution"] < 0:
            strongest_direction = result["negative_class_label"]
        else:
            strongest_direction = None
        if strongest_direction:
            st.info(
                f"The strongest influence is **{strongest['label']}** "
                f"(your value: **{strongest['value']}**). It moved the model "
                f"score by **{abs(strongest['contribution']) * 100:.1f} "
                f"percentage points** toward **{strongest_direction}** "
                "compared with the reference profile."
            )
        else:
            st.info(
                f"**{strongest['label']}** had no measurable effect on this "
                "model score compared with the reference profile."
            )

        shown = shown.head(5).copy()
        shown["score_shift_points"] = shown["contribution"].map(
            lambda value: round(value * 100, 2)
        )
        shown["effect_on_model"] = shown["contribution"].map(
            lambda value: (
                f"Toward {result['positive_class_label']}"
                if value > 0
                else (
                    f"Toward {result['negative_class_label']}"
                    if value < 0
                    else "No measurable shift"
                )
            )
        )
        st.plotly_chart(
            px.bar(
                shown.sort_values("score_shift_points"),
                x="score_shift_points",
                y="label",
                color="effect_on_model",
                orientation="h",
                labels={
                    "score_shift_points": "Score shift (percentage points)",
                    "label": "Factor",
                    "effect_on_model": "Direction",
                },
                title="How each factor moved this model's score",
            ),
            use_container_width=True,
        )
        st.dataframe(
            shown[["label", "value", "score_shift_points", "effect_on_model"]]
            .rename(
                columns={
                    "label": "Factor",
                    "value": "Your value",
                    "score_shift_points": "Score shift (points)",
                    "effect_on_model": "Direction",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.info(deterministic_explanation(result))
    st.caption(
        "These are patterns the model learned from its historical data. They "
        "do not show that a factor caused a loan outcome, predict what a lender "
        "would do, or establish fairness."
    )


def overview(artifact: dict[str, Any] | None) -> None:
    st.title("FinClarity")
    st.caption("Explainable Loan Eligibility Agent · Educational prototype")
    if artifact and artifact["metadata"].get("target_column") == "source_outcome":
        overview_text = (
            "FinClarity demonstrates how to inspect financial data, compare a "
            "profile with historical source groups, and explain the model's "
            "behavior in plain language. It does not estimate an actual lender "
            "decision."
        )
    else:
        overview_text = (
            "FinClarity demonstrates how to inspect financial data, predict a "
            "historical loan outcome, and explain the model's behavior in plain "
            "language."
        )
    st.markdown(
        "### Understand a model result—not just a score\n"
        + overview_text
    )
    st.warning(
        "This prototype is not a lender, does not determine loan eligibility, "
        "and must not be used for real credit decisions. The accepted and "
        "rejected files contain unmatched historical records; a source-file "
        "classifier is not an approval model."
    )

    first, second, third, fourth = st.columns(4)
    first.metric(
        "Dataset",
        "Available"
        if DATA_PATH.is_file() or DATASET_REPORT_PATH.is_file()
        else "Not found",
    )
    second.metric("Trained model", "Available" if artifact else "Not trained")
    third.metric(
        "Task",
        artifact["metadata"]["prediction_task"]
        if artifact
        else "Not approval prediction",
    )
    local_llm = ollama_status()
    fourth.metric(
        "Local LLM",
        "Connected" if local_llm["api_connected"] and local_llm["model_available"]
        else "Not connected",
    )

    try:
        registered = load_registry()
    except ValueError:
        registered = []
    if registered:
        st.subheader("Imported datasets")
        accepted_records = sum(int(item.get("row_count", 0) or 0) for item in registered if item.get("label") == "accepted")
        rejected_records = sum(int(item.get("row_count", 0) or 0) for item in registered if item.get("label") == "rejected")
        managed = st.columns(3)
        managed[0].metric("Registered CSV files", len(registered))
        managed[1].metric("Accepted-source records", f"{accepted_records:,}")
        managed[2].metric("Rejected-source records", f"{rejected_records:,}")

    st.subheader("How it works")
    st.markdown(
        "1. Inspect the actual local LendingClub CSV files and their distinct "
        "labels.\n"
        "2. Train and compare reproducible baseline models using only shared "
        "application-time fields.\n"
        "3. Evaluate on held-out data, including precision, recall, F1, "
        "ROC-AUC, average precision, confusion matrix, and calibration checks.\n"
        "4. Explain the saved model's output with SHAP; optionally ask your "
        "locally running Llama model through Ollama."
    )

    if artifact is None:
        st.info(
            "No prediction, score, metric, or SHAP value is shown until a real "
            "dataset is provided and training succeeds."
        )
        st.code(
            "python train_application_model.py "
            "--accepted 'C:\\path\\to\\accepted.csv' "
            "--rejected 'C:\\path\\to\\rejected.csv'",
            language="powershell",
        )
    else:
        st.success(
            f"Loaded the saved {artifact['metadata']['selected_model']} model. "
            "See Applicant assessment for an actual model prediction."
        )


def dataset_page() -> None:
    st.title("Dataset Explorer")
    try:
        managed_entries = load_registry()
    except ValueError as error:
        st.error(str(error))
        managed_entries = []
    if managed_entries:
        st.subheader("Imported CSVs")
        for item in managed_entries:
            with st.expander(f"{item['name']} | {item['label']} | {item['row_count']:,} rows"):
                left, right, third = st.columns(3)
                left.metric("Columns", item.get("column_count", len(item.get("columns", []))))
                right.metric("File size", f"{item.get('file_size_bytes', 0):,} bytes")
                third.metric("Duplicate rows", str(item.get("duplicate_rows") if item.get("duplicate_check_complete") else f"sample {item.get('duplicate_rows_in_checked_rows', 0)}"))
                st.caption(f"Imported {item.get('imported_at', 'unknown date')} | Source: {item.get('source_url') or 'manual upload'} | License: {item.get('license', 'not specified')}")
                st.json(item.get("mapping", {}))
                try:
                    source_path = Path(item["stored_file"])
                    st.dataframe(load_csv_preview(str(source_path), source_path.stat().st_mtime), hide_index=True, use_container_width=True)
                except (OSError, ValueError, pd.errors.ParserError) as error:
                    st.error(f"Could not preview this imported CSV: {error}")
    if DATASET_REPORT_PATH.is_file():
        st.caption("Showing the two source files recorded in the saved training report.")
        try:
            paired_report = json.loads(
                DATASET_REPORT_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not load the saved dataset report: {error}")
            return
        st.warning(paired_report["target_definition"])
        for source in ("accepted", "rejected"):
            details = paired_report[source]
            st.subheader(f"{source.title()} source file")
            left, right, third = st.columns(3)
            left.metric("Rows scanned", f"{details['total_rows']:,}")
            right.metric("Columns", details["column_count"])
            third.metric("Valid loan-amount rows", f"{details['rows_with_valid_loan_amount']:,}")
            st.caption(
                f"File: {details['file']} · "
                f"Size: {details['file_size_bytes']:,} bytes"
            )
            st.caption(
                f"Column types inferred from the first "
                f"{details['column_sample_rows']:,} rows (not the whole file)."
            )
            with st.expander("Full column-name and sampled-type schema"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "column": name,
                                "sampled_type": details["column_sample_dtypes"].get(
                                    name, "unknown"
                                ),
                            }
                            for name in details["columns"]
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            if details["potentially_sensitive_or_identifying_columns"]:
                st.warning(
                    "Potentially sensitive or identifying columns were found "
                    "(not used as model inputs): "
                    + ", ".join(
                        details["potentially_sensitive_or_identifying_columns"]
                    )
                )
            st.write("Selected-field missing and invalid counts")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Missing values": details[
                            "missing_counts_selected_columns"
                        ],
                        "Missing or invalid numeric values": details[
                            "invalid_numeric_counts"
                        ],
                    }
                ).fillna(0).astype("int64"),
                use_container_width=True,
            )
            if source == "accepted":
                st.write("Historical `loan_status` distribution")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"loan_status": label, "rows": count}
                            for label, count in details[
                                "accepted_loan_status_distribution"
                            ].items()
                        ]
                    ).sort_values("rows", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )
            st.caption(
                "Full-file duplicate rows were not assessed. The selected "
                "feature columns are shown in the saved report."
            )
        if paired_report.get("license"):
            st.caption(
                "Dataset license note: "
                + paired_report["license"]["kaggle_dataset_page"]
                + ". "
                + paired_report["license"]["caution"]
            )
        st.download_button(
            "Download dataset inspection report",
            data=json.dumps(paired_report, indent=2),
            file_name="dataset_inspection.json",
            mime="application/json",
        )
        return
    st.caption(f"Configured file: `{DATA_PATH}`")
    if not DATA_PATH.is_file():
        st.warning("The LendingClub dataset is not present at the configured path.")
        st.markdown(
            "Download the actual training CSV from the Kaggle competition after "
            "accepting its rules, then place it at "
            "`C:\\Users\\ragha\\Documents\\FinClarity\\data\\loanStats_train.csv`."
        )
        st.code(
            "$env:FINCLARITY_DATA_PATH = 'C:\\full\\path\\to\\loanStats_train.csv'\n"
            "python inspect_dataset.py",
            language="powershell",
        )
        st.info(
            "The absence of a local dataset means its exact schema, version, "
            "license, labels, and class distribution cannot yet be verified."
        )
        return
    try:
        report = inspect_dataset(DATA_PATH)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not inspect the CSV: {error}")
        return

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Rows", f"{report['rows']:,}")
    metric_b.metric("Columns", report["columns"])
    metric_c.metric("Duplicate rows", f"{report['duplicate_rows']:,}")
    st.write(f"File size: {report['file_size_bytes']:,} bytes")
    st.warning(report["approval_target_warning"])
    if report["target_definition"]:
        st.write(f"Detected target: {report['target_definition']}")
        st.dataframe(
            pd.DataFrame(report["target_distribution"]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning(
            "No supported historical outcome target was detected. Training will "
            "not guess a target or present this as an approval model."
        )
    st.write("Target candidates")
    st.dataframe(pd.DataFrame(report["target_candidates"]), hide_index=True)
    st.write("Column schema and missing values")
    st.dataframe(pd.DataFrame(report["column_report"]), hide_index=True)
    if report["sensitive_or_identifying_columns"]:
        st.warning(
            "Potentially sensitive/identifying column names found (not used by "
            "the model): "
            + ", ".join(report["sensitive_or_identifying_columns"])
        )
    st.download_button(
        "Download schema report",
        data=json.dumps(report, indent=2),
        file_name="dataset_inspection.json",
        mime="application/json",
    )


def dataset_management_page() -> None:
    st.title("Dataset Management")
    st.write("Upload and inspect local CSV files. FinClarity never guesses whether a file is accepted or rejected; assign its label before importing.")
    uploaded = st.file_uploader("Choose one or more CSV files", type=["csv"], accept_multiple_files=True, key="dataset_uploads")
    for file in uploaded or []:
        contents = file.getvalue()
        token = hashlib.sha256(contents).hexdigest()[:10]
        with st.expander(f"Preview | {file.name}", expanded=True):
            try:
                from dataset_manager import inspect_csv_bytes
                report = inspect_csv_bytes(contents)
                duplicate_status = "complete" if report["duplicate_check_complete"] else f"checked first {report['duplicate_check_rows']:,} rows"
                st.caption(f"{report['row_count']:,} rows | {report['column_count']} columns | duplicate check {duplicate_status}")
                st.dataframe(pd.read_csv(io.BytesIO(contents), nrows=25), hide_index=True, use_container_width=True)
                st.write("Detected columns and mappings")
                st.json(report["detected_mapping"])
                st.caption("Unmapped columns are retained in the original file: " + (", ".join(report["unmapped_columns"]) or "none"))
                if report.get("date_columns_detected"):
                    st.caption("Date-like columns inspected with mixed-format parsing (not automatically used for training): " + json.dumps(report["date_sample_report"]))
                blocked_mappings = POST_DECISION_FEATURES.intersection(report["detected_mapping"])
                if blocked_mappings:
                    st.warning("Detected post-decision fields are preserved for inspection but excluded from source-membership training: " + ", ".join(sorted(blocked_mappings)))
                label = st.selectbox("Assign this dataset's source label", ["Choose label", "accepted", "rejected"], key=f"dataset_label_{token}")
                st.caption("Label definitions must come from the source documentation or your explicit selection. These labels mean historical source membership only.")
                detected = report["detected_mapping"]
                column_options = ["(not mapped)"] + report["columns"]
                mapping_cols = st.columns(2)
                selected_mapping = {}
                for index, (feature, aliases) in enumerate(FEATURE_ALIASES.items()):
                    default_col = detected.get(feature, "(not mapped)")
                    selected = mapping_cols[index % 2].selectbox(
                        FEATURE_LABELS[feature], column_options,
                        index=column_options.index(default_col) if default_col in column_options else 0,
                        key=f"dataset_map_{token}_{feature}",
                    )
                    if selected != "(not mapped)":
                        selected_mapping[feature] = selected
                description = st.text_input("Dataset description (optional)", key=f"dataset_description_{token}")
                source_url = st.text_input("Source URL (optional)", key=f"dataset_source_{token}")
                license_name = st.text_input("License (confirm from source; leave blank if unknown)", key=f"dataset_license_{token}")
                if st.button("Import dataset", key=f"dataset_import_{token}", type="primary"):
                    if label not in {"accepted", "rejected"}:
                        st.error("Choose accepted or rejected explicitly before importing.")
                    elif len(set(selected_mapping.values())) != len(selected_mapping):
                        st.error("A source column cannot map to more than one feature.")
                    else:
                        try:
                            result = register_uploaded_csv(file.name, contents, label, mapping=selected_mapping, description=description, source_url=source_url, license_name=license_name)
                            st.success("This file is already registered." if result["duplicate_file"] else f"Imported {file.name} with explicit {label} source label.")
                            st.rerun()
                        except (OSError, ValueError) as error:
                            st.error(f"Import failed: {error}")
            except (OSError, ValueError) as error:
                st.error(f"CSV preview failed: {error}")
    try:
        entries = load_registry()
    except ValueError as error:
        st.error(str(error))
        return
    st.subheader("Local dataset registry")
    accepted_count = sum(item["label"] == "accepted" for item in entries)
    rejected_count = sum(item["label"] == "rejected" for item in entries)
    metrics = st.columns(3)
    metrics[0].metric("Registered files", len(entries))
    metrics[1].metric("Accepted-source files", accepted_count)
    metrics[2].metric("Rejected-source files", rejected_count)
    if entries:
        st.dataframe(pd.DataFrame([{key: item.get(key) for key in ("name", "label", "row_count", "file_size_bytes", "license", "source_url", "dataset_id")} for item in entries]), hide_index=True, use_container_width=True)
        selected_id = st.selectbox("Remove an imported dataset", [item["dataset_id"] for item in entries], format_func=lambda value: next(f"{row['name']} · {row['label']}" for row in entries if row["dataset_id"] == value))
        if st.button("Remove selected dataset", key="remove_registered_dataset"):
            try:
                remove_dataset(selected_id)
                st.success("Dataset removed from the local registry and managed storage.")
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"Could not remove the selected dataset: {error}")
    else:
        st.info("No user-managed datasets are registered yet. Existing project datasets and training workflows remain available separately.")


def dataset_discovery_page() -> None:
    st.title("Dataset Discovery and Download")
    st.write("FinClarity uses a small curated source list. A download is staged for review; it is never silently labeled or added to accepted/rejected training.")
    for source in CATALOG:
        with st.container(border=True):
            st.subheader(source["name"])
            st.caption(f"Source: {source['source']} | License: {source['license']}")
            st.write(source["description"])
            st.warning(source["task"])
            st.markdown(f"[Open source page]({source['url']})")
            if source["downloadable"]:
                st.caption("Kaggle requires an account, authentication, and acceptance of competition rules. Configure credentials locally (for example, KAGGLE_API_TOKEN or Kaggle's local credentials file); FinClarity never asks you to paste a token into the app.")
                button_label = "Stage accepted/rejected source files" if source.get("kind") == "dataset_pair" else "Download verified training CSV"
                if st.button(button_label, key=f"download_{source['id']}"):
                    try:
                        with st.status("Checking Kaggle access and staging the labeled training CSV...", expanded=True) as status:
                            record = download_curated_dataset(source["id"])
                            status.update(label="Already staged and checksum verified" if record["already_downloaded"] else "Download staged and checksum recorded", state="complete", expanded=False)
                        for staged in record["files"]:
                            st.success(f"{staged['filename']} | {staged['file_size_bytes']:,} bytes | SHA-256 {staged['sha256']}")
                        if source.get("kind") == "competition":
                            st.info("This file contains historical repayment outcomes. It is not included in accepted/rejected source-membership training. Use the separate historical-repayment workflow after reviewing the data dictionary and competition rules.")
                    except (OSError, ValueError, RuntimeError, ImportError) as error:
                        st.error(f"Download failed: {error}")
            else:
                st.info("This source is listed for discovery and provenance review only. Its feature schema or target definition is not compatible with the active model workflow, so no automatic download or training action is offered.")
    st.caption(f"Staged files and download manifests are stored under `{DISCOVERED_ROOT}`. Review provenance and schema before any separate training use.")
    manifest_path = DISCOVERED_ROOT / "downloads.json"
    try:
        downloads = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        downloads = []
    source_pair = next((item for item in downloads if item.get("source_id") == "lending-club-accepted-rejected"), None)
    if source_pair:
        st.subheader("Confirm and import staged source files")
        st.warning("The filenames suggest accepted/rejected, but you must confirm each label against the Kaggle source documentation. FinClarity will not assign them automatically.")
        for staged in source_pair.get("files", []):
            path = Path(staged["file_path"])
            token = staged["sha256"][:10]
            st.markdown(f"**{staged['filename']}** | {staged['file_size_bytes']:,} bytes | checksum `{staged['sha256']}`")
            label = st.selectbox("Confirm source label", ["Choose label", "accepted", "rejected"], key=f"staged_label_{token}")
            try:
                columns = list(pd.read_csv(path, nrows=0).columns)
                detected = detect_column_mapping(columns)
                st.caption("Columns: " + ", ".join(columns[:30]) + (" ..." if len(columns) > 30 else ""))
                mapped = {feature: column for feature, column in detected.items()}
                if st.button("Import this staged file with confirmed label", key=f"staged_import_{token}"):
                    if label not in {"accepted", "rejected"}:
                        st.error("Confirm accepted or rejected explicitly before import.")
                    else:
                        try:
                            with st.status("Validating this CSV in chunks and recording provenance...") as status:
                                imported = register_existing_csv(path, label, mapping=mapped, source_url=source_pair.get("source_url", ""), license_name=source_pair.get("license", "Review source license"), description=source_pair.get("description", ""))
                                status.update(label="File registered" if not imported["duplicate_file"] else "Checksum already registered", state="complete")
                            duplicate_status = "complete" if imported["duplicate_check_complete"] else f"sampled {imported['duplicate_check_rows']:,} rows"
                            st.success(f"{imported['row_count']:,} rows inspected; duplicate check {duplicate_status}.")
                            st.rerun()
                        except (OSError, ValueError) as error:
                            st.error(f"Could not import staged source: {error}")
            except (OSError, ValueError, pd.errors.ParserError) as error:
                st.error(f"Could not preview staged source file: {error}")


def model_training_page() -> None:
    st.title("Model Training Center")
    st.warning("Training predicts historical accepted-file versus rejected-file membership only. It does not predict loan approval, creditworthiness, or default.")
    st.caption("The original one-accepted/one-rejected trainer and the separate historical repayment trainer are preserved. This center uses only imported files with user-assigned labels.")
    try:
        entries = load_registry()
    except ValueError as error:
        st.error(str(error))
        entries = []
    history = model_history()
    active = get_artifact()
    if active:
        st.info(f"Active model version: {active.get('metadata', {}).get('version', 'legacy')} · {active.get('metadata', {}).get('selected_model', 'unknown')}")
    with st.expander("Version history and rollback"):
        if history:
            st.dataframe(pd.DataFrame([{"Version": item.get("version"), "Created": item.get("created_at"), "Model": item.get("selected_model"), "Validation AP": item.get("validation_average_precision"), "Test AP": (item.get("test_metrics") or {}).get("average_precision"), "Test F1": (item.get("test_metrics") or {}).get("f1"), "Rows": item.get("rows_used"), "Datasets": item.get("dataset_ids")} for item in history]), hide_index=True, use_container_width=True)
            versions = [item["version"] for item in history]
            chosen = st.selectbox("Activate a previous model version", versions, key="rollback_model_version")
            if st.button("Activate selected version"):
                try:
                    activate_model_version(chosen)
                    st.success("Model version activated.")
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(f"Could not activate model version: {error}")
        else:
            st.info("No versioned multi-dataset training runs exist yet.")
    if not entries:
        st.info("Import at least one CSV with the explicit accepted label and one with the rejected label in Dataset Management to train here.")
        return
    options = [item["dataset_id"] for item in entries]
    selected_ids = st.multiselect("Choose labeled datasets", options, format_func=lambda value: next(f"{row['name']} · {row['label']} ({row['row_count']:,} rows)" for row in entries if row["dataset_id"] == value))
    selected = [item for item in entries if item["dataset_id"] in selected_ids]
    if selected:
        st.dataframe(pd.DataFrame([{ "Dataset": item["name"], "Label": item["label"], "Rows": item["row_count"], "Mappings": ", ".join(item.get("mapping", {}).keys()), "Unmapped columns": ", ".join(item.get("unmapped_columns", []))} for item in selected]), hide_index=True, use_container_width=True)
        features = compatible_features(selected)
        if features:
            st.caption("Only features mapped in every selected source are compatible. Original files and unmapped columns remain preserved.")
            chosen_features = st.multiselect("Select compatible features", features, default=features, format_func=lambda value: FEATURE_LABELS[value])
            model_kind = st.selectbox("Baseline classifier", ["random_forest", "logistic_regression"], format_func=lambda value: "Random Forest" if value == "random_forest" else "Logistic Regression")
            max_per_class = st.number_input("Maximum sampled records per source label", min_value=100, max_value=200_000, value=20_000, step=100)
            train_fraction = st.slider("Training split", min_value=0.50, max_value=0.85, value=0.70, step=0.05)
            validation_fraction = st.slider("Validation split", min_value=0.05, max_value=0.30, value=0.15, step=0.05)
            test_fraction = 1.0 - train_fraction - validation_fraction
            if test_fraction < 0.10:
                st.error("Keep at least 10% of the sampled data for final testing.")
            else:
                st.caption(f"Final held-out test split: {test_fraction:.0%}")
            if st.button("Train and activate model", type="primary"):
                if len(chosen_features) < 2:
                    st.error("Select at least two compatible features.")
                elif {item["label"] for item in selected} != {"accepted", "rejected"}:
                    st.error("Select at least one accepted and one rejected dataset.")
                elif test_fraction < 0.10:
                    st.error("Adjust the splits to leave at least 10% for testing.")
                else:
                    try:
                        training_progress = st.progress(0.0, text="Preparing training run...")
                        with st.status("Sampling source files, fitting preprocessing on training data, and evaluating held-out records…", expanded=True) as status:
                            report = train_from_registry(selected, chosen_features, maximum_per_class=int(max_per_class), model_kind=model_kind, train_fraction=train_fraction, validation_fraction=validation_fraction, progress_callback=lambda value, message: training_progress.progress(value, text=message))
                            status.update(label="Training complete · model version activated", state="complete", expanded=False)
                        st.success(f"Created model version {report['version']} with {report['rows_used']:,} sampled rows.")
                        st.metric("Held-out average precision", f"{report['test']['metrics']['average_precision']:.3f}")
                        st.caption(f"Class counts: accepted {report['class_counts_before_split']['accepted']:,} | rejected {report['class_counts_before_split']['rejected']:,} | imbalance ratio {report['class_imbalance_ratio']:.2f}:1")
                        st.rerun()
                    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
                        st.error(f"Training failed; the active model was not reported as updated: {error}")
        else:
            st.warning("The selected datasets have no common mapped features. Confirm mappings in Dataset Management or select compatible files.")


def responsible_ai_page() -> None:
    st.title("Responsible AI")
    st.warning("FinClarity’s current accepted/rejected classifier estimates historical dataset-source membership. It is not an eligibility or lender decision system.")
    for item in (
        "Accepted and rejected files are unmatched populations; source labels are not counterfactual decisions for the same applicants.",
        "Dataset coverage, time periods, missingness, and feature measurement can differ between sources.",
        "SHAP and permutation importance describe model behavior, not causal effects or fairness.",
        "No group fairness audit is claimed unless a separately validated result is available.",
        "Do not use this educational prototype for real lending, eligibility, creditworthiness, or individual decisions.",
    ):
        st.markdown(f"- {item}")


def settings_page() -> None:
    st.title("Application Settings")
    status = ollama_status()
    st.metric("Local Ollama", "Connected" if status["api_connected"] else "Unavailable")
    st.write(f"**Primary model:** {status.get('model') or 'llama3.1:8b'}")
    st.write(f"**Base URL:** {status.get('base_url', 'http://localhost:11434')}")
    st.write("**Installed tags:** " + (", ".join(status.get("models", [])) or "No model tags detected"))
    st.caption("Set OLLAMA_BASE_URL and OLLAMA_MODEL in the PowerShell session before launching Streamlit to override these defaults. No cloud AI service or API key is used.")
    registry_path = PROJECT_ROOT / "data" / "managed"
    st.write(f"**Managed datasets:** `{registry_path}`")
    st.write(f"**Model version archive:** `{MODEL_VERSIONS_DIR}`")


def applicant_page(artifact: dict[str, Any] | None) -> None:
    st.title("Applicant assessment")
    if artifact is None:
        st.info(
            "Predictions are disabled until you provide the actual CSV and train "
            "the model. No sample prediction is substituted."
        )
        return

    metadata = artifact["metadata"]
    st.write(
        "Enter every listed feature used by this saved model. Inputs are processed "
        "in memory for this session and are not written to a file."
    )
    raw_values: dict[str, Any] = {}
    with st.form("applicant_input"):
        columns = st.columns(2)
        for index, (name, spec) in enumerate(metadata["feature_schema"].items()):
            column = columns[index % 2]
            with column:
                if spec["type"] == "number":
                    raw_values[name] = st.number_input(
                        spec["label"],
                        min_value=(
                            (0 if spec["integer"] else 0.0)
                            if spec["non_negative"]
                            else None
                        ),
                        value=None,
                        step=1 if spec["integer"] else 0.1,
                        format="%d" if spec["integer"] else "%.2f",
                        placeholder="Enter a value",
                        help="This feature is used by the trained model.",
                    )
                else:
                    options = [""] + spec["options"]
                    raw_values[name] = st.selectbox(
                        spec["label"],
                        options=options,
                        index=0,
                        help="Choose a value observed in the training data.",
                    )
        submitted = st.form_submit_button("Run model and explain")

    if submitted:
        valid_values, errors = validate_applicant_input(
            metadata["feature_schema"], raw_values
        )
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                st.session_state["latest_result"] = explain_prediction(
                    artifact, valid_values
                )
            except (ValueError, TypeError, KeyError, IndexError) as error:
                st.error(f"Could not generate the model explanation: {error}")

    result = st.session_state.get("latest_result")
    if result:
        show_prediction(result)


def explainability_page(artifact: dict[str, Any] | None) -> None:
    st.title("Explainability")
    if artifact is None:
        st.info("A trained model is required before any explanation is available.")
        return
    importance = global_importance(artifact)
    st.subheader("Global feature importance")
    st.caption(
        "Held-out permutation importance: change in average precision when a "
        "feature is shuffled. It is an empirical model-performance measure."
    )
    importance_frame = pd.DataFrame(importance).sort_values(
        "importance_mean", ascending=True
    )
    st.plotly_chart(
        px.bar(
            importance_frame,
            x="importance_mean",
            y="label",
            error_x="importance_std",
            orientation="h",
            labels={
                "importance_mean": "Change in average precision",
                "label": "Feature",
            },
            title="Permutation importance on a held-out sample",
        ),
        use_container_width=True,
    )
    st.dataframe(
        importance_frame.sort_values("importance_mean", ascending=False),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "A negative value means shuffling the feature improved the score in this "
        "sample. Importance is not causation or a fairness assessment."
    )

    result = st.session_state.get("latest_result")
    if result:
        st.subheader("Current applicant explanation")
        frame = pd.DataFrame(result["explanations"]).head(10)
        st.plotly_chart(
            px.bar(
                frame.sort_values("contribution"),
                x="contribution",
                y="label",
                color="direction",
                orientation="h",
                labels={
                    "contribution": (
                        "SHAP contribution to "
                        + result["positive_score_name"]
                    ),
                    "label": "Feature",
                },
                title="Local SHAP explanation",
            ),
            use_container_width=True,
        )
        st.caption(
            "Positive SHAP values move this model's positive-class score higher; "
            "negative values move it lower. They do not establish causation, "
            "fairness, or what a real lender would decide."
        )
    else:
        st.info("Run an assessment to see an individual SHAP explanation.")


def synthetic_examples_page(artifact: dict[str, Any] | None) -> None:
    st.title("Synthetic demonstrations")
    if artifact is None:
        st.info(
            "Examples are not generated without the trained model. No outcome "
            "or explanation is hardcoded."
        )
        return
    st.write(
        "Generate synthetic feature profiles from training-set summary ranges. "
        "The model chooses the lower- and higher-score profiles from actual "
        "predictions. These are not real customer records."
    )
    if st.button("Generate model-backed examples"):
        try:
            st.session_state["synthetic_examples"] = make_synthetic_examples(artifact)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            st.error(f"Could not generate the examples: {error}")
    examples = st.session_state.get("synthetic_examples")
    if examples:
        st.info(examples["message"])
        for item in examples["examples"]:
            with st.expander(item["name"], expanded=True):
                show_prediction(item["result"])
        if not examples["examples"]:
            st.warning(examples["message"])


def evaluation_page(artifact: dict[str, Any] | None) -> None:
    st.title("Model evaluation")
    if artifact is None:
        st.info("No held-out model evaluation is available before training.")
        return
    report = artifact["evaluation"]
    st.write(f"**Prediction target:** {report['target_definition']}")
    st.write(f"**Selected model:** {report['selected_model']}")
    st.write(f"**Split:** {report['split_method']}")
    st.write(
        f"**Rows used:** {report['rows_used']:,} "
        f"({report['rows_eligible']:,} eligible)"
    )
    metrics = report["test"]["metrics"]
    metric_meanings = {
        "precision": "Of records predicted in a class, the fraction matching that class in this held-out sample.",
        "recall": "Of records in a class, the fraction the classifier found.",
        "f1": "A combined summary of precision and recall.",
        "roc_auc": "How well scores rank positive above negative records across thresholds.",
        "average_precision": "A summary of the precision-recall curve; useful when class balance varies.",
        "brier_score": "Mean squared error of predicted class scores; lower is better, though scores are uncalibrated here.",
    }
    metric_columns = st.columns(3)
    for column, key in zip(
        metric_columns, ("precision", "recall", "f1")
    ):
        value = metrics[key]
        column.metric(key.replace("_", " ").title(), f"{value:.3f}")
        column.caption(metric_meanings[key])
    metric_columns = st.columns(3)
    for column, key in zip(
        metric_columns, ("roc_auc", "average_precision", "brier_score")
    ):
        value = metrics[key]
        column.metric(key.replace("_", " ").title(), "N/A" if value is None else f"{value:.3f}")
        column.caption(metric_meanings[key])

    matrix = report["test"]["confusion_matrix"]
    st.subheader("Held-out confusion matrix")
    matrix_frame = pd.DataFrame(
        matrix["values"],
        index=[f"Actual {label}" for label in matrix["labels"]],
        columns=[f"Predicted {label}" for label in matrix["labels"]],
    )
    st.dataframe(matrix_frame, use_container_width=True)

    calibration = report["test"]["calibration"]
    if calibration["mean_predicted"]:
        st.subheader("Calibration diagnostic (held-out test sample)")
        chart_frame = pd.DataFrame(
            {
                "Mean model score": calibration["mean_predicted"],
                f"Observed {matrix['labels'][1]} fraction": (
                    calibration["fraction_positive"]
                ),
            }
        )
        st.plotly_chart(
            px.scatter(
                chart_frame,
                x="Mean model score",
                y=f"Observed {matrix['labels'][1]} fraction",
                title="Observed positive-class fraction versus uncalibrated score",
            ),
            use_container_width=True,
        )
        st.caption(report["test"]["probability_warning"])

    st.subheader("Limitations")
    for limitation in report["limitations"]:
        st.markdown(f"- {limitation}")
    st.download_button(
        "Download model evaluation report",
        data=json.dumps(report, indent=2),
        file_name="model_evaluation.json",
        mime="application/json",
    )


def agent_page() -> None:
    st.title("AI Assistant")
    result = st.session_state.get("latest_result")
    st.write(
        "Ask general questions or explore FinClarity. The assistant retrieves saved project evidence when your question depends on actual app data."
    )
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []
    fingerprint = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest() if result else None
    if st.session_state.get("agent_assessment_fingerprint") != fingerprint:
        st.session_state["agent_history"] = []
        st.session_state["agent_assessment_fingerprint"] = fingerprint
    clear_col, service_col = st.columns([1, 3])
    if clear_col.button("Clear chat", key="clear_agent_chat"):
        st.session_state["agent_history"] = []
        st.rerun()
    status = ollama_status()
    selected_model = status.get("model")
    if status["models"]:
        configured = selected_model if selected_model in status["models"] else None
        default_index = status["models"].index(configured) if configured else 0
        selected_model = st.selectbox(
            "Installed local model",
            status["models"],
            index=default_index,
            key="agent_model_selector",
            help="Choose a model reported by your local Ollama installation. Set OLLAMA_MODEL to make a choice persistent.",
        )
        status = ollama_status(selected_model)
    model_name = selected_model or "Not selected"
    connection = (
        "Connected" if status["api_connected"] and status["model_available"]
        else "Disconnected" if not status["api_connected"]
        else "Model unavailable"
    )
    service_col.caption(
        f"LLM: {model_name} · Provider: Ollama · Status: {connection} · Execution: Local"
    )
    if not status["api_connected"]:
        st.info("Ollama is not responding at the configured local address. Start the Ollama app, or run `ollama serve`. The deterministic FinClarity fallback remains available.")
    elif not status["model_available"]:
        st.info("No installed Llama model is selected. Check `ollama list`, choose an installed model above, or set OLLAMA_MODEL.")
    for message in st.session_state["agent_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("evidence") and message["role"] == "assistant":
                st.caption(f"Evidence: {message['evidence']}")
    if not st.session_state["agent_history"]:
        st.caption("Try a question")
        prompts = ["What is machine learning?", "Explain Random Forest simply", "Write a Python average function", "Explain my current assessment", "What do my SHAP values mean?", "How does FinClarity work?"]
        prompt_cols = st.columns(3)
        for index, prompt in enumerate(prompts):
            if prompt_cols[index % 3].button(prompt, key=f"agent_suggest_{index}", use_container_width=True):
                st.session_state["agent_pending_question"] = prompt
                st.rerun()
    question = st.chat_input("Ask about an assessment, evidence, or model behavior", max_chars=2000)
    question = question or st.session_state.pop("agent_pending_question", None)
    if question:
        history = st.session_state["agent_history"]
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking locally and retrieving project evidence when needed..."):
                answer, evidence, notice = answer_chat(
                    question,
                    history[:-1],
                    result=result,
                    artifact=get_artifact(),
                    model=selected_model,
                )
            st.markdown(answer)
            st.caption(f"Evidence: {evidence}")
            if notice:
                st.info(notice)
        history.append({"role": "assistant", "content": answer, "evidence": evidence})


def main() -> None:
    dark = st.sidebar.toggle("Dark appearance", value=False, key="fc_dark_theme")
    apply_theme(dark)
    st.sidebar.title("FinClarity")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Dashboard",
            "Dataset Management",
            "Dataset Discovery",
            "Dataset Explorer",
            "Model Training Center",
            "Applicant assessment",
            "Explainability Center",
            "Synthetic demonstrations",
            "AI Assistant",
            "Model Performance",
            "Responsible AI",
            "Application Settings",
        ],
    )
    artifact = get_artifact()
    if page == "Executive Dashboard":
        overview(artifact)
    elif page == "Dataset Management":
        dataset_management_page()
    elif page == "Dataset Discovery":
        dataset_discovery_page()
    elif page == "Dataset Explorer":
        dataset_page()
    elif page == "Model Training Center":
        model_training_page()
    elif page == "Applicant assessment":
        applicant_page(artifact)
    elif page == "Explainability Center":
        explainability_page(artifact)
    elif page == "Synthetic demonstrations":
        synthetic_examples_page(artifact)
    elif page == "AI Assistant":
        agent_page()
    elif page == "Model Performance":
        evaluation_page(artifact)
    elif page == "Responsible AI":
        responsible_ai_page()
    else:
        settings_page()
    target_caution = (
        "Source-file membership, not loan approval."
        if artifact
        and artifact["metadata"].get("target_column") == "source_outcome"
        else "Historical loan outcome, not loan approval."
    )
    st.sidebar.caption(f"Educational prototype only · {target_caution}")


main()
