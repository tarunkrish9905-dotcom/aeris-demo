-- AERIS SQLite Database Schema v1
-- Authoritative schema definition for projects, datasets, mappings, configs, runs, jobs, and reports.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    analyst_name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE' -- ACTIVE, ARCHIVED, DELETED
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    storage_relative_path TEXT NOT NULL,
    sha256_checksum TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    detected_columns_json TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parameter_mappings (
    mapping_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    raw_column_name TEXT NOT NULL,
    target_parameter_id TEXT NOT NULL,
    source_unit TEXT,
    target_unit TEXT,
    data_type TEXT NOT NULL,
    is_timestamp INTEGER NOT NULL DEFAULT 0,
    UNIQUE(dataset_id, raw_column_name)
);

CREATE TABLE IF NOT EXISTS configuration_profiles (
    config_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    profile_type TEXT NOT NULL, -- VEHICLE, ENVELOPE, LIMITS, SOP
    profile_identifier TEXT NOT NULL,
    storage_relative_path TEXT NOT NULL,
    sha256_checksum TEXT NOT NULL,
    loaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    run_status TEXT NOT NULL, -- SUCCESS, WARNING, ERROR, PARTIAL, ABORTED, CANCELLED
    triggered_at TEXT NOT NULL,
    completed_at TEXT,
    execution_duration_ms INTEGER,
    total_issues_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    result_summary_json TEXT,
    provenance_id TEXT
);

CREATE TABLE IF NOT EXISTS background_jobs (
    job_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analysis_runs(analysis_id) ON DELETE CASCADE,
    state TEXT NOT NULL, -- QUEUED, RUNNING, SUCCEEDED, PARTIAL, FAILED, CANCELLED
    current_stage TEXT,
    completed_stages_count INTEGER NOT NULL DEFAULT 0,
    total_stages_count INTEGER NOT NULL DEFAULT 15,
    progress_percent INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exported_reports (
    report_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analysis_runs(analysis_id) ON DELETE CASCADE,
    report_format TEXT NOT NULL, -- HTML, MARKDOWN
    report_title TEXT NOT NULL,
    storage_relative_path TEXT NOT NULL,
    sha256_checksum TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    generated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_datasets_project ON datasets(project_id);
CREATE INDEX IF NOT EXISTS idx_mappings_dataset ON parameter_mappings(dataset_id);
CREATE INDEX IF NOT EXISTS idx_configs_project ON configuration_profiles(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_project ON analysis_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_jobs_analysis ON background_jobs(analysis_id);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON background_jobs(state);
CREATE INDEX IF NOT EXISTS idx_reports_analysis ON exported_reports(analysis_id);
