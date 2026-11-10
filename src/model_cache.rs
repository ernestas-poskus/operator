//! Locating an already-downloaded HF model cache before downloading anything.
//!
//! fastembed's `from_hf` loaders resolve their cache as `HF_HOME` first, then
//! `FASTEMBED_CACHE_DIR`, else a cwd-relative `.fastembed_cache`, and expect
//! hf-hub's `models--<org>--<name>/snapshots/...` layout directly under that
//! root. This module probes the common locations where such a cache may already
//! exist (macOS, Linux, Windows conventions included), picks the first root that
//! holds a complete snapshot of every required repo, and pins that root for the
//! process so the loaders use it deterministically.

use std::path::{Path, PathBuf};

use crate::config::Config;

/// The last candidate is always the download target, so `resolve` never returns
/// an empty list.
pub fn candidate_roots(cfg: &Config) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut push = |path: PathBuf| {
        if !roots.contains(&path) {
            roots.push(path);
        }
    };

    if let Some(dir) = &cfg.retrieval_model_cache_dir {
        push(dir.clone().into_path_buf());
    }
    if let Some(dir) = std::env::var_os("FASTEMBED_CACHE_DIR") {
        push(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE") {
        push(PathBuf::from(dir));
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        let home = PathBuf::from(home);
        push(home.join("hub"));
        push(home);
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        let xdg = PathBuf::from(xdg);
        push(xdg.join("huggingface/hub"));
        push(xdg.join("huggingface"));
    }
    if let Some(home) = home_dir() {
        push(home.join(".cache/huggingface/hub"));
        push(home.join(".cache/huggingface"));
        push(home.join(".cache/fastembed"));
        push(home.join(".fastembed_cache"));
    }
    if let Some(appdata) = std::env::var_os("LOCALAPPDATA") {
        let appdata = PathBuf::from(appdata);
        push(appdata.join("huggingface/hub"));
        push(appdata.join("huggingface"));
    }
    if let Some(userprofile) = std::env::var_os("USERPROFILE") {
        let userprofile = PathBuf::from(userprofile);
        push(userprofile.join(".cache/huggingface/hub"));
        push(userprofile.join(".cache/huggingface"));
        push(userprofile.join(".cache/fastembed"));
        push(userprofile.join(".fastembed_cache"));
    }
    push(cfg.workspace_root.join(".fastembed_cache"));
    // Last: the directory the operator owns and may create for a download.
    push(cfg.retrieval_cache_dir.join("models"));
    roots
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Whether `root` holds a complete hf-hub snapshot of `repo` (`org/name`).
pub fn repo_cached(repo: &str, root: &Path) -> bool {
    let Some(dir_name) = repo_dir_name(repo) else {
        return false;
    };
    let repo_dir = root.join(dir_name);
    let Ok(entries) = std::fs::read_dir(&repo_dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let snapshots = entry.path();
        if snapshots.file_name().and_then(|n| n.to_str()) != Some("snapshots") {
            continue;
        }
        let Ok(revs) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for rev in revs.flatten() {
            let dir = rev.path();
            let has = |name: &str| dir.join(name).is_file();
            let weights = has("model.safetensors") || has("model.safetensors.index.json");
            if has("config.json") && has("tokenizer.json") && weights {
                return true;
            }
        }
    }
    false
}

fn repo_dir_name(repo: &str) -> Option<String> {
    let (org, name) = repo.split_once('/')?;
    if org.is_empty() || name.is_empty() || name.contains('/') {
        return None;
    }
    Some(format!("models--{org}--{name}"))
}

/// The first candidate root that already holds every repo in `repos`, else the
/// final candidate (the operator's download directory, created on demand).
pub fn resolve(roots: &[PathBuf], repos: &[String]) -> PathBuf {
    for root in roots {
        if repos.iter().all(|repo| repo_cached(repo, root)) {
            return root.clone();
        }
    }
    roots
        .last()
        .cloned()
        .unwrap_or_else(|| PathBuf::from(".fastembed_cache"))
}

/// Pin the resolved cache root for this process. fastembed checks `HF_HOME`
/// first, so both variables are set to the same root; otherwise a user's
/// `HF_HOME` would silently win over the directory we resolved.
pub fn pin_env(root: &Path) {
    static PINNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    PINNED.get_or_init(|| {
        // SAFETY: called once, before any model loading thread starts; the
        // process does not spawn threads that read these variables concurrently.
        unsafe {
            std::env::set_var("HF_HOME", root);
            std::env::set_var("FASTEMBED_CACHE_DIR", root);
        }
    });
}

/// Probe and pin the cache root for `repos`.
pub fn resolve_and_pin(cfg: &Config, repos: &[String]) -> PathBuf {
    let roots = candidate_roots(cfg);
    let root = resolve(&roots, repos);
    pin_env(&root);
    root
}

#[cfg(test)]
mod tests {
    use super::{repo_cached, resolve};
    use std::path::PathBuf;

    fn fake_repo(root: &std::path::Path, repo: &str) {
        let (org, name) = repo.split_once('/').unwrap();
        let dir = root
            .join(format!("models--{org}--{name}"))
            .join("snapshots/abc123");
        std::fs::create_dir_all(&dir).unwrap();
        for file in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::write(dir.join(file), b"{}").unwrap();
        }
    }

    #[test]
    fn repo_cached_needs_tokenizer_and_weights() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!repo_cached("org/name", tmp.path()));
        fake_repo(tmp.path(), "org/name");
        assert!(repo_cached("org/name", tmp.path()));
        std::fs::remove_file(
            tmp.path()
                .join("models--org--name/snapshots/abc123/tokenizer.json"),
        )
        .unwrap();
        assert!(!repo_cached("org/name", tmp.path()));
    }

    #[test]
    fn resolve_prefers_the_first_complete_root_and_falls_back_to_the_last() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let c = tempfile::tempdir().unwrap();
        fake_repo(b.path(), "org/embed");
        let roots = vec![
            a.path().to_path_buf(),
            b.path().to_path_buf(),
            c.path().to_path_buf(),
        ];
        let repos = vec!["org/embed".to_string()];
        assert_eq!(resolve(&roots, &repos), b.path());
        let missing = vec!["org/other".to_string()];
        assert_eq!(resolve(&roots, &missing), c.path());
    }

    #[test]
    fn repo_dir_name_rejects_malformed_ids() {
        assert_eq!(super::repo_dir_name("a/b").as_deref(), Some("models--a--b"));
        assert!(super::repo_dir_name("a").is_none());
        assert!(super::repo_dir_name("a/b/c").is_none());
        assert!(super::repo_dir_name("").is_none());
    }

    #[test]
    fn candidate_roots_end_with_the_download_target() {
        let mut cfg = crate::config::Config::test_default();
        cfg.retrieval_model_cache_dir = None;
        let roots = super::candidate_roots(&cfg);
        assert_eq!(roots.last(), Some(&cfg.retrieval_cache_dir.join("models")));
        cfg.retrieval_model_cache_dir = Some(PathBuf::from("/explicit/cache").into());
        assert_eq!(
            super::candidate_roots(&cfg)[0],
            PathBuf::from("/explicit/cache")
        );
    }
}
