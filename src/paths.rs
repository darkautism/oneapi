use std::path::PathBuf;
use home::home_dir;

pub fn get_config_dir() -> PathBuf {
    let mut p = home_dir().expect("Could not find home directory");
    p.push(".oneapi");
    if !p.exists() {
        std::fs::create_dir_all(&p).expect("Could not create config directory");
    }
    p
}

pub fn get_path(filename: &str) -> String {
    let mut p = get_config_dir();
    p.push(filename);
    p.to_string_lossy().to_string()
}

pub fn write_atomic(filename: &str, content: &str) -> anyhow::Result<()> {
    let path = get_path(filename);
    let temp_path = format!("{}.tmp", path);
    
    // 1. 寫入臨時檔案
    use std::io::Write;
    let mut file = std::fs::File::create(&temp_path)?;
    file.write_all(content.as_bytes())?;
    
    // 2. 強制作業系統將資料寫入實體磁碟 (Flush & Sync)
    file.sync_all()?;
    
    // 3. 原子化重新命名（這在 Linux 下是原子操作）
    std::fs::rename(temp_path, path)?;
    Ok(())
}
