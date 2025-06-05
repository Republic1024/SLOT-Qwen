import re

# 常见的日志时间格式
TIME_PATTERNS = [
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]",       # [2025-05-08 15:34:22]
    r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})",            # 2025/05/08 15:34:22
    r"(\d{2}:\d{2}:\d{2})",                              # 15:34:22
]

# 日志等级关键词（不区分大小写）
LEVEL_KEYWORDS = [
    "INFO", "DEBUG", "ERROR", "WARN", "WARNING",
    "TRACE", "EXCEPTION", "FATAL", "CRITICAL"
]

def extract_time(line):
    for pattern in TIME_PATTERNS:
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    return None

def extract_level(line):
    upper_line = line.upper()
    for level in LEVEL_KEYWORDS:
        if level in upper_line:
            return level
    return "UNKNOWN"

def parse_log_line(line, source_file="unknown.log"):
    content = line.strip()
    if not content:
        return None

    time_str = extract_time(content)
    level = extract_level(content)

    return {
        "source_file": source_file,
        "timestamp": time_str,
        "level": level,
        "content": content
    }
