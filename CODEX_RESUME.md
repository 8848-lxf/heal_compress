# Codex resume note

Previous Codex thread became too long and showed context compaction warning.

Current goal:
- Continue the unfinished task only.
- Do not redo completed work unless tests reveal a concrete bug.

Completed:
- TODO: 写已经完成的任务列表

Unfinished:
- TODO: 写还没完成的任务

Important constraints:
- Preserve existing CLI/API behavior unless explicitly asked.
- Keep changes minimal and localized.
- Do not add generated files, caches, logs, model weights, or experiment outputs to git.

Files to inspect first:
- TODO: 写关键文件路径

Suggested workflow:
1. Run git status and git diff --stat.
2. Inspect the listed files.
3. Identify the exact unfinished integration point.
4. Make the minimal code change.
5. Run the smallest relevant test first.
6. Show git diff --stat and test results.
