---
name: document-changes
description: Generates a comprehensive change summary from git modifications
---

OBJECTIVE
---------
After I (the user) make code changes (add, edit, or delete files), invoke this skill to create a structured change summary document.

INSTRUCTIONS FOR THE AI
-----------------------
When this skill is invoked, produce a markformatted change summary report by following these steps:

1. **Gather change information** using git commands:
   ```bash
   git diff --cached HEAD  # staged changes
   git diff HEAD           # unstaged changes
   git status --short      # all modified/untracked files
   git diff --stat HEAD    # line change statistics
   git rev-parse --abbrev-ref HEAD  # current branch
   ```

2. **Analyze** the changes and create a summary document containing:
   - **Branch**: Current branch name
   - **Total files changed**: Count of all modified/added/deleted files
   - **Modified files**: List each file with a brief description of what changed
   - **Added files**: List new files with purpose
   - **Deleted files**: List removed files
   - **Untracked files**: List if any exist
   - **Diff statistics**: Include the git diff --stat output
   - **Impact**: Brief note on what the overall effect of these changes is
   - **Breaking changes?**: Note if any (e.g., API changes, removed functionality, config changes)
   - **Migration steps**: If applicable, list any steps needed to adopt these changes

3. **Format** the output as a clear markdown document suitable for PR descriptions, commit messages, or release notes.

EXAMPLES
--------
User types: /document-changes

Expected output:
```markdown
# Change Summary

**Branch:** feature/my-feature
**Total files changed:** 3

## Modified Files (2)
- `api/main.py` - Added CORS middleware configuration
- `agent/mcp_agent.py` - Updated Schema Inspector integration

## Added Files (1)
- `my_skill/change-summary/SKILL.md` - New skill for generating change summaries

## Deleted Files (0)

## Diff Statistics
 3 files changed, 45 insertions(+), 2 deletions(-)

## Impact
CORS now enabled for browser-based frontend clients. Added automated change documentation skill.

## Breaking Changes
None

## Migration Steps
Restart server after configuration changes.
```