# Professional Git & GitHub Cheat Sheet — RL-EMS Project

**Keshvendra Kumar Ramawat** · Always-open reference · Corrected & completed version

> All commands are copy-paste safe. Every long flag uses a **double dash** (`--`).
> Run them from the repo root: `RL-Based_Hybrid_vehicle_EMS/`.

---

## SECTION 0 — ONE-TIME SETUP (do this once, ever)

```bash
# Identity (used to sign every commit)
git config --global user.name  "Keshvendra Kumar Ramawat"
git config --global user.email "keshvendra.ramawat@gmail.com"

# Use VS Code for commit messages / interactive edits
git config --global core.editor "code --wait"

# New repos start on 'main', not 'master'
git config --global init.defaultBranch main

# Nicer output + safer pushes
git config --global pull.ff only          # refuse surprise merge commits on pull
git config --global push.autoSetupRemote true

# Confirm everything
git config --list
```

---

## SECTION 1 — BRANCH STRATEGY

| Branch | Purpose | Rule |
| --- | --- | --- |
| `main` | Stable, presentable, "known-good" history | Never commit directly. Only receives merges from `dev` at a release. |
| `dev`  | Day-to-day work: experiments, diagnostics, docs | This is where you live. Commit and push here every day. |

```
main   ●────────────────●────────────────●        <- tagged releases (v2.0, v4.0, ...)
        \              / \              /
dev      ●──●──●──●──●    ●──●──●──●──●
         daily work        daily work
```

- You do **not** need a separate `feature/...` branch for this project — working directly on `dev` is fine for a solo research repo.
- Start a throwaway branch only for a risky rewrite you might abandon: `git checkout -b spike/new-reward`.

---

## SECTION 2 — DAILY WORKFLOW

### Every morning
```bash
git checkout dev
git pull                 # get anything you pushed from another machine
```

### While working
```bash
git status               # what changed?
git diff                 # exact line-by-line changes
git add <file>           # stage one file  (preferred: deliberate)
git add .                # stage everything (fast: only when you know it's all wanted)
git diff --staged        # review what you're about to commit
git commit -m "type: short description"
```

### Every evening
```bash
git status               # nothing left behind?
git push                 # GitHub is now up to date
```

**Golden rule:** never end the day with unpushed commits or uncommitted work you care about.

---

## SECTION 3 — COMMIT MESSAGES

Format: **`type: short description in the imperative mood`**

| type | When to use | Example |
| --- | --- | --- |
| `feat:`       | New file, function, or capability            | `feat: add prioritized experience replay` |
| `fix:`        | Fixing a bug                                 | `fix: resolve NaN reward at low SOC` |
| `experiment:` | Running / saving an RL experiment or study   | `experiment: SAC vs n-step SAC on WLTP` |
| `refactor:`   | Restructure without changing behaviour       | `refactor: split sac.py into modules` |
| `docs:`       | README, comments, CHANGELOG, reports         | `docs: add Phase 3 results to README` |
| `model:`      | Save / update a trained checkpoint           | `model: save best SAC at episode 500` |
| `test:`       | Add or fix tests                             | `test: add unit test for reward function` |
| `chore:`      | Cleanup, config, maintenance                 | `chore: update .gitignore for .pkl files` |

**Good**
```
feat: add lookahead env with SOC prediction
fix: resolve SOC drift during highway segment
experiment: compare PER vs uniform on WLTP
```

**Bad**
```
update changes
fix stuff
done aaaaaa
```

Tips:
- Keep the subject line under ~60 chars. Add detail in a body if needed:
  ```bash
  git commit    # opens VS Code; line 1 = subject, blank line, then bullets
  ```
- One logical change per commit. If the description needs "and", consider two commits.

---

## SECTION 4 — GITHUB ISSUES (task tracker)

Create at: **GitHub repo → Issues → New issue**

| Label | Purpose | Example title |
| --- | --- | --- |
| `[EXPERIMENT]` | An RL run to do        | `[EXPERIMENT] SAC vs n-step SAC on WLTP` |
| `[BUG]`        | Something is broken     | `[BUG] SOC drifts below 0.3 on highway` |
| `[FEATURE]`    | New capability          | `[FEATURE] add FTP-75 evaluation pipeline` |
| `[DOCS]`       | Documentation           | `[DOCS] document reward function formula` |
| `[CHORE]`      | Maintenance             | `[CHORE] archive old model checkpoints` |

**The professional loop:**
1. Create an Issue on GitHub.
2. Work on `dev`.
3. Commit with a matching message, referencing the issue number:
   `experiment: test PER beta annealing (#12)`
4. Push.
5. Close the Issue with a short results comment (numbers, plots, conclusion).

Writing `Closes #12` in a commit or PR description auto-closes the issue when it lands on `main`.

---

## SECTION 5 — INSPECTING HISTORY

```bash
git log --oneline -10               # last 10 commits, compact
git log --oneline --graph --all     # visual branch/merge graph
git show <hash>                     # full diff of one commit
git blame <file>                    # who/when for each line
git diff main..dev                  # what's on dev but not main
```

---

## SECTION 6 — TAGS & RELEASES

Tag every completed phase. A tag is a permanent bookmark — `git checkout v5.0` gives you
the code exactly as it was then.

```bash
# Tag the current commit
git tag -a v3.0 -m "Phase 3: SAC with PER"

# Tag a past commit (backfilling)
git tag -a v2.0 <hash> -m "Phase 2 complete"

git push --tags                     # tags are NOT pushed by 'git push' alone

git tag                             # list all tags
git show v3.0                       # what the tag points at
```

### Version scheme

| Version | Meaning |
| --- | --- |
| `v1.0`, `v2.0` | Major phase complete |
| `v3.1`, `v3.2` | Minor improvement within a phase |
| `v3.1.1`       | Small bug fix on top of a tagged version |

---

## SECTION 7 — MERGING DEV → MAIN (stable release)

Do this when `dev` is at a clean, presentable state (usually right after tagging a phase).

```bash
# 1. Make sure dev is fully pushed
git checkout dev
git push

# 2. Switch to main and update it
git checkout main
git pull

# 3. Merge dev into main
git merge dev

# 4. Push main
git push

# 5. Tag the release (if not already tagged on dev)
git tag -a v4.0 -m "Phase 4 complete"
git push --tags

# 6. Back to work
git checkout dev
```

If `git merge dev` reports conflicts: open each conflicted file, resolve the
`<<<<<<< ======= >>>>>>>` markers, then `git add <file>` and `git commit`.

---

## SECTION 8 — EMERGENCY UNDO

| Command | What it does | Danger |
| --- | --- | --- |
| `git restore <file>`            | Discard unstaged changes in one file        | loses those edits |
| `git restore --staged <file>`   | Unstage a file (keep the edits)             | safe |
| `git commit --amend`            | Fix the **last** commit's message/content   | only if not pushed |
| `git reset --soft HEAD~1`       | Undo last commit, keep changes staged       | safe, local only |
| `git reset --hard HEAD~1`       | Undo last commit **and delete the changes** | ⚠️ irreversible |
| `git stash` / `git stash pop`   | Shelve work / bring it back                 | safe |
| `git revert <hash>`             | Make a **new** commit that undoes an old one| safe, history-preserving |
| `git reflog`                    | Show every HEAD move — recover "lost" commits| your safety net |

Rule: never `reset --hard` or force-push a branch that's already on GitHub and shared.
Use `git revert` instead.

---

## SECTION 9 — NOTIFICATIONS & STATUS

### GitHub email notifications
GitHub → Settings → Notifications → turn on **Issues**, **Pull Requests**, **Actions (CI)**.

### VS Code status bar (bottom-left)
```
 dev  ↑2 ↓0
  │    │  └── commits to PULL from GitHub
  │    └───── commits waiting to PUSH
  └────────── current branch
```
- `↑` visible → **push now**
- `↓` visible → **pull before working**

### Automation already set up in this repo
- `.github/workflows/ci.yml` — runs tests on every push.
- `.github/workflows/weekly_reminder.yml` — every Monday, opens a checklist Issue; you get an email.

---

## SECTION 10 — QUICK REFERENCE CARD

```bash
# --- start of day ---
git checkout dev && git pull

# --- while working ---
git status
git diff
git add <file>
git commit -m "type: description"

# --- end of day ---
git push

# --- finished a phase ---
git commit -m "docs: add Phase N report"
git tag -a vN.0 -m "Phase N complete"
git push --tags

# --- release to main ---
git checkout main && git pull && git merge dev && git push
git checkout dev

# --- oh no ---
git restore <file>            # undo file edits
git reset --soft HEAD~1       # undo last commit, keep work
git reflog                    # find anything "lost"
```

**The loop:** Issue → work on `dev` → `type:` commit → push → close Issue with results.
