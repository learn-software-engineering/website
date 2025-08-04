---
title: "Revert a local change in Git"
authors:
  - jnonino
description: >
  How to discard changes in Git before pushing them.
date: 2023-01-18
tags: ["Blog","Version Control", "VCS", "Git", "Revert"]
---

When we have created a commit locally but have not published it to the remote yet, we can use `git reset` to undo the commit and, if we wish, discard the changes.
Although there are several options for `git reset` the most used are:

> - `--soft`: Does not touch the index file or the working tree at all (but resets the head to <COMMIT>, just like all modes do). This leaves all your changed files "Changes to be committed", as git status would put it.
> - `--hard`: Resets the index and working tree. Any changes to tracked files in the working tree since <COMMIT> are discarded. Any untracked files or directories in the way of writing any tracked files are simply deleted.
>
> [Git reset documentation](https://git-scm.com/docs/git-reset)
> {: style="text-align: right;"}

Here there is an example about using `git reset`. We start by changing a file and creating a commit with the change.

```bash
bash-3.2$  cat README.md
# Index

1
bash-3.2$
bash-3.2$  cat README.md
# Index

1
2
bash-3.2$
bash-3.2$  git add README.md
bash-3.2$
bash-3.2$  git commit -m "Add number 2 in README.md"
[main 3734fd5] Add number 2 in README.md
 1 file changed, 1 insertion(+)
```

`git status` shows there is one commit pending to be published.

```bash
bash-3.2$  git status
On branch main
Your branch is ahead of 'origin/main' by 1 commit.
  (use "git push" to publish your local commits)

nothing to commit, working tree clean
```

Using `git reflog` to see the history.

```bash
3734fd5 (HEAD -> main) HEAD@{0}: commit: Add number 2 in README.md
866bfa8 (origin/main) HEAD@{1}: revert: Revert "Merge branch 'feature-1'"
e2f6d08 HEAD@{2}: merge feature-1: Merge made by the 'ort' strategy.
23644da HEAD@{3}: checkout: moving from feature-1 to main
```

Now we can use `git reset --soft <COMMIT_ID>` to undo the commit but keep the changes.

```bash
bash-3.2$  git reset --soft 866bfa8
bash-3.2$
bash-3.2$  git status
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.md
```

`git log` after using `git reset`

```bash
commit 866bfa8a952d11240707ebfc87f3266034d42443 (HEAD -> main, origin/main)
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 20:06:40 2023 -0300

    Revert "Merge branch 'feature-1'"

    This reverts commit e2f6d08d3b38a02a1c026cfb879f3131536757ac, reversing
    changes made to 23644dab9fc5828ecdd358c6d3acb4196ed23546.
```

We create a new commit so we can test the `git reset --hard` command.

```bash
bash-3.2$  git status
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.md

bash-3.2$  git commit -m "Add number 2 in README.md - NEW COMMIT"
[main 2e7193d] Add number 2 in README.md - NEW COMMIT
 1 file changed, 1 insertion(+)
```

`git log` now shows the new commit.

```bash
commit 2e7193db650b9ba0762fe73525df599a08f8577d (HEAD -> main)
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Thu Jan 19 08:32:57 2023 -0300

    Add number 2 in README.md - NEW COMMIT

commit 866bfa8a952d11240707ebfc87f3266034d42443 (origin/main)
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 20:06:40 2023 -0300

    Revert "Merge branch 'feature-1'"

    This reverts commit e2f6d08d3b38a02a1c026cfb879f3131536757ac, reversing
    changes made to 23644dab9fc5828ecdd358c6d3acb4196ed23546.
```

Now we can use `git reset --hard <COMMIT_ID>` to undo the commit and discard all the changes.

```bash
bash-3.2$  git reset --hard 866bfa8
HEAD is now at 866bfa8 Revert "Merge branch 'feature-1'"
bash-3.2$
bash-3.2$  git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

`git log` remains as it nothing had happened.

```bash
commit 866bfa8a952d11240707ebfc87f3266034d42443 (HEAD -> main, origin/main)
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 20:06:40 2023 -0300

    Revert "Merge branch 'feature-1'"

    This reverts commit e2f6d08d3b38a02a1c026cfb879f3131536757ac, reversing
    changes made to 23644dab9fc5828ecdd358c6d3acb4196ed23546.
```
