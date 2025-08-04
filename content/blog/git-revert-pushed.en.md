---
title: "Revert a pushed change in Git"
authors:
  - jnonino
description: >
  Have you ever pushed a change in Git and it was wrong? Here you'll learn how to revert the change, even if the problem was with a merge commit.
date: 2023-01-18
tags: ["Blog","Version Control", "VCS", "Git", "Revert"]
---

When we realized that the last commit was a mistake but we already published it, the command to use is `git revert <COMMIT_HASH>`.

- First we need to locate the ID of the commit we want to revert, it can be done with `git log` or `git reflog` commands.
- Then, run the `git revert <COMMIT_HASH>` command using the ID obtained in the previous step. Use the options `-e` or `--edit` to edit the commit message if we like.
- Push our changes so the revert is available for everyone in our group.

## Reverting multiple commits

If we need to revert multiple commits we can revert them one by one using the `--no-commit` option in order to create a single revert commit at the end.

Imagine the history is like the following and we need to go back to `COMMIT-3`.

```bash
COMMIT-1 -> COMMIT-2 -> COMMIT-3 -> COMMIT-4 -> COMMIT-5 -> COMMIT-6 -> HEAD
```

This sequence of commands will get our files to the version of `COMMIT-3`:

```bash
bash-3.2$  git revert --no-commit COMMIT-6
bash-3.2$  git revert --no-commit COMMIT-5
bash-3.2$  git revert --no-commit COMMIT-4
bash-3.2$  git commit -m "Revert to version in COMMIT-3"
bash-3.2$  git push
```

## Reverting a merge commit

>  -m parent-number, --mainline parent-number
>
>   Usually you cannot revert a merge because you do not know which side of the merge should be considered the mainline. This option specifies the parent number (starting from 1) of the mainline and allows revert to reverse the change relative to the specified parent.
>
> [Git revert documentation](https://git-scm.com/docs/git-revert)
> {: style="text-align: right;"}

When we need to revert a merge commit `git revert` command needs to be run with the `-m` or `--mainline` option to indicate the **parent number** because a `merge commit` has more than one parent and Git does not know which parent was target branch and which was the branch with the changes that should be reverted.

Here there is an example showing how to revert a merge commit.

Create the first commit in `main` branch.

```bash
bash-3.2$  cat README.md
# Index

1
bash-3.2$
bash-3.2$  git commit -m "Add number 1 in README.md - main branch"
[main (root-commit) 23644da] Add number 1 in README.md - main branch
 1 file changed, 3 insertions(+)
 create mode 100644 README.md
bash-3.2$
bash-3.2$  git push -u origin main
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Writing objects: 100% (3/3), 254 bytes | 254.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To github.com:jnonino/test-repo.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
bash-3.2$
```

The state of the `README.md` file in `main` branch.

```bash
bash-3.2$git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
bash-3.2$
bash-3.2$  cat README.md
# Index

1
```

Branch `feature-1` created and added one commit.

```bash
bash-3.2$  git checkout -b feature-1
Switched to a new branch 'feature-1'
bash-3.2$
bash-3.2$  cat README.md
# Index

1
2
bash-3.2$
bash-3.2$  git add README.md
bash-3.2$
bash-3.2$  git commit -m "Add number 2 in README.md - feature-1 branch"
[feature-1 83ea1a3] Add number 2 in README.md - feature-1 branch
 1 file changed, 1 insertion(+)
bash-3.2$
bash-3.2$  git push --set-upstream origin feature-1
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Writing objects: 100% (3/3), 292 bytes | 292.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
remote:
remote: Create a pull request for 'feature-1' on GitHub by visiting:
remote:      https://github.com/jnonino/test-repo/pull/new/feature-1
remote:
To github.com:jnonino/test-repo.git
 * [new branch]      feature-1 -> feature-1
branch 'feature-1' set up to track 'origin/feature-1'.
bash-3.2$
```

The state of `README.md` in the `feature-1` branch.

```bash
bash-3.2$  git status
On branch feature-1
Your branch is up to date with 'origin/feature-1'.

nothing to commit, working tree clean
bash-3.2$
bash-3.2$  cat README.md
# Index

1
2
```

Merge the `feature-1` branch into the `main` branch.

```bash
bash-3.2$  git checkout main
Switched to branch 'main'
Your branch is up to date with 'origin/main'.
bash-3.2$
bash-3.2$  git merge --no-ff feature-1
Merge made by the 'ort' strategy.
 README.md | 1 +
 1 file changed, 1 insertion(+)
bash-3.2$
bash-3.2$  git push
Enumerating objects: 1, done.
Counting objects: 100% (1/1), done.
Writing objects: 100% (1/1), 233 bytes | 233.00 KiB/s, done.
Total 1 (delta 0), reused 0 (delta 0), pack-reused 0
To github.com:jnonino/test-repo.git
   23644da..e2f6d08  main -> main
bash-3.2$
```

Current state of `README.md` in `main` branch.

```bash
bash-3.2$  git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
bash-3.2$
bash-3.2$  cat README.md
# Index

1
2
```

`git log` after merging `feature-1` into `main` branch.

```bash
commit e2f6d08d3b38a02a1c026cfb879f3131536757ac (HEAD -> main, origin/main)
Merge: 23644da 83ea1a3
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 19:58:19 2023 -0300

    Merge branch 'feature-1'

commit 83ea1a347e0e87b19a611997219089b5b9247d1f (origin/feature-1, feature-1)
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 19:53:38 2023 -0300

    Add number 2 in README.md - feature-1 branch

commit 23644dab9fc5828ecdd358c6d3acb4196ed23546
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 19:48:37 2023 -0300

    Add number 1 in README.md - main branch
```

To revert the merge commit, as it was stated above we need to pay attention to the `merge` field.

```bash
Merge: 23644da 83ea1a3
```

Running `git revert e2f6d08 -m 1` will reinstate the tree as it was in `23644da`, and `git revert e2f6d08 -m 2` will set the tree as it was in `83ea1a3`.
In this example we would like to leave the `main` branch as it was before the merge commit. For doing that, we need to run `git revert e2f6d08 -m 1`.

```bash
bash-3.2$  git revert e2f6d08 -m 1
[main 866bfa8] Revert "Merge branch 'feature-1'"
 1 file changed, 1 deletion(-)
bash-3.2$
bash-3.2$  git status
On branch main
Your branch is ahead of 'origin/main' by 1 commit.
  (use "git push" to publish your local commits)

nothing to commit, working tree clean
bash-3.2$
bash-3.2$  git push
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Writing objects: 100% (3/3), 344 bytes | 344.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To github.com:jnonino/test-repo.git
   e2f6d08..866bfa8  main -> main
```

`git log` after reverting the merge commit.

```bash
commit 866bfa8a952d11240707ebfc87f3266034d42443 (HEAD -> main, origin/main)
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 20:06:40 2023 -0300

    Revert "Merge branch 'feature-1'"

    This reverts commit e2f6d08d3b38a02a1c026cfb879f3131536757ac, reversing
    changes made to 23644dab9fc5828ecdd358c6d3acb4196ed23546.

commit e2f6d08d3b38a02a1c026cfb879f3131536757ac
Merge: 23644da 83ea1a3
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 19:58:19 2023 -0300

    Merge branch 'feature-1'

commit 83ea1a347e0e87b19a611997219089b5b9247d1f (origin/feature-1, feature-1)
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 19:53:38 2023 -0300

    Add number 2 in README.md - feature-1 branch

commit 23644dab9fc5828ecdd358c6d3acb4196ed23546
Author: Julian Nonino <learn.software.eng+jnonino@gmail.com>
Date:   Wed Jan 18 19:48:37 2023 -0300

    Add number 1 in README.md - main branch
```

Current state of `README.md` in `main` branch.

```bash
bash-3.2$  git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
bash-3.2$
bash-3.2$  cat README.md
# Index

1
```
