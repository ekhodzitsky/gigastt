# Homebrew release updates

Run **Homebrew Formula update** manually with an existing published release tag.
The workflow verifies published `SHA256SUMS.txt` and proposes only
`Formula/gigastt.rb` in a release-specific PR. Neither a tag push nor a successful
Release starts Homebrew automatically. Repeated manual runs reuse the same branch
and PR; missing, duplicate or malformed checksums stop the update before writing.
An older selected release can propose a downgrade, so review the version too.

```sh
gh workflow run homebrew.yml --ref main -f tag=v2.22.0
```

The repository must allow GitHub Actions to create pull requests. The proposal
job uses `contents: write` and `pull-requests: write`; it never approves reviews,
merges, pushes directly to main or changes branch protection. The commit-pinned
`peter-evans/create-pull-request` action handles repeatable branch and PR updates.
Downloaded release assets stay outside the checkout. No additional secret is
required.

After the proposal is ready, manually dispatch CI and Publication hygiene for
its branch, selecting the accepted ancestor SHA and proposal PR number. Review
and merge only after required checks pass. The formula retains its existing pins
until then. See [manual CI operation](performance-regressions.md#manual-ci-operation).

The manual path reads published checksums and does not depend on a previous
workflow's artifact or run attempt. Release metadata artifacts remain available
for provenance, but do not trigger a downstream workflow.
