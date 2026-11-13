# 0013: Keep the scoreboard schedule enabled and show stale data

Status: accepted.

## Context

GitHub disables scheduled workflows in public repositories after 60 days without repository activity (GitHub Actions documentation, read 2026-11-04). The scoreboard depends on the morning schedule, so a disabled workflow would leave the published document unchanged without any visible failure.

The keepalive-workflow action (version 2.0.10) relies on an enable call resetting that inactivity clock. GitHub does not document the reset as a guarantee, so the call is a mitigation rather than a promise.

## Decision

The last step of the morning score job calls:

```sh
gh api -X PUT "repos/${GITHUB_REPOSITORY}/actions/workflows/scoreboard.yml/enable"
```

The job has `actions: write` for that call. The published `scores.json` also carries `stale_since`, two days after `generated_at`. The page compares today with that date and says the data is stale when the schedule has not run. The page is the backstop because it reports the failure even if the enable call does not reset the clock.

## Evidence and limits

The enable call and the stale date are configuration and display logic. I have not run the scheduled workflow or observed GitHub disabling and re-enabling it. The 60-day rule and the keepalive mechanism are recorded from the cited documentation and the action's implementation, not from a local measurement.
