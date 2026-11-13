# Publication notes

Publication is pending. Before public release I need to:

- Configure an experiment archive with credentials limited to that archive.
- Confirm which free notebook accelerators are available; keep CPU execution reproducible.
- Validate accelerator execution separately from the measured local CPU run.
- Finish the method description and record limitations beside the results.
- Add the social preview and discussion categories.
- Configure package publication with a trusted publisher.
- Select a location for the long-form write-up.
- Check the demonstration's clarity with readers before choosing its final presentation.
- Prepare launch posts and keep time available to answer questions and issues.

Collection also needs activation: publish the repository, enable Actions,
dispatch `scoreboard.yml`, and verify today's `wn2/` file on `origin/data`.
I have not yet verified these remote steps.

## Required Pages assets

Pages deliberately fails until the Sandy home experiment is published. Export
the verified run, upload `dist/atlases/*.zip` to a GitHub release, then run
`uv run python scripts/fetch_atlases.py --lock <release-tag>` with
`site/hero.json` naming `sandy_east_300km`. Commit the resulting
`site/atlases.lock.json` with the release tag and SHA-256 hashes. The lock must
include the home experiment as well as every other intended published export.

The Pages job installs the frozen Python environment without model dependencies,
checks the required lock entries, and invokes `scripts/fetch_atlases.py` before
the site build. Missing assets, changed archive bytes or failed manifest checks
stop the job before upload or deployment. The atlas release is uploaded by hand
before the first deployment; a fixture-only build is not a successful deployment.
