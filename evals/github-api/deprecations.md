# Deprecations: 2022-11-28 → 2026-03-10

Every symbol below existed in API Version `2022-11-28` (old-version.md) and was removed or changed in `2026-03-10` (current-version.md). Any assistant answer that mentions one of these symbols as if it still works in the current version, without stating the migration note, is a version-confusion failure.

Table format is fixed (`| symbol | endpoint | migration_note |`) because this file is parsed directly by the eval's deterministic assertion code, not judged by the LLM.

| symbol | endpoint | migration_note |
|---|---|---|
| `rate` (top-level field) | `GET /rate_limit` | Read rate limit info from `resources.core` instead. |
| `permission` (request property) | `POST /orgs/{org}/teams` | Set repository permissions via the team-repository-permissions endpoint after creating the team. |
| `type: "file"` for submodules | `GET /repos/{owner}/{repo}/contents/{path}` | Submodule entries now report `type: "submodule"`; don't assume `"file"` covers them. |
| `has_downloads` | `GET /repos/{owner}/{repo}` | Removed with no replacement field. |
| `merge_commit_sha` | `GET /repos/{owner}/{repo}/pulls/{pull_number}` | No direct replacement; look up the merge commit via commit history. |
| `assignee` (singular field) | `GET /repos/{owner}/{repo}/issues/{issue_number}` | Use the `assignees` array instead. |
| `return_run_details` (request param) | `POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches` | Removed; the endpoint now always returns `200 OK` with run details in the body. |
| `204 No Content` (success response) | `DELETE /app/installations/{installation_id}` | Now returns `202 Accepted`; deletion is processed in the background. |
| `javascript` / `typescript` (separate enum values) | `GET /repos/{owner}/{repo}/code-scanning/default-setup` | Combined into a single `javascript-typescript` value. |
| `bundle` (inline field) | `GET /repos/{owner}/{repo}/attestations/{subject_digest}` | Use `bundle_url` to fetch the attestation bundle instead. |
