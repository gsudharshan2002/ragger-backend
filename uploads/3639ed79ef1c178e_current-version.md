# GitHub REST API Reference — current-version

**API Version:** `2026-03-10`

Send this version via the `X-GitHub-Api-Version: 2026-03-10` request header. This is the first API version to ship breaking changes since GitHub introduced calendar-versioned APIs. `2022-11-28` remains fully supported until at least 2028-03-10, and requests that omit the `X-GitHub-Api-Version` header still default to `2022-11-28`, **not** to this version — you must opt in explicitly to get the changes below.

## GET /rate_limit

The deprecated top-level **`rate` property has been removed.** Read rate limit information from `resources.core` instead.

```json
{
  "resources": {
    "core": { "limit": 5000, "remaining": 4999, "reset": 1730000000 }
  }
}
```

> **Migrating from 2022-11-28:** replace any code reading the top-level `rate` field with `resources.core`.

## POST /orgs/{org}/teams

The deprecated **`permission` property has been removed** from team creation. Set repository-level permissions via the dedicated "Add or update team repository permissions" endpoint after creating the team instead.

```json
{ "name": "core-team" }
```

> **Migrating from 2022-11-28:** stop sending `permission` in the request body; set repository permissions through the separate team-repository-permissions endpoint instead.

## GET /repos/{owner}/{repo}/contents/{path}

Submodule entries now correctly report **`"type": "submodule"`** instead of `"type": "file"`:

```json
{ "name": "vendor/lib", "type": "submodule", "submodule_git_url": "https://github.com/example/lib.git" }
```

> **Migrating from 2022-11-28:** any code that branched on `type === "file"` to detect regular files must now also exclude `"submodule"` explicitly, since submodules no longer masquerade as files.

## GET /repos/{owner}/{repo}

The deprecated **`has_downloads` property has been removed** from the repository object; there is no replacement field.

```json
{ "id": 1, "name": "example" }
```

> **Migrating from 2022-11-28:** remove any dependency on `has_downloads`.

## GET /repos/{owner}/{repo}/pulls/{pull_number}

The **`merge_commit_sha` field has been removed** from all pull request responses.

```json
{ "number": 42, "state": "closed", "merged": true }
```

> **Migrating from 2022-11-28:** there is no direct replacement field on the pull request object; look up the merge commit via the repository's commit history instead.

## GET /repos/{owner}/{repo}/issues/{issue_number}

The singular **`assignee` field has been removed.** Use the `assignees` array instead.

```json
{ "number": 10, "assignees": [{ "login": "octocat" }] }
```

> **Migrating from 2022-11-28:** read and write assignees exclusively through the `assignees` array.

## POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches

The **`return_run_details` parameter has been removed.** On success, this endpoint now always returns **`200 OK`** with the triggered workflow run's details in the response body (previously `204 No Content` with no body).

```json
{ "id": 123456, "status": "queued", "workflow_id": 1 }
```

> **Migrating from 2022-11-28:** stop passing `return_run_details`, and update any code that expected `204 No Content` to instead parse the `200 OK` response body.

## DELETE /app/installations/{installation_id}

This endpoint now returns **`202 Accepted`** instead of `204 No Content`; the deletion is processed in the background.

> **Migrating from 2022-11-28:** treat `202` as success, not just `204` — the installation may not be fully removed the instant the response is received.

## GET /repos/{owner}/{repo}/code-scanning/default-setup

The `languages` array no longer contains `javascript` and `typescript` as separate values — they are combined into a single **`javascript-typescript`** value, since CodeQL analyzes them together.

```json
{ "state": "configured", "languages": ["javascript-typescript", "python"] }
```

> **Migrating from 2022-11-28:** replace checks for `"javascript"` or `"typescript"` individually with a check for `"javascript-typescript"`.

## GET /repos/{owner}/{repo}/attestations/{subject_digest}

The **`bundle` property has been removed** from attestation list responses. Use **`bundle_url`** to retrieve the attestation bundle instead.

```json
{ "attestations": [{ "bundle_url": "https://api.github.com/repos/example/example/attestations/download/sha256:abc123" }] }
```

> **Migrating from 2022-11-28:** fetch the bundle from `bundle_url` rather than reading it inline from `bundle`.
