# GitHub REST API Reference — old-version

**API Version:** `2022-11-28`

Send this version explicitly via the `X-GitHub-Api-Version: 2022-11-28` request header, or omit the header entirely — `2022-11-28` is the default version GitHub uses when no header is sent. It remains fully supported for at least 24 months after any newer version ships.

This reference documents endpoint behavior as of API Version 2022-11-28, the version most existing integrations were built against before the 2026-03-10 breaking-changes release.

## GET /rate_limit

Returns your current rate limit status. The response body includes a **top-level `rate` object** (deprecated, but still present in this version) in addition to `resources.core`:

```json
{
  "resources": {
    "core": { "limit": 5000, "remaining": 4999, "reset": 1730000000 }
  },
  "rate": { "limit": 5000, "remaining": 4999, "reset": 1730000000 }
}
```

## POST /orgs/{org}/teams

Creates a team. Accepts a `permission` property to set the team's default repository permission:

```json
{ "name": "core-team", "permission": "push" }
```

## GET /repos/{owner}/{repo}/contents/{path}

Returns the contents of a file, directory, or submodule. A submodule entry reports `"type": "file"`:

```json
{ "name": "vendor/lib", "type": "file", "submodule_git_url": "https://github.com/example/lib.git" }
```

## GET /repos/{owner}/{repo}

Returns a repository. The response includes the deprecated `has_downloads` boolean:

```json
{ "id": 1, "name": "example", "has_downloads": true }
```

## GET /repos/{owner}/{repo}/pulls/{pull_number}

Returns a pull request. The response includes a `merge_commit_sha` field:

```json
{ "number": 42, "state": "closed", "merged": true, "merge_commit_sha": "6dcb09b5b57875f334f61aebed695e2e4193db5" }
```

## GET /repos/{owner}/{repo}/issues/{issue_number}

Returns an issue or pull request. The response includes a singular `assignee` field (nullable) alongside `assignees`:

```json
{ "number": 10, "assignee": { "login": "octocat" }, "assignees": [{ "login": "octocat" }] }
```

## POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches

Triggers a workflow run. Accepts an optional `return_run_details` parameter. On success, returns **`204 No Content`** with no response body.

## DELETE /app/installations/{installation_id}

Removes a GitHub App installation. On success, returns **`204 No Content`**.

## GET /repos/{owner}/{repo}/code-scanning/default-setup

Returns the default setup configuration for code scanning. The `languages` array may contain `javascript` and `typescript` as **separate** values:

```json
{ "state": "configured", "languages": ["javascript", "typescript", "python"] }
```

## GET /repos/{owner}/{repo}/attestations/{subject_digest}

Returns build attestations for a subject digest. Each attestation includes a `bundle` property containing the full attestation bundle inline:

```json
{ "attestations": [{ "bundle": { "mediaType": "application/vnd.dev.sigstore.bundle+json;version=0.3", "verificationMaterial": {} } }] }
```
