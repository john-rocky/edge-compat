"""Own `.litertlm` bundle list for the catalog regression (`data/regression/bundles.json`).

    uv run python tools/regression/list_bundles.py [--out data/regression/bundles.json]
    uv run python tools/regression/list_bundles.py --check data/regression/bundles.json

Rows come from three sources, all read from the real thing:

(a) the shipping record: every `.litertlm` variant of the generated manifests
    (`--manifests`, default `$LITERTLM_MANIFESTS` or
    `~/code/litertlm-convert/manifest/generated`);
(b) the lab cards: every `cards/*/card.json` artifact that is a `.litertlm`;
(c) the Hugging Face API (read-only): every public repo of the own account
    that holds a `.litertlm`, and every `.litertlm` in a public litert-community
    repo that a commit authored by the own account wrote (file-level, never
    repo-level: the org's repos mix the team's files and ours).

A token (`HF_TOKEN`, else the `token` file under `$HF_HOME`, default
`~/.cache/huggingface`) is sent as `Authorization: Bearer` when present, from
a header file curl reads (never on a command line, in a log or in the list):
it opens the commit history of gated repos. Private repos the token can see
are not shipped bundles and are skipped (counted in `meta`).

Every row carries the live Hub values read from the repo's tree at the commit
that `api/models/<repo>` names (`hub_commit`, LFS `sha256`, `size_bytes`),
and its repo's `role` from `roles.py` (`chat` unless the map names the repo).
A file that left the Hub keeps its row with `status: "hub_missing"` and its
last known bytes under `last_known` (its lab card stays on that row; no second
card-only row). A row keeps the `bundle_id` the previous list gave it (results
are filed under it). `--check` re-reads the Hub for every row and prints what
changed (exit 0 = nothing, 1 = something); for a hub_missing row only its
bytes reappearing counts, not the repo's head moving on.

Model-level inventory only: nothing here reads or writes `data/matrix`.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litert_compat.parser.litertlm import LitertlmParseError, parse_litertlm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import roles  # noqa: E402 - stdlib only; the role of each own repo

REPO_ROOT = Path(__file__).resolve().parents[2]
HUB = "https://huggingface.co"
OWN_ORG = "mlboydaisuke"
ORGS = (OWN_ORG, "litert-community")
# The own account is whoever authored this commit (the 2026-10-11 re-ship of
# Qwen3.5-2B_int8.litertlm); read live, never assumed.
OWN_SEED_REPO = "litert-community/Qwen3.5-2B"
OWN_SEED_COMMIT = "4523abeb8b87f381a64a6ddb4906a805b13938bc"
GENERATOR = "tools/regression/list_bundles.py"
SUITE_OF_KIND = {"chat": "v1", "chat+vision": "v1", "chat+audio": "v2", "other": "v2"}
HEADER_PROBE_BYTES = 16384


class HubError(RuntimeError):
    """The Hub answered something this tool cannot use."""


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def hub_token() -> tuple[str | None, str]:
    """(token, where it came from); (None, "none") = public reads only. The
    value is only ever written to a request's header file."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token, "HF_TOKEN"
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    try:
        token = (home / "token").read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    return (token, "token file") if token else (None, "none")


class Hub:
    """curl against the Hub API: one request per second, a 429 waits 60 s and
    is retried once, `Link: rel="next"` pagination. With a token, every
    request to the Hub carries it (curl drops a custom Authorization header on
    a redirect to another host)."""

    def __init__(self, interval: float = 1.0, log=None, token: str | None = None) -> None:
        self.interval = interval
        self.last = 0.0
        self.calls = 0
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
        self.env = dict(os.environ, HF_HUB_DISABLE_XET="1")
        self.token = token

    def _once(self, url: str, method: str, data: bytes | None, byte_range: str | None) -> Response:
        for attempt in range(3):
            try:
                return self._curl(url, method, data, byte_range)
            except HubError as exc:
                if attempt == 2:
                    raise
                self.log(f"{exc}; retrying in 5 s")
                time.sleep(5)
        raise AssertionError("unreachable")

    def _curl(self, url: str, method: str, data: bytes | None, byte_range: str | None) -> Response:
        wait = self.interval - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        with tempfile.TemporaryDirectory() as tmp:
            head, body = Path(tmp) / "h", Path(tmp) / "b"
            cmd = ["curl", "-sSL", "-D", str(head), "-o", str(body),
                   "-w", "%{http_code}", "-X", method]
            if self.token and url.startswith(HUB + "/"):
                auth = Path(tmp) / "auth"
                auth.touch(mode=0o600)
                auth.write_text(f"Authorization: Bearer {self.token}\n", encoding="utf-8")
                cmd += ["-H", f"@{auth}"]
            if data is not None:
                cmd += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
            if byte_range is not None:
                cmd += ["-r", byte_range]
            cmd.append(url)
            proc = subprocess.run(cmd, input=data, capture_output=True, env=self.env,
                                  check=False)
            self.last = time.monotonic()
            self.calls += 1
            if proc.returncode != 0:
                raise HubError(f"curl exit {proc.returncode} for {url}: "
                               f"{proc.stderr.decode(errors='replace').strip()[:200]}")
            status = int(proc.stdout.decode().strip() or 0)
            headers: dict[str, str] = {}
            # With -L every hop's headers land in the file; keep the last block.
            for line in head.read_text(errors="replace").splitlines():
                if line.startswith("HTTP/"):
                    headers = {}
                elif ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            return Response(status, headers, body.read_bytes() if body.exists() else b"")

    def request(self, url: str, *, method: str = "GET", payload: Any = None,
                byte_range: str | None = None) -> Response:
        if not url.startswith("http"):
            url = HUB + url
        data = None if payload is None else json.dumps(payload).encode()
        resp = self._once(url, method, data, byte_range)
        if resp.status == 429:
            self.log(f"429 from {url}; waiting 60 s, one retry")
            time.sleep(60)
            resp = self._once(url, method, data, byte_range)
            if resp.status == 429:
                raise HubError(f"429 twice for {url}")
        return resp

    def get_json(self, url: str, *, ok=(200,)) -> Any:
        resp = self.request(url)
        if resp.status not in ok:
            raise HubError(f"HTTP {resp.status} for {url}")
        return resp.json()

    def paginate(self, url: str) -> tuple[int, list[Any]]:
        """All pages of a list endpoint; (status of the first page, items)."""
        items: list[Any] = []
        first_status = 0
        while url:
            resp = self.request(url)
            if not first_status:
                first_status = resp.status
            if resp.status != 200:
                return resp.status, items
            items.extend(resp.json())
            url = _next_link(resp.headers.get("link", ""))
        return first_status, items


def _next_link(link: str) -> str:
    for part in link.split(","):
        m = re.match(r'\s*<([^>]+)>\s*;\s*rel="next"', part)
        if m:
            return m.group(1)
    return ""


def _q(s: str) -> str:
    return urllib.parse.quote(s, safe="/")


# --- local sources -----------------------------------------------------------


def kind_of(model_types: list[str]) -> str:
    types = [t for t in model_types if t]
    if not any("prefill_decode" in t for t in types):
        return "other"
    if any("audio" in t for t in types):
        return "chat+audio"
    if any("vision" in t for t in types):
        return "chat+vision"
    return "chat"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-.")


def read_manifests(root: Path
                   ) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[str]]]:
    """(repo, file) -> manifest facts; plus upstream base_model -> own repo."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    base_to_repo: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        repo = doc["repo"]
        base = (doc.get("model") or {}).get("base_model")
        if isinstance(base, str) and base and repo not in base_to_repo[base.lower()]:
            base_to_repo[base.lower()].append(repo)
        for variant in doc.get("variants") or []:
            name = variant["file"]
            if not name.endswith(".litertlm"):
                continue
            types = [s.get("model_type") or "" for s in variant.get("sections") or []
                     if s.get("type") == "TFLiteModel"]
            rows[(repo, name)] = {
                "manifest": _home(path),
                "sha256": variant.get("sha256"),
                "size_bytes": variant.get("size_bytes"),
                "kind": kind_of(types),
            }
    return rows, base_to_repo


@dataclass
class CardArtifact:
    card: str
    file: str
    sha256: str | None
    source_repo: str | None  # model.source_url when it is an own-org repo
    upstream: str | None  # model.source_url as <org>/<name> otherwise
    mentions: list[str] = field(default_factory=list)  # own-org repos named in the card text


def read_cards(cards_dir: Path) -> list[CardArtifact]:
    own_re = re.compile(r"huggingface\.co/((?:" + "|".join(ORGS) + r")/[A-Za-z0-9._-]+)")
    out: list[CardArtifact] = []
    for path in sorted(cards_dir.glob("*/card.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        artifacts = [a for a in doc.get("artifacts") or [] if str(a.get("file", "")).endswith(
            ".litertlm")]
        if not artifacts:
            continue
        url = str((doc.get("model") or {}).get("source_url") or "")
        m = re.match(r"https://huggingface\.co/([^/]+)/([^/?#]+)", url)
        source_repo = upstream = None
        if m and m.group(1) in ORGS:
            source_repo = f"{m.group(1)}/{m.group(2)}"
        elif m:
            upstream = f"{m.group(1)}/{m.group(2)}"
        mentions = sorted({x.rstrip(".") for x in own_re.findall(json.dumps(doc))})
        for art in artifacts:
            out.append(CardArtifact(doc["model"]["id"], art["file"], art.get("sha256"),
                                    source_repo, upstream, mentions))
    return out


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _home(path: Path | str) -> str:
    """A path as written into data: the home directory as `~`."""
    text, home = str(path), str(Path.home())
    return "~" + text[len(home):] if text == home or text.startswith(home + os.sep) else text


def local_blob(sha256: str | None) -> str | None:
    if not sha256:
        return None
    hits = sorted(hf_cache_dir().glob(f"models--*/blobs/{sha256}"))
    return _home(hits[0]) if hits else None


# --- Hub reads ---------------------------------------------------------------


@dataclass
class HubFile:
    repo: str
    file: str
    sha256: str | None
    size_bytes: int | None
    hub_commit: str
    owner_evidence: str | None = None  # hf-own-org / hf-commit-author:<user>
    notes: list[str] = field(default_factory=list)


def list_org_repos(hub: Hub, org: str) -> list[dict[str, Any]]:
    url = (f"/api/models?author={org}&limit=200&"
           "expand%5B%5D=siblings&expand%5B%5D=sha&expand%5B%5D=private")
    status, items = hub.paginate(url)
    if status != 200:
        raise HubError(f"HTTP {status} listing {org}")
    return items


def repo_head(hub: Hub, repo: str) -> tuple[int, str | None, bool]:
    """(HTTP status, head commit, private)."""
    resp = hub.request(f"/api/models/{_q(repo)}")
    if resp.status != 200:
        return resp.status, None, False
    info = resp.json()
    return 200, info.get("sha"), bool(info.get("private"))


def repo_litertlm(hub: Hub, repo: str, commit: str) -> dict[str, tuple[str | None, int | None]]:
    status, entries = hub.paginate(f"/api/models/{_q(repo)}/tree/{commit}?recursive=true")
    if status != 200:
        raise HubError(f"HTTP {status} reading the tree of {repo}@{commit}")
    out: dict[str, tuple[str | None, int | None]] = {}
    for e in entries:
        if e.get("type") == "file" and e["path"].endswith(".litertlm"):
            lfs = e.get("lfs") or {}
            out[e["path"]] = (lfs.get("oid"), lfs.get("size", e.get("size")))
    return out


def paths_oids(hub: Hub, repo: str, rev: str, paths: list[str]) -> dict[str, str | None]:
    resp = hub.request(f"/api/models/{_q(repo)}/paths-info/{rev}", method="POST",
                       payload={"paths": paths, "expand": False})
    if resp.status != 200:
        raise HubError(f"HTTP {resp.status} for paths-info {repo}@{rev}")
    found = {e["path"]: ((e.get("lfs") or {}).get("oid") or e.get("oid")) for e in resp.json()}
    return {p: found.get(p) for p in paths}


def writers_by_own(hub: Hub, repo: str, paths: list[str], own: set[str]
                   ) -> tuple[dict[str, list[tuple[str, str, str]]], dict[str, str]] | int:
    """For each path, the own-authored commits that wrote its bytes
    [(commit, user, oid)], judged by the path's oid at the commit vs at the
    next-older commit in the listing; plus commit -> authors for notes.
    The HTTP status (401 / 403) when the commit history is not readable
    (a gated repo without a token, or one the token may not read)."""
    status, commits = hub.paginate(f"/api/models/{_q(repo)}/commits/main")
    if status in (401, 403):
        return status
    if status != 200:
        raise HubError(f"HTTP {status} listing commits of {repo}")
    authors = {c["id"]: ",".join(sorted(a["user"] for a in c.get("authors") or []))
               for c in commits}
    wrote: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    cache: dict[str, dict[str, str | None]] = {}

    def oids(rev: str) -> dict[str, str | None]:
        if rev not in cache:
            cache[rev] = paths_oids(hub, repo, rev, paths)
        return cache[rev]

    for i, commit in enumerate(commits):
        users = sorted({a["user"] for a in commit.get("authors") or []} & own)
        if not users:
            continue
        at = oids(commit["id"])
        before = oids(commits[i + 1]["id"]) if i + 1 < len(commits) else dict.fromkeys(paths)
        for p in paths:
            oid = at[p]
            if oid is not None and oid != before[p]:
                wrote[p].append((commit["id"], users[0], oid))
        if all(wrote.get(p) for p in paths):
            break
    return wrote, authors


def header_kind(hub: Hub, repo: str, commit: str, name: str, size: int | None) -> str:
    """Section kinds from the container header alone (a ranged read of the
    first bytes, parsed by the lab parser over a zero-filled map of the full
    size; nothing else of the file is fetched)."""
    if not size:
        return "unverified"
    url = f"{HUB}/{repo}/resolve/{commit}/{_q(name)}"
    resp = hub.request(url, byte_range=f"0-{HEADER_PROBE_BYTES - 1}")
    if resp.status not in (200, 206) or len(resp.body) < 32:
        return "unverified"
    head = resp.body
    header_end = int.from_bytes(head[24:32], "little")
    if header_end > len(head):
        resp = hub.request(url, byte_range=f"0-{header_end - 1}")
        if resp.status not in (200, 206) or len(resp.body) < header_end:
            return "unverified"
        head = resp.body
    buf = mmap.mmap(-1, size)
    try:
        buf[: len(head)] = head
        bundle = parse_litertlm(buf)  # type: ignore[arg-type]
    except (LitertlmParseError, ValueError):
        return "unverified"
    finally:
        buf.close()
    return kind_of([s.model_type or "" for s in bundle.sections if s.data_type == 3])


# --- assembly ----------------------------------------------------------------


def build(args: argparse.Namespace, previous_ids: dict[tuple[str | None, str], str] | None = None
          ) -> dict[str, Any]:
    log = lambda msg: print(msg, file=sys.stderr)  # noqa: E731
    token, token_source = hub_token()
    hub = Hub(log=log, token=token)
    log(f"Hub auth: {token_source}")
    manifests, base_to_repo = read_manifests(Path(args.manifests).expanduser())
    cards = read_cards(REPO_ROOT / "cards")
    log(f"manifest variants {len(manifests)}; card artifacts {len(cards)}")

    # The own account, from the seed commit.
    status, seed_commits = hub.paginate(f"/api/models/{_q(OWN_SEED_REPO)}/commits/main")
    seed = next((c for c in seed_commits if c["id"] == OWN_SEED_COMMIT), None)
    if status != 200 or seed is None:
        raise HubError(f"seed commit {OWN_SEED_COMMIT[:8]} not found in {OWN_SEED_REPO}")
    own = {a["user"] for a in seed.get("authors") or []}
    log(f"own accounts: {sorted(own)}")

    repos: dict[str, dict[str, Any]] = {}  # repo -> {"org_listed": bool}
    not_litert_named: list[str] = []
    private: set[str] = set()
    for org in ORGS:
        for item in list_org_repos(hub, org):
            names = [s["rfilename"] for s in item.get("siblings") or []]
            if item.get("private"):
                private.add(item["id"])
                continue
            if any(n.endswith(".litertlm") for n in names):
                repos[item["id"]] = {"org": org}
                if org == OWN_ORG and "LiteRT" not in item["id"]:
                    not_litert_named.append(item["id"])
    for repo, _ in manifests:
        repos.setdefault(repo, {"org": repo.split("/")[0]})
    log(f"repos holding a .litertlm: {len(repos)}")

    hub_files: dict[tuple[str, str], HubFile] = {}
    heads: dict[str, str | None] = {}
    unreadable: list[str] = []
    skipped_not_own: list[tuple[str, str]] = []
    owner_unreadable: list[tuple[str, str]] = []
    for repo in sorted(repos):
        status, head, is_private = repo_head(hub, repo)
        if is_private:  # a manifest repo the token can see but the public cannot
            private.add(repo)
            heads[repo] = None
            continue
        heads[repo] = head
        if head is None:
            unreadable.append(f"{repo} (HTTP {status})")
            continue
        files = repo_litertlm(hub, repo, head)
        org = repo.split("/")[0]
        need_owner = sorted(f for f in files if org != OWN_ORG and (repo, f) not in manifests)
        wrote: dict[str, list[tuple[str, str, str]]] = {}
        authors: dict[str, str] = {}
        if need_owner:
            found = writers_by_own(hub, repo, need_owner, own)
            if isinstance(found, int):
                owner_unreadable.extend((repo, f) for f in need_owner)
                unreadable.append(f"{repo} (commit history: HTTP {found}, "
                                  + ("with a token)" if token else "no token)"))
            else:
                wrote, authors = found
        for name, (sha, size) in sorted(files.items()):
            hf = HubFile(repo, name, sha, size, head)
            if org == OWN_ORG:
                hf.owner_evidence = "hf-own-org"
            elif name in need_owner:
                if (repo, name) in owner_unreadable:
                    continue
                if not wrote.get(name):
                    skipped_not_own.append((repo, name))
                    continue
                commit, user, oid = wrote[name][0]
                hf.owner_evidence = f"hf-commit-author:{user}"
                hf.notes.append(f"written by own commit {commit[:8]} ({authors.get(commit)})")
                if oid != sha:
                    hf.notes.append("the live bytes come from a later commit, not that one")
            hub_files[(repo, name)] = hf
    log(f"own Hub files: {len(hub_files)}; skipped (not own): {len(skipped_not_own)}; "
        f"unreadable repos: {len(unreadable)}")

    rows: dict[tuple[str | None, str], dict[str, Any]] = {}

    def row_for(repo: str | None, name: str) -> dict[str, Any]:
        key = (repo, name)
        if key not in rows:
            rows[key] = {"hf_repo": repo, "file": name, "evidence": set(), "notes": [],
                         "lab_card_candidates": [], "manifest": None, "kind": None,
                         "manifest_sha256": None}
        return rows[key]

    for (repo, name), facts in manifests.items():
        row = row_for(repo, name)
        row["evidence"].add("manifest")
        row["manifest"] = facts["manifest"]
        row["kind"] = facts["kind"]
        row["manifest_sha256"] = facts["sha256"]
    for (repo, name), hf in hub_files.items():
        row = row_for(repo, name)
        if hf.owner_evidence:
            row["evidence"].add(hf.owner_evidence)
        row["notes"].extend(hf.notes)

    # Cards: which row each card artifact names. Strongest rule first.
    by_file: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
    for key in rows:
        by_file[key[1]].append(key)
    not_own_keys = set(skipped_not_own)
    not_own_names = {name for _, name in skipped_not_own}
    cards_not_own: list[str] = []
    for art in cards:
        cands = sorted(by_file.get(art.file, []), key=lambda k: (k[0] or "", k[1]))
        weak = sorted(base_to_repo.get((art.upstream or "").lower(), []))
        chosen, rule = None, ""
        # 1. the card's own source_url names an own repo that holds the file
        if art.source_repo and (art.source_repo, art.file) in rows:
            chosen, rule = (art.source_repo, art.file), "card source_url"
        # 2. the card's upstream is the base_model of exactly one own repo holding the file
        if chosen is None:
            hits = [(r, art.file) for r in weak if (r, art.file) in rows]
            if len(hits) == 1:
                chosen, rule = hits[0], "card upstream = manifest base_model"
        # 3. the card's sha256 is the bytes of exactly one row (manifest or Hub)
        if chosen is None and art.sha256:
            hits = [k for k in cands if art.sha256 in (
                rows[k]["manifest_sha256"], hub_files[k].sha256 if k in hub_files else None)]
            if len(hits) == 1:
                chosen, rule = hits[0], "sha256"
        if chosen is None and not art.source_repo:
            # 4. the file name is unique among own rows; 5. one candidate repo is named
            if len(cands) == 1:
                chosen, rule = cands[0], "file name unique"
            else:
                hits = [k for k in cands if k[0] in art.mentions]
                if len(hits) == 1:
                    chosen, rule = hits[0], "own repo named in the card"
        if chosen is not None:
            rows[chosen]["lab_card_candidates"].append((art.card, rule, art.sha256))
            continue
        # The card names a repo (or one upstream) whose own files do not include it:
        # a team file there, or a file that is not on the Hub (a card-only row).
        home = art.source_repo or (weak[0] if len(weak) == 1 else None)
        if home is None and cands:
            for k in cands:
                rows[k]["notes"].append(
                    f"lab card {art.card} names this file name but its repo is not "
                    f"determined (candidates: {', '.join(c[0] or '?' for c in cands)})")
            continue
        if (home and (home, art.file) in not_own_keys) or (
                home is None and art.file in not_own_names):
            cards_not_own.append(f"{art.card}:{art.file}")
            continue
        if home is None and len(art.mentions) == 1:
            home = art.mentions[0]
        row = row_for(home, art.file)
        row["evidence"].add("lab-card")
        row["lab_card_candidates"].append((art.card, "card only", art.sha256))
        row["card_sha256"] = art.sha256

    # Finish every row.
    out_rows: list[dict[str, Any]] = []
    for (repo, name), row in rows.items():
        hf = hub_files.get((repo, name)) if repo else None
        cands = row.pop("lab_card_candidates")
        lab_card = None
        if len(cands) == 1:
            lab_card = cands[0][0]
            row["notes"].append(f"lab card {lab_card} matched by {cands[0][1]}")
            if cands[0][2] and hf and hf.sha256 and cands[0][2] != hf.sha256:
                row["notes"].append(f"lab card sha256 {cands[0][2][:12]}... differs from the "
                                    "Hub (the card measured an earlier file)")
        elif len(cands) > 1:
            row["notes"].append("lab card not unique (candidates: " + ", ".join(
                f"{c[0]} by {c[1]}" for c in cands) + ")")
        if lab_card:
            row["evidence"].add("lab-card")
        manifest_sha = row.pop("manifest_sha256")
        card_sha = row.pop("card_sha256", None)
        kind = row.pop("kind")
        if hf is not None:
            status = "ok"
            sha, size, commit = hf.sha256, hf.size_bytes, hf.hub_commit
            if manifest_sha and sha and manifest_sha != sha:
                row["sha_differs_from_manifest"] = True
                row["manifest_sha256"] = manifest_sha
            if kind is None:
                kind = header_kind(hub, repo, commit, name, size)  # type: ignore[arg-type]
        else:
            status = "hub_missing"
            sha = size = None
            commit = heads.get(repo) if repo else None
            last = {"sha256": manifest_sha or card_sha, "source": "manifest" if manifest_sha
                    else "lab card"}
            row["last_known"] = last
            row["notes"].append("not on the Hub" + (f" ({repo}@{commit[:8]})" if repo and commit
                                                    else ""))
            if kind is None:
                kind = "unverified"
        row.update({
            "bundle_id": None, "hub_commit": commit, "sha256": sha, "size_bytes": size,
            "kind": kind, "suite": SUITE_OF_KIND.get(kind), "role": roles.role_of(repo),
            "lab_card": lab_card, "status": status,
            "local": local_blob(sha or (row.get("last_known") or {}).get("sha256")),
            "evidence": sorted(row["evidence"]), "notes": "; ".join(row["notes"]),
        })
        out_rows.append(row)

    assign_ids(out_rows, previous_ids or {})
    out_rows.sort(key=lambda r: (r["hf_repo"] is None, r["hf_repo"] or "", r["file"]))
    counts = count(out_rows, len(skipped_not_own))
    return {
        "meta": {
            "generated_at": now_iso(),
            "generator": GENERATOR,
            "hub_auth": token_source,
            "private_repos_skipped": len(private),
            "own_accounts": sorted(own),
            "own_seed": f"{OWN_SEED_REPO}@{OWN_SEED_COMMIT}",
            "counts": counts,
            "counts_note": (
                "variants = lab_card_matched + manifest_only + lab_card_only + hf_only: "
                "lab_card_matched = manifest rows with a lab card, manifest_only = manifest "
                "rows without one, lab_card_only = card rows not in the manifest, hf_only = "
                "rows found only through the Hub API; repos = distinct hf_repo values"),
            "own_org_repos_without_LiteRT_in_name": sorted(not_litert_named),
            "unreadable_repos": sorted(unreadable),
            "files_with_unreadable_owner": len(owner_unreadable),
            "lab_cards_naming_a_team_file": sorted(cards_not_own),
            "hub_api_calls": hub.calls,
        },
        "bundles": out_rows,
    }


def assign_ids(rows: list[dict[str, Any]],
               previous: dict[tuple[str | None, str], str] | None = None) -> None:
    """A row the previous list had keeps its id (results are filed under it).
    A new row gets the lab card id when exactly one row names that card and
    the id is free, else `<repo slug>__<file stem slug>`, made unique; no new
    id reuses an id of the previous list or a lab card id."""
    previous = previous or {}
    cards: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["lab_card"]:
            cards[row["lab_card"]] += 1
    taken = set(previous.values()) | set(cards)
    used: set[str] = set()
    ordered = sorted(rows, key=lambda r: (r["hf_repo"] or "", r["file"]))
    for row in ordered:
        old = previous.get((row["hf_repo"], row["file"]))
        if old and old not in used:
            row["bundle_id"] = old
            used.add(old)
    for row in ordered:
        if row.get("bundle_id"):
            continue
        card = row["lab_card"]
        if card and cards[card] == 1 and card not in used:
            row["bundle_id"] = card
            used.add(card)
            continue
        repo_name = (row["hf_repo"] or "unknown/unknown").split("/")[1]
        base = f"{slug(repo_name)}__{slug(Path(row['file']).stem)}"
        bid = base
        if bid in taken or bid in used:
            bid = f"{slug((row['hf_repo'] or 'unknown').replace('/', '-'))}__" \
                  f"{slug(Path(row['file']).stem)}"
        n = 2
        while bid in taken or bid in used:
            bid = f"{base}-{n}"
            n += 1
        used.add(bid)
        row["bundle_id"] = bid


def count(rows: list[dict[str, Any]], skipped_not_own: int) -> dict[str, int]:
    def has(r: dict[str, Any], ev: str) -> bool:
        return ev in r["evidence"]

    return {
        "repos": len({r["hf_repo"] for r in rows if r["hf_repo"]}),
        "variants": len(rows),
        "v1": sum(r["suite"] == "v1" for r in rows),
        "v2": sum(r["suite"] == "v2" for r in rows),
        "lab_card_matched": sum(has(r, "manifest") and bool(r["lab_card"]) for r in rows),
        "manifest_only": sum(has(r, "manifest") and not r["lab_card"] for r in rows),
        "lab_card_only": sum(not has(r, "manifest") and has(r, "lab-card") for r in rows),
        "hf_only": sum(not has(r, "manifest") and not has(r, "lab-card") for r in rows),
        "hub_missing": sum(r["status"] == "hub_missing" for r in rows),
        "skipped_not_own": skipped_not_own,
    }


def now_iso() -> str:
    out = subprocess.run(["date", "+%Y-%m-%dT%H:%M:%S%z"], capture_output=True, text=True,
                         check=True).stdout.strip()
    return out[:-2] + ":" + out[-2:] if re.search(r"[+-]\d{4}$", out) else out


def dumps(doc: Any) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --- --check -----------------------------------------------------------------


def check(path: Path) -> int:
    doc = json.loads(path.read_text(encoding="utf-8"))
    hub = Hub(token=hub_token()[0])
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc["bundles"]:
        if row["hf_repo"]:
            by_repo[row["hf_repo"]].append(row)
    changes: list[tuple[str, str, str, str]] = []
    new_files: list[str] = []
    for repo in sorted(by_repo):
        status, head, is_private = repo_head(hub, repo)
        if head is None or is_private:
            for row in by_repo[repo]:
                changes.append((row["bundle_id"], "repo", row["hub_commit"] or "-",
                                "private" if is_private else f"HTTP {status}"))
            continue
        files = repo_litertlm(hub, repo, head)
        listed = {r["file"] for r in by_repo[repo]}
        if repo.split("/")[0] == OWN_ORG:  # every file there is own by definition
            new_files += [f"{repo}/{f}" for f in sorted(files) if f not in listed]
        for row in by_repo[repo]:
            # A hub_missing row's hub_commit is the last commit that had the file;
            # the repo moving on is not a change of the list, the bytes reappearing is.
            if head != row["hub_commit"] and row["status"] != "hub_missing":
                changes.append((row["bundle_id"], "hub_commit", _short(row["hub_commit"]),
                                _short(head)))
            sha, size = files.get(row["file"], (None, None))
            if sha != row["sha256"]:
                changes.append((row["bundle_id"], "sha256", _short(row["sha256"]), _short(sha)))
            if size != row["size_bytes"]:
                changes.append((row["bundle_id"], "size_bytes", str(row["size_bytes"]),
                                str(size)))
    print(f"| bundle_id | field | listed | hub ({now_iso()}) |")
    print("|---|---|---|---|")
    for c in changes:
        print("| " + " | ".join(c) + " |")
    if not changes:
        print("| (none) | | | |")
    for f in new_files:
        print(f"new .litertlm in a listed own-org repo (not in the list): {f}")
    print(f"{len(changes)} change(s), {len(new_files)} new own-org file(s), {hub.calls} API "
          "calls; files added to litert-community repos need the ownership check of a full "
          "regeneration (list_bundles.py without --check)")
    return 1 if changes or new_files else 0


def _short(value: str | None) -> str:
    return value[:12] if value else "-"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "regression" / "bundles.json"))
    ap.add_argument("--manifests", default=os.environ.get(
        "LITERTLM_MANIFESTS", "~/code/litertlm-convert/manifest/generated"))
    ap.add_argument("--check", metavar="BUNDLES_JSON")
    args = ap.parse_args(argv)
    if args.check:
        return check(Path(args.check))
    out = Path(args.out)
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    previous_ids = {(r["hf_repo"], r["file"]): r["bundle_id"]
                    for r in (previous or {}).get("bundles", [])}
    doc = build(args, previous_ids)
    if previous:
        keep_vanished(doc, previous)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(doc), encoding="utf-8")
    print(json.dumps(doc["meta"]["counts"], sort_keys=True))
    return 0


def keep_vanished(doc: dict[str, Any], previous: dict[str, Any]) -> None:
    """A row of the previous list that this scan no longer finds stays, as
    hub_missing with its last known bytes: the list never forgets a file.
    Its role is read from the map again. The lab card of a vanished file stays
    with that row: the scan's repo-less card-only row for the same file (the
    card no longer finds the file on the Hub) is dropped, so one file never
    has two rows with one `bundle_id` (2026-10-15, Nemotron-H-4B)."""
    seen = {(r["hf_repo"], r["file"]) for r in doc["bundles"]}
    for old in previous.get("bundles", []):
        if (old["hf_repo"], old["file"]) in seen:
            continue
        doc["bundles"] = [r for r in doc["bundles"] if not (
            r["hf_repo"] is None and r["file"] == old["file"] and (
                (old["lab_card"] and r["lab_card"] == old["lab_card"])
                or r["bundle_id"] == old["bundle_id"]))]
        row = dict(old, role=roles.role_of(old["hf_repo"]))
        if row["status"] != "hub_missing":
            row["last_known"] = {"sha256": old["sha256"], "size_bytes": old["size_bytes"],
                                 "hub_commit": old["hub_commit"], "source": "previous list"}
            row.update({"status": "hub_missing", "sha256": None, "size_bytes": None})
        doc["bundles"].append(row)
    doc["bundles"].sort(key=lambda r: (r["hf_repo"] is None, r["hf_repo"] or "", r["file"]))
    doc["meta"]["counts"] = count(doc["bundles"], doc["meta"]["counts"]["skipped_not_own"])


if __name__ == "__main__":
    sys.exit(main())
