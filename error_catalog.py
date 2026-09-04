"""
Lodestar — Error Signature Catalog
=========================================

This is the knowledge base for the support chatbot. Each entry maps a recognizable
error signature to its likely causes, SAFE (read-only) diagnostic commands, and a
resolution path.

DESIGN PRINCIPLES (important — do not violate):
- Diagnostic commands must be READ-ONLY (get, describe, logs, cat, echo, file, \\du).
  Never put destructive commands (delete, rm, drop, uninstall) in `diagnostics`.
- Destructive fixes go in `resolution_steps` and MUST be flagged with requires_human=True.
- When multiple root causes are possible, list them ALL. The bot must not guess a
  single cause when the error text alone cannot disambiguate.
- `escalate_when` describes the conditions under which the bot should stop and hand
  off to a human engineer instead of continuing.

Each entry is intentionally derived from REAL incidents so the catalog reflects
what customers actually hit, not theoretical failures.
"""

from dataclasses import dataclass, field
from typing import List
import re


@dataclass
class DiagnosticCommand:
    command: str
    purpose: str
    # read_only must always be True for anything the bot suggests autonomously
    read_only: bool = True


@dataclass
class ResolutionStep:
    instruction: str
    command: str = ""
    # If True, the bot must NOT present this as an autonomous instruction.
    # It must show it only with an explicit human-approval warning.
    requires_human: bool = False


@dataclass
class ErrorEntry:
    id: str
    title: str
    # Regex patterns that identify this error from raw text / OCR output.
    patterns: List[str]
    # Short, plain-English explanation of what the error means.
    meaning: str
    # ALL plausible root causes — never collapse to one when text is ambiguous.
    possible_causes: List[str]
    diagnostics: List[DiagnosticCommand] = field(default_factory=list)
    resolution_steps: List[ResolutionStep] = field(default_factory=list)
    escalate_when: List[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        for p in self.patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False


CATALOG: List[ErrorEntry] = [
    ErrorEntry(
        id="EXEC_FORMAT_ERROR",
        title="Binary architecture mismatch (Exec format error)",
        patterns=[
            r"exec format error",
            r"cannot execute binary file",
        ],
        meaning=(
            "The kernel cannot run one of the installer binaries because it was compiled "
            "for a different CPU architecture than this server. Shell scripts still "
            "run; only the compiled binaries (kubectl, helm, helmfile, mc, helm-diff) fail."
        ),
        possible_causes=[
            "The installation package is x86-64 but the server is aarch64 (ARM), or vice versa.",
            "The package was prepared/downloaded on a machine with a different arch "
            "(e.g. an Apple Silicon Mac) than the target server.",
            "Only some binaries were replaced, leaving others in the wrong architecture.",
        ],
        diagnostics=[
            DiagnosticCommand("uname -m",
                              "Show the server CPU architecture (x86_64 vs aarch64)."),
            DiagnosticCommand("file /home/<user>/<installer>/bin/*",
                              "Show the architecture of each top-level installer binary."),
            DiagnosticCommand(
                "find /home/<user>/<installer> -type f -exec file {} \\; | grep x86-64",
                "Recursively find ALL x86-64 binaries, including helm plugins under bin/."),
        ],
        resolution_steps=[
            ResolutionStep(
                "Confirm whether the vendor ships a build for this server's architecture. "
                "If the server is ARM, obtain the aarch64 build rather than patching "
                "binaries one by one — this is the only sustainable fix."),
            ResolutionStep(
                "If there is no build for this architecture, run the installer from a server whose "
                "architecture matches the package (e.g. an x86-64 bastion)."),
        ],
        escalate_when=[
            "The package contains vendor-specific compiled binaries with no upstream "
            "ARM build available.",
            "It is unclear whether the product officially supports this architecture — this must "
            "be confirmed with the product team.",
        ],
    ),

    ErrorEntry(
        id="INSTALLER_CLI_NOT_FOUND",
        title="installer-cli: command not found",
        patterns=[
            r"installer-cli: command not found",
            r"bash: .*installer-cli.*No such file or directory",
        ],
        meaning=(
            "The shell cannot find or execute the `installer-cli` command. Usually the symlink "
            "in /usr/local/bin points to a wrong or non-existent path, or the shell has "
            "a stale command cache."
        ),
        possible_causes=[
            "The symlink was created with $(pwd) from the wrong working directory, so it "
            "points to a path where installer-cli.sh does not exist (dangling symlink).",
            "/usr/local/bin is not on PATH.",
            "The shell's command hash cache is stale after creating the symlink.",
            "installer-cli.sh is missing the execute bit.",
        ],
        diagnostics=[
            DiagnosticCommand("ls -l /usr/local/bin/installer-cli",
                              "Show where the symlink points and whether it is dangling."),
            DiagnosticCommand("find / -name installer-cli.sh 2>/dev/null",
                              "Find the real location of installer-cli.sh."),
            DiagnosticCommand("echo $PATH | tr ':' '\\n' | grep -n local",
                              "Confirm /usr/local/bin is on PATH."),
        ],
        resolution_steps=[
            ResolutionStep(
                "Recreate the symlink with the absolute path to the real installer-cli.sh, "
                "then refresh the shell cache.",
                command="sudo ln -sf /home/<user>/<installer>/installer-cli.sh /usr/local/bin/installer-cli && "
                        "chmod +x /home/<user>/<installer>/installer-cli.sh && hash -r"),
        ],
        escalate_when=[
            "installer-cli.sh cannot be found anywhere on the filesystem (package extraction failed).",
        ],
    ),

    ErrorEntry(
        id="MISSING_CRYPTO_KEY_LENGTH_64",
        title="Missing/invalid encryption master key (length 64 required)",
        patterns=[
            r"must be set to an alphanumeric.*length 64",
            r"sessionEncryptionMasterKey:\s*''",
            r"dbEncryptionMasterKey:\s*''",
            r"Missing or incorrect configuration found",
        ],
        meaning=(
            "A chart requires a 64-character alphanumeric encryption master key, but the "
            "value rendered empty. The key is read from a file under the environment's "
            ".crypto/ directory via the values template's readFile."
        ),
        possible_causes=[
            "The .crypto key file is missing or empty.",
            "The key file exists but the value contains a trailing newline or a "
            "non-alphanumeric character, so it fails the length/format check.",
            "The LODESTAR_ENV_DIR environment variable is not set, so the template's readFile "
            "resolves to a wrong path and returns empty. (Common when running helm/helmfile "
            "manually instead of via installer-cli.)",
            "A manual edit to values.yaml broke the override chain so the readFile value "
            "no longer reaches the chart's common.* values.",
        ],
        diagnostics=[
            DiagnosticCommand('echo "LODESTAR_ENV_DIR=[$LODESTAR_ENV_DIR]"',
                              "Check whether LODESTAR_ENV_DIR is set (empty is a common cause)."),
            DiagnosticCommand(
                "ls -la /home/<user>/.<installer>/environments/<env>/.crypto/",
                "List the crypto key files and their sizes."),
            DiagnosticCommand(
                "tr -d '\\n' < /home/<user>/.<installer>/environments/<env>/.crypto/"
                "session-encryption-master.key | wc -c",
                "Count key characters WITHOUT the newline — must be exactly 64."),
            DiagnosticCommand(
                "grep -qE '^[A-Za-z0-9]{64}$' <keyfile> && echo OK || echo FORMAT_PROBLEM",
                "Verify the key is exactly 64 alphanumeric characters."),
        ],
        resolution_steps=[
            ResolutionStep(
                "If running helm/helmfile manually, prefer running via installer-cli, which sets "
                "LODESTAR_ENV_DIR and the environment for you.",
                command="installer-cli install <product> --env <env>"),
            ResolutionStep(
                "If LODESTAR_ENV_DIR must be set manually for a manual run:",
                command="export LODESTAR_ENV_DIR=/home/<user>/.<installer>/environments"),
            ResolutionStep(
                "If the key file is genuinely missing/malformed, regenerate a 64-char "
                "key WITHOUT a trailing newline. NOTE: this key encrypts data — store it "
                "safely; regenerating it on an existing environment can make previously "
                "encrypted data unreadable.",
                command="openssl rand -hex 32 | tr -d '\\n' > <keyfile>",
                requires_human=True),
        ],
        escalate_when=[
            "The environment already holds encrypted data and regenerating the key could "
            "cause data loss — a human must decide.",
            "The key file looks correct (exactly 64 alphanumeric chars) but the error "
            "persists, indicating an LODESTAR_ENV_DIR / override-chain issue that needs review.",
        ],
    ),

    ErrorEntry(
        id="DB_NO_SUCH_USER",
        title="Liquibase/PostgreSQL: FATAL: no such user",
        patterns=[
            r"FATAL:\s*no such user",
            r"Connection could not be created to jdbc:postgresql",
        ],
        meaning=(
            "Liquibase (DB migration) cannot connect to PostgreSQL because the database "
            "user it is authenticating as is not recognized — often through a PgBouncer "
            "pooler service (…-pooler-session)."
        ),
        possible_causes=[
            "The application's DB user was never created on the PostgreSQL cluster.",
            "The pooler (PgBouncer) userlist/auth config does not include this user, "
            "even though it exists on the backend PostgreSQL.",
            "The credential in the K8s secret does not match a user defined in the DB.",
            "A manual edit to values.yaml changed/removed the DB username, so the "
            "connecting user no longer matches the created user.",
        ],
        diagnostics=[
            DiagnosticCommand("kubectl get pods -n postgres",
                              "Check PostgreSQL cluster and pooler pod status."),
            DiagnosticCommand("kubectl get svc -n postgres",
                              "Confirm the pooler service name/endpoint."),
            DiagnosticCommand(
                "kubectl exec -it <postgres-pod> -n postgres -- psql -U postgres -c '\\du'",
                "List DB roles to check whether the expected user exists."),
            DiagnosticCommand(
                "cd /home/<user>/<installer> && git diff | grep -iE 'user|postgres|password'",
                "Check whether a manual values edit changed DB credentials."),
        ],
        resolution_steps=[
            ResolutionStep(
                "Identify which PostgreSQL operator is in use (Zalando / CloudNativePG), "
                "then verify the expected user is declared and created.",
                command="kubectl get postgresql -n postgres 2>/dev/null; "
                        "kubectl get cluster -n postgres 2>/dev/null"),
            ResolutionStep(
                "If a manual values edit removed/altered the DB user, revert it.",
                command="git checkout -- <values-file>",
                requires_human=True),
        ],
        escalate_when=[
            "The user exists on the backend but the pooler still rejects it — pooler auth "
            "configuration needs engineer review.",
            "It is unclear whether the user should be created manually or by the operator.",
        ],
    ),

    ErrorEntry(
        id="CHART_SCHEMA_ADDITIONAL_PROPS",
        title="Helm chart schema rejection (additional properties not allowed)",
        patterns=[
            r"additional propert(y|ies).*not allowed",
            r"values don't meet the specifications of the schema",
        ],
        meaning=(
            "The values passed to a chart include properties that the chart's "
            "values.schema.json does not allow, so rendering fails validation."
        ),
        possible_causes=[
            "Helm/chart version mismatch: the installed helm binary is a different "
            "version than the one the installer was tested with, and the chart's schema changed "
            "between versions.",
            "A manual values edit introduced properties the chart does not accept.",
        ],
        diagnostics=[
            DiagnosticCommand("/home/<user>/<installer>/bin/helm version",
                              "Show the current helm version."),
            DiagnosticCommand(
                "grep -rniE 'helm.*version|helmVersion' /home/<user>/<installer>/scripts/ 2>/dev/null",
                "Look for the helm version the installer expects."),
        ],
        resolution_steps=[
            ResolutionStep(
                "Pin helm to the version the installer was built/tested with rather than the latest "
                "upstream release. Obtaining the correct installer build for this architecture "
                "provides the matching helm version automatically."),
        ],
        escalate_when=[
            "The correct helm version for this release is unknown and must be "
            "confirmed with the product team.",
        ],
    ),
]


def find_matches(text: str) -> List[ErrorEntry]:
    """Return all catalog entries whose signature matches the given text."""
    return [e for e in CATALOG if e.matches(text)]
