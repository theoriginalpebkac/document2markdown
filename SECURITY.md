# Security Policy

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report them privately through GitHub's
[private vulnerability reporting](https://github.com/theoriginalpebkac/document2markdown/security/advisories/new):
**Security → Advisories → Report a vulnerability**.

We'll acknowledge your report as soon as we can and keep you updated on the fix.

## Scope

`doc2md.py` shells out to external tools (`pandoc`, `pdftotext`) and processes
arbitrary user-supplied documents. When reporting, please note the input file
type and the command used to reproduce.
