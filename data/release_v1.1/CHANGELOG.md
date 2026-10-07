# Changelog

## v1.1 (4 October 2026)

Corrections from a post-release audit of v1.0. The benchmark results in the accompanying paper refer to **v1.0**.

- **Deduplication.** Posts of the same victim by the same group are now merged within 7 days (v1.0: 3 days). Victim-name keys where one is a prefix of the other (titles truncated by the leak site, at least 8 characters) are now linked. This follows an audit of candidate duplicate pairs: 57 of 97 decided pairs were the same claim. Events: 37,385 → 36,810.
- **Country from ccTLD.**
  - Domains taken from RansomLook titles truncated right after the domain (`acme.ne...`) are discarded.
  - ccTLDs commonly used generically or by free domains (`.ms`, `.nu`, `.tk`, `.ga`, `.cf`, `.gq`, `.pw`, `.vc`, `.st`) are no longer read as countries.
  - US local-government domains with a state code (`ci.<city>.ca`) map to US.
  - Countries imputed from the ccTLD: 109 → 82.
- **New flag `not_a_victim`.** Marks 25 posts that do not name a victim: forums, the group itself or another group, generic announcements, multi-organisation configuration dumps. These events are kept.

| | v1.0 | v1.1 |
|---|---|---|
| events | 37,385 | 36,810 |
| country missing (observed / enriched) | 24.87% / 13.55% | 24.25% / 13.21% |
| sector missing (observed / enriched) | 14.94% / 5.03% | 14.3% / 4.59% |

## v1.0 (3 October 2026)

First release, used for all benchmarks in the paper.
