# Documentation Sync Checklist Template

Use this checklist when syncing docs with code. Copy and fill out for each sync session.

## Session Info
- **Date:** YYYY-MM-DD
- **Branch:**
- **Scope:** (e.g., "All design docs", "Container API only")
- **Trigger:** (e.g., "Major refactor", "Release prep", "Code review")

---

## Files to Review

### Design Documents
- [ ] `design/main_specs.md`
- [ ] `design/docker_wrapper.md`
- [ ] `design/vm_wrapper.md`
- [ ] `design/user_experience.md`
- [ ] Other: _______________

### Code Files
- [ ] `src/cube/core.py`
- [ ] `src/cube/benchmark.py`
- [ ] `src/cube/task.py`
- [ ] `src/cube/tool.py`
- [ ] `src/cube/containers.py`
- [ ] `src/cube/server.py`
- [ ] Other: _______________

---

## Comparison Checklist (Per Document)

### Document: _________________

#### Class/Interface Definitions
- [ ] Class names match
- [ ] Base classes/inheritance match
- [ ] Abstract vs concrete status matches

#### Method Signatures
- [ ] Method names match
- [ ] Parameter names match
- [ ] Parameter types match
- [ ] Return types match
- [ ] Optional vs required parameters match

#### Properties vs Methods
- [ ] Properties in code documented as properties
- [ ] Methods in code documented as methods
- [ ] Property return types match

#### Fields/Attributes
- [ ] All code fields documented
- [ ] Field types match
- [ ] Optional/required status matches
- [ ] Default values match (if specified)

#### Examples
- [ ] Import statements correct
- [ ] Method calls use correct names
- [ ] Parameters passed correctly
- [ ] Return values handled correctly
- [ ] Example code would actually run

#### API Endpoints (if applicable)
- [ ] HTTP methods match (GET/POST)
- [ ] Endpoint paths match
- [ ] Request/response types match
- [ ] Python method mappings correct

#### Diagrams
- [ ] Class diagram reflects actual classes
- [ ] Method signatures in diagram match
- [ ] Relationships accurate
- [ ] Inheritance correct

---

## Issues Found

### Critical (Breaks understanding)
1.
2.
3.

### Major (Significantly misleading)
1.
2.
3.

### Minor (Small discrepancies)
1.
2.
3.

---

## Changes Made

### Renamed Methods
| Doc Location | Old Name | New Name | Status |
|--------------|----------|----------|--------|
| | | | ✅ |
| | | | ✅ |

### Signature Changes
| Doc Location | Old Signature | New Signature | Status |
|--------------|---------------|---------------|--------|
| | | | ✅ |
| | | | ✅ |

### Added to Docs (in code but missing)
| Item | Location | Status |
|------|----------|--------|
| | | ✅ |
| | | ✅ |

### Marked as Future (in docs but not code)
| Item | Location | Status |
|------|----------|--------|
| | | ✅ |
| | | ✅ |

### Examples Updated
| Location | Change Description | Status |
|----------|-------------------|--------|
| | | ✅ |
| | | ✅ |

---

## Statistics

- **Documents updated:** __/__
- **Code files reviewed:** __/__
- **Classes updated:** __
- **Methods renamed:** __
- **New types added:** __
- **Items marked as future:** __
- **Examples updated:** __
- **Diagrams updated:** __

---

## Validation

- [ ] All examples use correct current API
- [ ] Future features clearly marked
- [ ] Status indicators added where appropriate
- [ ] Cross-references still valid
- [ ] Diagrams consistent with text
- [ ] No broken internal links
- [ ] Terminology consistent throughout

---

## Follow-up Actions

- [ ] Create GitHub issue for incomplete implementations
- [ ] Update CHANGELOG.md if API changed
- [ ] Notify team of documentation updates
- [ ] Schedule next sync review
- [ ] Other: _______________

---

## Notes

(Any additional observations, patterns noticed, or recommendations for future syncs)
