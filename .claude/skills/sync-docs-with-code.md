# Sync Design Docs with Code

Synchronize design documentation with actual code implementation, ensuring docs accurately reflect the codebase while preserving documented future features.

## When to Use

Use this skill when:
- Code has evolved and design docs are out of date
- Conducting a documentation review
- Preparing for a release or code review
- New team members report confusion between docs and code

## Principles

**Source of Truth Hierarchy:**
1. **Code is primary source of truth** - Document what's actually implemented
2. **Preserve future features** - Keep documented features not yet in code (mark as "future" or "planned")
3. **Resolve conflicts** - When docs and code conflict, update docs to match code
4. **Add missing docs** - Document implemented features not yet in docs

## Process

### 1. Discovery Phase

**Identify design documents:**
```bash
find design/ -name "*.md" -type f
```

**Identify code files:**
```bash
find src/ -name "*.py" -type f | grep -v __pycache__
```

### 2. Analysis Phase

For each design document, identify what needs review:

**Read design docs systematically:**
- Core abstractions (base classes, interfaces)
- Method signatures (names, parameters, return types)
- Class attributes and properties
- Examples and usage patterns
- API mappings (if RPC/REST endpoints exist)

**Read corresponding code files:**
- Import statements and dependencies
- Class definitions and inheritance
- Method implementations
- Type hints and return types
- Docstrings

**Create comparison checklist:**
- [ ] Class/interface names match
- [ ] Method names match
- [ ] Method signatures match (params, return types)
- [ ] Properties vs methods consistency
- [ ] Field/attribute names and types match
- [ ] Examples use correct API
- [ ] Documented features exist in code or marked as future

### 3. Update Phase

**For each inconsistency found:**

**A. Method name changes:**
```markdown
❌ OLD: `benchmark.start()`
✅ NEW: `benchmark.setup()`
```

**B. Signature changes:**
```markdown
❌ OLD: `make(runtime_info: Dict[str, Any])`
✅ NEW: `make(runtime_context: RuntimeContext, container_backend: ContainerBackend | None)`
```

**C. Property vs method changes:**
```markdown
❌ OLD: `@property runtime_info -> Dict`
✅ NEW: `def get_runtime_info() -> RuntimeContext`
```

**D. Missing fields (add to docs):**
```markdown
# Add if in code but not docs
metadata: BenchmarkMetadata  # NEW: Benchmark metadata
```

**E. Not yet implemented (mark as future):**
```markdown
### TaskLogic (Future Design)

> **Status:** Not currently implemented.
> **Current Approach:** [describe current approach]
> **Future Consideration:** [why it might be added later]
```

**F. Implementation status annotations:**
```markdown
**Status:** ✓ Fully implemented
**Status:** ✓ Abstract class defined, implementations pending
**Status:** ⚠️ Stub only - full implementation pending
```

### 4. Update Examples

**Update all code examples to match current API:**

1. Update method calls
2. Update parameter names
3. Update return value handling
4. Update class instantiation
5. Update import statements (if needed)

**Before:**
```python
benchmark.start()
tasks = benchmark.get_task_list()
runtime_info = benchmark.runtime_info
```

**After:**
```python
benchmark.setup()
tasks = benchmark.load_tasks()
runtime_context = benchmark.get_runtime_info()
```

### 5. Update Diagrams

**Class diagrams:**
- Add new classes/methods
- Remove obsolete elements
- Update method signatures
- Update relationships

**Sequence diagrams:**
- Update method call sequences
- Update parameter passing

### 6. Cross-Reference Updates

**Update links between docs:**
- If method names changed, update references
- If files renamed, update cross-references
- Ensure consistency across related docs

## Common Inconsistency Patterns

### Pattern 1: Method Name Evolution
- `start()` → `setup()`
- `stop()` → `close()`
- `reset()` → `setup()` (in Task)
- `get_task_list()` → `load_tasks()`

### Pattern 2: Type Evolution
- `Dict[str, Any]` → Typed dataclass (e.g., `RuntimeContext`)
- `str` → `Literal["option1", "option2"]`
- Optional parameters added

### Pattern 3: Return Type Changes
- Tuple → Dataclass
- `(obs, reward, done, truncated, info)` → `EnvironmentOutput`

### Pattern 4: Architectural Changes
- Combined class split into multiple
- Separate classes merged
- Abstraction added/removed (e.g., TaskLogic deferred)

### Pattern 5: Field Changes
- Optional → Required or vice versa
- Field added/removed
- Field renamed

## Documentation Standards

### Status Markers
```markdown
**Status:** ✓ Implemented
**Status:** ✓ Defined, implementations pending
**Status:** ⚠️ Stub only
**Status:** 📋 Planned/Future
```

### Future Features
```markdown
> **Status:** Not currently implemented.
> **Current Approach:** [describe what exists now]
> **Future Consideration:** [why/when it might be added]
```

### Implementation Notes
```markdown
**Note:** This is the current implementation.
**Future:** Additional methods planned (see below).
**Planned methods (to be implemented):**
```

## Validation

After updates, verify:

1. **Consistency:** All examples work with current API
2. **Completeness:** All implemented features documented
3. **Clarity:** Future vs current clearly marked
4. **Accuracy:** Type signatures match code
5. **Links:** Cross-references still valid

## Output Format

Provide a summary:

```markdown
## Documentation Sync Summary

### Updated Files
- design/main_specs.md
- design/docker_wrapper.md
- design/vm_wrapper.md

### Key Changes
1. Benchmark API: start()→setup(), stop()→close()
2. Task API: reset()→setup(), step() returns EnvironmentOutput
3. Added: BenchmarkMetadata, TaskMetadata, RuntimeContext
4. Marked as future: TaskLogic class

### Statistics
- Classes updated: 3
- Methods renamed: 8
- New types added: 3
- Examples updated: 5
- Diagrams updated: 2
```

## Tips

- **Start with core abstractions** - Base classes first, then concrete implementations
- **Check both ways** - Code→docs and docs→code
- **Preserve intent** - Keep "why" even if "how" changed
- **Use tools** - Compare side-by-side in IDE
- **Ask when unclear** - If you can't tell if something is future or obsolete, ask
- **Update incrementally** - One document at a time
- **Test examples** - Ensure code examples would actually run

## Related

- **Code Review Skill** - Review code for inconsistencies
- **Documentation Linter** - Check doc formatting
- **API Changelog Generator** - Generate changelog from doc diffs
