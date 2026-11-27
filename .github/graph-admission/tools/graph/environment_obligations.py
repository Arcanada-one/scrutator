"""Supplemental bounded source observations, never a config/admission exemption.

No environment values, inspected consumer commands or shell scripts are executed.
The scalar config_schema verdict and declared-source requirement remain binding.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def observe(source: str, path: str, compiler: str | None) -> dict:
    """Use the genuine selected compiler's API; absence/failure stays unknown."""
    unavailable = {'status': 'not_measured', 'reason': 'typescript_ast_provider_unavailable'}
    if not compiler:
        return unavailable
    binary = Path(compiler)
    candidates = [binary.parent.parent / 'typescript/lib/typescript.js',
                  binary.resolve().parent.parent / 'lib/typescript.js']
    module = next((p for p in candidates if p.is_file()), None)
    node = binary.parent / 'node'
    executable = str(node) if node.is_file() else shutil.which('node')
    if module is None or executable is None:
        return unavailable
    observer = Path(__file__).with_suffix('.cjs')
    try:
        run = subprocess.run([executable, str(observer), str(module.resolve())],
                             input=json.dumps({'source': source, 'path': path}),
                             text=True, capture_output=True, timeout=15)
        if run.returncode or len(run.stdout) > 1024 * 1024:
            return {'status': 'not_measured', 'reason': 'typescript_ast_observation_failed'}
        facts = json.loads(run.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {'status': 'not_measured', 'reason': 'typescript_ast_observation_failed'}
    facts['status'] = 'not_measured' if facts['parse_error_count'] else 'measured'
    facts['provider'] = {
        'typescript_sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
        'observer_sha256': hashlib.sha256(observer.read_bytes()).hexdigest(),
        'node_sha256': hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
    }
    return facts


def evidence(tree, path: str, compiler: str | None, *, base_owner: str | None,
             head_owner: str | None, native_test: bool, inferred_boundary: bool) -> dict:
    """Separate supported parent facts from unresolved transport/child duties."""
    source = tree.text(path)
    facts = observe(source, path, compiler)
    reasons = []
    if base_owner not in {'library', 'shared_package'} or head_owner != base_owner:
        reasons.append('ownership_not_stable_shared_source')
    if not native_test:
        reasons.append('not_native_test_source')
    if inferred_boundary:
        reasons.append('inferred_external_boundary')
    if facts['status'] != 'measured':
        reasons.append(facts.get('reason', 'source_syntax_not_measured'))
    readers = facts.get('named_parent_reads', [])
    if not readers:
        reasons.append('nonempty_named_reader_obligation_unmeasured')
    if any(r['key'] != 'PATH' for r in readers):
        reasons.append('custom_or_mixed_named_reader')
    if facts.get('parent_unknowns'):
        reasons.append('unresolved_parent_access')
    children = []
    for reference in facts.get('shell_references', []):
        child = reference['path']
        # A filename observation may bind only a unique tracked suffix; this
        # still does not resolve call flow or certify shell completeness.
        matches = [p for p in tree.paths if p == child or p.endswith('/' + child)]
        if len(matches) == 1:
            child = matches[0]
        if child not in tree.paths:
            children.append({'path': child, 'status': 'not_measured', 'reason': 'child_source_unavailable'})
            continue
        raw = tree.text(child)
        # Names observed in source, not a Bash parser/completeness/flow claim.
        names = sorted(set(re.findall(r'\$(?:\{)?([A-Z][A-Z0-9_]*)', raw)))
        children.append({'path': child, 'source_sha256': hashlib.sha256(tree.files[child]).hexdigest(),
                         'observed_variable_names': names, 'status': 'not_measured',
                         'reason': 'shell_reader_completeness_unsupported'})
    return {
        'source_sha256': hashlib.sha256(tree.files[path]).hexdigest(),
        'scope': 'named_parent_reader_source_only',
        'parent_reader': {'status': 'verified' if not reasons else 'not_measured',
                          'reasons': reasons, 'observations': readers,
                          'unknowns': facts.get('parent_unknowns', [])},
        'ambient_inheritance': {'status': 'not_measured',
                                'observations': facts.get('ambient_inheritance', []),
                                'reason': 'transport_observation_does_not_measure_receiver_configuration'},
        'outgoing_overrides': {'status': 'not_measured',
                               'observations': facts.get('outgoing_overrides', []),
                               'reason': 'override_values_and_receiver_contract_not_certified'},
        'child_configuration': {'status': 'not_measured',
                                'calls': facts.get('child_calls', []), 'sources': children,
                                'reason': 'child_flow_and_runtime_not_certified'},
        'provider': facts.get('provider'),
        'aggregate_effect': 'none; existing config_schema verdict remains binding',
    }


def for_verifier(runner, entities: list[str], find_bin, deployable_of, is_test_path) -> dict:
    result = {}
    for eid in entities:
        entity = runner.entities[eid]
        node = entity['node']
        path = node.get('path')
        if node.get('type') != 'code_unit' or not path or not re.search(r'\.[cm]?[jt]sx?$', path):
            continue
        if path not in runner.tree_head.paths:
            result[eid] = {'status': 'not_measured', 'reason': 'source_absent'}
            continue
        # An incomplete caller cannot certify owner or parser availability.
        if not all(hasattr(runner, x) for x in ('idx', 'head_nodes', 'deployables', 'a', 'exec_root', 'top')):
            result[eid] = {'status': 'not_measured', 'reason': 'source_context_unavailable'}
            continue
        base_deps = {n['path'].rstrip('/'): n for n in runner.idx.nodes.values()
                     if n.get('type') == 'deployable_unit' and n.get('path')}
        head_deps = {n['path'].rstrip('/'): n for n in runner.head_nodes.values()
                     if n.get('type') == 'deployable_unit' and n.get('path')}
        base_dep = deployable_of(path, base_deps)
        head_dep = deployable_of(path, head_deps)
        def kind(deps, dep):
            return deps.get(dep, {}).get('kind')
        if not is_test_path(path) or kind(base_deps, base_dep) not in {'library', 'shared_package'} or \
                kind(head_deps, head_dep) != kind(base_deps, base_dep) or entity.get('inferred_boundary'):
            result[eid] = {'status': 'not_measured', 'reason': 'outside_bounded_parent_reader_scope'}
            continue
        compiler = find_bin('tsc', getattr(runner.a, 'tsc', None), runner.exec_root,
                            runner.top, [])
        result[eid] = evidence(runner.tree_head, path, compiler,
                               base_owner=kind(base_deps, base_dep), head_owner=kind(head_deps, head_dep),
                               native_test=is_test_path(path), inferred_boundary=entity.get('inferred_boundary', False))
    return result
