/* Parse inspected bytes; never execute consumer code or emit environment values. */
'use strict';
const fs = require('fs');
const modulePath = process.argv[2];
if (!modulePath) throw new Error('TypeScript provider path required');
/** @type {typeof import('typescript')} */
const ts = require(modulePath);
/** @type {{path: string, source: string}} */
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = ts.createSourceFile(input.path, input.source, ts.ScriptTarget.Latest,
  true, input.path.endsWith('.tsx') ? ts.ScriptKind.TSX : input.path.endsWith('.jsx') ? ts.ScriptKind.JSX :
    /\.[cm]?js$/.test(input.path) ? ts.ScriptKind.JS : ts.ScriptKind.TS);
// Public compiler API reports syntax errors without resolving imports or
// loading/executing any consumer file other than the submitted source bytes.
const options = {allowJs: true, noLib: true, noResolve: true};
const host = ts.createCompilerHost(options);
host.getSourceFile = (/** @type {string} */ name) => name === input.path ? source : undefined;
const program = ts.createProgram([input.path], options, host);
/** @type {{
 * named_parent_reads: Array<{key: string, line: number, column: number}>,
 * parent_unknowns: Array<{reason: string, line: number, column: number}>,
 * ambient_inheritance: Array<{kind: string, line: number, column: number}>,
 * outgoing_overrides: Array<{keys: string[], unknowns: Array<{reason: string, line: number, column: number}>, line: number, column: number}>,
 * child_calls: Array<{operation: string, target: string, callable_identity: string, line: number, column: number}>,
 * shell_references: Array<{path: string, line: number, column: number}>,
 * parse_error_count: number
 * }} */
const result = {named_parent_reads: [], parent_unknowns: [], ambient_inheritance: [],
  outgoing_overrides: [], child_calls: [], shell_references: [],
  parse_error_count: program.getSyntacticDiagnostics(source).length};
/** @param {import('typescript').Node} n */
const location = n => {
  const p = source.getLineAndCharacterOfPosition(n.getStart(source));
  return {line: p.line + 1, column: p.character + 1};
};
/** @param {import('typescript').Node | undefined} n @param {string} text
 * @returns {n is import('typescript').Identifier} */
const identifier = (n, text) => !!n && ts.isIdentifier(n) && n.text === text;
/** @param {import('typescript').Node | undefined} n
 * @returns {n is import('typescript').StringLiteral | import('typescript').NoSubstitutionTemplateLiteral} */
const literal = n => !!n && (ts.isStringLiteral(n) || ts.isNoSubstitutionTemplateLiteral(n));
/** @param {import('typescript').Node} n */
const env = n => ts.isPropertyAccessExpression(n) && identifier(n.expression, 'process') && n.name.text === 'env';
/** @param {import('typescript').Node} n */
const computedEnv = n => ts.isElementAccessExpression(n) && identifier(n.expression, 'process') && literal(n.argumentExpression) && n.argumentExpression.text === 'env';
/** @param {import('typescript').PropertyName | undefined} n */
const propertyName = n => n && (ts.isIdentifier(n) || literal(n)) ? n.text : null;
/** @param {string} reason @param {import('typescript').Node} n */
const unknown = (reason, n) => result.parent_unknowns.push({reason, ...location(n)});
/** @param {import('typescript').Node} n */
const mutation = n => (ts.isBinaryExpression(n.parent) && n.parent.left === n &&
  n.parent.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
  n.parent.operatorToken.kind <= ts.SyntaxKind.LastAssignment) ||
  ts.isDeleteExpression(n.parent) || ts.isPostfixUnaryExpression(n.parent) || ts.isPrefixUnaryExpression(n.parent);

/** @param {import('typescript').Node} n */
function visit(n) {
  if (env(n)) {
    const p = n.parent;
    if (ts.isPropertyAccessExpression(p) && p.expression === n) {
      if (mutation(p))
        unknown('environment_mutation', p);
      else result.named_parent_reads.push({key: p.name.text, ...location(p)});
    } else if (ts.isElementAccessExpression(p) && p.expression === n) {
      if (mutation(p)) unknown('environment_mutation', p);
      else if (literal(p.argumentExpression)) result.named_parent_reads.push({key: p.argumentExpression.text, ...location(p)});
      else unknown('computed_parent_key', p);
    } else if (ts.isSpreadAssignment(p)) {
      result.ambient_inheritance.push({kind: 'whole_parent_spread', ...location(p)});
    } else unknown('aliased_or_unresolved_whole_environment', n);
  }
  if (computedEnv(n)) unknown('computed_environment_object', n);
  if (identifier(n, 'process') && !(ts.isPropertyAccessExpression(n.parent) && n.parent.expression === n) &&
      !(ts.isElementAccessExpression(n.parent) && n.parent.expression === n))
    unknown('unresolved_process_reference', n);
  if (ts.isImportDeclaration(n) && literal(n.moduleSpecifier) &&
      ['process', 'node:process'].includes(n.moduleSpecifier.text)) unknown('imported_environment_alias', n);
  if (ts.isIdentifier(n) && /configservice/i.test(n.text)) unknown('custom_configuration_reference', n);
  // Conservatively hold all parent attribution when process can be rebound.
  if ((ts.isVariableDeclaration(n) || ts.isParameter(n) || ts.isBindingElement(n) ||
       ts.isImportClause(n) || ts.isImportSpecifier(n) || ts.isFunctionDeclaration(n) ||
       ts.isClassDeclaration(n)) && n.name && identifier(n.name, 'process'))
    unknown('shadowed_process_binding', n);
  if (ts.isBinaryExpression(n) && identifier(n.left, 'process')) unknown('process_binding_assignment', n);
  if (ts.isCallExpression(n)) {
    const name = ts.isIdentifier(n.expression) ? n.expression.text :
      ts.isPropertyAccessExpression(n.expression) ? n.expression.name.text : '';
    if (['spawn', 'spawnSync', 'exec', 'execSync', 'execFile', 'execFileSync', 'fork'].includes(name)) {
      // Name alone is not resolved callable identity or an executable-trust claim.
      result.child_calls.push({operation: name, target: literal(n.arguments[0]) ? 'literal' : 'dynamic',
        callable_identity: 'not_measured', ...location(n)});
    }
    if (ts.isPropertyAccessExpression(n.expression) &&
        /configservice/i.test(n.expression.expression.getText(source)))
      unknown('custom_configuration_accessor', n);
  }
  if (ts.isPropertyAssignment(n) && propertyName(n.name) === 'env') {
    /** @type {{keys: string[], unknowns: Array<{reason: string, line: number, column: number}>, line: number, column: number}} */
    const override = {keys: [], unknowns: [], ...location(n)};
    if (ts.isObjectLiteralExpression(n.initializer)) {
      for (const p of n.initializer.properties) {
        if (ts.isSpreadAssignment(p)) {
          if (!env(p.expression)) override.unknowns.push({reason: 'dynamic_override_spread', ...location(p)});
        } else {
          const key = propertyName(p.name);
          if (key === null) override.unknowns.push({reason: 'computed_override_key', ...location(p)});
          else override.keys.push(key);
        }
      }
    } else override.unknowns.push({reason: 'dynamic_environment_expression', ...location(n)});
    result.outgoing_overrides.push(override);
  }
  if (literal(n) && /(?:^|\/)\w[^\n]*\.sh$/.test(n.text) && !n.text.includes('..'))
    result.shell_references.push({path: n.text, ...location(n)});
  ts.forEachChild(n, visit);
}
visit(source);
process.stdout.write(JSON.stringify(result));
