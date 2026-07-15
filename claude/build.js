#!/Users/grady/.nvm/versions/node/v24.18.0/bin/node
// Build script: injects prompts from markdown files into the HTML template.
// Run from the repo root: node claude/build.js
// Output: furniture-playground.html at the repo root.

const fs = require('fs');
const path = require('path');

const dir = __dirname;

const qualityPrompt = fs.readFileSync(path.join(dir, 'prompts/quality-verifier.md'), 'utf8').trim();
const trainingPrompt = fs.readFileSync(path.join(dir, 'prompts/training-assistant.md'), 'utf8').trim();

function escapeForTemplateLiteral(s) {
  return s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${');
}

let html = fs.readFileSync(path.join(dir, 'furniture-agent-playground.html'), 'utf8');
html = html.replace('__QUALITY_PROMPT__', escapeForTemplateLiteral(qualityPrompt));
html = html.replace('__TRAINING_PROMPT__', escapeForTemplateLiteral(trainingPrompt));

const out = path.join(dir, '../furniture-playground.html');
fs.writeFileSync(out, html);
console.log(`Built furniture-playground.html`);
