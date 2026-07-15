# Furniture Tools Project — Claude Code Context

This file provides background for Claude Code sessions working on the East Africa furniture agent playground. Read this before making any changes to files in this folder.

---

## Project overview

We are a development economics research team (University of Chicago) building two AI tools for East African furniture markets, initially targeting Uganda and eventually expanding to Kenya. Both tools are built on the Anthropic Claude API.

### Tool 1: Quality verification tool
- **Users:** Furniture buyers (customers), primarily in East Africa
- **Purpose:** Help customers identify whether furniture is high or low quality; explain specific defects and their practical consequences (durability, safety, comfort)
- **Delivery:** WhatsApp and/or a dedicated app
- **Literacy context:** Users have varying literacy levels; language should be simple, direct, and non-technical
- **Key behavior:** When images are provided, describe what is observed and what it means for quality; flag serious defects clearly

### Tool 2: Training assistant
- **Users:** Artisanal furniture manufacturers (small workshop owners and craftspeople)
- **Purpose:** Improve production quality and business profitability — better production methods, tool upgrade decisions, pricing, materials sourcing, customer feedback
- **Delivery:** WhatsApp (primary), eventually an app
- **Literacy context:** Users are skilled tradespeople, not necessarily highly literate; treat them as professionals learning new techniques
- **Key behavior:** Step-by-step practical guidance, economically realistic recommendations, encouraging tone

---

## Local context (East Africa)

- Common furniture types: wooden chairs, tables, sofas, beds, wardrobes
- Common materials: local hardwoods, softwoods, plywood, MDF, fabric, foam
- Key quality concerns: joinery quality, finishing, durability across humid/dry seasonal conditions, wood moisture management
- Maker constraints: small workshops, often hand tools or basic power tools, tight budgets — prioritize high-impact, low-cost improvements
- WhatsApp note: the WhatsApp Business API compresses images, which can degrade fine detail in defect photos — lean on text descriptions of defects for the training tool rather than relying on image quality

---

## Playground tool (furniture-playground.html)

The file `furniture-playground.html` in this folder is a self-contained browser-based testing tool for iterating on system prompts for both tools. It is deployed as a page on the personal GitHub Pages website.

### What it does
- Two-tool switcher (quality verifier / training assistant), each with its own independent conversation history and system prompt
- Editable system prompt per tool (with starter prompts pre-loaded)
- Config panel: model selector, persona/user notes, target region — all appended to the system prompt at runtime
- Context files tab: upload background images or PDFs (labeled defect photos, research papers) that are sent silently with every message
- Per-message image attachment (📎 button): simulates how a buyer would send a photo via WhatsApp
- "Apply & reset" clears conversation history so each test starts clean — this is intentional and important for prompt iteration
- Uploaded images display as thumbnails in the chat thread and are clickable to enlarge
- API key field at top of page (standalone version only) — key is used only in the browser, sent only to api.anthropic.com

### Models available
- `claude-sonnet-4-6` — default, balanced
- `claude-haiku-4-5-20251001` — fast and cheap, good for high-volume iteration
- `claude-opus-4-6` — most capable, use for final quality checks

### Two versions
1. **Standalone HTML** (`furniture-playground.html` in this folder): requires user to enter their own Anthropic API key. This is the version deployed to GitHub Pages for sharing with co-authors.
2. **Artifact version** (rendered inside a Claude chat): no API key needed, uses claude.ai's built-in API access. Render on demand in Claude Code by asking Claude to display the current file as an artifact.

---

## Technical notes and future plans

### Prompt caching
We plan to implement Anthropic's prompt caching for the production tools. The static part of the system prompt (including any reference images of defects) will be cached so that only the first message in a session is slow; subsequent messages retrieve cached tokens. Cached tokens are ~90% cheaper and significantly faster. This is especially relevant for the quality verifier, which may carry several labeled defect reference images in context.

### Image context strategy
There is a deliberate tension between including reference images (useful for model accuracy) and response latency. The planned approach:
- A small set (5–10) of the most diagnostically distinct defect images cached in the system context
- Rich text descriptions for remaining defects
- User-submitted images come in per-message
- For WhatsApp delivery: favor text descriptions over reference images due to WhatsApp's image compression

### Context files we have / plan to use
- Labeled defect photos from Uganda (to be added to this folder)
- Prior academic papers on furniture quality and East African markets (to be added)
- These will be tested via the Context files tab in the playground before being baked into production system prompts

### App integration
The tools will eventually use dedicated Android apps. The training tool will require user login and track user characteristics to better customized advice. The apps are not yet built; the playground simulates the interaction pattern.

---

## Repo structure (this folder)

This repo has materials for the authors personal website that you can ignore. Focus on the materials in the `/claude/` folder only. 

```
/claude/
  CLAUDE.md                              ← this file
  build.js                               ← build script (see Deployment section)
  furniture-agent-playground.html        ← HTML template (do not edit prompts here — edit the .md files)
  prompts/
    quality-verifier.md                  ← default system prompt for the quality verifier tool
    training-assistant.md                ← default system prompt for the training assistant tool
  context/
    quality-verifier/                    ← context files (images, PDFs) for the quality verifier
    training-assistant/                  ← context files (images, PDFs) for the training assistant
```

The built output is `/furniture-playground.html` at the repo root (deployed to GitHub Pages). **Never edit that file directly** — it is generated by `build.js`.

---

## Deployment

The playground is a single static HTML file built from a template and deployed to the GitHub Pages site at `/furniture-playground`.

### Build step (required before every deploy)

The prompts are stored as separate markdown files and injected into the HTML at build time. To rebuild the output file after editing prompts or the HTML template:

```bash
~/.nvm/versions/node/v24.18.0/bin/node claude/build.js
```

This reads `claude/prompts/quality-verifier.md` and `claude/prompts/training-assistant.md`, injects them into `claude/furniture-agent-playground.html`, and writes the result to `furniture-playground.html` at the repo root.

### Workflow for prompt iteration

1. Edit the relevant file in `claude/prompts/`
2. Run `build.js` to regenerate `furniture-playground.html`
3. Test in the browser (open the file locally)
4. Commit both the `.md` file and the rebuilt `furniture-playground.html` to a feature branch
5. Open a PR — do not push directly to master
6. User reviews and merges the PR manually on GitHub
7. After user confirms the merge, pull master: `git pull origin master`

### Rules for Claude Code
- **Never push directly to master.** Always create a feature branch and open a PR.
- **Never check whether the GitHub Pages deployment is live** — the user handles that.
- After the user confirms a PR is merged, pull master to stay in sync.

---

## Key priorities when making changes

- **Always preserve the "Apply & reset" clean-slate behavior** — this is the core workflow for prompt iteration and must not be broken
- **Keep the tool switcher state isolated** — each tool has its own conversation history and system prompt; switching tools resets the chat for that tool only
- **Per-message images and context file images are separate systems** — do not conflate them
- **The standalone HTML must remain fully self-contained** — no external dependencies except the Anthropic API endpoint; everything else inline
- **Test in both light and dark mode** if making UI changes
