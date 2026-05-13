---
name: create-report
description: Create a general-purpose Word DOCX report, brief, memo, article, or structured written document in the user workspace.
---

# Create Report

Use this skill when the user asks for a report, brief, memo, article, analysis document, study note, proposal, documentation, or written summary that should become a reusable Word document.

## Workflow

1. Infer the document purpose, audience, tone, structure, and level of detail from the request.
2. Choose the format that best fits the task: narrative paragraphs, concise sections, bullets, recommendations, references, appendix notes, or any mix.
3. If key details are missing, make reasonable assumptions and include them naturally only when helpful.
4. Run the script from this skill with `skill_bash`.
5. Save the output as a `.docx` file in `$WORKSPACE_DIR`.

## Content Model

- Use `--paragraph` for standalone intro, summary, or closing text.
- Use `--section "Heading::body"` for titled sections.
- Use `--bullet "Heading::item"` for bullet lists under a heading.
- Repeat arguments as needed. The order in the final document follows the argument order.
- The script creates a valid `.docx`; the LLM is responsible for choosing appropriate content, style, headings, and filenames.

## Script

```bash
python create-report/scripts/create_report.py \
  --title "Q2 Launch Readiness" \
  --paragraph "This brief summarizes launch readiness across product, support, and go-to-market workstreams." \
  --section "Current State::Beta users respond well to onboarding, but support volume remains above the target threshold." \
  --bullet "Recommended Actions::Staff an expanded support rotation for the first two weeks." \
  --bullet "Recommended Actions::Resolve the three highest-volume help center gaps before launch." \
  --output "q2-launch-readiness.docx"
```

The script prints the created workspace path.
