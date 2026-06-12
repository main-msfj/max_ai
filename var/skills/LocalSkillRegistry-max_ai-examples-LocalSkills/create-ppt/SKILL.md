---
name: create-ppt
description: Create a general-purpose PowerPoint PPTX presentation in the user workspace for pitches, lessons, reports, briefs, workshops, or storytelling decks.
---

# Create PPT

Use this skill when the user asks for a slide deck, presentation, pitch, training deck, lesson, workshop, report deck, visual brief, or PowerPoint file.

## Workflow

1. Infer the purpose, audience, tone, narrative arc, slide count, and level of detail from the request.
2. Choose the deck style that fits: executive summary, teaching deck, pitch, proposal, research recap, project update, workshop, or any other useful format.
3. Write slide titles and bodies directly; include bullets, short speaker prompts, examples, or calls to action when useful.
4. Run the script with `bash`.
5. The script writes a `.pptx` presentation into the user workspace.

## Content Model

- Use `--subtitle` for optional title-slide context.
- Use `--slide "Title::body"` for each slide.
- Use new lines inside the body when useful; PowerPoint will preserve them.
- The script creates a valid `.pptx`; the LLM is responsible for choosing appropriate slide count, structure, tone, and filenames.

## Script

```bash
python "skills/create-ppt/scripts/create_ppt.py" \
  --title "Customer Renewal Plan" \
  --subtitle "A practical operating plan for the sales team" \
  --slide "Current Renewal Risk::Three enterprise accounts need executive attention this quarter." \
  --slide "Account Prioritization::Focus first on renewal value, product fit, and unresolved support history." \
  --slide "Next Actions::Confirm owners, schedule customer check-ins, and review progress weekly." \
  --output "customer-renewal-plan.pptx"
```

The generated file is a PowerPoint `.pptx`.
