---
name: create-ppt
description: Create a PowerPoint PPTX presentation in the user workspace from a title, audience, and slide topics.
---

# Create PPT

Use this skill when the user asks for a slide deck, presentation, pitch, training deck, or PowerPoint file.

## Workflow

1. Decide the title, audience, and main slide points.
2. Run the script with `skill_bash`.
3. The script writes a `.pptx` presentation into `$WORKSPACE_DIR`.
4. Use the `workspace` tool to list or retrieve the generated file later.

## Script

```bash
python create-ppt/scripts/create_ppt.py \
  --title "Customer Renewal Plan" \
  --audience "Sales team" \
  --slide "Current renewal risk" \
  --slide "Account prioritization" \
  --slide "Next actions" \
  --output "customer-renewal-plan.pptx"
```

The generated file is a PowerPoint `.pptx`.
