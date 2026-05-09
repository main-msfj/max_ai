---
name: create-report
description: Create a Word DOCX report in the user workspace from a topic, audience, findings, and recommendations.
---

# Create Report

Use this skill when the user asks for a report, brief, memo, analysis document, or written summary that should become a reusable Word document.

## Workflow

1. Gather the report topic, audience, findings, and recommendations from the conversation.
2. If details are missing, make reasonable assumptions and keep them explicit in the report.
3. Run the script from this skill with `skill_bash`.
4. Save the output as a `.docx` file in `$WORKSPACE_DIR`.

## Script

```bash
python create-report/scripts/create_report.py \
  --title "Q2 Launch Readiness" \
  --audience "Product leadership" \
  --finding "Beta users like the onboarding flow" \
  --finding "Support volume is still high" \
  --recommendation "Ship with a staffed support rotation" \
  --output "q2-launch-readiness.docx"
```

The script prints the created workspace path.
