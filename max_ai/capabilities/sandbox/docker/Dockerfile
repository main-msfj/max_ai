# The image of the sandbox when you give neither ``image`` nor ``dockerfile``.
FROM python:3.12-slim
RUN pip install --no-cache-dir pypdf
# The user the agent's commands run as. /home/agent is not created here:
# the volume mounted there must belong to root, so only the harness writes
# setting.json and the skills.
RUN useradd --uid 1000 --no-create-home agent
