#!/bin/sh
set -e

echo "Pulling models: ${MODELS_TO_PULL}"

IFS=','
for model in ${MODELS_TO_PULL}; do
  model=$(echo "${model}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [ -z "${model}" ] && continue

  if ollama list | awk '{print $1}' | grep -Fxq "${model}"; then
    echo "  [skip] ${model} already present"
  else
    echo "  [pull] ${model}"
    ollama pull "${model}"
  fi
done

echo ""
echo "Done. Installed models:"
ollama list