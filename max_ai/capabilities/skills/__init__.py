"""Where skills come from."""

from .github import GitHubSkills, GitHubSkillsConfig
from .local import LocalSkills, LocalSkillsConfig

__all__ = ["GitHubSkills", "GitHubSkillsConfig", "LocalSkills", "LocalSkillsConfig"]
