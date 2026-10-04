"""Agent profiles: one module per agent. A new agent needs only a new profile."""

from collections.abc import Iterator, Sequence
from typing import Any

from blackbox.otlp.decode import SpanData
from blackbox.profiles.base import Profile, StartRequest
from blackbox.profiles.opsdesk import OpsDeskProfile
from blackbox.profiles.paperpilot import PaperPilotProfile

__all__ = ["Profile", "ProfileRegistry", "StartRequest", "default_registry"]

BUILTIN: list[type[Profile]] = [PaperPilotProfile, OpsDeskProfile]


class ProfileRegistry:
    def __init__(self, profiles: Sequence[Profile] = ()) -> None:
        self._profiles: dict[str, Profile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: Profile) -> None:
        if not profile.name:
            raise ValueError("a profile needs a name")
        self._profiles[profile.name] = profile

    def get(self, name: str) -> Profile:
        try:
            return self._profiles[name]
        except KeyError:
            raise KeyError(f"no profile {name!r} (known: {', '.join(sorted(self._profiles))})") from None

    def find(self, name: str | None) -> Profile | None:
        return self._profiles.get(name) if name else None

    def match(self, spans: Sequence[SpanData]) -> Profile | None:
        for profile in self._profiles.values():
            if profile.matches(spans):
                return profile
        return None

    def names(self) -> list[str]:
        return sorted(self._profiles)

    def __iter__(self) -> Iterator[Profile]:
        return iter(self._profiles.values())


def default_registry(options: dict[str, dict[str, Any]] | None = None) -> ProfileRegistry:
    options = options or {}
    return ProfileRegistry([cls(options.get(cls.name)) for cls in BUILTIN])
