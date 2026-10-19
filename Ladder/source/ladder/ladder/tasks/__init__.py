"""Task registrations for Ladder."""

_BLACKLIST_PKGS = ["utils"]

try:
    from isaaclab_tasks.utils import import_packages
except ModuleNotFoundError:  # pragma: no cover - pure preprocessing does not need IsaacLab.
    import_packages = None

if import_packages is not None:
    import_packages(__name__, _BLACKLIST_PKGS)
