import platform
import re

def translate_path(path: str) -> str:
	if platform.system() == "Windows":
		return path
	elif platform.system() == "Linux":
		# Translate a Windows path to a WSL path
		pattern = re.compile(r"^(?P<drive>[A-Z]):\\(?P<path>.+)")
		match = pattern.match(path)
		if match is None:
			raise Exception(f"Invalid Windows path: {path}")
		drive = match.group("drive").lower()
		windows_path = match.group("path").replace("\\", "/")
		output = f"/mnt/{drive}/{windows_path}"
		return output
	else:
		raise Exception("Unknown operating system")