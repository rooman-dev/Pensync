from __future__ import annotations

import base64
import shutil
from pathlib import Path

import webview


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "ui" / "index.html"
PENDING_DIR = BASE_DIR / "data" / "pending"
VERIFIED_DIR = BASE_DIR / "data" / "verified"
REJECTED_DIR = BASE_DIR / "data" / "rejected"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class PensyncAPI:
    def _safe_label(self, char: str) -> str:
        if not char:
            return ""

        symbol = char[0]
        if symbol.isupper():
            return f"upper_{symbol}"
        if symbol.islower():
            return f"lower_{symbol}"
        if symbol.isdigit():
            return symbol
        return f"sym_{ord(symbol)}"

    def _ensure_directories(self) -> None:
        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    def _is_image_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES

    def _unique_destination(self, folder: Path, filename: str) -> Path:
        destination = folder / filename
        if not destination.exists():
            return destination

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        counter = 1
        while True:
            candidate = folder / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _move_with_unique_name(self, source: Path, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = self._unique_destination(destination_dir, source.name)
        shutil.move(str(source), str(destination))
        return destination

    def test_connection(self, message):
        print(f"PensyncAPI received: {message}")
        return "Python backend received your message successfully."

    def get_dashboard_stats(self):
        unique_chars = 0
        total_variants = 0

        try:
            if VERIFIED_DIR.exists():
                for entry in VERIFIED_DIR.iterdir():
                    if not entry.is_dir():
                        continue

                    unique_chars += 1
                    for file_path in entry.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                            total_variants += 1
        except Exception as exc:
            print(f"PensyncAPI.get_dashboard_stats error: {exc}")

        return {"unique_chars": unique_chars, "total_variants": total_variants}

    def get_verification_queue(self):
        self._ensure_directories()

        mock_labels = ["A", "b", "."]
        queue = []

        try:
            pending_files = [path for path in sorted(PENDING_DIR.iterdir(), key=lambda item: item.name.lower()) if self._is_image_file(path)]
            for index, image_path in enumerate(pending_files[:10]):
                image_bytes = image_path.read_bytes()
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                queue.append(
                    {
                        "id": image_path.name,
                        "label": mock_labels[index % len(mock_labels)],
                        "image_base64": encoded_image,
                    }
                )
        except Exception as exc:
            print(f"PensyncAPI.get_verification_queue error: {exc}")

        return queue

    def resolve_verification(self, filename, label, action):
        self._ensure_directories()

        safe_filename = Path(filename).name
        source_path = PENDING_DIR / safe_filename
        normalized_action = str(action).strip().lower()

        if not source_path.exists():
            return {"ok": False, "error": f"Pending file not found: {safe_filename}"}

        try:
            if normalized_action == "approve":
                safe_label = self._safe_label(str(label))
                target_dir = VERIFIED_DIR / safe_label
                moved_path = self._move_with_unique_name(source_path, target_dir)
                return {"ok": True, "action": "approve", "destination": str(moved_path)}

            if normalized_action == "reject":
                moved_path = self._move_with_unique_name(source_path, REJECTED_DIR)
                return {"ok": True, "action": "reject", "destination": str(moved_path)}

            return {"ok": False, "error": f"Unsupported action: {action}"}
        except Exception as exc:
            print(f"PensyncAPI.resolve_verification error: {exc}")
            return {"ok": False, "error": str(exc)}


def main() -> None:
    if not HTML_PATH.exists():
        raise FileNotFoundError(f"UI entry file not found: {HTML_PATH}")

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Launching Pensync V2 UI from {HTML_PATH}")
    webview.create_window(
        title="Pensync V2",
        url=HTML_PATH.as_uri(),
        width=1440,
        height=900,
        resizable=True,
        confirm_close=True,
        js_api=PensyncAPI(),
    )

    try:
        webview.start()
    except KeyboardInterrupt:
        print("Pensync V2 UI closed by user.")


if __name__ == "__main__":
    main()
